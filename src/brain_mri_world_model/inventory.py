"""Resumable file/header inventory; no row implies that an image is training-ready."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
import datetime as dt
import fnmatch
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
import uuid


SCANNER_VERSION = "2"
EXECUTION_VERSION = "3"
IMAGE_SUFFIXES = (".nii.gz", ".nii", ".mgz", ".mgh")
ARCHIVE_SUFFIXES = (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst", ".tgz", ".tar", ".zip", ".7z", ".rar", ".zst")
MODALITY_PATTERNS = (
    ("FLAIR", r"flair"),
    ("T1w", r"(?:^|[^a-z0-9])t1(?:w|weighted)?(?:$|[^a-z0-9])|mprage|mp2rage|spgr|t1c"),
    ("T2w", r"(?:^|[^a-z0-9])t2(?:w|weighted)?(?:$|[^a-z0-9])|t2_space"),
    ("PD", r"(?:^|[^a-z0-9])pd(?:w)?(?:$|[^a-z0-9])|proton.density"),
    ("SWI", r"(?:^|[^a-z0-9])swi(?:$|[^a-z0-9])|susceptibility.weighted"),
    ("T2star", r"t2star|t2\*|t2_star"),
    ("DWI", r"(?:^|[^a-z0-9])(?:dwi|dti|diffusion|b0|b1000)(?:$|[^a-z0-9])"),
    ("fMRI", r"(?:^|[^a-z0-9])(?:bold|fmri|rest)(?:$|[^a-z0-9])"),
    ("TOF", r"(?:^|[^a-z0-9])tof(?:$|[^a-z0-9])|time.of.flight"),
    ("ASL", r"(?:^|[^a-z0-9])asl(?:$|[^a-z0-9])|arterial.spin"),
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def file_kind(path):
    name = str(path).lower()
    if name.endswith(IMAGE_SUFFIXES):
        return "image"
    if name.endswith(ARCHIVE_SUFFIXES):
        return "unexpanded_archive"
    return None


def subject_hint(path):
    """Only explicit BIDS subject labels; no arbitrary folder-name inference."""
    for part in Path(path).parts:
        match = re.fullmatch(r"sub-([A-Za-z0-9]+)", part)
        if match:
            return "sub-" + match.group(1), "explicit_bids_directory"
    match = re.match(r"sub-([A-Za-z0-9]+)(?:_|\.)", Path(path).name)
    return (("sub-" + match.group(1), "explicit_bids_filename") if match else (None, "unknown"))


def classify_candidate(path):
    """Conservative hints, never a certified modality or brain/anatomy label."""
    name = Path(path).name
    for suffix in IMAGE_SUFFIXES:
        if name.lower().endswith(suffix):
            name = name[:-len(suffix)]
            break
    sidecar_path = Path(path).with_name(name + ".json")
    sources = [("filename", name)]
    sidecar_status = "absent"
    if sidecar_path.is_file():
        try:
            if sidecar_path.stat().st_size > 2 * 1024**2:
                sidecar_status = "too_large_not_read"
            else:
                metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if not isinstance(metadata, dict):
                    raise ValueError("sidecar is not an object")
                sidecar_status = "read"
                # Free text may contain sensitive information: never persist it.
                for field in ("SeriesDescription", "ProtocolName", "SequenceName", "ImageType"):
                    value = metadata.get(field)
                    if isinstance(value, list):
                        value = " ".join(str(item) for item in value)
                    if isinstance(value, str):
                        sources.append(("sidecar:" + field, value))
        except Exception as error:
            sidecar_status = "unreadable:" + type(error).__name__
    joined = " ".join(value for _, value in sources).lower()
    # Exclusion classes take precedence over apparent structural contrast.
    if re.search(r"\.(?:dscalar|dtseries|dconn|dlabel|pconn|ptseries|pdconn|pscalar)\.nii$", str(path).lower()):
        image_class = "cifti_nonvolume"
    elif re.search(r"(?:^|[^a-z0-9])(?:warp|warp_coef|warpfield|deformation|displacement|flow|jac|jacobian)(?:$|[^a-z0-9])", joined):
        image_class = "transform_or_jacobian_candidate"
    elif re.search(r"(?:^|[^a-z0-9])(?:mask|brainmask|brain_mask|seg|segmentation|dseg|aseg|aparc|label|labels|parcellation|probseg|probability|skull|fov)(?:$|[^a-z0-9])", joined):
        image_class = "mask_label_or_segmentation_candidate"
    elif re.search(r"fieldmap|field_map|(?:^|[^a-z0-9])(?:fmap|phasediff|phase1|phase2|magnitude1|magnitude2)(?:$|[^a-z0-9])", joined):
        image_class = "fieldmap_candidate"
    elif re.search(r"(?:^|[^a-z0-9])(?:adc|fa|md|rd|ad|trace|tracew|cbf|cbv|tmax|r2star|r2s|dti|noddi|icvf|isovf|od|odi|l1|l2|l3|vbm|gm_den|gm_mod|skeletonised|skeletonized)(?:$|[^a-z0-9])", joined):
        image_class = "derived_map_candidate"
    else:
        image_class = "mri_or_other_image_unverified"
    matches, evidence = set(), set()
    for source, value in sources:
        for modality, pattern in MODALITY_PATTERNS:
            if re.search(pattern, value.lower()):
                matches.add(modality)
                evidence.add(source)
    if image_class == "derived_map_candidate":
        derived_patterns = (
            ("ADC", r"(?:^|[^a-z0-9])adc(?:$|[^a-z0-9])"),
            ("FA", r"(?:^|[^a-z0-9])fa(?:$|[^a-z0-9])"),
            ("MD", r"(?:^|[^a-z0-9])md(?:$|[^a-z0-9])"),
            ("DTI_L1", r"(?:^|[^a-z0-9])l1(?:$|[^a-z0-9])"),
            ("DTI_L2", r"(?:^|[^a-z0-9])l2(?:$|[^a-z0-9])"),
            ("DTI_L3", r"(?:^|[^a-z0-9])l3(?:$|[^a-z0-9])"),
            ("NODDI_ICVF", r"icvf"), ("NODDI_ISOVF", r"isovf"),
            ("NODDI_OD", r"(?:^|[^a-z0-9])(?:od|odi)(?:$|[^a-z0-9])"),
            ("GM_density", r"gm_den"), ("GM_modulated", r"gm_mod"),
        )
        derived_matches = {label for label, pattern in derived_patterns if re.search(pattern, joined)}
        if derived_matches:
            matches = derived_matches
            evidence.add("derived_map_name_candidate")
    # A FLAIR filename often includes "T2"; retain its more specific label.
    if "FLAIR" in matches:
        matches.discard("T2w")
    modality = next(iter(matches)) if len(matches) == 1 else ("ambiguous" if matches else "unknown")
    return dict(
        image_class=image_class,
        modality_candidate=modality,
        modality_candidates=sorted(matches),
        modality_certainty="heuristic_candidate" if len(matches) == 1 else "uncertain",
        modality_evidence=sorted(evidence),
        sidecar_status=sidecar_status,
        sidecar_path=str(sidecar_path) if sidecar_path.exists() else None,
        brain_anatomy_status="unverified",
        registration_status="unverified",
        skullstrip_status="unverified",
    )


def path_selected(relative_path, include_globs, exclude_globs):
    """Patterns apply to the root-relative POSIX path; '*' may span directories."""
    relative_path = str(relative_path).replace(os.sep, "/")
    return ((not include_globs or any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in include_globs))
            and not any(fnmatch.fnmatchcase(relative_path, pattern) for pattern in exclude_globs))


def validation_fingerprint(path, validate_voxels, max_validation_gib):
    stat = os.stat(path)
    sidecar = Path(path)
    for suffix in IMAGE_SUFFIXES:
        if str(sidecar).lower().endswith(suffix):
            sidecar = Path(str(sidecar)[:-len(suffix)] + ".json")
            break
    sidecar_stat = sidecar.stat() if sidecar.exists() else None
    return json.dumps(dict(
        version=SCANNER_VERSION, size=stat.st_size, mtime_ns=stat.st_mtime_ns,
        observed_path=str(path),
        sidecar_size=sidecar_stat.st_size if sidecar_stat else None,
        sidecar_mtime_ns=sidecar_stat.st_mtime_ns if sidecar_stat else None,
        validate_voxels=validate_voxels, max_validation_gib=max_validation_gib,
    ), sort_keys=True)


def inspect_image(path, validate_voxels=False, max_validation_gib=2.0):
    import nibabel as nib
    import numpy as np

    result = classify_candidate(path)
    subject, subject_basis = subject_hint(path)
    result.update(subject_id=subject, subject_basis=subject_basis,
                  header_status="not_checked", header_candidate_3d=False,
                  voxel_status="not_requested", training_ready=False)
    try:
        image = nib.load(str(path), mmap="r", keep_file_open=False)
        if isinstance(image, nib.Cifti2Image):
            result.update(image_class="cifti_nonvolume", header_status="excluded_cifti",
                          dimensionality="nonvolume_cifti")
            return result
        shape = tuple(int(value) for value in image.shape)
        result.update(shape=list(shape), dtype=str(image.get_data_dtype()),
                      voxel_sizes=[float(value) for value in image.header.get_zooms()],
                      affine=np.asarray(image.affine).tolist())
        result["dimensionality"] = (
            "3d" if len(shape) == 3 and min(shape) > 1 else
            "4d_singleton" if len(shape) == 4 and shape[3] == 1 and min(shape[:3]) > 1 else
            "4d_multivolume" if len(shape) == 4 and shape[3] > 1 else "unsupported"
        )
        affine = np.asarray(image.affine)
        affine_valid = bool(affine.shape == (4, 4) and np.isfinite(affine).all()
                            and abs(np.linalg.det(affine[:3, :3])) > 1e-12
                            and np.allclose(affine[3], [0, 0, 0, 1]))
        spacing_valid = bool(len(result["voxel_sizes"]) >= 3 and all(
            math.isfinite(value) and value > 0 for value in result["voxel_sizes"][:3]))
        result.update(affine_valid=affine_valid, spacing_valid=spacing_valid)
        try:
            result["orientation"] = list(nib.aff2axcodes(affine)) if affine_valid else None
        except Exception:
            result["orientation"] = None
        result.update(qform_code=None, sform_code=None, qform_sform_disagree=None,
                      spatial_units=None, temporal_units=None)
        if hasattr(image.header, "get_xyzt_units"):
            result["spatial_units"], result["temporal_units"] = image.header.get_xyzt_units()
        if hasattr(image, "get_qform"):
            qform, qcode = image.get_qform(coded=True)
            sform, scode = image.get_sform(coded=True)
            result.update(qform_code=int(qcode), sform_code=int(scode))
            if qcode and scode:
                result["qform_sform_disagree"] = not bool(np.allclose(qform, sform, atol=1e-3))
        result["header_candidate_3d"] = bool(
            result["dimensionality"] in ("3d", "4d_singleton") and affine_valid and spacing_valid
            and result["image_class"] not in ("cifti_nonvolume", "transform_or_jacobian_candidate",
                                             "mask_label_or_segmentation_candidate", "fieldmap_candidate"))
        if len(shape) == 3 and min(shape) == 1 and str(path).lower().endswith((".mgh", ".mgz")):
            result["image_class"] = "surface_or_nonvolumetric_mgh"
        result["header_status"] = "read"
        if not validate_voxels:
            return result
        if not result["header_candidate_3d"]:
            result["voxel_status"] = "not_checked_ineligible_header"
            return result
        n_voxels = math.prod(shape)
        result["n_voxels"] = n_voxels
        if n_voxels * 4 > max_validation_gib * 1024**3:
            result["voxel_status"] = "deferred_memory_limit"
            return result
        # Read every voxel, including gzip decompression; no sampling is used.
        array = image.get_fdata(dtype=np.float32, caching="unchanged").reshape(-1)
        nonfinite_count, nonzero_count = 0, 0
        minimum, maximum = math.inf, -math.inf
        for start in range(0, array.size, 1024**2):
            chunk = array[start:start + 1024**2]
            finite = np.isfinite(chunk)
            nonfinite_count += int(chunk.size - np.count_nonzero(finite))
            values = chunk[finite]
            if values.size:
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
                nonzero_count += int(np.count_nonzero(values))
        result.update(nonfinite_count=nonfinite_count, nonzero_count=nonzero_count,
                      value_min=minimum if math.isfinite(minimum) else None,
                      value_max=maximum if math.isfinite(maximum) else None)
        result["voxel_status"] = (
            "failed_nonfinite" if nonfinite_count else
            "failed_constant" if minimum == maximum else "read_passed_basic_checks")
    except Exception as error:
        # Keep a failed row rather than silently dropping a damaged file.
        result["error_type"] = type(error).__name__
        result["error"] = str(error)[:1000]
        if result["header_status"] != "read":
            result["header_status"] = "failed"
        else:
            result["voxel_status"] = "failed_read"
    return result


def create_database(path):
    connection = sqlite3.connect(str(path), timeout=60)
    connection.executescript("""
        PRAGMA busy_timeout=60000;
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, dataset TEXT NOT NULL, root TEXT NOT NULL,
            started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
            options_json TEXT NOT NULL, counters_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS files (
            dataset TEXT NOT NULL, canonical_path TEXT NOT NULL, observed_path TEXT NOT NULL,
            kind TEXT NOT NULL, fingerprint TEXT NOT NULL, result_json TEXT NOT NULL,
            checked_at TEXT NOT NULL, PRIMARY KEY (dataset, canonical_path)
        );
        CREATE TABLE IF NOT EXISTS observations (
            run_id TEXT NOT NULL, dataset TEXT NOT NULL, canonical_path TEXT NOT NULL,
            PRIMARY KEY (run_id, canonical_path)
        );
        CREATE TABLE IF NOT EXISTS traversal_errors (
            run_id TEXT NOT NULL, path TEXT NOT NULL, error_type TEXT NOT NULL, error TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS observations_run_idx ON observations(run_id);
    """)
    connection.commit()
    return connection


def walk_entries(root, follow_directory_symlinks, on_error, counters):
    stack, visited = [str(root)], set()
    while stack:
        directory = stack.pop()
        try:
            stat = os.stat(directory)
            identity = (stat.st_dev, stat.st_ino)
            if identity in visited:
                counters["duplicate_or_loop_directories_skipped"] += 1
                continue
            visited.add(identity)
            with os.scandir(directory) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=True):
                            if entry.is_symlink() and not follow_directory_symlinks:
                                counters["directory_symlinks_skipped"] += 1
                            else:
                                stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=True):
                            counters["all_file_entries_seen"] += 1
                            if file_kind(entry.path):
                                yield entry.path
                        elif entry.is_symlink():
                            counters["broken_or_nonfile_symlinks"] += 1
                    except OSError as error:
                        on_error(entry.path, error)
        except OSError as error:
            on_error(directory, error)


def scan_dataset(connection, dataset, root, args):
    root = Path(root).expanduser().absolute()
    run_id = uuid.uuid4().hex
    counters = dict(all_file_entries_seen=0, unique_supported_files=0, duplicate_file_aliases=0,
                    cached_rows_reused=0, rows_inspected=0, traversal_errors=0,
                    directory_symlinks_skipped=0, duplicate_or_loop_directories_skipped=0,
                    broken_or_nonfile_symlinks=0, supported_file_entries_discovered=0,
                    excluded_by_glob=0)
    options = dict(validate_voxels=args.validate_voxels,
                   max_validation_gib=args.max_validation_gib,
                   follow_directory_symlinks=args.follow_directory_symlinks,
                   include_globs=args.include_glob, exclude_globs=args.exclude_glob,
                   commit_every=args.commit_every,
                   scanner_version=SCANNER_VERSION, execution_version=EXECUTION_VERSION,
                   workers=args.workers)
    connection.execute("INSERT INTO runs(run_id,dataset,root,started_at,status,options_json) VALUES(?,?,?,?,?,?)",
                       (run_id, dataset, str(root), utc_now(), "running", json.dumps(options)))
    connection.commit()

    def checkpoint(status=None):
        if status:
            connection.execute("UPDATE runs SET finished_at=?,status=?,counters_json=? WHERE run_id=?",
                               (utc_now(), status, json.dumps(counters), run_id))
        else:
            connection.execute("UPDATE runs SET counters_json=? WHERE run_id=?", (json.dumps(counters), run_id))
        connection.commit()

    def on_error(path, error):
        counters["traversal_errors"] += 1
        connection.execute("INSERT INTO traversal_errors VALUES(?,?,?,?)",
                           (run_id, str(path), type(error).__name__, str(error)[:1000]))
        checkpoint()

    def prepare_file(path):
        canonical = str(Path(path).resolve(strict=True))
        return dict(path=path, canonical=canonical, kind=file_kind(path),
                    fingerprint=validation_fingerprint(path, args.validate_voxels, args.max_validation_gib),
                    size_bytes=os.stat(path).st_size)

    def inspect_prepared(prepared):
        if prepared["kind"] == "image":
            result = inspect_image(prepared["path"], args.validate_voxels, args.max_validation_gib)
        else:
            result = dict(archive_status="not_expanded_not_counted_as_images",
                          header_candidate_3d=False, training_ready=False)
        result["size_bytes"] = prepared["size_bytes"]
        return result

    def record_file(prepared, result=None):
        if result is not None:
            connection.execute("""INSERT INTO files VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(dataset,canonical_path) DO UPDATE SET observed_path=excluded.observed_path,
                kind=excluded.kind,fingerprint=excluded.fingerprint,result_json=excluded.result_json,
                checked_at=excluded.checked_at""",
                (dataset, prepared["canonical"], prepared["path"], prepared["kind"], prepared["fingerprint"],
                 json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False), utc_now()))
            counters["rows_inspected"] += 1
        else:
            counters["cached_rows_reused"] += 1
        connection.execute("INSERT OR IGNORE INTO observations VALUES(?,?,?)", (run_id, dataset, prepared["canonical"]))
        counters["unique_supported_files"] += 1
        progress_due = counters["unique_supported_files"] % args.progress_every == 0
        if counters["unique_supported_files"] % args.commit_every == 0 or progress_due:
            checkpoint()
        if progress_due:
            print(json.dumps(dict(dataset=dataset, run_id=run_id, **counters)), flush=True)

    seen, pending = set(), {}
    executor = ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="mri-inventory")
    try:
        if not root.is_dir():
            on_error(root, FileNotFoundError("dataset root is not an accessible directory"))
        else:
            paths = iter(walk_entries(root, args.follow_directory_symlinks, on_error, counters))
            exhausted = False
            while pending or not exhausted:
                while len(pending) < 2 * args.workers and not exhausted:
                    try:
                        path = next(paths)
                    except StopIteration:
                        exhausted = True
                        break
                    counters["supported_file_entries_discovered"] += 1
                    if not path_selected(os.path.relpath(path, root), args.include_glob, args.exclude_glob):
                        counters["excluded_by_glob"] += 1
                        continue
                    pending[executor.submit(prepare_file, path)] = ("prepare", path)
                if not pending:
                    continue
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    stage, payload = pending.pop(future)
                    try:
                        value = future.result()
                        if stage == "inspect":
                            record_file(payload, value)
                            continue
                        prepared = value
                        canonical = prepared["canonical"]
                        if canonical in seen:
                            counters["duplicate_file_aliases"] += 1
                            continue
                        seen.add(canonical)
                        prior = connection.execute("SELECT fingerprint FROM files WHERE dataset=? AND canonical_path=?",
                                                   (dataset, canonical)).fetchone()
                        if prior and prior[0] == prepared["fingerprint"] and not args.force:
                            record_file(prepared)
                        else:
                            pending[executor.submit(inspect_prepared, prepared)] = ("inspect", prepared)
                    except OSError as error:
                        on_error(payload if stage == "prepare" else payload["path"], error)
        status = "completed" if counters["traversal_errors"] == 0 else "completed_with_traversal_errors"
    except BaseException:
        # Preserve successfully processed rows in the last partial batch on an orderly
        # exception/KeyboardInterrupt. A process kill may lose the current batch only.
        checkpoint("interrupted")
        raise
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)
    checkpoint(status)
    return run_id


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def export_reports(connection, output_dir, run_ids):
    columns = ["dataset", "canonical_path", "observed_path", "kind", "checked_at", "subject_id",
               "subject_basis", "image_class", "modality_candidate", "modality_certainty",
               "modality_evidence", "shape", "voxel_sizes", "dtype", "orientation", "spatial_units",
               "temporal_units", "affine", "affine_valid", "spacing_valid", "qform_code", "sform_code",
               "qform_sform_disagree", "dimensionality", "header_status", "header_candidate_3d",
               "voxel_status", "n_voxels", "nonfinite_count", "nonzero_count", "value_min", "value_max",
               "brain_anatomy_status", "registration_status", "skullstrip_status", "sidecar_status",
               "size_bytes", "archive_status", "training_ready", "error_type", "error"]
    summaries = []
    csv_path = output_dir / "inventory.csv"
    temporary_csv = csv_path.with_suffix(".csv.tmp")
    with temporary_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for run_id in run_ids:
            run = connection.execute("SELECT dataset,root,started_at,finished_at,status,counters_json,options_json FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if not run:
                continue
            summary = dict(run_id=run_id, dataset=run[0], root=run[1], started_at=run[2],
                           finished_at=run[3], status=run[4], traversal=json.loads(run[5]),
                           options=json.loads(run[6]), image_files=0, unexpanded_archives=0,
                           header_candidates_3d=0, voxel_checked_candidates_3d=0,
                           voxel_basic_check_passes_3d=0, training_ready_images=0,
                           explicit_subject_ids_among_3d_candidates=0,
                           candidates_3d_without_explicit_subject_id=0, counts_by_candidate_modality={})
            subjects = set()
            rows = connection.execute("""SELECT f.dataset,f.canonical_path,f.observed_path,f.kind,f.checked_at,f.result_json
                 FROM observations o JOIN files f ON f.dataset=o.dataset AND f.canonical_path=o.canonical_path
                 WHERE o.run_id=? ORDER BY f.canonical_path""", (run_id,))
            for dataset, canonical, observed, kind, checked_at, result_json in rows:
                result = json.loads(result_json)
                row = dict(result, dataset=dataset, canonical_path=canonical, observed_path=observed,
                           kind=kind, checked_at=checked_at)
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                                 for key, value in row.items() if key in columns})
                if kind != "image":
                    summary["unexpanded_archives"] += 1
                    continue
                summary["image_files"] += 1
                modality = result.get("modality_candidate", "unknown")
                bucket = summary["counts_by_candidate_modality"].setdefault(modality,
                    dict(image_files=0, header_candidates_3d=0, voxel_basic_check_passes_3d=0,
                         excluded_class_candidates_3d=0))
                bucket["image_files"] += 1
                if result.get("header_candidate_3d"):
                    summary["header_candidates_3d"] += 1
                    bucket["header_candidates_3d"] += 1
                    if result.get("image_class") != "mri_or_other_image_unverified":
                        bucket["excluded_class_candidates_3d"] += 1
                    if result.get("subject_id"):
                        subjects.add(result["subject_id"])
                    else:
                        summary["candidates_3d_without_explicit_subject_id"] += 1
                    if result.get("voxel_status", "").startswith(("read_", "failed_")):
                        summary["voxel_checked_candidates_3d"] += 1
                    if result.get("voxel_status") == "read_passed_basic_checks":
                        summary["voxel_basic_check_passes_3d"] += 1
                        bucket["voxel_basic_check_passes_3d"] += 1
            summary["explicit_subject_ids_among_3d_candidates"] = len(subjects)
            summary["subject_count_is_complete"] = summary["candidates_3d_without_explicit_subject_id"] == 0
            summaries.append(summary)
    temporary_csv.replace(csv_path)
    report = dict(generated_at=utc_now(), scanner_version=SCANNER_VERSION, datasets=summaries,
                  scope_notes=[
                      "Header candidates are geometric file candidates, not certified brain MRI or training-ready images.",
                      "Voxel checks cover every voxel of eligible volumes when requested, within the configured memory limit.",
                      "Modality and image-class labels are heuristic candidates; review real dataset metadata before selection.",
                      "Explicit BIDS subject labels are counted within each dataset; no cross-cohort identity matching is inferred.",
                      "Archives are listed but not expanded or counted as image volumes.",
                      "Paths are deduplicated by realpath within a dataset; copied/hardlinked content and cross-dataset duplicates require later checks.",
                      "Glob filters restrict header checks and counts; traversal counters separately report discovered supported entries and filtered entries.",
                      "Registration, skull stripping, brain anatomy and visual quality remain unverified.",
                  ])
    atomic_json(output_dir / "summary.json", report)
    return report


def parse_dataset(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=ROOT")
    name, root = value.split("=", 1)
    if not name.strip() or not root.strip():
        raise argparse.ArgumentTypeError("NAME and ROOT must be nonempty")
    return name, root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=parse_dataset, action="append", required=True,
                        help="Dataset name and root, NAME=ROOT; repeat for multiple datasets.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Private output directory for resumable SQLite, CSV and summary JSON.")
    parser.add_argument("--validate-voxels", action="store_true",
                        help="Fully read eligible 3D/singleton-4D volumes for basic numeric checks.")
    parser.add_argument("--max-validation-gib", type=float, default=2.0,
                        help="Maximum float32 voxel array size per image; larger images are deferred (default 2 GiB).")
    parser.add_argument("--follow-directory-symlinks", action="store_true",
                        help="Traverse linked directories, with inode-based loop protection; root links always work.")
    parser.add_argument("--include-glob", action="append", default=[],
                        help="Only inspect matching root-relative paths; repeat for OR matching. Shell-quote patterns. '*' spans directories.")
    parser.add_argument("--exclude-glob", action="append", default=[],
                        help="Skip matching root-relative paths before header reads; repeat for OR matching. Exclusions take priority.")
    parser.add_argument("--force", action="store_true", help="Reinspect unchanged files instead of using the cache.")
    parser.add_argument("--progress-every", type=int, default=1000,
                        help="Print progress every N unique image/archive files (default 1000).")
    parser.add_argument("--commit-every", type=int, default=256,
                        help="Commit each N complete rows and at progress/final checkpoints (default 256).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Concurrent file metadata/header readers; SQLite writes stay in the main thread, at most 2*N pending tasks (default 8).")
    args = parser.parse_args(argv)
    if args.max_validation_gib <= 0 or args.progress_every <= 0 or args.commit_every <= 0 or args.workers <= 0:
        parser.error("--max-validation-gib, --progress-every, --commit-every and --workers must be positive")
    if len({name for name, _ in args.dataset}) != len(args.dataset):
        parser.error("each --dataset NAME must be unique within one invocation")
    # Fail before recording an apparently successful all-failure scan if packages are absent.
    try:
        import nibabel  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as error:
        parser.error(f"Missing required dependency: {error}. Install nibabel and numpy in the project Conda environment.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    connection = create_database(args.output_dir / "inventory.sqlite")
    run_ids = []
    started = time.monotonic()
    try:
        for dataset, root in args.dataset:
            run_ids.append(scan_dataset(connection, dataset, root, args))
    except KeyboardInterrupt:
        print("Interrupted; completed rows remain in inventory.sqlite. Rerun the same command to resume.", file=sys.stderr)
        return 130
    finally:
        if run_ids:
            report = export_reports(connection, args.output_dir, run_ids)
            print(json.dumps(dict(summary_path=str(args.output_dir / "summary.json"),
                                  elapsed_seconds=round(time.monotonic() - started, 3),
                                  datasets=[dict(dataset=item["dataset"], status=item["status"],
                                                 image_files=item["image_files"],
                                                 header_candidates_3d=item["header_candidates_3d"])
                                            for item in report["datasets"]]), ensure_ascii=False), flush=True)
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
