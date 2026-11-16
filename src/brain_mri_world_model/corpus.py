"""合并多个完整编码批次；保留各批合同，不读取原始 MRI 或临床标签。"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable

CORPUS_VERSION = "brain_mri_corpus_v1"
CORPUS_COMPLETE_VERSION = "brain_mri_corpus_complete_v1"
ENCODING_VERSION = "brain_mri_encoding_v2"
SAFE_ROW_FIELDS = frozenset({
    "dataset", "subject_id", "image_id", "modality", "axis", "split",
    "latent_path", "latent_shape", "approved", "qc_passed",
    "encoding_contract_sha256", "latent_sha256",
})
_AXIS_FIELDS = frozenset({"axis", "latent_path", "latent_shape", "latent_sha256"})
_CHILD_FIELDS = frozenset({
    "manifest_path", "manifest_sha256", "encoding_contract_path", "encoding_contract_sha256",
    "complete_path", "complete_sha256", "records", "axes",
})
_COMPATIBILITY_FIELDS = (
    "vae_checkpoint_sha256", "space", "spatial_shape", "affine", "dtype", "padding",
    "valid_token_mask_definition",
    "video_spatial_scale", "video_raster_policy",
)
SingleLoader = Callable[[str | Path], tuple[list[dict[str, Any]], str]]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_sha(value: Any, name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"{name} 必须为小写 SHA-256")
    return value


def _json_file(path: Path) -> tuple[dict[str, Any], str]:
    from .integrity import file_sha256, fingerprint
    before = fingerprint(path)
    data = path.read_bytes()
    result = json.loads(data)
    if not isinstance(result, dict):
        raise ValueError(f"合同或完成标记必须为 JSON 对象：{path.name}")
    sha = file_sha256(path)
    if fingerprint(path)!=before:
        raise ValueError('控制文件在读取核验期间发生变化')
    return result, sha


def _resolve_path(value: Any, base: Path, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须为非空路径字符串")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def _source_key(row: dict[str, Any]) -> str:
    encoded = json.dumps([row["dataset"], row["image_id"]], ensure_ascii=False, separators=(",", ":"))
    return _sha(encoded.encode())


def _expected_shape(shape: list[int], axis: int, scale: int = 1) -> list[int]:
    from .wan import latent_shape_for_video
    return latent_shape_for_video(shape,axis,scale)


def _verified_child(manifest_path: Path, loader: SingleLoader, *, audit_latents: bool = False) -> dict[str, Any]:
    """默认只验证清单/合同/账本；显式audit才读latent，始终不读原MRI。"""
    from .wan import (CANONICAL_DATASETS, WAN22_VAE_SHA256, _audit_latent_file,file_sha256,verify_used_images,
        representation_of,encoding_reference_geometry,expected_record_latent_shape)
    manifest_path = manifest_path.resolve()
    if (manifest_path.parent / "CORPUS_COMPLETE").exists() or (manifest_path.parent / "corpus_contract.json").exists():
        raise ValueError("不允许把 corpus 作为编码子批次，禁止嵌套或循环")
    records, manifest_sha = loader(manifest_path)
    if any("source_encoding_manifest" in row for row in records):
        raise ValueError("编码子批次不能携带 corpus 来源字段")
    contract_path, complete_path = manifest_path.parent / "encoding_contract.json", manifest_path.parent / "COMPLETE"
    if not contract_path.is_file() or not complete_path.is_file():
        raise ValueError("只接受有 encoding_contract.json 与 COMPLETE 的完整 prepare 批次")
    contract, contract_sha = _json_file(contract_path)
    complete, complete_sha = _json_file(complete_path)
    if contract.get("schema_version") != ENCODING_VERSION:
        raise ValueError("编码子合同版本不支持")
    if (complete.get("schema_version") != "brain_mri_encoding_complete_v2" or
            complete.get("manifest_sha256") != manifest_sha or complete.get("contract_sha256") != contract_sha or
            type(complete.get("records")) is not int or complete["records"] != len(records)):
        raise ValueError("编码子批次的 COMPLETE、合同、清单或数量不一致")
    axes = contract.get("axes")
    if (not isinstance(axes, list) or not axes or any(type(axis) is not int or axis not in (0, 1, 2) for axis in axes) or
            len(set(axes)) != len(axes)):
        raise ValueError("编码子合同必须声明不重复的 axes 0/1/2")
    sources = contract.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise ValueError("编码子合同缺少完整 sources 摘要记录")
    compatibility = {key: contract.get(key) for key in _COMPATIBILITY_FIELDS}
    representation=representation_of(contract)
    compatibility.update(encoding_reference_geometry(contract))
    if representation!='volume_xyz' and axes!=[0]:raise ValueError('View sequence is a single video, axes=[0]')
    scale = contract.get('video_spatial_scale',1)
    raster_policy = contract.get('video_raster_policy','native')
    if type(scale) is not int or scale not in (1,2):
        raise ValueError('video_spatial_scale 必须为1或2')
    if representation=='mra_native_2d':
        if scale!=1 or raster_policy!='native_2d_aspect_resize_then_pad' or contract.get('resize_max_side')!=512:
            raise ValueError('Native2D protocol requires scale1/aspect resize<=512 then padding')
        compatibility.pop('spatial_shape',None);compatibility.pop('affine',None)
        compatibility.update(resize_max_side=512,frame_geometry='native_2D_MRA_views',
                             frame_axis_kinds=['ordered_views','projection_angle'])
    elif raster_policy not in ('native','pad_then_bilinear_spatial_scale') or (scale==2 and raster_policy!='pad_then_bilinear_spatial_scale'):
        raise ValueError('video raster policy与空间scale不一致')
    compatibility['video_spatial_scale']=scale
    compatibility['video_raster_policy']=raster_policy
    if representation!='mra_native_2d':
        resources = contract.get("resource_sha256")
        if not isinstance(resources, dict):raise ValueError("编码子合同缺少目标模板与脑掩膜资源摘要")
        compatibility["template_sha256"] = _require_sha(resources.get("template"), "template_sha256")
        compatibility["template_brain_mask_sha256"] = _require_sha(resources.get("brain_mask"), "template_brain_mask_sha256")
    if (compatibility["vae_checkpoint_sha256"] != WAN22_VAE_SHA256 or
            compatibility["dtype"] != "float32" or compatibility["padding"] != "pad_only_T4k_plus1_HW32" or
            not isinstance(compatibility["valid_token_mask_definition"], str) or not compatibility["valid_token_mask_definition"]):
        raise ValueError("编码子批次不是兼容的 Wan2.2 5B / FSL MNI152 2mm / FP32 latent 协议")
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    used_sources = set()
    for row in records:
        if not SAFE_ROW_FIELDS<=set(row) or set(row)-SAFE_ROW_FIELDS-{'representation'}:
            raise ValueError("corpus 只接受 prepare 的固定安全行字段；缺失字段或临床标签/未知附加字段不能进入")
        for key in ("image_id", "subject_id", "dataset", "modality"):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f"编码行 {key} 必须为非空字符串")
        if row["dataset"] not in CANONICAL_DATASETS:
            raise ValueError("编码行必须使用 canonical 队列命名空间")
        if row["encoding_contract_sha256"] != contract_sha:
            raise ValueError("编码行指向了其他子批次合同")
        _require_sha(row["latent_sha256"], "latent_sha256")
        if row["axis"] not in axes:
            raise ValueError("编码行的 axis 不在该批次声明中")
        source_key = _source_key(row)
        source = sources.get(source_key)
        if not isinstance(source, dict):
            raise ValueError("编码行在该批次编码合同 sources 中没有对应影像")
        if row['latent_shape']!=expected_record_latent_shape(contract,source,row):
            raise ValueError('Latent shape differs from its volume-axis or projection frame_shape compression protocol')
        source_sha = _require_sha(source.get("source_sha256"), "source_sha256")
        source_component = source.get("source_component_index")
        source_format = source.get("source_format", "nifti")
        if source_format == "npy_masked_maps":
            if type(source_component) is not int or source_component < 0:
                raise ValueError("Masked-container scalar map requires an explicit component index")
            _require_sha(source.get("source_mapping_sha256"), "source_mapping_sha256")
        elif source_format=='mra_projection_frames':
            if representation!='mra_projection' or source_component is not None:
                raise ValueError('Projection frame source must have projection representation and no component')
            _require_sha(source.get('source_mapping_sha256'),'source_mapping_sha256')
            _require_sha(source.get('projection_metadata_sha256'),'projection_metadata_sha256')
        elif source_format=='mra_2d_collection':
            if representation!='mra_native_2d' or source_component is not None:
                raise ValueError('Native2D collection has no component and requires native representation')
        elif source_format != "nifti" or source_component is not None:
            raise ValueError("Invalid source format/component provenance")
        source_identity = (source_sha, source_format, source_component,representation)
        used_sources.add(source_key)
        identity = (row["dataset"], row["image_id"])
        common = {key: value for key, value in row.items() if key not in _AXIS_FIELDS}
        common['representation']=representation_of(row)
        group = groups.setdefault(identity, {"metadata": common, "axes": set(), "source_sha256": source_sha,
                                             "source_identity": source_identity})
        if group["metadata"] != common or group["source_identity"] != source_identity:
            raise ValueError("同一 image_id 的不同 axis 元数据不一致")
        if row["axis"] in group["axes"]:
            raise ValueError("同一影像的 axis 重复")
        group["axes"].add(row["axis"])
    if used_sources != set(sources):
        raise ValueError("编码清单未完整覆盖原子合同 sources，不能把缺少整幅影像的批次标为完整")
    if any(group["axes"] != set(axes) for group in groups.values()):
        raise ValueError("至少一幅影像缺少该批次声明的 axis")
    audit = verify_used_images(manifest_path, records, contract, complete)
    if audit is not None:
        for identity, digest in audit['image_voxel_sha256'].items():
            groups[identity]['preprocessed_volume_sha256'] = digest
    if file_sha256(manifest_path) != manifest_sha:
        raise ValueError("编码子清单在校验过程中发生变化")
    if audit_latents:
        for row in records:
            _audit_latent_file(row['latent_path'],row['latent_sha256'])
    entry = {"manifest_path": str(manifest_path), "manifest_sha256": manifest_sha,
             "encoding_contract_path": str(contract_path), "encoding_contract_sha256": contract_sha,
             "complete_path": str(complete_path), "complete_sha256": complete_sha,
             "records": len(records), "axes": sorted(axes)}
    return {"entry": entry, "records": records, "images": groups, "compatibility": compatibility,
            'representation':representation}


def _assemble(children: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .wan import representation_of
    rows, seen_paths, subject_splits, image_owners, hash_owners = [], set(), {}, {}, {}
    latent_hashes = set()
    volume_splits = {}
    voxel_audited_images = 0
    counts: Counter[str] = Counter()
    first=children[0]['compatibility']
    common={key:first.get(key) for key in ('vae_checkpoint_sha256','dtype')};common['latent_channels']=48
    protocols={}
    for child in children:
        protocol=child['compatibility'];representation=child.get('representation','volume_xyz')
        if any(protocol.get(key)!=common[key] for key in ('vae_checkpoint_sha256','dtype')):
            raise ValueError('子批次VAE48/dtype协议不兼容')
        if representation in protocols and protocols[representation]!=protocol:
            raise ValueError('同representation的space/scale/resize/padding/token-mask协议语义不兼容')
        protocols[representation]=protocol
        owner = child["entry"]["manifest_path"]
        for identity, image in child["images"].items():
            if identity in image_owners:
                raise ValueError("同一 canonical image_id 在不同编码批次重复")
            image_owners[identity] = owner
            source_identity = image["source_identity"]
            if source_identity in hash_owners:
                raise ValueError("不同影像重复引用同一 source_sha256/source_format/source_component_index")
            hash_owners[source_identity] = identity
            volume_sha = image.get('preprocessed_volume_sha256')
            if volume_sha is not None:
                split = image['metadata']['split']
                volume_identity=(representation_of(image['metadata']),volume_sha)
                if volume_identity in volume_splits and volume_splits[volume_identity] != split:
                    raise ValueError('相同标准图体素摘要跨train/val/test，拒绝精确影像重复泄露')
                volume_splits[volume_identity] = split
                voxel_audited_images += 1
        for record in child["records"]:
            subject = (record["dataset"], record["subject_id"])
            if subject in subject_splits and subject_splits[subject] != record["split"]:
                raise ValueError("同一 canonical 受试者在不同子批次跨 train/val/test")
            subject_splits[subject] = record["split"]
            path = record["latent_path"]
            if path in seen_paths:
                raise ValueError("同一 latent_path 在 corpus 重复")
            seen_paths.add(path)
            if record["latent_sha256"] in latent_hashes:
                raise ValueError("不同 latent_path 出现完全相同的 latent_sha256，拒绝复制的编码文件")
            latent_hashes.add(record["latent_sha256"])
            rows.append({**record, "source_encoding_manifest": owner})
            counts[record["dataset"] + "/" + record["split"]] += 1
    compatibility={'common':common,'representations':dict(sorted(protocols.items()))}
    summary = {"records": len(rows), "source_images": len(image_owners), "subjects": len(subject_splits),
               "counts_by_dataset_split": dict(sorted(counts.items())), "compatibility": compatibility,
               "axis_coverage_policy": "each_image_matches_its_child_contract_axes",
               "voxel_hash_audited_images": voxel_audited_images,
               "duplicate_screening": "source_container_component_and_cross_split_exact_preprocessed_voxel_sha256; no_fuzzy_matching"}
    return rows, summary


def _atomic_new_file(path: Path, content: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=".corpus-write-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(record: Any) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def compose_corpus(manifests: list[str | Path], output_dir: str | Path, *, audit_latents: bool = False) -> Path:
    """默认元数据核验后发布corpus；audit_latents显式强制读每幅latent摘要。"""
    from .wan import load_manifest
    if not manifests:
        raise ValueError("至少提供一个完整编码批次清单")
    paths = [Path(path).resolve() for path in manifests]
    if len(set(paths)) != len(paths):
        raise ValueError("输入 manifest 路径重复")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError("corpus 输出目录必须尚不存在，拒绝覆盖已有目录")
    children = [_verified_child(path, load_manifest, audit_latents=audit_latents) for path in paths]
    rows, summary = _assemble(children)
    contract = {"schema_version": CORPUS_VERSION, "manifest_file": "corpus.jsonl",
                "children": [child["entry"] for child in children], **summary,
                "composer_code_sha256": _sha(Path(__file__).read_bytes()),
                "record_order": "input_manifest_order_then_child_row_order"}
    contract_bytes = _json_bytes(contract)
    manifest_bytes = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows).encode()
    # mkdir is exclusive; each file is published atomically without overwriting.
    # CORPUS_COMPLETE is last, so a crash cannot expose a complete partial corpus.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    _atomic_new_file(output / "corpus_contract.json", contract_bytes)
    _atomic_new_file(output / "corpus.jsonl", manifest_bytes)
    _atomic_new_file(output / "CORPUS_COMPLETE", _json_bytes({
        "schema_version": CORPUS_COMPLETE_VERSION, "contract_sha256": _sha(contract_bytes),
        "manifest_sha256": _sha(manifest_bytes), "records": len(rows), "children": len(children),
    }))
    return output / "corpus.jsonl"


def load_corpus_manifest(manifest_path: str | Path, *, single_loader: SingleLoader, audit_latents: bool = False) -> tuple[list[dict[str, Any]], str]:
    """WAN 入口调用：核验 parent/child 摘要并逐行确认等于已验证来源行。"""
    manifest_path = Path(manifest_path).resolve()
    contract_path, complete_path = manifest_path.parent / "corpus_contract.json", manifest_path.parent / "CORPUS_COMPLETE"
    if not contract_path.is_file() or not complete_path.is_file():
        raise ValueError("corpus 尚未完整发布：缺少 corpus_contract.json 或 CORPUS_COMPLETE")
    from .integrity import file_sha256, fingerprint
    before = fingerprint(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = file_sha256(manifest_path)
    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    contract, contract_sha = _json_file(contract_path)
    complete, _ = _json_file(complete_path)
    if (contract.get("schema_version") != CORPUS_VERSION or contract.get("manifest_file") != manifest_path.name or
            complete.get("schema_version") != CORPUS_COMPLETE_VERSION or
            complete.get("manifest_sha256") != manifest_sha or complete.get("contract_sha256") != contract_sha or
            type(complete.get("records")) is not int or complete["records"] != len(rows)):
        raise ValueError("corpus parent 合同、清单、完成标记的摘要或数量不一致")
    entries = contract.get("children")
    if (not isinstance(entries, list) or not entries or type(complete.get("children")) is not int or
            complete["children"] != len(entries)):
        raise ValueError("corpus 子批次目录不完整或数量不一致")
    children, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != _CHILD_FIELDS:
            raise ValueError("corpus 子批次记录字段不完整")
        if type(entry["records"]) is not int or not isinstance(entry["axes"], list) or any(type(axis) is not int for axis in entry["axes"]):
            raise ValueError("corpus 子批次 records/axes 必须保留整数类型")
        child_path = _resolve_path(entry["manifest_path"], manifest_path.parent, "child manifest_path")
        if child_path in seen:
            raise ValueError("corpus 子批次 manifest 重复")
        seen.add(child_path)
        child = _verified_child(child_path, single_loader, audit_latents=audit_latents)
        expected = child["entry"]
        normalized = dict(entry)
        for key in ("manifest_path", "encoding_contract_path", "complete_path"):
            normalized[key] = str(_resolve_path(entry[key], manifest_path.parent, key))
        if normalized != expected:
            raise ValueError("corpus 记录的子清单、合同、COMPLETE 摘要/数量/axes 与实际来源不一致")
        children.append(child)
    expected_rows, summary = _assemble(children)
    if len(rows) != len(expected_rows) or any(
            not isinstance(row, dict) or _json_bytes(row) != _json_bytes(expected)
            for row, expected in zip(rows, expected_rows)):
        raise ValueError("corpus 行必须逐条等于原完整子批次行加 source_encoding_manifest；不能重标 split、身份或 latent")
    if any(type(contract.get(key)) is not int for key in ("records", "source_images", "subjects")):
        raise ValueError("corpus 派生数量必须为整数")
    comparison=dict(summary)
    stored=contract.get('compatibility')
    if isinstance(stored,dict) and 'common' not in stored:
        # 已存在的legacy corpus用单一完整兼容dict；保持只读接受，不重写旧文件。
        protocols=list(summary['compatibility']['representations'].values())
        if any(protocol!=stored for protocol in protocols):raise ValueError('Legacy corpus协议与实际子批次不一致')
        comparison['compatibility']=stored
    if any(contract.get(key) != value for key, value in comparison.items()):
        raise ValueError("corpus parent 合同中的派生数量或协议与已验证子批次不一致")
    if fingerprint(manifest_path)!=before:
        raise ValueError('corpus清单在读取核验期间发生变化')
    return expected_rows, manifest_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="完整 prepare latents.jsonl；可重复，按输入顺序合并")
    parser.add_argument("--output-dir", type=Path, required=True, help="尚不存在的新 corpus 输出目录")
    parser.add_argument('--audit-latents',action='store_true',help='显式逐latent force SHA审计；默认只读清单/合同/账本，不遍历latent文件')
    arguments = parser.parse_args(argv)
    try:
        result = compose_corpus(arguments.manifest, arguments.output_dir,audit_latents=arguments.audit_latents)
        print(json.dumps({"manifest": str(result), "status": "complete"}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"corpus 整合失败：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
