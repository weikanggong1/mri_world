"""显式已选 recipes -> 持久MNI2mm数据；不运行VAE、不批准训练。"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
import math
import multiprocessing
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sqlite3
import sys
import tempfile
import time
import uuid

from .integrity import file_sha256, fingerprint, record_verified_sha256

VERSION = 'brain_mri_preprocess_phase1_v2'
SOURCE_FIELDS = ('source_path', 'source_mask_path', 'source_mask_reference_path', 'source_mapping_path', 'source_fov_mask_path')
TERMINAL = ('PASSED', 'REJECTED')
CPU_THREADS = 8
DEFAULT_MAX_IMAGES = 128


class IntegrityError(RuntimeError):
    pass


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def publish_json(path, value, *, exclusive=False):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    content = json_bytes(value)
    if exclusive and path.exists():
        if path.read_bytes() != content:
            raise IntegrityError('Existing frozen receipt/contract differs; refusing overwrite')
        return sha_bytes(content)
    descriptor, name = tempfile.mkstemp(prefix='.preprocess-write-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        if exclusive:os.link(temporary, path)
        else:temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    digest = sha_bytes(content)
    record_verified_sha256(path, digest, fingerprint(path))
    return digest


@contextmanager
def lock(path, *, shared=False, nonblocking=True):
    import fcntl
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        flags = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
        if nonblocking:flags |= fcntl.LOCK_NB
        try:fcntl.flock(stream, flags)
        except BlockingIOError as error:raise RuntimeError('An active controller/worker holds this run') from error
        try:yield
        finally:fcntl.flock(stream, fcntl.LOCK_UN)


def connect(path, *, readonly=False):
    if readonly:
        db = sqlite3.connect(Path(path).absolute().as_uri()+'?mode=ro', uri=True, timeout=30)
    else:
        db = sqlite3.connect(path, timeout=30)
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
    db.row_factory = sqlite3.Row
    return db


def meta_get(db, key):
    row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def meta_set(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES(?,?)', (key, json.dumps(value, ensure_ascii=False, sort_keys=True)))


def absolute_lexical(value, base):
    """原MRI路径仅做字符串归一化；正常resume不能resolve/stat原图。"""
    if not isinstance(value, str) or not value.strip():raise ValueError('Input path must be nonempty string')
    return os.path.abspath(os.path.join(str(base), value))


def validate_recipe(raw, base):
    from .prepare import CANONICAL_DATASETS, image_identifier
    if not isinstance(raw, dict):raise ValueError('Recipe must be a JSON object')
    forbidden = {'diagnosis', 'diagnoses', 'diag', 'report', 'radiologyreport', 'pathology', 'outcome', 'targetlabel', 'clinicalreport', 'label', 'labels', 'target', 'y', 'phenotype', 'findings', 'impression'}
    if any(''.join(c for c in key.casefold() if c.isalnum()) in forbidden for key in raw):
        raise ValueError('Clinical reports/labels are not accepted in preprocessing recipes')
    if raw.get('selected_by_user') is not True or raw.get('pipeline_protocol_reviewed') is not True:
        raise ValueError('Recipe requires selected_by_user=true and pipeline_protocol_reviewed=true')
    required = ('source_path', 'dataset', 'subject_id', 'image_id', 'modality', 'split', 'preprocess_mode')
    if any(not isinstance(raw.get(key), str) or not raw[key].strip() for key in required):
        raise ValueError('Missing canonical source recipe fields')
    if raw['dataset'] not in CANONICAL_DATASETS or raw['split'] not in ('train', 'val', 'test'):
        raise ValueError('Invalid canonical dataset or source split')
    if raw['preprocess_mode'] not in ('already_mni', 'fnit_joint') or type(raw.get('source_skull_stripped')) is not bool:
        raise ValueError('Explicit preprocessing route and boolean skull flag required')
    if raw['preprocess_mode'] == 'already_mni' and (raw.get('source_space') != 'FSL_MNI152' or raw.get('registration_qc_passed') is not True):
        raise ValueError('already_mni requires proven FSL_MNI152 space and registration_qc_passed=true')
    coverage = raw.get('minimum_brain_fov_coverage', .95)
    if type(coverage) not in (int, float) or not math.isfinite(coverage) or not 0 < coverage <= 1:
        raise ValueError('Invalid explicitly reviewed FOV coverage threshold')
    source_format = raw.get('source_format', 'nifti')
    if source_format not in ('nifti', 'npy_masked_maps'):raise ValueError('Unsupported source_format')
    if source_format == 'npy_masked_maps' and (type(raw.get('source_component_index')) is not int or raw['source_component_index'] < 0 or not raw.get('source_mapping_path') or not raw.get('source_mask_path')):
        raise ValueError('Masked maps require explicit scalar component, audited mapping and original mask')
    row = dict(raw)
    for field in SOURCE_FIELDS:
        if row.get(field):row[field] = absolute_lexical(row[field], base)
    if source_format == 'npy_masked_maps':
        if not row['source_path'].lower().endswith('.npy') or not row['source_mapping_path'].lower().endswith('.json'):
            raise ValueError('Masked maps require .npy numeric source and .json audited mapping')
    elif not row['source_path'].lower().endswith(('.nii', '.nii.gz', '.mgh', '.mgz')):
        raise ValueError('Source must be an MRI volume, not report/CSV metadata')
    for field in ('source_mask_path', 'source_mask_reference_path', 'source_fov_mask_path'):
        if row.get(field) and not row[field].lower().endswith(('.nii', '.nii.gz', '.mgh', '.mgz')):
            raise ValueError('MRI auxiliary inputs must be image volumes')
    identifier = image_identifier(row)
    source_key = sha_bytes(json_bytes([row['source_path'], source_format, row.get('source_component_index')]))
    return row, identifier, source_key


def protocol(args):
    """仅绑定资源/代码；不导入模型、不初始化CUDA，也不访问患者源文件。"""
    from .prepare import verify_resources
    resources = verify_resources(args.resource_manifest)
    runtime = Path(args.fnit_runtime).resolve()
    source_manifest_path = runtime/'SOURCE_MANIFEST.json'
    source_manifest = json.loads(source_manifest_path.read_text())
    monitored = {}
    runtime_files = {}
    for name, expected in source_manifest['files'].items():
        path = (runtime/name).resolve()
        if not path.is_relative_to(runtime) or file_sha256(path) != expected:
            raise IntegrityError('Frozen FNIT runtime source differs from its manifest')
        monitored[str(path)] = {'sha256': expected, 'fingerprint': list(fingerprint(path))}
        runtime_files[name] = expected
    package = Path(__file__).resolve().parent
    code = {}
    for name in ('preprocess_runner.py', 'preprocess_adoption.py', 'prepare.py', 'registered.py', 'masked_volume.py', 'integrity.py'):
        path = package/name;code[name] = file_sha256(path)
        monitored[str(path)] = {'sha256': code[name], 'fingerprint': list(fingerprint(path))}
    for path in (Path(args.resource_manifest).resolve(), source_manifest_path):
        monitored[str(path)] = {'sha256': file_sha256(path), 'fingerprint': list(fingerprint(path))}
    for value in resources.values():
        monitored[value['path']] = {'sha256': value['sha256'], 'fingerprint': list(fingerprint(value['path']))}
    result = {'resource_manifest': str(Path(args.resource_manifest).resolve()), 'resource_manifest_sha256': file_sha256(args.resource_manifest),
        'resources': resources, 'fnit_runtime': str(runtime), 'fnit_source_manifest_sha256': file_sha256(source_manifest_path),
        'fnit_source_sha256': runtime_files, 'code_sha256': code,
        'software': {name: version(name) for name in ('torch', 'numpy', 'nibabel', 'scipy', 'h5py')},
        'python': str(Path(sys.executable).resolve()), 'registered_output_dir': str(Path(args.registered_output_dir).resolve()),
        'extent': args.extent, 'minimum_mask_dice': args.minimum_mask_dice, 'gpu_budget_gb': args.gpu_budget_gb,
        'cpu_threads': CPU_THREADS, 'dtype': 'float32', 'tf32': True, 'require_review': False,
        'minimum_free_gib': args.minimum_free_gib, 'stable_free_seconds': args.stable_free_seconds,
        'minimum_registered_free_gib': args.minimum_registered_free_gib,
        'normalization': 'deferred_to_root_global_freeze_and_release', 'training_approval': False}
    return result, monitored


def settings(args):
    if args.extent not in (192, 256) or not 0 < args.minimum_mask_dice <= 1 or not math.isfinite(args.minimum_mask_dice):
        raise ValueError('Invalid extent or Dice threshold')
    if not 0 < args.gpu_budget_gb <= 20 or not math.isfinite(args.gpu_budget_gb):raise ValueError('FNIT allocator budget must be >0 and <=20 decimal GB')
    if type(args.max_images) is not int or args.max_images < 0:raise ValueError('max_images must be 0 (all) or a positive invocation quota')
    if not math.isfinite(args.minimum_free_gib) or args.minimum_free_gib < 30:raise ValueError('GPU free-space floor must be at least30 GiB')
    if not math.isfinite(args.stable_free_seconds) or args.stable_free_seconds < 15:raise ValueError('GPU free-space stability window must be at least15 seconds')
    if not math.isfinite(args.minimum_registered_free_gib) or args.minimum_registered_free_gib < 30:raise ValueError('Registered disk free-space floor must be at least30 GiB')
    if len(set(args.gpu_indices)) != len(args.gpu_indices) or any(type(index) is not int or index < 0 for index in args.gpu_indices):
        raise ValueError('Explicit physical GPU indices must be unique nonnegative integers')
    if 'latents' in Path(args.registered_output_dir).parts:raise ValueError('Registered volumes must not be published into latents')


def initialize(args):
    """首次一遍读取全局recipes并建SQLite；后续FP未变完全不再读取list。"""
    settings(args)
    root = Path(args.output_dir).resolve();root.mkdir(parents=True, exist_ok=True);os.chmod(root, 0o700)
    manifest = Path(args.recipes).resolve()
    current_protocol, monitored = protocol(args)
    old_run=getattr(args,'adopt_from_stopped_run',None)
    adoption_source=None
    if old_run is not None:
        old_run=Path(old_run).resolve()
        if old_run==root:raise IntegrityError('Adoption source and target run must differ')
        adoption_source={'run_dir':str(old_run),'contract_sha256':file_sha256(old_run/'preprocess_contract.json')}
    with lock(root/'.initialize.lock'):
        db_path = root/'state.sqlite3';contract_path = root/'preprocess_contract.json'
        if db_path.exists():
            db = connect(db_path)
            try:
                contract = meta_get(db, 'contract')
                if not contract or not meta_get(db, 'initialization_complete'):raise IntegrityError('Incomplete SQLite initialization cannot be adopted')
                if (contract['protocol'] != current_protocol or contract['recipes_path'] != str(manifest)
                        or contract.get('adoption_source')!=adoption_source):
                    raise IntegrityError('Existing preprocessing run has a different recipe/resource/code/parameter contract')
                current_fp = list(fingerprint(manifest));old_fp = meta_get(db, 'recipes_fingerprint')
                if current_fp != old_fp:
                    if file_sha256(manifest) != contract['recipes_sha256']:raise IntegrityError('Global source recipes changed; use a new reviewed run')
                    meta_set(db, 'recipes_fingerprint', current_fp);db.commit()
                publish_json(contract_path, contract, exclusive=True)
                publish_json(root/'protocol_files.json', monitored)
                return root, contract, sha_bytes(json_bytes(contract))
            finally:db.close()
        allowed = {'.initialize.lock', '.runner.lock', '.workers.lock', 'protocol_files.json'}
        if any(path.name not in allowed and not path.name.startswith('.state-building-') for path in root.iterdir()):
            raise IntegrityError('Nonempty output directory has no compatible state.sqlite3')
        descriptor, name = tempfile.mkstemp(prefix='.state-building-', suffix='.sqlite3', dir=root);os.close(descriptor)
        temporary = Path(name);db = connect(temporary)
        try:
            db.executescript('''CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
              CREATE TABLE subjects(dataset TEXT,subject_id TEXT,split TEXT,PRIMARY KEY(dataset,subject_id));
              CREATE TABLE recipes(position INTEGER PRIMARY KEY,identifier TEXT UNIQUE NOT NULL,source_key TEXT UNIQUE NOT NULL,
                recipe_json TEXT NOT NULL,recipe_sha256 TEXT NOT NULL,state TEXT NOT NULL DEFAULT 'PENDING',attempt INTEGER NOT NULL DEFAULT 0,
                receipt_path TEXT,receipt_sha256 TEXT,receipt_fingerprint TEXT,receipt_json TEXT,source_proof TEXT,owner TEXT,last_invocation TEXT);
              CREATE INDEX recipe_state ON recipes(state,position);
              CREATE INDEX recipe_dispatch ON recipes(position) WHERE state IN('PENDING','FAILED');
              CREATE TABLE dispatch(invocation TEXT PRIMARY KEY,quota INTEGER NOT NULL,claimed INTEGER NOT NULL DEFAULT 0,stopped INTEGER NOT NULL DEFAULT 0);
              CREATE TABLE invocations(token TEXT PRIMARY KEY,started REAL NOT NULL,finished REAL,status TEXT,info_json TEXT);''')
            db.execute('CREATE TABLE adoptions(identifier TEXT PRIMARY KEY,producer_contract_sha256 TEXT NOT NULL,provenance_json TEXT NOT NULL)')
            before = fingerprint(manifest);digest = hashlib.sha256();count = 0
            with manifest.open('rb') as stream:
                for line in stream:
                    digest.update(line)
                    if not line.strip():continue
                    raw = json.loads(line);row, identifier, source_key = validate_recipe(raw, manifest.parent)
                    existing = db.execute('SELECT split FROM subjects WHERE dataset=? AND subject_id=?', (row['dataset'], row['subject_id'])).fetchone()
                    if existing and existing['split'] != row['split']:raise ValueError('Canonical subject appears in more than one split')
                    db.execute('INSERT OR IGNORE INTO subjects VALUES(?,?,?)', (row['dataset'], row['subject_id'], row['split']))
                    count += 1;body = json_bytes(row)
                    db.execute('INSERT INTO recipes(position,identifier,source_key,recipe_json,recipe_sha256) VALUES(?,?,?,?,?)',
                               (count, identifier, source_key, body.decode(), sha_bytes(body)))
                    if count % 1024 == 0:db.commit()
            if count == 0 or fingerprint(manifest) != before:raise IntegrityError('Source list is empty or changed during first registration')
            manifest_sha = record_verified_sha256(manifest, digest.hexdigest(), before)
            contract = {'schema_version': VERSION, 'recipes_path': str(manifest), 'recipes_sha256': manifest_sha,
                        'recipe_count': count, 'protocol': current_protocol,'adoption_source':adoption_source,
                        'outputs_never_auto_released': True, 'completion_scope': 'all_explicit_recipes_numeric_preprocessing_only'}
            meta_set(db, 'contract', contract);meta_set(db, 'recipes_fingerprint', list(before));meta_set(db, 'initialization_complete', True)
            db.commit();db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        finally:db.close()
        try:os.link(temporary, db_path)
        finally:temporary.unlink(missing_ok=True)
        publish_json(contract_path, contract, exclusive=True);publish_json(root/'protocol_files.json', monitored)
        return root, contract, sha_bytes(json_bytes(contract))


def verify_protocol_files(root):
    for path, proof in json.loads((Path(root)/'protocol_files.json').read_text()).items():
        if tuple(proof['fingerprint']) != fingerprint(path) and file_sha256(path) != proof['sha256']:
            raise IntegrityError('Resource or preprocessing implementation changed')


def counts(db):
    result = {key: 0 for key in ('PENDING', 'RUNNING', 'PASSED', 'REJECTED', 'FAILED')}
    result.update({row[0]: row[1] for row in db.execute('SELECT state,COUNT(*) FROM recipes GROUP BY state')})
    return result


def validate_receipt(value, item, contract_sha):
    if (value.get('schema_version') != VERSION or value.get('contract_sha256') != contract_sha or
            value.get('identifier') != item['identifier'] or value.get('recipe_sha256') != item['recipe_sha256'] or
            value.get('attempt') != item['attempt'] or value.get('state') not in ('PASSED', 'REJECTED', 'FAILED')):
        raise IntegrityError('Receipt identity/contract/state does not match SQLite')
    output = value.get('output_record', {})
    if (output.get('approved') is not False or output.get('manual_release') is not False or
            output.get('ready_for_encoding') is not (value['state'] == 'PASSED')):
        raise IntegrityError('Phase1 receipt incorrectly releases an image')


def interrupted_failure_receipt(root, item, contract_sha):
    """claim后失联的attempt用SQLite技术证据记FAILED；不访问任何源/registered体积。"""
    if not item['receipt_path'] or item['attempt'] < 1:
        raise IntegrityError('Interrupted attempt has no recoverable receipt identity')
    record = json.loads(item['recipe_json'])
    if sha_bytes(json_bytes(record)) != item['recipe_sha256']:
        raise IntegrityError('Indexed interrupted recipe changed')
    proof = json.loads(item['source_proof']) if item['source_proof'] else None
    error = {'stage': 'interrupted_before_atomic_receipt', 'error_type': 'InterruptedAttempt',
             'message': 'Previous worker exited before publishing its attempt receipt; recovery did not inspect source images'}
    output = {key: record[key] for key in ('dataset', 'subject_id', 'image_id', 'modality', 'split')}
    output.update(selected_by_user=True, pipeline_protocol_reviewed=True, ready_for_encoding=False,
        manual_release=False, release_for_encoding=False, approved=False, qc_passed=False,
        preprocess_qc_reviewed=False, numeric_qc_passed=False, phase1_state='FAILED', permanently_rejected=False,
        original_source_path=record['source_path'],
        original_source_sha256=proof.get('source_path', {}).get('sha256') if proof else None,
        original_source_format=record.get('source_format', 'nifti'),
        original_source_component_index=record.get('source_component_index'),
        normalization_status='deferred_to_root_global_freeze_and_release',
        preprocess_receipt_path=str(Path(root)/item['receipt_path']), error=error)
    receipt = {'schema_version': VERSION, 'contract_sha256': contract_sha, 'identifier': item['identifier'],
        'recipe_sha256': item['recipe_sha256'], 'attempt': item['attempt'], 'state': 'FAILED',
        'source_provenance': proof, 'source_recipe': record, 'preprocessing_route': record['preprocess_mode'],
        'preprocessor_metadata': {}, 'registered': {}, 'output_record': output,
        'worker': {'previous_invocation': item['owner']}, 'elapsed_seconds': None, 'error': error,
        'recovered_interrupted_attempt_without_source_io': True, 'no_vae_or_training_release': True}
    publish_json(Path(root)/item['receipt_path'], receipt, exclusive=True)


def reconcile(root, contract_sha, *, interrupted=False):
    """只读小回执；不读取、不resolve、不stat任何原MRI/原mask/registered体积。"""
    root = Path(root);db = connect(root/'state.sqlite3')
    try:
        def receipt_pages():
            position = 0
            while True:
                page = db.execute('SELECT * FROM recipes WHERE receipt_path IS NOT NULL AND position>? ORDER BY position LIMIT 256', (position,)).fetchall()
                if not page:break
                for value in page:
                    position = value['position'];yield value
        for item in receipt_pages():
            path = root/item['receipt_path']
            adoption=db.execute('SELECT producer_contract_sha256 FROM adoptions WHERE identifier=?',(item['identifier'],)).fetchone()
            if not path.exists():
                if adoption:raise IntegrityError('Original adopted producer receipt is missing; it will not be rewritten')
                if item['receipt_json']:
                    cached = json.loads(item['receipt_json'])
                    if sha_bytes(json_bytes(cached)) != item['receipt_sha256']:raise IntegrityError('Cached receipt body differs')
                    publish_json(path, cached, exclusive=True)
                elif item['state'] in ('RUNNING', 'FAILED') and interrupted:
                    interrupted_failure_receipt(root, item, contract_sha)
                else:continue
            current_fp = list(fingerprint(path))
            if item['receipt_sha256'] and item['state'] != 'RUNNING' and json.loads(item['receipt_fingerprint'] or 'null') == current_fp:
                # Receipt内容由第一次发布+SQLite事务绑定；FP未变不再读。
                continue
            digest = file_sha256(path)
            if item['receipt_sha256'] and digest != item['receipt_sha256']:
                raise IntegrityError('Previously published receipt changed; it will not be overwritten')
            value = json.loads(path.read_text());validate_receipt(value, item, adoption[0] if adoption else contract_sha)
            db.execute('UPDATE recipes SET state=?,receipt_sha256=?,receipt_fingerprint=?,receipt_json=?,owner=NULL WHERE identifier=?',
                (value['state'], digest, json.dumps(current_fp), json_bytes(value).decode(), item['identifier']))
        if interrupted and db.execute("SELECT COUNT(*) FROM recipes WHERE state='RUNNING'").fetchone()[0]:
            raise IntegrityError('Running state lacks a recoverable attempt receipt path')
        db.commit()
    finally:db.close()


def source_proof(record, previous=None):
    """仅未完成图调用；共享SHA cache和全局bind锁避免同NPY的多个组件并发重hash。"""
    files = {}
    previous = previous or {}
    for field in SOURCE_FIELDS:
        if not record.get(field):continue
        path = record[field];before = fingerprint(path);digest = file_sha256(path)
        if fingerprint(path) != before:raise IntegrityError('Source changed while first provenance was bound')
        expected = record.get(field[:-5]+'_sha256') if field.endswith('_path') else None
        if expected and expected != digest:raise IntegrityError('Recipe source/auxiliary SHA does not match current file')
        if field in previous and previous[field]['sha256'] != digest:
            raise IntegrityError('A previously bound source changed; review into a new recipe/run')
        files[field] = {'path': path, 'sha256': digest, 'fingerprint': list(before)}
    return files


def sources_unchanged(proof):
    for value in proof.values():
        if tuple(value['fingerprint']) != fingerprint(value['path']):
            raise IntegrityError('Source changed during preprocessing; not publishing a success')


def _error(error, stage):
    import re
    message = re.sub(r'hf_[A-Za-z0-9]+', '[REDACTED]', str(error))[:2000]
    return {'stage': stage, 'error_type': type(error).__name__, 'message': message}


def _claim_item(db, item, invocation):
    attempt = item['attempt']+1
    receipt = f"receipts/{item['identifier']}/attempt-{attempt:06d}.json"
    db.execute("UPDATE recipes SET state='RUNNING',attempt=?,receipt_path=?,receipt_sha256=NULL,receipt_fingerprint=NULL,receipt_json=NULL,owner=?,last_invocation=? WHERE identifier=?",
               (attempt, receipt, invocation, invocation, item['identifier']))
    return {**dict(item), 'attempt': attempt, 'receipt_path': receipt, 'owner': invocation}


def claim(root, identifier, invocation):
    """单图控制接口；生产worker通过claim_next共享SQLite限额。"""
    db = connect(Path(root)/'state.sqlite3')
    try:
        db.execute('BEGIN IMMEDIATE')
        if meta_get(db, 'active_invocation') != invocation:raise IntegrityError('Stale worker invocation cannot claim sources')
        item = db.execute('SELECT * FROM recipes WHERE identifier=?', (identifier,)).fetchone()
        if item is None:raise IntegrityError('Unknown explicitly indexed source')
        if item['state'] in TERMINAL:db.rollback();return None
        if item['state'] == 'RUNNING':raise IntegrityError('Source already claimed by an active worker')
        result = _claim_item(db, item, invocation);db.commit();return result
    finally:db.close()


def claim_next(root, invocation):
    """BEGIN IMMEDIATE下一张；每worker只持有一个recipe，全局0限额不生成ID列表。"""
    db = connect(Path(root)/'state.sqlite3')
    try:
        db.execute('BEGIN IMMEDIATE')
        if meta_get(db, 'active_invocation') != invocation:raise IntegrityError('Stale worker invocation cannot claim sources')
        dispatch = db.execute('SELECT * FROM dispatch WHERE invocation=?', (invocation,)).fetchone()
        if dispatch is None:raise IntegrityError('Invocation has no frozen dispatch quota')
        if dispatch['stopped'] or (dispatch['quota'] > 0 and dispatch['claimed'] >= dispatch['quota']):
            db.rollback();return None
        item = db.execute("SELECT * FROM recipes INDEXED BY recipe_dispatch WHERE state IN('PENDING','FAILED') AND (last_invocation IS NULL OR last_invocation<>?) ORDER BY position LIMIT 1", (invocation,)).fetchone()
        if item is None:db.rollback();return None
        result = _claim_item(db, item, invocation)
        db.execute('UPDATE dispatch SET claimed=claimed+1 WHERE invocation=?', (invocation,));db.commit()
        return result
    finally:db.close()


def registered_space_gate(directory, minimum_free_gib):
    path = Path(directory)
    while not path.exists():
        if path == path.parent:raise OSError('Cannot identify registered output filesystem')
        path = path.parent
    free = shutil.disk_usage(path).free
    if free < minimum_free_gib*1024**3:
        raise OSError('Registered output filesystem has insufficient reserved free space')
    return free


def gpu_state(index):
    output = subprocess.check_output(['nvidia-smi', f'--id={index}',
        '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    number, device_uuid, free = [value.strip() for value in output.strip().split(',')]
    if int(number) != index or not device_uuid.startswith('GPU-'):
        raise IntegrityError('nvidia-smi physical GPU mapping differs')
    return {'physical_index': index, 'uuid': device_uuid, 'free_bytes': int(free)*1024**2, 'time_unix': time.time()}


def wait_gpu(root, index, invocation, parameters):
    """每worker模型建立前仅排队一次，UUID绑定GPU；不在逐图循环重新等待。"""
    started = time.monotonic();good_since = None;minimum = None
    event_path = Path(root)/'gpu_events'/invocation/f'GPU{index}.json'
    while True:
        value = gpu_state(index);now = time.monotonic()
        if value['free_bytes'] >= parameters['minimum_free_gib']*1024**3:
            if good_since is None:good_since = now;minimum = value['free_bytes']
            minimum = min(minimum, value['free_bytes'])
            if now-good_since >= parameters['stable_free_seconds']:
                publish_json(event_path, {'status': 'ready', 'last_sample': value,
                    'queue_seconds': now-started, 'minimum_observed_free_bytes': minimum})
                return value
        else:good_since = None;minimum = None
        publish_json(event_path, {'status': 'queued', 'last_sample': value, 'queue_seconds': now-started})
        time.sleep(1)


def process_one(root, identifier, contract, contract_sha, invocation, worker, get_preprocessor, publisher, *, claimed_item=None):
    """一个attempt；preprocessor的QC ValueError与publisher/IO/资源错误分开。"""
    from .prepare import image_identifier
    root = Path(root);item = claimed_item if claimed_item is not None else claim(root, identifier, invocation)
    if item is None:return 'SKIPPED'
    record = json.loads(item['recipe_json'])
    if sha_bytes(json_bytes(record)) != item['recipe_sha256']:raise IntegrityError('Indexed source recipe changed')
    started = time.monotonic();stage = 'source_provenance';proof = None;metadata = {};stored = {};error_info = None
    state = 'FAILED'
    try:
        stage = 'registered_disk_space'
        registered_space_gate(contract['protocol']['registered_output_dir'], contract['protocol']['minimum_registered_free_gib'])
        stage = 'source_provenance'
        with lock(root/'.source_bind.lock', nonblocking=False):
            proof = source_proof(record, json.loads(item['source_proof']) if item['source_proof'] else None)
        db = connect(root/'state.sqlite3')
        try:db.execute('UPDATE recipes SET source_proof=? WHERE identifier=?', (json.dumps(proof, sort_keys=True), identifier));db.commit()
        finally:db.close()
        stage = 'model_initialization';preprocessor = get_preprocessor()
        stage = 'numeric_preprocessing'
        try:
            array, mask, metadata = preprocessor(record)
        except ValueError as error:
            sources_unchanged(proof)
            state = 'REJECTED';error_info = _error(error, 'numeric_preprocessing')
        else:
            sources_unchanged(proof)
            stage = 'registered_disk_space_before_publish'
            registered_space_gate(contract['protocol']['registered_output_dir'], contract['protocol']['minimum_registered_free_gib'])
            stage = 'publish_registered'
            stored = publisher(contract['protocol']['registered_output_dir'], record, array, mask, metadata,
                               image_identifier(record), preprocessor.last_fov_mask)
            sources_unchanged(proof)
            state = 'PASSED'
    except Exception as error:
        state = 'FAILED';error_info = _error(error, stage)
    identity = {key: record[key] for key in ('dataset', 'subject_id', 'image_id', 'modality', 'split')}
    output = {**identity, 'selected_by_user': True, 'pipeline_protocol_reviewed': True,
              'ready_for_encoding': state == 'PASSED', 'manual_release': False, 'release_for_encoding': False,
              'approved': False, 'qc_passed': False, 'preprocess_qc_reviewed': False,
              'numeric_qc_passed': state == 'PASSED', 'phase1_state': state,
              'permanently_rejected': False, 'original_source_path': record['source_path'],
              'original_source_sha256': proof.get('source_path', {}).get('sha256') if proof else None,
              'original_source_format': record.get('source_format', 'nifti'),
              'original_source_component_index': record.get('source_component_index'),
              'normalization_status': 'deferred_to_root_global_freeze_and_release',
              'preprocess_receipt_path': str(root/item['receipt_path'])}
    if state == 'PASSED':
        output.update(stored)
        output.update(source_path=stored['registered_image_path'], source_format='nifti', source_skull_stripped=True,
                      source_component_index=None, source_mask_path=stored['registered_brain_mask_path'], source_fov_mask_path=stored['registered_fov_mask_path'],
                      source_space='FSL_MNI152', preprocess_mode='already_mni', registration_qc_passed=False,
                      mask_vs_template_dice=metadata.get('mask_vs_template_dice'), brain_fov_coverage=metadata.get('brain_fov_coverage'),
                      coverage_type=metadata.get('coverage_type'), fov_mask_kind=metadata.get('fov_mask_kind'), preprocess_qc_passed=False,
                      intensity_statistics=metadata.get('intensity_statistics'))
    for hint in ('intensity_unit', 'normalization_mode', 'intensity_range', 'nonfinite_policy', 'maximum_invalid_voxel_fraction'):
        if hint in record:output[hint] = record[hint]
    for hint in ('source_nonfinite_policy', 'source_nan_replaced_with_zero', 'source_nan_fraction',
                 'source_nan_fill_retains_anatomical_mask'):
        if hint in metadata:output[hint] = metadata[hint]
    if error_info:output['error'] = error_info
    receipt = {'schema_version': VERSION, 'contract_sha256': contract_sha, 'identifier': identifier,
        'recipe_sha256': item['recipe_sha256'], 'attempt': item['attempt'], 'state': state,
        'source_provenance': proof, 'source_recipe': record, 'preprocessing_route': record['preprocess_mode'],
        'preprocessor_metadata': metadata, 'registered': stored, 'output_record': output,
        'worker': worker, 'elapsed_seconds': time.monotonic()-started, 'error': error_info,
        'no_vae_or_training_release': True}
    digest = publish_json(root/item['receipt_path'], receipt, exclusive=True)
    db = connect(root/'state.sqlite3')
    try:
        db.execute('UPDATE recipes SET state=?,receipt_sha256=?,receipt_fingerprint=?,receipt_json=?,owner=NULL WHERE identifier=?',
            (state, digest, json.dumps(list(fingerprint(root/item['receipt_path']))), json_bytes(receipt).decode(), identifier))
        if state == 'FAILED':db.execute('UPDATE dispatch SET stopped=1 WHERE invocation=?', (invocation,))
        db.commit()
    finally:db.close()
    print(json.dumps({'event': 'image_preprocessed', 'identifier': identifier, 'state': state,
                      'ready_for_encoding': state == 'PASSED', 'approved': False,
                      'error_type': error_info['error_type'] if error_info else None}), flush=True)
    return state


def _worker_entry(result_queue, root, contract, contract_sha, invocation, physical_gpu, cpu_affinity):
    worker = {'pid': os.getpid(), 'physical_gpu': physical_gpu, 'cpu_threads': CPU_THREADS, 'cpu_affinity': cpu_affinity}
    result = {'worker': worker, 'counts': {}, 'status': 'FAILED'}
    try:
        with lock(Path(root)/'.workers.lock', shared=True, nonblocking=False):
            db = connect(Path(root)/'state.sqlite3', readonly=True)
            try:
                if meta_get(db, 'active_invocation') != invocation:raise IntegrityError('Stale spawn worker was superseded')
            finally:db.close()
            gate = wait_gpu(root, physical_gpu, invocation, contract['protocol'])
            worker.update(gpu_uuid=gate['uuid'])
            os.environ['CUDA_VISIBLE_DEVICES'] = gate['uuid']
            for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                os.environ[key] = str(CPU_THREADS)
            os.sched_setaffinity(0, set(cpu_affinity))
            verify_protocol_files(root)
            sys.path.insert(0, str(Path(contract['protocol']['fnit_runtime'])/'src'))
            import torch
            from .prepare import Preprocessor
            from .registered import publish_registered
            torch.set_num_threads(CPU_THREADS);torch.set_num_interop_threads(1);torch.set_grad_enabled(False)
            torch.set_default_dtype(torch.float32)
            torch.backends.cuda.matmul.allow_tf32 = True;torch.backends.cudnn.allow_tf32 = True
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1., contract['protocol']['gpu_budget_gb']*1e9/total), 0)
            worker.update(cuda_device_name=torch.cuda.get_device_name(0), allocator_cap_bytes=min(total, contract['protocol']['gpu_budget_gb']*1e9))
            model = None
            def get_preprocessor():
                nonlocal model
                if model is None:
                    model = Preprocessor(contract['protocol']['resources'], 'cuda:0', contract['protocol']['extent'],
                                         contract['protocol']['minimum_mask_dice'], require_review=False)
                return model
            outcomes = Counter()
            while True:
                item = claim_next(root, invocation)
                if item is None:break
                state = process_one(root, item['identifier'], contract, contract_sha, invocation, worker, get_preprocessor, publish_registered, claimed_item=item)
                outcomes[state] += 1
                if state == 'FAILED':break
            result.update(status='FAILED' if outcomes['FAILED'] else 'DONE', counts=dict(outcomes),
                          preprocessor_instances_created=int(model is not None))
    except Exception as error:
        result.update(status='FAILED', error=_error(error, 'worker_initialization_or_receipt_io'))
    finally:
        result_queue.put(result)


def spawn_workers(root, contract, contract_sha, invocation, gpu_indices):
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < CPU_THREADS:raise RuntimeError('Worker requires at least8 allowed CPU cores')
    context = multiprocessing.get_context('spawn');results_queue = context.Queue();processes = []
    for index, gpu in enumerate(gpu_indices):
        start = (index*CPU_THREADS) % len(allowed)
        affinity = [allowed[(start+offset) % len(allowed)] for offset in range(CPU_THREADS)]
        process = context.Process(target=_worker_entry, args=(results_queue, str(root), contract, contract_sha,
                    invocation, gpu, affinity), name=f'brain-preprocess-GPU{gpu}')
        process.start();processes.append(process)
    results = []
    try:
        while len(results) < len(processes):
            try:results.append(results_queue.get(timeout=.5))
            except queue.Empty:
                if all(not process.is_alive() for process in processes):break
        for process in processes:process.join()
        while len(results) < len(processes):
            try:results.append(results_queue.get_nowait())
            except queue.Empty:break
        if len(results) != len(processes) or any(process.exitcode != 0 for process in processes):
            results.append({'status': 'FAILED', 'error': {'stage': 'spawn_worker_exit', 'error_type': 'WorkerExitedWithoutReceipt'}})
    finally:results_queue.close()
    return results


def export_results(root, contract, contract_sha, *, worker_results=None, invocation=None):
    """SQLite流式导出，不重写全局逐图JSON journal，不碰原图。"""
    root = Path(root);db = connect(root/'state.sqlite3')
    try:
        totals = counts(db)
        exports = {}
        for state, name in (('PASSED', 'passed.jsonl'), ('REJECTED', 'rejected.jsonl'), ('FAILED', 'failed.jsonl')):
            target = root/name;descriptor, temporary_name = tempfile.mkstemp(prefix='.export-', dir=root)
            digest = hashlib.sha256();written = 0
            try:
                with os.fdopen(descriptor, 'wb') as stream:
                    for item in db.execute('SELECT receipt_json,receipt_sha256 FROM recipes WHERE state=? AND receipt_json IS NOT NULL ORDER BY position', (state,)):
                        receipt = json.loads(item['receipt_json'])
                        if sha_bytes(json_bytes(receipt)) != item['receipt_sha256']:raise IntegrityError('SQLite cached receipt body changed')
                        row = {**receipt['output_record'], 'preprocess_receipt_sha256': item['receipt_sha256']}
                        adoption=db.execute('SELECT provenance_json FROM adoptions WHERE identifier=?',(receipt['identifier'],)).fetchone()
                        if adoption:row['phase1_adoption_provenance']=json.loads(adoption[0])
                        content = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
                        stream.write(content);digest.update(content);written += 1
                    stream.flush();os.fsync(stream.fileno())
                if written != totals[state]:
                    raise IntegrityError('Terminal state has missing receipts; reconcile before export')
                Path(temporary_name).replace(target)
            finally:Path(temporary_name).unlink(missing_ok=True)
            value = record_verified_sha256(target, digest.hexdigest(), fingerprint(target))
            exports[name] = {'sha256': value, 'fingerprint': list(fingerprint(target)), 'records': written}
        worker_failed = any(value.get('status') == 'FAILED' for value in (worker_results or []))
        complete = not worker_failed and totals['PENDING'] == totals['RUNNING'] == totals['FAILED'] == 0 and sum(totals.values()) == contract['recipe_count']
        adopted=db.execute('SELECT COUNT(*) FROM adoptions').fetchone()[0]
        summary = {'schema_version': VERSION, 'contract_sha256': contract_sha, 'state': 'PREPROCESS_COMPLETE' if complete else 'PARTIAL',
            'counts': totals, 'declared_recipes': contract['recipe_count'], 'ready_for_encoding': totals['PASSED'],
            'manual_released_images': 0, 'training_approved_images': 0, 'normalization_frozen': False,
            'adopted_previous_PASSED':adopted,'newly_preprocessed_PASSED':totals['PASSED']-adopted,
            'exports': exports, 'worker_results': worker_results or [], 'invocation': invocation,
            'execution_failed': worker_failed or totals['FAILED'] > 0,
            'original_mri_or_masks_inspected_during_resume': False, 'completion_is_numeric_phase1_only': True}
        publish_json(root/'summary.json', summary);publish_json(root/'RUN_STATUS.json', summary)
        if complete:
            publish_json(root/'PREPROCESS_COMPLETE', {'schema_version': VERSION, 'contract_sha256': contract_sha,
                'counts': totals, 'recipe_count': contract['recipe_count'], 'exports': exports,
                'normalization_frozen': False, 'manual_released_images': 0, 'training_approved_images': 0}, exclusive=True)
        return summary
    finally:db.close()


def completed_summary(root, contract_sha):
    root = Path(root);marker = json.loads((root/'PREPROCESS_COMPLETE').read_text())
    if marker['contract_sha256'] != contract_sha:raise IntegrityError('Completion marker belongs to another protocol')
    for name, proof in marker['exports'].items():
        path = root/name
        if tuple(proof['fingerprint']) != fingerprint(path) and file_sha256(path) != proof['sha256']:
            raise IntegrityError('Published phase1 export changed')
    summary = json.loads((root/'summary.json').read_text())
    if summary.get('contract_sha256') != contract_sha or summary.get('state') != 'PREPROCESS_COMPLETE' or summary.get('counts') != marker['counts'] or summary.get('training_approved_images') != 0 or summary.get('manual_released_images') != 0:
        raise IntegrityError('Completion summary identity/counts/release flags changed')
    return summary


def run(args):
    root, contract, contract_sha = initialize(args)
    if (root/'PREPROCESS_COMPLETE').exists():return completed_summary(root, contract_sha)
    if not args.execute:
        db = connect(root/'state.sqlite3', readonly=True)
        try:totals = counts(db)
        finally:db.close()
        return {'schema_version': VERSION, 'state': 'PLAN_ONLY', 'counts': totals, 'declared_recipes': contract['recipe_count'],
                'cuda_initialized': False, 'models_loaded': 0, 'training_approved_images': 0,
                'next_invocation_max_images': args.max_images, 'gpu_indices_if_execute': args.gpu_indices}
    if getattr(args,'adopt_from_stopped_run',None) is not None:
        from .preprocess_adoption import adopt_from_stopped_run
        adopt_from_stopped_run(root,args.adopt_from_stopped_run)
    with lock(root/'.runner.lock'):
        # 活着的旧worker持SH锁；父进程崩溃也不能让新控制器并发重做。
        with lock(root/'.workers.lock'):
            reconcile(root, contract_sha, interrupted=True)
            db = connect(root/'state.sqlite3')
            try:
                remaining = db.execute("SELECT COUNT(*) FROM recipes WHERE state NOT IN('PASSED','REJECTED')").fetchone()[0]
                quota = min(remaining, args.max_images) if args.max_images else remaining
                if remaining and not args.gpu_indices:raise ValueError('--execute unfinished sources requires explicit --gpu-indices')
                invocation = uuid.uuid4().hex
                meta_set(db, 'active_invocation', invocation)
                db.execute('INSERT INTO dispatch(invocation,quota) VALUES(?,?)', (invocation, args.max_images))
                db.execute('INSERT INTO invocations VALUES(?,?,NULL,?,?)', (invocation, time.time(), 'RUNNING', json.dumps({'gpu_indices': args.gpu_indices, 'max_images': args.max_images})));db.commit()
            finally:db.close()
        worker_results = spawn_workers(root, contract, contract_sha, invocation, args.gpu_indices[:quota]) if remaining else []
        with lock(root/'.workers.lock'):
            reconcile(root, contract_sha, interrupted=True)
            summary = export_results(root, contract, contract_sha, worker_results=worker_results, invocation=invocation)
            db = connect(root/'state.sqlite3')
            try:db.execute('UPDATE invocations SET finished=?,status=?,info_json=? WHERE token=?', (time.time(), summary['state'], json.dumps(worker_results), invocation));db.commit()
            finally:db.close()
        return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipes', type=Path, required=True, help='显式selected_by_user/pipeline_protocol_reviewed recipes JSONL')
    parser.add_argument('--resource-manifest', type=Path, required=True, help='已核许可/大小/SHA的MNI/FNIT权重资源JSON')
    parser.add_argument('--fnit-runtime', type=Path, required=True, help='成熟FNIT冻结源码根，含SOURCE_MANIFEST.json/src')
    parser.add_argument('--output-dir', type=Path, required=True, help='SQLite状态/逐图回执/候选导出目录')
    parser.add_argument('--registered-output-dir', type=Path, required=True, help='持久MNI2mm brain/mask/FOV目录，禁止latents')
    parser.add_argument('--gpu-indices', type=int, nargs='*', default=[], help='执行时显式物理GPU，每卡一个spawn worker')
    parser.add_argument('--extent', type=int, choices=[192, 256], default=192, help='SynthMorph extent；不自动降参数')
    parser.add_argument('--minimum-mask-dice', type=float, default=.65, help='brainmask/template最低Dice')
    parser.add_argument('--gpu-budget-gb', type=float, default=20., help='每卡PyTorch allocator十进制GB上限，最大20')
    parser.add_argument('--max-images', type=int, default=DEFAULT_MAX_IMAGES, help='本次未完成recipe配额；默认128；0逐张处理全部显式recipes，有限额时保持PARTIAL')
    parser.add_argument('--adopt-from-stopped-run',type=Path,help='显式绑定原producer；plan-only只预建，execute前从已停止run复用相同recipe的PASSED')
    parser.add_argument('--minimum-free-gib', type=float, default=30., help='模型建立前GPU最低free GiB，至少30')
    parser.add_argument('--stable-free-seconds', type=float, default=15., help='GPU连续free窗口秒数，至少15；仅每worker开始等待一次')
    parser.add_argument('--minimum-registered-free-gib', type=float, default=1024., help='每张处理前/发布前注册盘剩余GiB阈值，至少30；不足记FAILED并停止当前dispatch')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='明确执行才允许GPU，默认只建计划')
    mode.add_argument('--plan-only', action='store_true', help='默认模式；只建/检查源清单与协议，完全不GPU')
    return parser.parse_args(argv)


def main(argv=None):
    try:
        result = run(parse_args(argv))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        return 1 if result.get('execution_failed') else 0
    except Exception as error:
        detail = _error(error, 'controller_setup_or_resume')
        print(json.dumps({'state': 'FAILED_CONTROLLER', 'error_type': type(error).__name__, 'reason': detail['message'], 'stage': detail['stage'], 'approved': False}), flush=True)
        return 1


if __name__ == '__main__':raise SystemExit(main())
