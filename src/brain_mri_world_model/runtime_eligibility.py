"""Mandatory immutable runtime training/evaluation/source-finalize eligibility.

Only small JSON/JSONL metadata is opened. No tensor/MRI paths are statted or read.
"""
from __future__ import annotations
import fcntl,hashlib,json,os
from pathlib import Path

REGISTRY_VERSION='brain_mri_required_runtime_eligibility_registry_v1'
CONTEXT_VERSION='brain_mri_runtime_eligibility_applied_v1'


def fingerprint(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


def read_metadata(path,proof=None):
    path=Path(path).absolute();before=fingerprint(path);data=path.read_bytes()
    if fingerprint(path)!=before:raise ValueError('Runtime eligibility metadata changed while reading')
    digest=hashlib.sha256(data).hexdigest()
    if proof is not None and (str(path)!=proof['path'] or before!=proof['fingerprint'] or digest!=proof['sha256']):raise ValueError('Runtime eligibility metadata proof differs')
    return data,dict(path=str(path),sha256=digest,fingerprint=before)


def source_identity(row):
    phase=row.get('phase1_provenance') or {}
    def get(name):return phase.get(name) if phase.get(name) is not None else row.get(name)
    return dict(sha256=get('original_source_sha256'),format=get('original_source_format') or 'nifti',
                component=get('original_source_component_index'),frame=get('original_source_selected_frame_zero_based'),
                path=get('original_source_path'))


class RuntimeEligibility:
    def __init__(self,registry_path):
        if not registry_path:raise ValueError('Mandatory --runtime-eligibility-registry is required')
        self._lock=Path(str(Path(registry_path).absolute())+'.lock').open('a+')
        fcntl.flock(self._lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        raw,registry_proof=read_metadata(registry_path);registry=json.loads(raw)
        if registry.get('schema_version')!=REGISTRY_VERSION or registry.get('required') is not True:raise ValueError('A mandatory runtime eligibility registry is required')
        raw,context_proof=read_metadata(registry['applied']['path'],registry['applied']);context=json.loads(raw)
        if context.get('schema_version')!=CONTEXT_VERSION or context.get('state')!='APPLIED' or context.get('training_evaluation_downstream_hold_applied') is not True:raise ValueError('Runtime context is not APPLIED')
        raw,_=read_metadata(context['subjects']['path'],context['subjects']);subjects=[json.loads(v) for v in raw.splitlines() if v.strip()]
        raw,_=read_metadata(context['source_records']['path'],context['source_records']);records=[json.loads(v) for v in raw.splitlines() if v.strip()]
        self.subjects={(r['dataset'],r['subject_id']) for r in subjects}
        self.records={(r['dataset'],r['image_id']):r for r in records}
        if len(subjects)!=context['subjects']['records'] or len(self.subjects)!=len(subjects) or len(records)!=context['source_records']['records'] or len(self.records)!=len(records):raise ValueError('Runtime subject/source record count differs')
        if any((r['dataset'],r['subject_id']) not in self.subjects for r in records):raise ValueError('Runtime canonical hold lacks its subject closure')
        self.registry_path=Path(registry_path).absolute();self.registry_proof=registry_proof;self.context=context;self.context_proof=context_proof
        self.binding=dict(schema_version=REGISTRY_VERSION,required=True,registry=registry_proof,applied=context_proof,
                          version=context['version'],subjects_sha256=context['subjects']['sha256'],source_records_sha256=context['source_records']['sha256'],
                          subject_count=len(subjects),canonical_source_holds=len(records))

    def close(self):
        if getattr(self,'_lock',None) is not None:self._lock.close();self._lock=None

    def __del__(self):
        self.close()

    def assert_current(self):
        if fingerprint(self.registry_path)!=self.registry_proof['fingerprint'] or fingerprint(self.context_proof['path'])!=self.context_proof['fingerprint']:
            raise ValueError('Runtime eligibility version changed; rebuild/rebind metadata before use')

    def reason(self,row,*,strict_source=False):
        identity=(row.get('dataset'),row.get('image_id'));subject=(row.get('dataset'),row.get('subject_id'))
        held=self.records.get(identity)
        if held is not None:
            if any(row.get(k)!=held[k] for k in ('dataset','image_id','subject_id','modality','split')):raise ValueError('Runtime held canonical metadata/source identity changed')
            current=source_identity(row);expected=held['source']
            if strict_source and not current['sha256']:raise ValueError('Runtime held source requires original source proof')
            if current['sha256'] is not None and any(current[k]!=expected[k] for k in ('sha256','format','component','frame')):raise ValueError('Runtime held canonical source proof changed; no silent bypass')
            if strict_source and current['path'] is not None and os.path.normpath(current['path'])!=os.path.normpath(expected['path']):raise ValueError('Runtime held original source locator changed')
        return 'applied_runtime_subject_identity_training_eval_downstream_hold' if subject in self.subjects or held is not None else None

    def assert_allowed(self,row):
        if self.reason(row) is not None:raise ValueError('Runtime eligibility blocked held source before tensor access')

    def validate_finalizer_scope(self,passed_manifest,contract_sha):
        scope=self.context['producer_scope']
        if str(Path(passed_manifest).absolute().parent)!=scope['output_dir'] or contract_sha!=scope['preprocess_contract_sha256']:raise ValueError('Runtime-aware8764 finalizer requires its exact original complete producer')

    def validate_bound_context(self,marker):
        if marker.get('mandatory_runtime_eligibility')!=self.binding or marker.get('runtime_hold_applied') is not True:raise ValueError('InputReservations lacks mandatory current APPLIED runtime context')

    def excluded_image_indices(self,db):
        """Scalar indexed subject/canonical query only; never latent metadata files."""
        clauses=[];args=[]
        for dataset,sid in sorted(self.subjects):clauses.append('(dataset=? AND subject_id=?)');args.extend((dataset,sid))
        for dataset,iid in sorted(self.records):clauses.append('(dataset=? AND image_id=?)');args.extend((dataset,iid))
        if not clauses:return set()
        return {v[0] for v in db.execute('SELECT image_index FROM images WHERE '+' OR '.join(clauses),args)}


_WORKER_RUNTIME=None
_WORKER_PATH=None
def required_worker_runtime():
    global _WORKER_RUNTIME,_WORKER_PATH
    path=os.environ.get('BWM_RUNTIME_ELIGIBILITY_REGISTRY')
    if not path:raise ValueError('Mandatory runtime registry is missing from worker environment')
    if _WORKER_RUNTIME is None or path!=_WORKER_PATH:_WORKER_RUNTIME=RuntimeEligibility(path);_WORKER_PATH=path
    return _WORKER_RUNTIME


def filter_metadata(records,runtime):
    selected=[];held=0
    for row in records:
        if runtime.reason(row) is not None:held+=1
        else:selected.append(row)
    return selected,held


class RuntimeMetadataRecords(list):
    """Small-manifest metadata view pins the same mandatory context as samplers."""
    def __init__(self,rows,runtime):super().__init__(rows);self.runtime=runtime


def load_eligible_feature_records(feature_directory,runtime_eligibility_registry,*,runtime=None):
    """Mandatory downstream metadata entry; never opens feature tensors."""
    if runtime is None:runtime=RuntimeEligibility(runtime_eligibility_registry)
    elif not isinstance(runtime,RuntimeEligibility) or runtime.registry_path!=Path(runtime_eligibility_registry).absolute():raise ValueError('Downstream runtime registry differs')
    root=Path(feature_directory)
    body,_=read_metadata(root/'provenance.json');provenance=json.loads(body)
    body,_=read_metadata(root/'COMPLETE');complete=json.loads(body)
    if provenance.get('mandatory_runtime_eligibility')!=runtime.binding or complete.get('mandatory_runtime_eligibility')!=runtime.binding:
        raise ValueError('Feature metadata eligibility is pending current binding; verify/rebind metadata before downstream use; numerical features unchanged')
    def rows():
        count=0;before=fingerprint(root/'index.jsonl')
        with (root/'index.jsonl').open() as h:
            for line in h:
                if line.strip():
                    value=json.loads(line);runtime.assert_allowed(value);count+=1;yield value
        runtime.assert_current()
        if fingerprint(root/'index.jsonl')!=before or count!=complete['records']:raise ValueError('Feature metadata eligibility/count changed')
    return rows(),runtime.binding
