"""从phase1完成清单的脑内统计拟合归一化候选；不读取MRI/latent、不批准编码。"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

POLICY_VERSION='brain_mri_normalization_fit_policy_v1'
CANDIDATE_VERSION='brain_mri_normalization_candidate_v1'
PHASE1_VERSION='brain_mri_preprocess_phase1_v2'
DATASETS=frozenset({'ukb','mrrate','tiantan','fomo'})
IDENTITY=('dataset','subject_id','image_id','modality','split')
PROBABILITIES=[.005,.5,.995]


def _json_bytes(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()


def _number(value):
    return type(value) in (int,float) and math.isfinite(value)


def _sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _small_json(path):
    from .integrity import fingerprint,record_verified_sha256
    path=Path(path);before=fingerprint(path);body=path.read_bytes()
    value=json.loads(body);digest=hashlib.sha256(body).hexdigest()
    record_verified_sha256(path,digest,before)
    return value,digest


def _policy(value):
    if isinstance(value,(str,Path)):
        if Path(value).suffix.lower()!='.json':raise ValueError('Normalization policy must be a JSON metadata file')
        value,_=_small_json(value)
    if not isinstance(value,dict) or value.get('schema_version')!=POLICY_VERSION:
        raise ValueError('Expected an explicit normalization fit policy schema')
    if value.get('fit_splits',['train'])!=['train']:
        raise ValueError('Normalization fitting is train-only; val/test fit requests are forbidden')
    minimum=value.get('minimum_train_images',32)
    outer=value.get('outer_quantiles',[.01,.99])
    rules=value.get('rules')
    if not isinstance(rules,list) or not rules:raise ValueError('Explicit dataset/modality rules are required')
    result={}
    for raw in rules:
        if not isinstance(raw,dict):raise ValueError('Each normalization rule must be an object')
        key=(raw.get('dataset'),raw.get('modality'))
        if not isinstance(key[0],str) or key[0] not in DATASETS or not isinstance(key[1],str) or not key[1] or key in result:
            raise ValueError('Each rule needs one unique canonical dataset/modality')
        if raw.get('fit_splits',['train'])!=['train']:
            raise ValueError('Rule-level normalization fitting is train-only; val/test requests are forbidden')
        strategy=raw.get('strategy')
        if strategy not in ('robust_quantile','train_fixed_range','fixed_range'):
            raise ValueError('Explicit robust_quantile/train_fixed_range/fixed_range strategy is required')
        count=raw.get('minimum_train_images',minimum)
        if type(count) is not int or count<1:raise ValueError('minimum_train_images must be a positive integer')
        unit=raw.get('intensity_unit','source-intensity units')
        confirmed=raw.get('physical_unit_confirmed',False)
        evidence=raw.get('unit_evidence')
        if not isinstance(unit,str) or not unit or type(confirmed) is not bool:
            raise ValueError('Explicit intensity unit and boolean confirmation are required')
        if confirmed and (not isinstance(evidence,str) or not evidence.strip()):
            raise ValueError('Confirmed units require explicit source evidence, never a modality-name inference')
        rule=dict(dataset=key[0],modality=key[1],strategy=strategy,minimum_train_images=count,
                  intensity_unit=unit,physical_unit_confirmed=confirmed,unit_evidence=evidence)
        if strategy=='robust_quantile':
            if raw.get('quantiles',[.005,.995])!=[.005,.995]:
                raise ValueError('Available phase1 statistics support robust quantiles005/995 only')
            rule['quantiles']=[.005,.995]
        elif strategy=='train_fixed_range':
            probabilities=raw.get('outer_quantiles',outer)
            if (not isinstance(probabilities,list) or len(probabilities)!=2 or
                    not all(_number(x) for x in probabilities) or not 0<=probabilities[0]<probabilities[1]<=1):
                raise ValueError('outer_quantiles must be two ordered finite probabilities in[0,1]')
            rule['outer_quantiles']=probabilities
        else:
            bounds=raw.get('intensity_range')
            if (not isinstance(bounds,list) or len(bounds)!=2 or not all(_number(x) for x in bounds) or bounds[1]<=bounds[0]):
                raise ValueError('Explicit fixed_range requires a finite increasing intensity_range')
            provenance=raw.get('range_evidence')
            if not isinstance(provenance,str) or not provenance.strip():
                raise ValueError('Externally fixed ranges require explicit range_evidence')
            rule.update(intensity_range=bounds,range_evidence=provenance)
        result[key]=rule
    return value,result


def _statistics(value):
    if (not isinstance(value,dict) or value.get('space')!='preprocessed_brain' or value.get('finite') is not True
            or type(value.get('count')) is not int or value['count']<2
            or not _number(value.get('min')) or not _number(value.get('max')) or value['max']<=value['min']):
        raise ValueError('Missing finite, nonconstant preprocessed_brain intensity statistics')
    probabilities=value.get('quantile_probabilities');quantiles=value.get('quantiles')
    if (not isinstance(probabilities,list) or probabilities!=PROBABILITIES or any(not _number(p) for p in probabilities)
            or not isinstance(quantiles,list) or len(quantiles)!=3 or not all(_number(q) for q in quantiles)
            or [value['min'],*quantiles,value['max']]!=sorted([value['min'],*quantiles,value['max']])):
        raise ValueError('Intensity statistics require ordered005/50/995 quantiles within min/max')
    return quantiles


def _fallback_statistics(row,manifest,contract_sha):
    path=row.get('preprocess_receipt_path');expected=row.get('preprocess_receipt_sha256')
    if not isinstance(path,str) or not _sha(expected):raise ValueError('Receipt fallback requires a bound receipt path/SHA')
    path=Path(path);path=path if path.is_absolute() else manifest.parent/path
    if path.suffix!='.json':raise ValueError('Receipt fallback opens JSON metadata only')
    receipt,digest=_small_json(path)
    output=receipt.get('output_record',{})
    if (digest!=expected or receipt.get('schema_version')!=PHASE1_VERSION or receipt.get('state')!='PASSED'
            or receipt.get('contract_sha256')!=contract_sha or any(output.get(k)!=row[k] for k in IDENTITY)):
        raise ValueError('Receipt fallback identity/state/contract or digest differs from passed export')
    return receipt.get('preprocessor_metadata',{}).get('intensity_statistics')


def _linear_quantile(db,key,column,probability,count):
    if column not in ('qlow','qhigh'):raise ValueError('Unsupported quantile column')
    position=(count-1)*probability;lower=math.floor(position);upper=math.ceil(position)
    values=[row[0] for row in db.execute(
        f'SELECT {column} FROM samples WHERE dataset=? AND modality=? ORDER BY {column},image_id LIMIT ? OFFSET ?',
        (*key,upper-lower+1,lower))]
    return values[0]+(values[-1]-values[0])*(position-lower)


def _fit_rule(db,key,rule):
    count,subjects,voxels,minlow,maxhigh,degenerate=db.execute('''SELECT COUNT(*),COUNT(DISTINCT subject_id),SUM(voxels),
      MIN(qlow),MAX(qhigh),SUM(qlow>=qhigh) FROM samples WHERE dataset=? AND modality=?''',key).fetchone()
    if count<rule['minimum_train_images']:
        raise ValueError(f'{key}: insufficient train statistics: {count} < {rule["minimum_train_images"]}')
    identities=hashlib.sha256()
    for subject,image in db.execute('SELECT subject_id,image_id FROM samples WHERE dataset=? AND modality=? ORDER BY subject_id,image_id',key):
        identities.update(_json_bytes([key[0],subject,image])+b'\n')
    result=dict(rule,status='candidate_pending_review_and_freeze',approved=False,
        fitted_from_data=rule['strategy']=='train_fixed_range',
        train_images=count,train_subjects=subjects,train_voxels=voxels,train_ids_sha256=identities.hexdigest(),
        train_ids_digest_format='sha256 of sorted JSON[dataset,subject_id,image_id] lines; UTF8 compact separators',
        train_q005_min=minlow,train_q995_max=maxhigh,train_degenerate_q005_q995_images=degenerate,
        fit_splits=['train'],statistics_space='preprocessed_brain')
    if rule['strategy']=='robust_quantile':
        if degenerate:raise ValueError(f'{key}: robust normalization has {degenerate} degenerate per-image005/995 ranges')
        result.update(normalization_mode='robust_quantile',intensity_range=None,
                      estimator='per_image_preprocessed_brain_q005_q995; no globally fitted bounds')
    else:
        if rule['strategy']=='train_fixed_range':
            low=_linear_quantile(db,key,'qlow',rule['outer_quantiles'][0],count)
            high=_linear_quantile(db,key,'qhigh',rule['outer_quantiles'][1],count)
            if not math.isfinite(low) or not math.isfinite(high) or high<=low:
                raise ValueError(f'{key}: fitted fixed range is degenerate')
            bounds=[low,high]
            result['estimator']='linear outer quantiles of train per-volume q005 and q995 distributions; each MRI equal weight'
        else:
            bounds=rule['intensity_range'];result['estimator']='explicit_source_defined_range_not_fitted_to_data'
        result.update(normalization_mode='fixed_range',intensity_range=bounds,
            train_images_q005_below_range=db.execute('SELECT COUNT(*) FROM samples WHERE dataset=? AND modality=? AND qlow<?',(*key,bounds[0])).fetchone()[0],
            train_images_q995_above_range=db.execute('SELECT COUNT(*) FROM samples WHERE dataset=? AND modality=? AND qhigh>?',(*key,bounds[1])).fetchone()[0])
    return result


def _publish(path,value):
    from .integrity import record_verified_sha256,fingerprint
    content=json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()+b'\n'
    descriptor,name=tempfile.mkstemp(prefix='.normalization-candidate-',dir=path.parent)
    temporary=Path(name)
    try:
        with os.fdopen(descriptor,'wb') as stream:
            stream.write(content);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:temporary.unlink(missing_ok=True)
    record_verified_sha256(path,hashlib.sha256(content).hexdigest(),fingerprint(path))


def fit_from_passed_manifest(passed_manifest,policy,output,*,allow_receipt_fallback=False):
    """流式读取一个/多个完整phase1 passed.jsonl，返回待审核policy对象并独占发布JSON。

    默认只用行内intensity_statistics；allow_receipt_fallback=True才打开旧receipt JSON。
    policy为显式dict或JSON路径；output为新JSON路径。没有MRI、latent或模型I/O。
    """
    from .integrity import fingerprint,record_verified_sha256,file_sha256
    raw_policy,rules=_policy(policy)
    if type(allow_receipt_fallback) is not bool:raise ValueError('allow_receipt_fallback must be boolean')
    output=Path(output).absolute()
    if output.exists():raise FileExistsError('Normalization candidate output already exists')
    output.parent.mkdir(parents=True,exist_ok=True)
    manifests=[passed_manifest] if isinstance(passed_manifest,(str,Path)) else list(passed_manifest)
    if not manifests:raise ValueError('At least one completed passed manifest is required')
    paths=[Path(value).absolute() for value in manifests]
    if len(set(paths))!=len(paths):raise ValueError('Repeated passed manifest input')
    sources=[];counts=Counter({name:0 for name in ('input_records','train_statistics_used','skipped_val',
        'skipped_test','skipped_train_without_policy','receipt_fallback_records')})
    with tempfile.TemporaryDirectory(prefix='.normalization-fit-',dir=output.parent) as temporary:
        db=sqlite3.connect(str(Path(temporary)/'stats.sqlite3'))
        try:
            db.executescript('''CREATE TABLE identities(dataset TEXT,image_id TEXT,subject_id TEXT,modality TEXT,split TEXT,PRIMARY KEY(dataset,image_id));
              CREATE TABLE subjects(dataset TEXT,subject_id TEXT,split TEXT,PRIMARY KEY(dataset,subject_id));
              CREATE TABLE samples(dataset TEXT,modality TEXT,image_id TEXT,subject_id TEXT,qlow REAL,qmedian REAL,qhigh REAL,voxels INTEGER,
                PRIMARY KEY(dataset,image_id));
              CREATE INDEX sample_low ON samples(dataset,modality,qlow,image_id);
              CREATE INDEX sample_high ON samples(dataset,modality,qhigh,image_id);''')
            for path in paths:
                if path.name!='passed.jsonl':raise ValueError('Input must be the completed phase1 passed.jsonl export')
                marker,marker_sha=_small_json(path.parent/'PREPROCESS_COMPLETE')
                contract,contract_sha=_small_json(path.parent/'preprocess_contract.json')
                proof=marker.get('exports',{}).get('passed.jsonl',{})
                expected=proof.get('sha256');stored_fp=proof.get('fingerprint')
                terminal_counts=marker.get('counts',{})
                if (marker.get('schema_version')!=PHASE1_VERSION or marker.get('contract_sha256')!=contract_sha
                        or contract.get('schema_version')!=PHASE1_VERSION or not _sha(expected)
                        or not isinstance(stored_fp,list) or len(stored_fp)!=5 or any(type(x) is not int for x in stored_fp)
                        or type(proof.get('records')) is not int or proof['records']<0
                        or terminal_counts.get('PASSED')!=proof['records']
                        or any(type(terminal_counts.get(key)) is not int or terminal_counts[key]<0
                               for key in ('PASSED','REJECTED','FAILED','PENDING','RUNNING'))
                        or any(terminal_counts[key]!=0 for key in ('FAILED','PENDING','RUNNING'))
                        or type(marker.get('recipe_count')) is not int
                        or marker['recipe_count']!=contract.get('recipe_count')
                        or marker['recipe_count']!=terminal_counts['PASSED']+terminal_counts['REJECTED']):
                    raise ValueError('PREPROCESS_COMPLETE does not bind the phase1 contract/passed export')
                before=fingerprint(path);digest=hashlib.sha256();rows=0
                with path.open('rb') as stream:
                    for body in stream:
                        digest.update(body)
                        if not body.strip():continue
                        row=json.loads(body);rows+=1;counts['input_records']+=1
                        if (not isinstance(row,dict) or any(not isinstance(row.get(k),str) or not row[k] for k in IDENTITY)
                                or row['dataset'] not in DATASETS or row['split'] not in ('train','val','test')
                                or row.get('phase1_state')!='PASSED' or row.get('numeric_qc_passed') is not True
                                or row.get('ready_for_encoding') is not True):
                            raise ValueError('Only canonical numeric-PASSED phase1 records are accepted')
                        existing=db.execute('SELECT split FROM subjects WHERE dataset=? AND subject_id=?',(row['dataset'],row['subject_id'])).fetchone()
                        if existing and existing[0]!=row['split']:raise ValueError('Canonical subject crosses train/val/test splits')
                        db.execute('INSERT OR IGNORE INTO subjects VALUES(?,?,?)',(row['dataset'],row['subject_id'],row['split']))
                        try:db.execute('INSERT INTO identities VALUES(?,?,?,?,?)',tuple(row[k] for k in ('dataset','image_id','subject_id','modality','split')))
                        except sqlite3.IntegrityError as error:raise ValueError('Duplicate canonical image across passed inputs') from error
                        if row['split']!='train':counts['skipped_'+row['split']]+=1;continue
                        key=(row['dataset'],row['modality'])
                        if key not in rules:counts['skipped_train_without_policy']+=1;continue
                        value=row.get('intensity_statistics')
                        if value is None:
                            if not allow_receipt_fallback:
                                raise ValueError('Missing inline intensity_statistics: pending statistics; explicit receipt fallback is disabled')
                            value=_fallback_statistics(row,path,contract_sha);counts['receipt_fallback_records']+=1
                        quantiles=_statistics(value)
                        db.execute('INSERT INTO samples VALUES(?,?,?,?,?,?,?,?)',(*key,row['image_id'],row['subject_id'],*quantiles,value['count']))
                        counts['train_statistics_used']+=1
                        if rows%1024==0:db.commit()
                actual=digest.hexdigest()
                if rows!=proof['records'] or actual!=expected or fingerprint(path)!=before:
                    raise ValueError('Passed export content/count/fingerprint changed or differs from PREPROCESS_COMPLETE')
                record_verified_sha256(path,actual,before)
                sources.append(dict(manifest_path=str(path),manifest_sha256=actual,manifest_fingerprint=list(before),
                    fingerprint_matches_complete=list(before)==stored_fp,records=rows,
                    complete_sha256=marker_sha,preprocess_contract_sha256=contract_sha))
            db.commit()
            policies=[_fit_rule(db,key,rules[key]) for key in sorted(rules)]
        finally:db.close()
    result=dict(schema_version=CANDIDATE_VERSION,status='candidate_pending_review_and_freeze',approved=False,
        normalization_frozen=False,release_for_encoding=False,training_approval=False,
        normalization_fitter_code_sha256=file_sha256(__file__),
        fit_splits=['train'],policy_spec_sha256=hashlib.sha256(_json_bytes(raw_policy)).hexdigest(),
        source_manifests=sources,counts=dict(counts),policies=policies,
        receipt_fallback_enabled=allow_receipt_fallback,
        source_volume_or_latent_files_opened=0,model_or_GPU_used=False,
        limitations=['Receipt/inline statistics reflect phase1 numeric processing, not manual release.',
                    'Per-volume quantile summaries do not measure pooled voxel tails or exact clipping fractions.',
                    'Source units remain unconfirmed unless explicit evidence was supplied; modality names do not establish units.',
                    'This candidate must be reviewed and frozen before VAE release.'])
    for fitted in result['policies']:
        fitted['fit_provenance']=dict(fit_splits=['train'],train_images=fitted['train_images'],
            train_subjects=fitted['train_subjects'],train_ids_sha256=fitted['train_ids_sha256'],
            statistics_space='preprocessed_brain',summary_quantile_probabilities=PROBABILITIES,
            estimator=fitted['estimator'],source_passed_manifest_sha256=[source['manifest_sha256'] for source in sources],
            policy_spec_sha256=result['policy_spec_sha256'],
            normalization_fitter_code_sha256=result['normalization_fitter_code_sha256'])
    _publish(output,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passed-manifest',type=Path,action='append',required=True)
    parser.add_argument('--policy',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-receipt-fallback',action='store_true')
    args=parser.parse_args(argv)
    result=fit_from_passed_manifest(args.passed_manifest,args.policy,args.output,allow_receipt_fallback=args.allow_receipt_fallback)
    print(json.dumps(dict(status=result['status'],policies=len(result['policies']),counts=result['counts'])),flush=True)


if __name__=='__main__':main()
