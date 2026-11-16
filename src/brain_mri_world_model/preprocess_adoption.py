"""已停止phase1结果→新脑部位过滤run的metadata复用；不读取或重哈希MRI。"""
from __future__ import annotations

import argparse
from contextlib import ExitStack,contextmanager
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import time

from .integrity import file_sha256,fingerprint
from . import preprocess_runner as p

VERSION='brain_mri_preprocess_adoption_v1'
NUMERICAL_CODE=('prepare.py','registered.py','masked_volume.py','integrity.py')
NUMERICAL_FIELDS=('resource_manifest_sha256','fnit_source_manifest_sha256','fnit_source_sha256',
    'software','extent','minimum_mask_dice','gpu_budget_gb','cpu_threads','dtype','tf32','require_review','normalization')
VOXEL_FIELDS=('preprocessed_volume_sha256','preprocessed_brain_mask_sha256','preprocessed_fov_sha256')


@contextmanager
def _old_lock(path):
    """源run锁文件必须已存在；不创建或改写旧run的任何文件内容。"""
    try:stream=Path(path).open('r+')
    except FileNotFoundError as error:raise p.IntegrityError('Old producer lock is missing: '+str(path)) from error
    try:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('Old producer/controller/worker is still active; adoption refused') from error
        yield
    finally:stream.close()


def numerical_protocol_equal(old,new):
    for key in NUMERICAL_FIELDS:
        if key not in old or key not in new or p.json_bytes(old[key])!=p.json_bytes(new[key]):
            raise p.IntegrityError('Adoption numerical/resource protocol differs: '+key)
    old_resources=old.get('resources');new_resources=new.get('resources')
    if not isinstance(old_resources,dict) or not old_resources or not isinstance(new_resources,dict) or set(old_resources)!=set(new_resources):
        raise p.IntegrityError('Adoption requires the same explicit resources')
    for key in old_resources:
        left={k:v for k,v in old_resources[key].items() if k!='path'}
        right={k:v for k,v in new_resources[key].items() if k!='path'}
        if not left or p.json_bytes(left)!=p.json_bytes(right):raise p.IntegrityError('Adoption resource content differs: '+key)
    for name in NUMERICAL_CODE:
        left=old.get('code_sha256',{}).get(name);right=new.get('code_sha256',{}).get(name)
        if not isinstance(left,str) or len(left)!=64 or left!=right:
            raise p.IntegrityError('Adoption numerical implementation differs: '+name)
    old_code=old.get('code_sha256',{});new_code=new.get('code_sha256',{})
    for name in (set(old_code)|set(new_code))-{'preprocess_runner.py','preprocess_adoption.py'}:
        if name not in old_code or name not in new_code or old_code[name]!=new_code[name]:
            raise p.IntegrityError('Unreviewed producer code difference: '+name)
    # Paths for output/recipes/Python/runtime, controller code, GPU scheduling,
    # free-space guards are bookkeeping. Numeric code/resources/parameters above
    # must still be identical. Do not infer equivalence from matching filenames.


def _validated_old_receipt(item,new_item,contract_sha):
    old_recipe=json.loads(item['recipe_json']);new_recipe=json.loads(new_item['recipe_json'])
    if (item['recipe_sha256']!=new_item['recipe_sha256'] or p.json_bytes(old_recipe)!=p.json_bytes(new_recipe)
            or p.sha_bytes(p.json_bytes(old_recipe))!=item['recipe_sha256']):return None
    if item['state']!='PASSED':return None
    if not item['receipt_json'] or not item['receipt_sha256']:raise p.IntegrityError('Selected old PASSED lacks bound receipt')
    receipt=json.loads(item['receipt_json'])
    if p.sha_bytes(p.json_bytes(receipt))!=item['receipt_sha256']:raise p.IntegrityError('Old cached receipt body SHA differs')
    p.validate_receipt(receipt,item,contract_sha)
    if receipt.get('source_recipe')!=old_recipe:raise p.IntegrityError('Old receipt/source recipe differ')
    output=receipt.get('output_record',{});registered=receipt.get('registered',{})
    for key in ('dataset','subject_id','image_id','modality','split'):
        if output.get(key)!=new_recipe.get(key):raise p.IntegrityError('Old processed identity differs from new recipe')
    for key in VOXEL_FIELDS:
        value=registered.get(key)
        if not isinstance(value,str) or len(value)!=64 or output.get(key)!=value:
            raise p.IntegrityError('Old standard brain/mask/FOV voxel provenance is incomplete')
        recorded=receipt.get('preprocessor_metadata',{}).get(key)
        if recorded is not None and recorded!=value:raise p.IntegrityError('Old standard voxel provenance contradicts producer metadata')
    proof=receipt.get('source_provenance')
    if not isinstance(proof,dict) or not proof:raise p.IntegrityError('Old PASSED has no original source proof')
    if item['source_proof'] and json.loads(item['source_proof'])!=proof:raise p.IntegrityError('Old original-source proof differs from its receipt')
    return receipt


def adopt_from_stopped_run(new_run,old_run):
    """只导入selected PASSED；资源/recipe严格一致，旧REJECTED/新增项仍PENDING。

    不读取原图/标准图、不核它们当前fingerprint、不改旧SQLite或旧receipt。
    原producer contract与receipt保留原值，新adoptions表只记bookkeeping lineage。
    """
    new_run,old_run=(Path(value).resolve() for value in (new_run,old_run))
    if new_run==old_run:raise ValueError('New and old phase1 runs must differ')
    if not old_run.is_dir():raise FileNotFoundError(old_run)
    if (new_run/'state.sqlite3').is_file() and (old_run/'state.sqlite3').is_file() and os.path.samefile(new_run/'state.sqlite3',old_run/'state.sqlite3'):
        raise p.IntegrityError('New state must not be a hardlink/symlink to the old SQLite')
    started=time.monotonic()
    with ExitStack() as stack:
        # Fail before any target mutation when the source is still active.
        for name in ('.initialize.lock','.runner.lock','.workers.lock'):stack.enter_context(_old_lock(old_run/name))
        for name in ('.initialize.lock','.runner.lock','.workers.lock'):stack.enter_context(p.lock(new_run/name))
        old_contract=json.loads((old_run/'preprocess_contract.json').read_text());old_sha=file_sha256(old_run/'preprocess_contract.json')
        new_contract=json.loads((new_run/'preprocess_contract.json').read_text());new_sha=file_sha256(new_run/'preprocess_contract.json')
        declaration=new_contract.get('adoption_source')
        if declaration!={'run_dir':str(old_run),'contract_sha256':old_sha}:
            raise p.IntegrityError('New contract does not explicitly bind this stopped producer')
        if old_contract.get('schema_version')!=p.VERSION or new_contract.get('schema_version')!=p.VERSION:
            raise p.IntegrityError('Unsupported phase1 producer schema')
        if old_contract.get('adoption_source') is not None:
            raise p.IntegrityError('Recursive adoption producers are not supported; select the original producer run')
        numerical_protocol_equal(old_contract['protocol'],new_contract['protocol'])
        old_state=old_run/'state.sqlite3';before=list(fingerprint(old_state))
        old=p.connect(old_state,readonly=True);new=p.connect(new_run/'state.sqlite3')
        try:
            if p.meta_get(old,'contract')!=old_contract or not p.meta_get(old,'initialization_complete'):
                raise p.IntegrityError('Old SQLite/contract is not a complete initialized producer')
            if p.meta_get(new,'contract')!=new_contract or not p.meta_get(new,'initialization_complete'):
                raise p.IntegrityError('New SQLite/contract is not a complete initialized filtered run')
            old_adoptions=old.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='adoptions'").fetchone()
            if old_adoptions and old.execute('SELECT COUNT(*) FROM adoptions').fetchone()[0]:
                raise p.IntegrityError('Recursive adoption rows are not supported; select original producers')
            previous=p.meta_get(new,'adoption_complete')
            if previous:
                if previous['old_contract_sha256']!=old_sha or previous['old_SQLite_fingerprint']!=before:
                    raise p.IntegrityError('Stopped source changed after adoption')
                p.publish_json(new_run/'ADOPTION_COMPLETE.json',previous,exclusive=True)
                return dict(previous,metadata_only_resume=True)
            if new.execute("SELECT COUNT(*) FROM recipes WHERE state='RUNNING'").fetchone()[0]:
                raise p.IntegrityError('New run has unresolved RUNNING rows; reconcile before adoption')
            old.execute('BEGIN');new.execute('BEGIN IMMEDIATE')
            adopted=0;outside_selection=0;recipe_mismatch=0;position=0
            while True:
                # Scan only the already-completed old index, rather than doing
                # hundreds of thousands of lookups for every new PENDING image.
                page=old.execute("SELECT * FROM recipes WHERE state='PASSED' AND position>? ORDER BY position LIMIT 256",(position,)).fetchall()
                if not page:break
                for producer in page:
                    position=producer['position']
                    item=new.execute("SELECT * FROM recipes WHERE identifier=? AND state='PENDING'",(producer['identifier'],)).fetchone()
                    if item is None:outside_selection+=1;continue
                    receipt=_validated_old_receipt(producer,item,old_sha)
                    if receipt is None:recipe_mismatch+=1;continue
                    receipt_path=Path(producer['receipt_path']);receipt_path=receipt_path if receipt_path.is_absolute() else old_run/receipt_path
                    lineage=dict(schema_version=VERSION,origin='selected_PASSED_from_stopped_run',metadata_only_adoption=True,
                        new_preprocessing_performed=False,producer_run_dir=str(old_run),producer_contract_sha256=old_sha,
                        producer_receipt_path=str(receipt_path),producer_receipt_sha256=producer['receipt_sha256'],
                        producer_recipe_sha256=producer['recipe_sha256'],producer_attempt=producer['attempt'],
                        adopted_into_run_dir=str(new_run),adopted_into_contract_sha256=new_sha)
                    new.execute('INSERT INTO adoptions(identifier,producer_contract_sha256,provenance_json) VALUES(?,?,?)',
                        (item['identifier'],old_sha,json.dumps(lineage,sort_keys=True)))
                    new.execute("UPDATE recipes SET state='PASSED',attempt=?,receipt_path=?,receipt_sha256=?,receipt_fingerprint=?,receipt_json=?,source_proof=?,owner=NULL WHERE identifier=?",
                        (producer['attempt'],str(receipt_path),producer['receipt_sha256'],producer['receipt_fingerprint'],producer['receipt_json'],
                         producer['source_proof'] or json.dumps(receipt['source_provenance'],sort_keys=True),item['identifier']))
                    adopted+=1
            if list(fingerprint(old_state))!=before:raise p.IntegrityError('Stopped producer SQLite changed during adoption')
            remaining=new.execute("SELECT COUNT(*) FROM recipes WHERE state='PENDING'").fetchone()[0]
            result=dict(schema_version=VERSION,state='ADOPTION_COMPLETE',old_run_dir=str(old_run),old_contract_sha256=old_sha,
                old_SQLite_fingerprint=before,new_run_dir=str(new_run),new_contract_sha256=new_sha,
                adopted_selected_PASSED=adopted,pending_new_or_old_nonPASSED=remaining-recipe_mismatch,pending_recipe_mismatch=recipe_mismatch,
                old_PASSED_outside_new_pending_selection=outside_selection,
                MRI_or_mask_or_latent_files_opened=0,new_MRI_processing_performed=False,old_receipts_rewritten=0,
                seconds=time.monotonic()-started)
            p.meta_set(new,'adoption_complete',result);new.commit()
        except Exception:new.rollback();raise
        finally:new.close();old.close()
        p.publish_json(new_run/'ADOPTION_COMPLETE.json',result,exclusive=True)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--new-run',type=Path,required=True);parser.add_argument('--old-run',type=Path,required=True)
    args=parser.parse_args(argv)
    result=adopt_from_stopped_run(args.new_run,args.old_run);print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':main()
