"""数值预处理结果+已冻结归一化规则→可编码3D清单，不复制原始MRI。"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .integrity import file_sha256
from .prepare import atomic_json, normalization_spec


def read_policy(path):
    policy = json.loads(Path(path).read_text())
    if (policy.get('schema_version') != 'brain_mri_encoding_release_policy_v1' or
            policy.get('protocol_reviewed') is not True or policy.get('normalization_frozen') is not True):
        raise ValueError('Encoding requires a reviewed spatial protocol and frozen normalization policy')
    rules = {}
    for rule in policy.get('rules', []):
        key = (rule.get('dataset'), rule.get('modality'))
        if any(not isinstance(value, str) or not value for value in key) or key in rules:
            raise ValueError('Normalization rules require unique explicit dataset/modality keys')
        normalization_spec(rule)
        if rule.get('normalization_mode', 'robust_quantile') == 'fixed_range' and rule.get('fitted_from_data') is True:
            if rule.get('fit_splits') != ['train'] or not rule.get('fit_provenance'):
                raise ValueError('A fitted fixed intensity range must retain train-only provenance')
        rules[key] = rule
    if not rules:
        raise ValueError('No frozen normalization rules')
    dice = policy.get('minimum_mask_dice', .65)
    if type(dice) not in (int, float) or not math.isfinite(dice) or not 0 < dice <= 1:
        raise ValueError('Invalid reviewed minimum mask Dice')
    return policy, rules


def finalized_record(row, rule, policy):
    """只处理PASSED的标准图，源byte SHA首次绑定后由共享缓存复用。"""
    from .selection_utils import split_for_subject
    if (row.get('phase1_state') != 'PASSED' or row.get('ready_for_encoding') is not True or
            row.get('numeric_qc_passed') is not True or row.get('source_space') != 'FSL_MNI152' or
            row.get('source_skull_stripped') is not True or row.get('preprocess_mode') != 'already_mni'):
        raise ValueError('Only successfully published MNI2mm skull-stripped phase1 records can be finalized')
    if row['split'] != split_for_subject(row['dataset'], row['subject_id']):
        raise ValueError('Phase1 record differs from the common subject split')
    dice, coverage = row.get('mask_vs_template_dice'), row.get('brain_fov_coverage')
    minimum_coverage = rule.get('minimum_brain_fov_coverage', policy.get('minimum_brain_fov_coverage', .95))
    if type(minimum_coverage) not in (int, float) or not math.isfinite(minimum_coverage) or not 0 < minimum_coverage <= 1:
        raise ValueError('Invalid reviewed field coverage threshold')
    if (type(dice) not in (int, float) or not math.isfinite(dice) or not policy.get('minimum_mask_dice', .65) <= dice <= 1 or
            type(coverage) not in (int, float) or not math.isfinite(coverage) or not minimum_coverage <= coverage <= 1):
        raise ValueError('Phase1 spatial QA fails the frozen release threshold')
    fields = ('dataset', 'subject_id', 'image_id', 'modality', 'split', 'source_path', 'source_format',
              'source_component_index', 'source_mask_path', 'source_fov_mask_path', 'source_space',
              'source_skull_stripped', 'preprocess_mode')
    result = {name: row.get(name) for name in fields}
    result.update(approved=True, preprocess_qc_reviewed=True, registration_qc_passed=True,
                  minimum_brain_fov_coverage=minimum_coverage,
                  source_fov_mask_kind=row.get('fov_mask_kind', 'input_grid_domain'),
                  original_source_path=row.get('original_source_path'),
                  original_source_sha256=row.get('original_source_sha256'),
                  original_source_format=row.get('original_source_format'),
                  original_source_component_index=row.get('original_source_component_index'),
                  phase1_preprocess_receipt_path=row.get('preprocess_receipt_path'),
                  phase1_preprocess_receipt_sha256=row.get('preprocess_receipt_sha256'),
                  phase1_intensity_statistics=row.get('intensity_statistics'))
    result['source_grid_is_acquisition_grid'] = row.get('coverage_type') in ('complete_brain', 'partial_brain_slab')
    result['phase1_provenance'] = {key: row.get(key) for key in (
        'original_source_path', 'original_source_sha256', 'original_source_format', 'original_source_component_index',
        'preprocess_receipt_path', 'preprocess_receipt_sha256', 'preprocessed_volume_sha256',
        'preprocessed_brain_mask_sha256', 'preprocessed_fov_sha256', 'source_nonfinite_policy',
        'source_nan_replaced_with_zero', 'source_nan_fraction', 'source_nan_fill_retains_anatomical_mask')}
    adoption=row.get('phase1_adoption_provenance')
    if adoption is not None:
        if (not isinstance(adoption,dict) or adoption.get('metadata_only_adoption') is not True
                or adoption.get('new_preprocessing_performed') is not False
                or adoption.get('schema_version')!='brain_mri_preprocess_adoption_v1'
                or adoption.get('origin')!='selected_PASSED_from_stopped_run'
                or adoption.get('producer_receipt_path')!=row.get('preprocess_receipt_path')
                or adoption.get('producer_receipt_sha256')!=row.get('preprocess_receipt_sha256')
                or any(not isinstance(adoption.get(key),str) or len(adoption[key])!=64
                       or any(c not in '0123456789abcdef' for c in adoption[key])
                       for key in ('producer_contract_sha256','producer_receipt_sha256','producer_recipe_sha256','adopted_into_contract_sha256'))):
            raise ValueError('Adopted phase1 result must retain explicit original-producer lineage')
        result['phase1_provenance']['adoption']=adoption
    for name in ('normalization_mode', 'intensity_unit', 'intensity_range'):
        if name in rule:
            result[name] = rule[name]
    for path_key, digest_key in (('source_path', 'source_sha256'), ('source_mask_path', 'source_mask_sha256'),
                                 ('source_fov_mask_path', 'source_fov_mask_sha256')):
        if not isinstance(result.get(path_key), str) or not result[path_key]:
            raise ValueError('Published standard brain/mask/FOV path missing')
        result[digest_key] = file_sha256(result[path_key])
    return result


def finalize(passed_manifest, reviewed_policy, output_dir):
    from .preprocess_runner import publish_json
    passed_manifest, reviewed_policy, output_dir = map(Path, (passed_manifest, reviewed_policy, output_dir))
    if output_dir.exists():
        raise FileExistsError('Encoding list output exists; finalized lists are immutable')
    from .preprocess_runner import VERSION
    marker = json.loads((passed_manifest.parent / 'PREPROCESS_COMPLETE').read_text())
    contract_path = passed_manifest.parent / 'preprocess_contract.json'
    contract = json.loads(contract_path.read_text())
    counts = marker.get('counts', {})
    if (marker.get('schema_version') != VERSION or contract.get('schema_version') != VERSION or
            marker.get('contract_sha256') != file_sha256(contract_path) or not isinstance(counts, dict) or
            any(type(counts.get(key)) is not int or counts[key] < 0 for key in ('PENDING','RUNNING','PASSED','REJECTED','FAILED')) or
            any(counts[key] for key in ('PENDING','RUNNING','FAILED')) or
            marker.get('recipe_count') != contract.get('recipe_count') or
            sum(counts.values()) != contract.get('recipe_count')):
        raise ValueError('Phase1 complete marker does not bind a complete producer contract/state')
    expected = marker.get('exports', {}).get(passed_manifest.name)
    if (not isinstance(expected, dict) or not isinstance(expected.get('sha256'), str) or
            expected.get('records') != counts['PASSED']):
        raise ValueError('Passed list lacks its complete phase1 export provenance')
    policy, rules = read_policy(reviewed_policy)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.'+output_dir.name+'-pending-', dir=output_dir.parent))
    target, pending = staging/'selected.jsonl', staging/'pending.jsonl'
    source_digest, selected_digest = hashlib.sha256(), hashlib.sha256()
    selected_count = pending_count = total = 0
    try:
        with passed_manifest.open('rb') as source, target.open('xb') as selected, pending.open('x') as deferred:
            for line in source:
                source_digest.update(line)
                if not line.strip():
                    continue
                row = json.loads(line);total += 1
                adoption=row.get('phase1_adoption_provenance')
                if adoption is not None and (not isinstance(adoption,dict) or adoption.get('adopted_into_contract_sha256')!=marker['contract_sha256']):
                    raise ValueError('Adopted result is bound to another bookkeeping run')
                rule = rules.get((row.get('dataset'), row.get('modality')))
                if rule is None:
                    deferred.write(json.dumps({'dataset':row.get('dataset'), 'image_id':row.get('image_id'),
                        'modality':row.get('modality'), 'reason':'no_frozen_normalization_rule'}, ensure_ascii=False)+'\n')
                    pending_count += 1
                    continue
                record = finalized_record(row, rule, policy)
                content = (json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)+'\n').encode()
                selected.write(content);selected_digest.update(content);selected_count += 1
            selected.flush();os.fsync(selected.fileno())
            deferred.flush();os.fsync(deferred.fileno())
        if source_digest.hexdigest() != expected['sha256'] or total != expected['records']:
            raise ValueError('Phase1 export changed after publication')
        result = {'schema_version':'brain_mri_encoding_release_v1', 'selected_images':selected_count,
                  'pending_images':pending_count, 'phase1_passed_records':total,
                  'phase1_passed_path':str(passed_manifest.absolute()), 'phase1_passed_sha256':source_digest.hexdigest(),
                  'reviewed_policy_path':str(reviewed_policy.absolute()), 'reviewed_policy_sha256':file_sha256(reviewed_policy),
                  'selected_sha256':selected_digest.hexdigest(), 'source_MRI_or_masks_decoded':0,
                  'file_digest_policy':'first standard-file binding only; unchanged files use shared fingerprint cache'}
        publish_json(staging/'FINALIZED', result, exclusive=True)
        if output_dir.exists():
            raise FileExistsError('Another finalizer published this output')
        staging.rename(output_dir)  # Only complete approved lists become visible at their final path.
        return result
    except BaseException:
        atomic_json(staging/'INCOMPLETE', {'status':'failed_before_finalized_marker'})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--passed-manifest', type=Path, required=True)
    parser.add_argument('--reviewed-policy', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(finalize(args.passed_manifest, args.reviewed_policy, args.output_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
