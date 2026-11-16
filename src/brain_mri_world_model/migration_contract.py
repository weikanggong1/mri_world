"""Bind an isolated destination contract to the unchanged scientific cohort/settings."""
import hashlib,json,pathlib
SCIENCE_KEYS=('epochs','micro_batch','gradient_accumulation','learning_rate','weight_decay','warmup_fraction','max_grad_norm','sigma_distribution','sigma_shift','epoch_axis_policy','seed','validation_seed','gradient_checkpointing','mode','optimizer','context_sha256')
def bind_contract(contract,origin_checkpoint,input_attestation,physical_world_size):
    origin=pathlib.Path(origin_checkpoint);raw=(origin/'metadata.json').read_bytes();m=json.loads(raw);old=m['contract'];a=json.loads(pathlib.Path(input_attestation).read_text())
    if m['world_size']!=6 or physical_world_size!=8:raise ValueError('EXPLICIT_SIX_TO_EIGHT_TOPOLOGY_REQUIRED')
    if old['micro_batch']!=1:raise ValueError('MIGRATION_MICRO_BATCH_ONE_REQUIRED')
    for key in SCIENCE_KEYS:
        if contract.get(key)!=old.get(key):raise ValueError('ORIGINAL_SCIENTIFIC_TRAINING_CONFIGURATION_CHANGED:'+key)
    for key in('torch','diffusers'):
        if contract['software'][key]!=old['software'][key]:raise ValueError('ORIGINAL_SOFTWARE_VERSION_REQUIRED:'+key)
    for key in('config_sha256','index_sha256','parameter_dtype','tf32','attention','pet_loader_sha256'):
        if contract['model'].get(key)!=old['model'].get(key):raise ValueError('ORIGINAL_MODEL_CONFIGURATION_REQUIRED:'+key)
    if a.get('state')!='SOURCE_TARGET_SEMANTIC_INPUT_ORDER_IDENTICAL' or a.get('source_manifest_sha256')!=old['manifest_sha256']or a.get('target_manifest_sha256')!=contract['manifest_sha256']:raise ValueError('ACTUAL_SOURCE_TARGET_INPUT_ORDER_ATTESTATION_REQUIRED')
    if contract['data_loading']!=old['data_loading']:raise ValueError('ORIGINAL_LOADER_CONFIGURATION_REQUIRED')
    if ('input_index' in old)!=('input_index' in contract):raise ValueError('ORIGINAL_INPUT_INDEX_PROTOCOL_REQUIRED')
    if 'input_index' in old:
        previous=old['input_index'];current=contract['input_index']
        if previous['runtime_eligible_summary']!=current['runtime_eligible_summary']:raise ValueError('ORIGINAL_EFFECTIVE_COHORT_COUNTS_CHANGED')
        for key in ('version','subjects_sha256','source_records_sha256','subject_count','canonical_source_holds'):
            if previous['mandatory_runtime_eligibility'][key]!=current['mandatory_runtime_eligibility'][key]:raise ValueError('ORIGINAL_EFFECTIVE_COHORT_MEMBERSHIP_CHANGED:'+key)

    contract=dict(contract);contract['migration']={'schema_version':'mri_world_original_optimizer_window_cursor_v1','origin_metadata_sha256':hashlib.sha256(raw).hexdigest(),'origin_contract':old,'logical_world_size':6,'physical_world_size':8,'logical_cursor_unit':'original_global_batch','same_valid_examples_per_optimizer_update':True,'semantic_input_sha256':a['semantic_rows_sha256'],'source_manifest_sha256':old['manifest_sha256'],'target_manifest_sha256':contract['manifest_sha256'],'RNG_assignment_and_reduction_order_may_change':True}
    contract['migration']['adapter_source_sha256'] = {name: hashlib.sha256(pathlib.Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('migration_contract.py','migration_cursor.py','migration_model.py')}
    return contract
