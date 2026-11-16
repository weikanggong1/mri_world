"""Create a new private package; frozen input/public/active source is never edited."""
import argparse,ast,hashlib,json,pathlib,shutil
ROOT=pathlib.Path(__file__).parent
EXACT_TRAIN_SHA='54a63fa9b0b237f8f9238259c51336fe5e13eb6e26beecef49efe1ebfdca1a82'
def once(source,old,new):
 if source.count(old)!=1:raise ValueError('EXACT_SINGLE_MATURE_TRAINING_ANCHOR_REQUIRED:'+old[:100])
 return source.replace(old,new,1)
def render(source_package,output_package):
 source_package=pathlib.Path(source_package);output_package=pathlib.Path(output_package);source=(source_package/'train.py').read_text()
 if hashlib.sha256(source.encode()).hexdigest()!=EXACT_TRAIN_SHA:raise ValueError('EXACT_REVIEWED_MATURE_TRAIN_SOURCE_REQUIRED')
 original=source
 source=once(source,'from .data_loading import epoch_axis_indices, make_loader, CudaPrefetcher','from .data_loading import epoch_axis_indices, make_loader, CudaPrefetcher\nfrom .migration_cursor import window_plan, flatten_physical_batches\nfrom .migration_contract import bind_contract')
 anchor='    run_dir = Path(args.output_dir).resolve()\n    validate_run_directory'
 injected='''    contract = bind_contract(contract, args.migration_origin_checkpoint,
                             args.migration_input_attestation, world_size)
    if args.migration_contract_output:
        with Path(args.migration_contract_output).open('x') as stream:
            json.dump(contract, stream, indent=2)
        return 0
'''
 source=once(source,anchor,injected+anchor)

 old_cuda='''    if not torch.cuda.is_available():
        raise RuntimeError("全量微调要求 CUDA；未自动回退 CPU 或降低精度")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")'''
 new_cuda='''    if args.migration_contract_output:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Contract-only mode uses one CPU process, no torchrun")
        world_size = 8
        device = torch.device("cpu")
    else:
'''+ '\n'.join('    '+line for line in old_cuda.splitlines())
 source=once(source,old_cuda,new_cuda)
 source=once(source,'    torch.cuda.manual_seed(args.seed + rank)','    if not args.migration_contract_output:\n        torch.cuda.manual_seed(args.seed + rank)')

 source=once(source,'    model = load_backbone(args.model_dir, True, pet_code_root=args.pet_code_root)',
     '    from .migration_model import initialize_from_config\n    model = initialize_from_config(args.model_dir, args.pet_code_root)')
 source=once(source,'    if not args.migration_contract_output:\n        torch.cuda.manual_seed(args.seed + rank)',
     '    if not args.migration_contract_output:\n        if not args.resume:\n            raise ValueError("Converted full checkpoint resume is mandatory; no random/weights-only start")\n        torch.cuda.manual_seed(args.seed + rank)')
 source=once(source,'nominal_update_examples = args.micro_batch * world_size * args.gradient_accumulation','nominal_update_examples = args.micro_batch * 6 * args.gradient_accumulation')
 source=once(source,'global_batches = batches_for_epoch(training, args.micro_batch, world_size,','global_batches = batches_for_epoch(training, args.micro_batch, 6,')
 source=once(source,'        batch_sampler.set_batches(global_batches[start_batch:])','''        migration_windows = list(window_plan(global_batches, source_world_size=6,
            target_world_size=world_size, micro_batch=args.micro_batch,
            gradient_accumulation=args.gradient_accumulation, next_batch=start_batch))
        batch_sampler.set_batches(flatten_physical_batches(migration_windows))''')
 source=once(source,'''        for window_start in range(start_batch, len(global_batches), args.gradient_accumulation):
            started = time.monotonic()
            window = global_batches[window_start:window_start + args.gradient_accumulation]''','''        for migration_window in migration_windows:
            window_start = migration_window['logical_start']
            started = time.monotonic()
            window = migration_window['physical_batches']''')
 source=once(source,'next_batch=window_start + len(window),','next_batch=migration_window[\'logical_stop\'],')
 source=once(source,'    args = parser.parse_args()','''    parser.add_argument('--migration-origin-checkpoint', required=True)
    parser.add_argument('--migration-input-attestation', required=True)
    parser.add_argument('--migration-contract-output')
    args = parser.parse_args()''')
 tree=ast.parse(source);oldtree=ast.parse(original);same={}
 for name in('batches_for_epoch','local_batch','flow_losses','sample_sigmas','rng_state','restore_rng','read_resume_metadata','restore_checkpoint','training_progress_from_checkpoint','validate','save_checkpoint'):
  a=next(n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name==name);b=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name);same[name]=ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
 if not all(same.values()):raise ValueError('ORIGINAL_NUMERICAL_OR_RESUME_GUARD_FUNCTION_CHANGED')
 loop_same={}
 def assignments(tree,name):
  return [ast.dump(n,include_attributes=False)for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(z,ast.Name)and z.id==name for z in n.targets)]
 for name in ('valid_count','losses','loss','finite','grad_norm','scheduler'):
  loop_same[name]=assignments(tree,name)==assignments(oldtree,name)
 for owner in ('optimizer','scheduler'):
  def steps(t):return[ast.dump(n,include_attributes=False)for n in ast.walk(t)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id==owner and n.func.attr=='step']
  loop_same[owner+'.step']=steps(tree)==steps(oldtree)
 if not all(loop_same.values()):raise ValueError('ORIGINAL_LOSS_NORMALIZATION_OR_OPTIMIZER_SCHEDULER_UPDATE_CHANGED')

 shutil.copytree(source_package,output_package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'));(output_package/'train.py').write_text(source)
 shutil.copy2(ROOT/'migration_cursor.py',output_package/'migration_cursor.py');shutil.copy2(ROOT/'migration_contract.py',output_package/'migration_contract.py');shutil.copy2(ROOT/'migration_model.py',output_package/'migration_model.py')
 return {'state':'ISOLATED_TRAINING_DERIVATIVE_CREATED_NOT_RUN','original_train_sha256':EXACT_TRAIN_SHA,'new_train_sha256':hashlib.sha256(source.encode()).hexdigest(),'unchanged_function_AST':same,'unchanged_loop_math_AST':loop_same,'logical_world_size':6,'physical_world_size':8,'same_valid_examples_per_optimizer_update':True,'bitwise_trajectory_claimed':False}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-package');p.add_argument('--output-package');p.add_argument('--execute',action='store_true');a=p.parse_args()
 if not a.execute:print(json.dumps({'state':'SOURCE_ONLY_NO_RENDER_NO_GPU'}));return
 if not a.source_package or not a.output_package:p.error('--source-package and --output-package are required with --execute')
 print(json.dumps(render(a.source_package,a.output_package),indent=2))
if __name__=='__main__':main()
