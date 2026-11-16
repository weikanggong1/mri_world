"""Six -> eight FSDP Adam state conversion. No training/forward/backward step.

Run collect under six ranks, then reshard under eight. Default only describes.
The source checkpoint is trusted private input, never an arbitrary pickle.
"""
import argparse,errno,fcntl,hashlib,json,os,pathlib,random
P=pathlib.Path

def need(ok,why):
    if not ok:raise ValueError(why)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def fp(p):
    s=p.stat();return[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def read_json(p):
    need(p.is_file()and not p.is_symlink()and p.stat().st_size<4*1024*1024,'BOUNDED_PRIVATE_CONTROL_REQUIRED');a=fp(p);raw=p.read_bytes();need(fp(p)==a,'CONTROL_CHANGED_DURING_READ');return json.loads(raw),hashlib.sha256(raw).hexdigest()
def preserve_model_copy(source,destination,expected_sha256):
    # Same project strategy as frozen fresh_epoch_archive.copy_file: CoW first,
    # one streamed copy otherwise. Never hardlink: that changes origin ctime.
    need(isinstance(expected_sha256,str)and len(expected_sha256)==64,'EXISTING_ACCEPTED_MODEL_SHA256_REQUIRED')
    before=fp(source);mode='CoW_clone_from_accepted_immutable_source';h=None;total=0
    srcfd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW)
    with os.fdopen(srcfd,'rb')as src,destination.open('xb')as dst:
        need(fp(source)==before,'SOURCE_MODEL_CHANGED_BEFORE_COPY')
        try:fcntl.ioctl(dst.fileno(),0x40049409,src.fileno());total=before[2]
        except OSError as e:
            need(e.errno in {errno.EXDEV,errno.EINVAL,errno.ENOTTY,errno.EOPNOTSUPP,errno.ENOSYS},'MODEL_CLONE_UNEXPECTED_ERROR')
            mode='single_stream_copy_SHA_during_write';h=hashlib.sha256();dst.truncate(0);src.seek(0)
            while True:
                block=src.read(8*1024*1024)
                if not block:break
                dst.write(block);h.update(block);total+=len(block)
        dst.flush();os.fsync(dst.fileno())
    need(fp(source)==before,'SOURCE_MODEL_CHANGED_DURING_COPY')
    need(destination.stat().st_size==total==before[2]and fp(destination)[:2]!=before[:2],'INDEPENDENT_COMPLETE_MODEL_FILE_REQUIRED')
    if h is not None:need(h.hexdigest()==expected_sha256,'MODEL_FIRST_WRITE_STREAM_SHA_MISMATCH')
    return {'mode':mode,'source_file_state_unchanged':before,'destination_file_state':fp(destination),'accepted_source_sha256':expected_sha256,'separate_postwrite_rehash':False,'SHA_computed_in_copy_write':h is not None}

def source_metadata(root, *, expected_metadata_sha256, expected_training_code_sha256):
    need(not root.is_symlink()and(root/'COMPLETE').read_text().strip()=='brain_mri_world_model_checkpoint_v1','COMPLETE_ORIGINAL_CHECKPOINT_REQUIRED')
    m,h=read_json(root/'metadata.json');need(m['world_size']==6,'ORIGINAL_SIX_RANK_CHECKPOINT_REQUIRED')
    need(h==expected_metadata_sha256,'EXACT_ACCEPTED_ORIGIN_METADATA_REQUIRED_NO_REWRITE')
    need(m['contract']['micro_batch']==1,'CURRENT_WINDOW_MIGRATION_REQUIRES_MICRO_BATCH_ONE')
    need(m['contract'].get('optimizer')=='AdamW(0.9,0.999,1e-8)'and m['contract'].get('mode')=='full_dit','ORIGINAL_FULL_DIT_ADAMW_ABI_REQUIRED')
    need(m['contract'].get('training_code_sha256')==expected_training_code_sha256,'EXACT_ORIGINAL_FSDP_PARAMETER_ORDER_SOURCE_REQUIRED')
    need(root.name=='checkpoint-'+format(m['optimizer_step'],'09d'),'CHECKPOINT_STEP_NAME_MISMATCH')
    for k in('epoch','next_batch','optimizer_step','train_examples_seen'):need(type(m[k])is int and m[k]>=0,'ORIGINAL_CURSOR_REQUIRED:'+k)
    for n in['model.pt']+['rank-'+format(r,'05d')+'.pt'for r in range(6)]:need((root/n).is_file()and(root/n).stat().st_size>0 and not(root/n).is_symlink(),'COMPLETE_MODEL_AND_ALL_ORIGINAL_OPTIMIZER_RNG_SHARDS_REQUIRED:'+n)
    return m,h

def initialize(expected_world):
    import torch
    import torch.distributed as dist
    need(torch.__version__=='2.6.0+cu124','ORIGINAL_TORCH_260_CU124_ENVIRONMENT_REQUIRED')
    need(int(os.environ.get('WORLD_SIZE','0'))==expected_world,'EXPLICIT_CORRECT_TORCHRUN_WORLD_SIZE_REQUIRED')
    local=int(os.environ['LOCAL_RANK']);torch.set_num_threads(1);torch.cuda.set_device(local);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    dist.init_process_group('nccl');return torch,dist,dist.get_rank(),torch.device('cuda',local)

def model_and_optimizer(torch,device,model_config,checkpoint,m):
    import functools,importlib.metadata
    from diffusers import WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    need(importlib.metadata.version('diffusers')=='0.35.2','ORIGINAL_DIFFUSERS_0352_REQUIRED')
    config,_=read_json(model_config)
    expected={'in_channels':48,'out_channels':48,'num_layers':30,'num_attention_heads':24,'attention_head_dim':128,'text_dim':4096}
    need(all(config.get(k)==v for k,v in expected.items())and tuple(config.get('patch_size',[]))==(1,2,2),'EXACT_ORIGINAL_WAN22_TI2V5B_ARCHITECTURE')
    need(hashlib.sha256(model_config.read_bytes()).hexdigest()==m['contract']['model']['config_sha256'],'ORIGINAL_MODEL_CONFIG_CONTENT_REQUIRED')
    # Same Diffusers class/config/attention/FP32 as mature wan.load_backbone.
    # Full trained checkpoint overwrites every parameter before FSDP/optimizer.
    model=WanTransformer3DModel.from_config(config).float();model.enable_gradient_checkpointing();model.set_attn_processor(WanAttnProcessor());model.requires_grad_(True).train()
    before=fp(checkpoint/'model.pt');state=torch.load(checkpoint/'model.pt',map_location='cpu',weights_only=True,mmap=True)
    need(all(not t.is_floating_point()or t.dtype==torch.float32 for t in state.values()),'FULL_FP32_MODEL_REQUIRED')
    model.load_state_dict(state,strict=True);del state;need(fp(checkpoint/'model.pt')==before,'ORIGINAL_MODEL_CHANGED_DURING_MMAP_LOAD')
    policy=functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={type(model.blocks[0])})
    model=FSDP(model,auto_wrap_policy=policy,sharding_strategy=ShardingStrategy.FULL_SHARD,device_id=device,use_orig_params=True,sync_module_states=True,limit_all_gathers=True)
    c=m['contract'];optim=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],betas=(0.9,0.999),eps=1e-8,weight_decay=c['weight_decay'])
    return model,optim

def full_optimizer(model,optim):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,StateDictType,FullStateDictConfig,FullOptimStateDictConfig
    with FSDP.state_dict_type(model,StateDictType.FULL_STATE_DICT,FullStateDictConfig(offload_to_cpu=True,rank0_only=True),FullOptimStateDictConfig(offload_to_cpu=True,rank0_only=True)):
        return FSDP.optim_state_dict(model,optim)
def same_tree(a,b,torch,path='root'):
    if torch.is_tensor(a):
        need(torch.is_tensor(b)and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),'OPTIMIZER_ROUNDTRIP_TENSOR_DIFFERENCE:'+path);return
    need(type(a)is type(b),'OPTIMIZER_ROUNDTRIP_TYPE_DIFFERENCE:'+path)
    if isinstance(a,dict):
        need(a.keys()==b.keys(),'OPTIMIZER_ROUNDTRIP_KEYS_DIFFERENCE:'+path)
        for k in a:same_tree(a[k],b[k],torch,path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        need(len(a)==len(b),'OPTIMIZER_ROUNDTRIP_LENGTH_DIFFERENCE:'+path)
        for i,(x,y)in enumerate(zip(a,b)):same_tree(x,y,torch,path+'/'+str(i))
    else:need(a==b,'OPTIMIZER_ROUNDTRIP_SCALAR_DIFFERENCE:'+path)
def cpu_clone(x,torch):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return{k:cpu_clone(v,torch)for k,v in x.items()}
    if isinstance(x,list):return[cpu_clone(v,torch)for v in x]
    if isinstance(x,tuple):return tuple(cpu_clone(v,torch)for v in x)
    return x

def collect(args):
    m,msha=source_metadata(args.checkpoint,expected_metadata_sha256=args.expected_metadata_sha256,expected_training_code_sha256=args.expected_training_code_sha256);torch,dist,rank,device=initialize(6)
    try:
        if rank==0:args.export_dir.mkdir(parents=True,exist_ok=False)
        dist.barrier();model,optim=model_and_optimizer(torch,device,args.model_config,args.checkpoint,m)
        path=args.checkpoint/('rank-'+format(rank,'05d')+'.pt');before=fp(path)
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True);need(fp(path)==before,'SOURCE_RANK_PAYLOAD_CHANGED')
        need(set(payload)=={'optimizer','scheduler','rng'},'ORIGINAL_COMPLETE_RANK_PAYLOAD_ABI_REQUIRED');optim.load_state_dict(payload['optimizer']);need(fp(path)==before,'SOURCE_RANK_PAYLOAD_CHANGED_DURING_STATE_LOAD')
        scheduler_states=[None]*6;rng_states=[None]*6
        dist.all_gather_object(scheduler_states,payload['scheduler']);dist.all_gather_object(rng_states,payload['rng'])
        need(all(digest(s)==digest(scheduler_states[0])for s in scheduler_states),'ORIGINAL_SCHEDULER_STATES_MUST_AGREE')
        full=full_optimizer(model,optim)
        if rank==0:
            out=args.export_dir/'full_optimizer.pt';torch.save({'optimizer':full,'scheduler':scheduler_states[0],'original_rank_rng':rng_states,'source_metadata':m},out)
            marker={'schema_version':'mri_world_six_rank_named_optimizer_export_v1','source_checkpoint':str(args.checkpoint.resolve()),'source_metadata_sha256':msha,'source_model_file_state':fp(args.checkpoint/'model.pt'),'accepted_source_model_sha256':args.source_model_sha256,'full_optimizer_file_state':fp(out),'source_world_size':6,'optimizer_step':m['optimizer_step'],'no_optimizer_updates_executed':True,'scheduler_state_sha256':digest(scheduler_states[0])}
            with(args.export_dir/'EXPORT_COMPLETE.json').open('x')as f:json.dump(marker,f,indent=2)
        dist.barrier()
    finally:dist.destroy_process_group()

def rng_mapping(source):
    return [{'physical_rank':rank,'kind':'original_rank_rng','source_rank':rank}if rank<6 else
            {'physical_rank':rank,'kind':'explicit_derived_new_rank_rng','seed':int(digest([source['contract']['seed'],source['optimizer_step'],'new_topology_rank',rank])[:16],16)%(2**63-1)}for rank in range(8)]
def derived_rng(torch,rank,source):
    import numpy as np
    seed=rng_mapping(source)[rank]['seed']
    cpu=torch.Generator(device='cpu').manual_seed(seed);cuda=torch.Generator(device=torch.device('cuda',torch.cuda.current_device())).manual_seed(seed)
    return {'python':random.Random(seed).getstate(),'numpy':np.random.RandomState(seed%2**32).get_state(),'torch':cpu.get_state(),'cuda':cuda.get_state()}
def reshard(args):
    m,msha=source_metadata(args.checkpoint,expected_metadata_sha256=args.expected_metadata_sha256,expected_training_code_sha256=args.expected_training_code_sha256);marker,_=read_json(args.export_dir/'EXPORT_COMPLETE.json');need(marker['source_metadata_sha256']==msha and marker['source_checkpoint']==str(args.checkpoint.resolve())and marker['source_model_file_state']==fp(args.checkpoint/'model.pt'),'EXACT_ORIGINAL_METADATA_EXPORT_AND_MODEL_REQUIRED');path=args.export_dir/'full_optimizer.pt';need(fp(path)==marker['full_optimizer_file_state'],'FULL_OPTIMIZER_EXPORT_CHANGED')
    contract,_=read_json(args.destination_contract);need(contract['migration']['origin_metadata_sha256']==msha and contract['migration']['physical_world_size']==8,'ACTUAL_DESTINATION_MIGRATION_CONTRACT_REQUIRED')
    torch,dist,rank,device=initialize(8)
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        model,optim=model_and_optimizer(torch,device,args.model_config,args.checkpoint,m)
        # Only rank0 keeps the complete moments. Broadcast small scheduler/RNG controls.
        data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)if rank==0 else None
        if rank==0:need(data['source_metadata']==m and fp(path)==marker['full_optimizer_file_state'],'EXACT_EXPORTED_ORIGIN_METADATA_AND_FILE_STATE_REQUIRED')
        control=[{'scheduler':data['scheduler'],'original_rank_rng':data['original_rank_rng']}if rank==0 else None];dist.broadcast_object_list(control,src=0);control=control[0]
        local=FSDP.scatter_full_optim_state_dict(data['optimizer']if rank==0 else None,model,optim=optim);optim.load_state_dict(local);del local
        # Compare all named state tensors, Adam steps and complete parameter groups
        # before writing any destination checkpoint. No optimizer.step is called.
        roundtrip=full_optimizer(model,optim);validation=[None]
        if rank==0:
            try:same_tree(data['optimizer'],roundtrip,torch);validation[0]={'passed':True}
            except Exception as e:validation[0]={'passed':False,'error':str(e)}
        dist.broadcast_object_list(validation,src=0);need(validation[0]['passed'],'FULL_OPTIMIZER_ROUNDTRIP_MUST_BE_EXACT:'+str(validation[0]));del roundtrip
        temp=args.output_run_dir/(args.checkpoint.name+'.writing')
        if rank==0:
            args.output_run_dir.mkdir(parents=True,exist_ok=False);temp.mkdir();(args.output_run_dir/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
            model_copy=preserve_model_copy(args.checkpoint/'model.pt',temp/'model.pt',marker['accepted_source_model_sha256'])
            (temp/'MODEL_COPY_PROVENANCE.json').write_text(json.dumps(model_copy,indent=2)+'\n')
        dist.barrier();rng=control['original_rank_rng'][rank]if rank<6 else derived_rng(torch,rank,m)
        torch.save({'optimizer':cpu_clone(optim.state_dict(),torch),'scheduler':control['scheduler'],'rng':rng},temp/('rank-'+format(rank,'05d')+'.pt'))
        dist.barrier()
        if rank==0:
            metadata=dict(m,world_size=8,contract=contract,migration={'origin_metadata_sha256':msha,'named_optimizer_roundtrip_exact':True,'logical_cursor_preserved':True,'first_six_rank_rng_preserved':True,'new_rank_rng_scheme':'SHA256(original_seed,step,new_topology_rank,rank)','physical_rank_rng_mapping':rng_mapping(m),'original_six_rng_archive':str(args.export_dir/'full_optimizer.pt'),'bitwise_training_trajectory_claimed':False})
            (temp/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n');(temp/'COMPLETE').write_text('brain_mri_world_model_checkpoint_v1\n');temp.replace(args.output_run_dir/args.checkpoint.name)
            (args.output_run_dir/'latest.json').write_text(json.dumps({'checkpoint':args.checkpoint.name})+'\n')
            (args.output_run_dir/'MIGRATION_COMPLETE.json').write_text(json.dumps({'schema_version':'mri_world_state_preserving_worldsize_migration_v1','source_metadata_sha256':msha,'source_world_size':6,'destination_world_size':8,'optimizer_state_exact_roundtrip':True,'cursor':{k:m[k]for k in('epoch','next_batch','optimizer_step','train_examples_seen')},'new_optimizer_updates':0,'bitwise_training_trajectory_claimed':False},indent=2)+'\n')
        dist.barrier()
    finally:dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('collect6','reshard8'))
    p.add_argument('--project-root',type=P,help='Explicit local project root containing all inputs and new outputs')
    p.add_argument('--checkpoint',type=P)
    p.add_argument('--model-config',type=P)
    p.add_argument('--export-dir',type=P)
    p.add_argument('--destination-contract',type=P)
    p.add_argument('--output-run-dir',type=P)
    p.add_argument('--source-model-sha256',help='Previously accepted model SHA; never recompute it here')
    p.add_argument('--expected-metadata-sha256',help='Previously accepted original metadata.json SHA')
    p.add_argument('--expected-training-code-sha256',help='Training source digest recorded in the original checkpoint')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:
        print(json.dumps({'state':'SOURCE_ONLY_NO_IO_NO_GPU','training_started':False,'weights_only_reset_allowed':False}));return
    need(a.phase and a.project_root and a.checkpoint and a.model_config and a.export_dir,'EXPLICIT_PROJECT_CHECKPOINT_CONFIG_AND_EXPORT_REQUIRED')
    for digest_value in (a.expected_metadata_sha256,a.expected_training_code_sha256):
        need(isinstance(digest_value,str)and len(digest_value)==64 and all(c in '0123456789abcdef'for c in digest_value),'EXPLICIT_ACCEPTED_ORIGIN_SHA256_REQUIRED')
    root=a.project_root.absolute()
    need(root.is_dir()and not root.is_symlink(),'EXISTING_EXPLICIT_PROJECT_ROOT_REQUIRED')
    for path in (root,a.checkpoint,a.model_config,a.export_dir,a.destination_contract,a.output_run_dir):
        if path is None:continue
        need(path.is_absolute()and path.is_relative_to(root),'ALL_INPUT_AND_OUTPUT_PATHS_MUST_BE_WITHIN_PROJECT_ROOT')
        for parent in (path,*path.parents):need(not parent.is_symlink(),'NO_SYMLINK_INPUT_OR_OUTPUT_PATH')
    if a.phase=='collect6':
        need(not a.export_dir.exists()and isinstance(a.source_model_sha256,str)and len(a.source_model_sha256)==64 and all(c in '0123456789abcdef'for c in a.source_model_sha256),'NEW_EXPORT_AND_EXISTING_ACCEPTED_MODEL_SHA_REQUIRED')
        collect(a)
    else:
        need(a.destination_contract and a.output_run_dir and not a.output_run_dir.exists(),'NEW_DESTINATION_CONTRACT_AND_RUN_REQUIRED')
        reshard(a)
if __name__=='__main__':main()
