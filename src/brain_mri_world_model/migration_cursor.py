"""Preserve old optimizer windows while assigning their work to a new rank count.

Software scheduling metadata only. With micro_batch=1, different ranks may read
volumes of different shapes in a wave; the FSDP parameter collective order stays
unchanged. Sample-level RNG assignment changes with topology and is explicit.
"""
import hashlib,json,math

def window_plan(logical_batches, *, source_world_size, target_world_size,
                micro_batch, gradient_accumulation, next_batch=0):
    if micro_batch!=1:raise ValueError('MIGRATION_CURRENT_PROTOCOL_REQUIRES_MICRO_BATCH_ONE')
    if not(0<source_world_size<=target_world_size and gradient_accumulation>0):raise ValueError('POSITIVE_TOPOLOGY_AND_ACCUMULATION_REQUIRED')
    if not 0<=next_batch<=len(logical_batches):raise ValueError('ORIGINAL_LOGICAL_CURSOR_OUTSIDE_EPOCH')
    if next_batch!=len(logical_batches)and next_batch%gradient_accumulation:raise ValueError('CHECKPOINT_MUST_BE_AT_ORIGINAL_OPTIMIZER_WINDOW_BOUNDARY')
    width=source_world_size*micro_batch;physical=target_world_size*micro_batch
    for start in range(next_batch,len(logical_batches),gradient_accumulation):
        stop=min(start+gradient_accumulation,len(logical_batches));old=logical_batches[start:stop]
        if any(len(batch)!=width for batch in old):raise ValueError('ORIGINAL_LOGICAL_BATCH_WIDTH_CHANGED')
        flat=[(int(index),float(weight))for batch in old for index,weight in batch]
        if not flat or any(weight not in(0.,1.)for _,weight in flat):raise ValueError('ORIGINAL_ZERO_ONE_PADDING_PROTOCOL_REQUIRED')
        valid=sum(weight for _,weight in flat)
        if valid<=0:raise ValueError('NO_VALID_EXAMPLES_IN_ORIGINAL_WINDOW')
        padded=list(flat);padded.extend([(flat[-1][0],0.)]*((-len(flat))%physical))
        waves=[padded[i:i+physical]for i in range(0,len(padded),physical)]
        yield {'logical_start':start,'logical_stop':stop,'physical_batches':waves,
               'valid_count':int(valid),'source_entries':len(flat),'padding_added':len(padded)-len(flat)}

def flatten_physical_batches(plan):
    return [batch for window in plan for batch in window['physical_batches']]

def protocol(*,source_world_size,target_world_size,micro_batch,gradient_accumulation):
    return {'schema_version':'mri_world_original_optimizer_window_cursor_v1',
            'logical_world_size':source_world_size,'physical_world_size':target_world_size,
            'micro_batch':micro_batch,'original_gradient_accumulation':gradient_accumulation,
            'cursor_unit':'original_global_batch','same_valid_example_multiset_per_optimizer_update':True,
            'RNG_assignment_and_floating_point_reduction_order_may_change':True}
