"""对已审核 latent 清单进行 Wan2.2 5B 全量 DiT 微调；显式运行才启动。"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import functools
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist

from .wan import (backbone_provenance, fixed_seed, load_backbone, load_batch,
                  load_context, load_manifest)
from .data_loading import epoch_axis_indices, make_loader, CudaPrefetcher


def batches_for_epoch(records, micro_batch, world_size, seed, epoch, shuffle, axis_policy='all_axes'):
    """同形状组成全局批次；尾部占位权重为零，不重复计入训练/验证。"""
    groups = defaultdict(list)
    if axis_policy not in ('one_per_image', 'all_axes'):
        raise ValueError('Unknown epoch axis policy')
    if hasattr(records, 'epoch_view_metadata'):
        views = records.epoch_view_metadata(seed, epoch if shuffle else 0, axis_policy)
    else:
        indices = (epoch_axis_indices(records, seed, epoch if shuffle else 0)
                   if axis_policy == 'one_per_image' else range(len(records)))
        views = ((index, records[index]['latent_shape']) for index in indices)
    for index, shape in views:
        groups[tuple(shape)].append(index)
    generator = random.Random(seed + epoch * 1000003)
    batches = []
    width = micro_batch * world_size
    for shape in sorted(groups):
        indices = list(groups[shape])
        if shuffle:
            generator.shuffle(indices)
        for start in range(0, len(indices), width):
            members = [(index, 1.0) for index in indices[start:start + width]]
            members.extend([(members[-1][0], 0.0)] * (width - len(members)))
            batches.append(members)
    if shuffle:
        generator.shuffle(batches)
    return batches


def local_batch(global_batch, records, rank, micro_batch):
    local = global_batch[rank * micro_batch:(rank + 1) * micro_batch]
    return [records[index] for index, _ in local], [weight for _, weight in local]

def examples_per_epoch(records,seed,axis_policy):
    """The LR denominator counts eligible MRI/video groups, never XYZ rows or padding."""
    if axis_policy=='one_per_image':
        if hasattr(records,'eligible_image_count'):return records.eligible_image_count
        if hasattr(records,'image_count'):return records.image_count
        return len(epoch_axis_indices(records,seed,0))
    if axis_policy=='all_axes':return getattr(records,'eligible_count',len(records))
    raise ValueError('Unknown epoch axis policy')


def flow_losses(model, clean, token_mask, context, sigmas, noise):
    expanded = sigmas.reshape(-1, 1, 1, 1, 1)
    noisy = (1 - expanded) * clean + expanded * noise
    with torch.autocast(clean.device.type, enabled=False):
        prediction = model(hidden_states=noisy, timestep=sigmas * 1000,
                           encoder_hidden_states=context, return_dict=False)[0]
        if prediction.dtype != torch.float32 or prediction.shape != clean.shape:
            raise RuntimeError("模型输出必须为同形状 FP32 flow velocity")
        mask = token_mask.repeat_interleave(2, -2).repeat_interleave(2, -1).unsqueeze(1)
        squared_error = (prediction - (noise - clean)).square()
        denominator = mask.flatten(1).sum(1) * clean.shape[1]
        return (squared_error * mask).flatten(1).sum(1) / denominator


def sample_sigmas(batch_size, device, distribution, shift):
    values = (torch.randn(batch_size, device=device).sigmoid() if distribution == "logit_normal"
              else torch.rand(batch_size, device=device))
    return shift * values / (1 + (shift - 1) * values)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def fsdp_class():
    from torch.distributed.fsdp import FullyShardedDataParallel
    return FullyShardedDataParallel


def model_state_context(model, rank0_only):
    from torch.distributed.fsdp import FullStateDictConfig, StateDictType
    fsdp = fsdp_class()
    if isinstance(model, fsdp):
        return fsdp.state_dict_type(model, StateDictType.FULL_STATE_DICT,
                                   FullStateDictConfig(offload_to_cpu=True, rank0_only=rank0_only))
    return nullcontext()


def barrier(world_size):
    if world_size > 1:
        dist.barrier()


def save_checkpoint(model, optimizer, scheduler, run_dir, metadata,
                    rank, world_size, keep_last):
    """完整 FP32 模型 + 每 rank 分片 optimizer/RNG；只在优化步边界写入。"""
    destination = run_dir / f"checkpoint-{metadata['optimizer_step']:09d}"
    temporary = run_dir / (destination.name + ".writing")
    if rank == 0:
        if destination.exists() or temporary.exists():
            raise FileExistsError(f"checkpoint 已存在，拒绝覆盖：{destination}")
        temporary.mkdir(parents=True)
    barrier(world_size)
    with model_state_context(model, True):
        state = model.state_dict()
        if rank == 0:
            torch.save({key: value.detach().cpu() for key, value in state.items()},
                       temporary / "model.pt")
    del state
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng_state()}, temporary / f"rank-{rank:05d}.pt")
    barrier(world_size)
    if rank == 0:
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (temporary / "COMPLETE").write_text("brain_mri_world_model_checkpoint_v1\n")
        temporary.replace(destination)
        latest = run_dir / "latest.json.tmp"
        latest.write_text(json.dumps({"checkpoint": destination.name}))
        latest.replace(run_dir / "latest.json")
        if metadata.get("is_best"):
            best_pointer = run_dir / "best.json.tmp"
            best_pointer.write_text(json.dumps({"checkpoint": destination.name,
                                                "val_loss": metadata["val_loss"]}))
            best_pointer.replace(run_dir / "best.json")
        checkpoints = sorted(path for path in run_dir.glob("checkpoint-*")
                             if path.is_dir() and (path / "COMPLETE").is_file())
        keep_names = {path.name for path in checkpoints[-keep_last:]}
        if (run_dir / "best.json").is_file():
            keep_names.add(json.loads((run_dir / "best.json").read_text())["checkpoint"])
        for old in checkpoints:
            if old.name in keep_names:
                continue
            if (old / "COMPLETE").read_text().strip() == "brain_mri_world_model_checkpoint_v1":
                shutil.rmtree(old)
    barrier(world_size)


def read_resume_metadata(path, contract, world_size):
    """恢复前只读核对元数据与所有必要文件，不改写既有运行记录。"""
    path = Path(path).resolve()
    complete = path / "COMPLETE"
    if (not complete.is_file() or
            complete.read_text().strip() != "brain_mri_world_model_checkpoint_v1"):
        raise ValueError("checkpoint 尚未完整写入或格式不匹配")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["contract"] != contract or metadata["world_size"] != world_size:
        raise ValueError("resume 要求相同清单、训练配置、模型来源和 world_size")
    if (not (path / "model.pt").is_file() or any(
            not (path / f"rank-{rank:05d}.pt").is_file() for rank in range(world_size))):
        raise ValueError("checkpoint 缺少模型或某个 rank 的 optimizer/RNG 文件")
    return metadata


def validate_run_directory(run_dir, contract, resume, world_size):
    """纯只读启动检查；resume 只能继续所属 run，不混合别的 run 日志。"""
    run_dir = Path(run_dir).resolve()
    if run_dir.exists() and not run_dir.is_dir():
        raise NotADirectoryError(run_dir)
    contract_path = run_dir / "contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("已有运行 contract.json 与本次配置不一致；未改写任何运行文件")
    if resume is None:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError("输出目录非空；恢复时必须显式指定 --resume")
        return None
    checkpoint = Path(resume).resolve()
    if checkpoint.parent != run_dir:
        raise ValueError("--resume 必须属于 --output-dir，不能把不同运行的 checkpoint 和日志混合")
    if not contract_path.is_file():
        raise ValueError("恢复目录缺少原始 contract.json；未新建或改写运行文件")
    return read_resume_metadata(checkpoint, contract, world_size)


def write_new_run_contract(run_dir, contract):
    """新运行原子发布合同；目标已存在时绝不覆盖。"""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = run_dir / f".contract-{os.getpid()}.tmp"
    created = False
    try:
        with temporary.open("x") as stream:
            created = True
            stream.write(json.dumps(contract, indent=2, ensure_ascii=False))
        # 同目录 hard link 原子发布，若已有合同则 FileExistsError，原文件保持不变。
        os.link(temporary, run_dir / "contract.json")
    finally:
        if created:
            temporary.unlink(missing_ok=True)


def restore_checkpoint(path, model, optimizer, scheduler, contract, rank, world_size):
    path = Path(path).resolve()
    metadata = read_resume_metadata(path, contract, world_size)
    from .checkpoint_io import load_model_state_mmap, load_rank_payload_mmap
    state = load_model_state_mmap(path / "model.pt")
    with model_state_context(model, False):
        model.load_state_dict(state, strict=True)
    del state
    # 此文件仅接受本工具在私有工作目录生成的可信 checkpoint。
    payload = load_rank_payload_mmap(path / f"rank-{rank:05d}.pt")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    return metadata


def training_progress_from_checkpoint(metadata):
    """Validation-only flags must never leak into a later mid-epoch checkpoint."""
    fields = ('epoch', 'next_batch', 'optimizer_step', 'best_val_loss', 'train_examples_seen')
    return {name: metadata[name] for name in fields}


def training_inputs(args):
    """CPU metadata only; indexed views do not build per-rank full record dictionaries."""
    os.environ['BWM_RUNTIME_ELIGIBILITY_REGISTRY']=str(args.runtime_eligibility_registry)
    if getattr(args, 'input_index', None):
        from .indexed_input import load_index
        records, manifest_sha = load_index(args.input_index,runtime_eligibility_registry=args.runtime_eligibility_registry)
        return records, records.for_split('train'), records.for_split('val'), manifest_sha, records.binding
    records, manifest_sha = load_manifest(args.manifest)
    from .runtime_eligibility import RuntimeEligibility,RuntimeMetadataRecords,filter_metadata
    runtime=RuntimeEligibility(args.runtime_eligibility_registry);records,excluded=filter_metadata(records,runtime)
    records=RuntimeMetadataRecords(records,runtime)
    return (records,RuntimeMetadataRecords([r for r in records if r['split']=='train'],runtime),
            RuntimeMetadataRecords([r for r in records if r['split']=='val'],runtime),manifest_sha,
            {'mandatory_runtime_eligibility':runtime.binding,'excluded_records':excluded})


@torch.no_grad()
def validate(model, records, context, args, device, rank, world_size, loader=None, sampler=None):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    batches = batches_for_epoch(records, args.micro_batch, world_size,
                                args.validation_seed, 0, False, args.epoch_axis_policy)
    if loader is None or sampler is None:
        raise ValueError('Production validation requires the parallel prefetch loader')
    sampler.records = records
    sampler.set_batches(batches)
    for batch in CudaPrefetcher(loader, device):
        selected, weight_tensor = batch['records'], batch['weights']
        clean, masks, embeddings = batch['clean'], batch['mask'], batch['context']
        noises, sigma_values = [], []
        for record, latent in zip(selected, clean):
            generator = torch.Generator(device=device).manual_seed(
                fixed_seed(record, args.validation_seed, "validation"))
            value = (torch.randn((), generator=generator, device=device).sigmoid()
                     if args.sigma_distribution == "logit_normal"
                     else torch.rand((), generator=generator, device=device))
            sigma_values.append(args.sigma_shift * value / (1 + (args.sigma_shift - 1) * value))
            noises.append(torch.randn(latent.shape, generator=generator, device=device))
        losses = flow_losses(model, clean, masks, embeddings,
                             torch.stack(sigma_values), torch.stack(noises))
        totals[0] += (losses * weight_tensor).double().sum()
        totals[1] += weight_tensor.double().sum()
    if world_size > 1:
        dist.all_reduce(totals)
    model.train()
    if totals[1] <= 0 or not torch.isfinite(totals).all():
        raise FloatingPointError("验证损失不是有限值或验证集为空")
    return float(totals[0] / totals[1])


def run(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    records, training, validation, manifest_sha, input_binding = training_inputs(args)
    if not torch.cuda.is_available():
        raise RuntimeError("全量微调要求 CUDA；未自动回退 CPU 或降低精度")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.set_num_threads(1)
    if not training or not validation or getattr(training,'eligible_count',len(training))==0 or getattr(validation,'eligible_count',len(validation))==0:
        raise ValueError("必须同时包含按受试者划分的 train 和 val；test 从不参与训练/选模")
    context = load_context(args.context_path)
    provenance = backbone_provenance(args.model_dir, args.pet_code_root)
    from .wan import file_sha256
    contract = {"manifest_sha256": manifest_sha, "model": provenance,
                "training_code_sha256": file_sha256(Path(__file__)),
                "software": {"torch": torch.__version__,
                             "diffusers": importlib.metadata.version("diffusers")},
                "context_sha256": file_sha256(args.context_path) if args.context_path else None,
                "epochs": args.epochs, "micro_batch": args.micro_batch,
                "gradient_accumulation": args.gradient_accumulation,
                "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                "warmup_fraction": args.warmup_fraction, "max_grad_norm": args.max_grad_norm,
                "sigma_distribution": args.sigma_distribution, "sigma_shift": args.sigma_shift,
                "epoch_axis_policy": args.epoch_axis_policy,
                "data_loading_code_sha256": file_sha256(Path(__file__).with_name('data_loading.py')),
                "data_loading": {"workers_per_rank": args.loader_workers, "prefetch_factor": args.prefetch_factor,
                                 "pin_memory": True, "persistent_workers": args.loader_workers > 0,
                                 "worker_start_method": "spawn", "cuda_transfer": "separate_stream_nonblocking"},
                "seed": args.seed, "validation_seed": args.validation_seed,
                "gradient_checkpointing": True, "mode": "full_dit", "optimizer": "AdamW(0.9,0.999,1e-8)"}
    if input_binding is not None:
        contract['input_index'] = input_binding
    contract['input_and_checkpoint_code_sha256'] = {
        name: file_sha256(Path(__file__).with_name(name))
        for name in ('indexed_input.py', 'checkpoint_io.py', 'wan.py', 'corpus.py')}
    run_dir = Path(args.output_dir).resolve()
    validate_run_directory(run_dir, contract, args.resume, world_size)
    barrier(world_size)
    if rank == 0 and args.resume is None:
        write_new_run_contract(run_dir, contract)
    barrier(world_size)
    model = load_backbone(args.model_dir, True, pet_code_root=args.pet_code_root)
    model.requires_grad_(True).train()
    if world_size > 1:
        from torch.distributed.fsdp import ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        policy = functools.partial(transformer_auto_wrap_policy,
                                   transformer_layer_cls={type(model.blocks[0])})
        model = fsdp_class()(model, auto_wrap_policy=policy,
                             sharding_strategy=ShardingStrategy.FULL_SHARD,
                             device_id=device, use_orig_params=True,
                             sync_module_states=True, limit_all_gathers=True)
    else:
        model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.999), eps=1e-8, weight_decay=args.weight_decay)
    images_per_epoch = examples_per_epoch(training,args.seed,args.epoch_axis_policy)
    total_examples = images_per_epoch * args.epochs
    warmup_examples = round(total_examples * args.warmup_fraction)
    nominal_update_examples = args.micro_batch * world_size * args.gradient_accumulation
    estimated_total_steps = math.ceil(images_per_epoch / nominal_update_examples) * args.epochs
    state = {"epoch": 0, "next_batch": 0, "optimizer_step": 0, "best_val_loss": None,
             "train_examples_seen": 0}

    def learning_rate_factor(step):
        # Count valid MRI examples, excluding zero-weight bucket padding. This
        # keeps the schedule exact when random axes change bucket tail sizes.
        seen = state['train_examples_seen']
        if warmup_examples and seen < warmup_examples:
            return min(1., seen / warmup_examples)
        progress = min(1., max(0., (seen - warmup_examples) / max(1, total_examples - warmup_examples)))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    if args.resume:
        state.update(training_progress_from_checkpoint(restore_checkpoint(
            args.resume, model, optimizer, scheduler, contract, rank, world_size)))
    # One persistent worker pool serves train and validation; there are eight
    # CPU workers per rank in total, rather than a second idle pool for val.
    loader, batch_sampler = make_loader(records, args.context_path, rank, args.micro_batch,
                                       args.loader_workers, args.prefetch_factor, args.seed)
    event_log = run_dir / "metrics.jsonl"

    def emit(event):
        if rank == 0:
            event["time_unix"] = time.time()
            with event_log.open("a") as stream:
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)

    emit({"event": "training_start", "train_images_axes": len(training),
          "val_images_axes": len(validation), "test_images_axes": len(records) - len(training) - len(validation),
          "world_size": world_size, "epochs": args.epochs, "estimated_total_optimizer_steps": estimated_total_steps,
          "planned_valid_training_examples": total_examples, "scheduler_progress": "valid_MRI_examples",
          "epoch_axis_policy": args.epoch_axis_policy,
          "train_MRI_images": (training.eligible_image_count if hasattr(training, 'eligible_image_count')
                               else len({(row['dataset'], row['image_id']) for row in training})),
          "workers_per_rank": args.loader_workers, "prefetch_factor": args.prefetch_factor,
          "parameter_dtype": "float32", "tf32": True, "real_data_benchmark": "pending"})
    for epoch in range(state["epoch"], args.epochs):
        global_batches = batches_for_epoch(training, args.micro_batch, world_size,
                                           args.seed, epoch, True, args.epoch_axis_policy)
        start_batch = state["next_batch"] if epoch == state["epoch"] else 0
        batch_sampler.records = training
        batch_sampler.set_batches(global_batches[start_batch:])
        prefetched_batches = iter(CudaPrefetcher(loader, device))
        for window_start in range(start_batch, len(global_batches), args.gradient_accumulation):
            started = time.monotonic()
            window = global_batches[window_start:window_start + args.gradient_accumulation]
            valid_count = sum(weight for batch in window for _, weight in batch)
            if warmup_examples and state['train_examples_seen'] < warmup_examples:
                factor = min(1., (state['train_examples_seen'] + valid_count) / warmup_examples)
                for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                    group['lr'] = base_lr * factor
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            for global_batch in window:
                batch = next(prefetched_batches)
                selected, weight_tensor = batch['records'], batch['weights']
                clean, masks, embeddings = batch['clean'], batch['mask'], batch['context']
                sigma = sample_sigmas(len(selected), device, args.sigma_distribution, args.sigma_shift)
                losses = flow_losses(model, clean, masks, embeddings, sigma, torch.randn_like(clean))
                loss = (losses * weight_tensor).sum() * world_size / valid_count
                finite = torch.tensor(int(torch.isfinite(loss)), device=device)
                if world_size > 1:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("非有限训练 loss，未更新 optimizer")
                # 每次 backward 都同步，避免 no_sync 暂存完整 5B 梯度造成峰值显存。
                loss.backward()
                totals[0] += (losses.detach() * weight_tensor).double().sum()
                totals[1] += weight_tensor.double().sum()
            grad_norm = (model.clip_grad_norm_(args.max_grad_norm) if world_size > 1
                         else torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("非有限梯度，未更新 optimizer")
            optimizer.step()
            state['train_examples_seen'] += int(valid_count)
            scheduler.step()
            state.update(epoch=epoch, next_batch=window_start + len(window),
                         optimizer_step=state["optimizer_step"] + 1)
            if world_size > 1:
                dist.all_reduce(totals)
            if state["optimizer_step"] % args.log_every == 0:
                torch.cuda.synchronize(device)
                emit({"event": "train", "epoch": epoch + 1, "optimizer_step": state["optimizer_step"],
                      "loss": float(totals[0] / totals[1]), "learning_rate": optimizer.param_groups[0]["lr"],
                      "gradient_norm": float(grad_norm), "step_seconds": time.monotonic() - started,
                      "rank0_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9})
            if args.checkpoint_every and state["optimizer_step"] % args.checkpoint_every == 0 and state["next_batch"] < len(global_batches):
                metadata = {**state, "world_size": world_size, "contract": contract}
                save_checkpoint(model, optimizer, scheduler, run_dir, metadata,
                                rank, world_size, args.keep_last)
        validation_loss = validate(model, validation, context, args, device, rank, world_size, loader, batch_sampler)
        best = state["best_val_loss"]
        is_best = best is None or validation_loss < best
        state.update(epoch=epoch + 1, next_batch=0,
                     best_val_loss=validation_loss if is_best else best)
        emit({"event": "validation", "epoch": epoch + 1, "optimizer_step": state["optimizer_step"],
              "val_loss": validation_loss, "best_val_loss": state["best_val_loss"], "is_best": is_best})
        metadata = {**state, "world_size": world_size, "contract": contract,
                    "val_loss": validation_loss, "is_best": is_best}
        save_checkpoint(model, optimizer, scheduler, run_dir, metadata,
                        rank, world_size, args.keep_last)
    emit({"event": "training_complete", "completed_epochs": args.epochs,
          "optimizer_step": state["optimizer_step"], "best_val_loss": state["best_val_loss"]})
    if world_size > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest", help="小清单保留原完整latent JSONL路径")
    inputs.add_argument("--input-index", help="完整producer编译并有INDEX_COMPLETE的只读SQLite索引目录")
    parser.add_argument('--runtime-eligibility-registry',required=True,help='Mandatory current APPLIED runtime eligibility registry')
    parser.add_argument("--model-dir", required=True, help="本地 Wan2.2 TI2V-5B Diffusers 根目录")
    parser.add_argument("--pet-code-root", help="可选：既有 PET 代码根目录，含 pet_wan/model.py")
    parser.add_argument("--context-path", help="固定真实 T5 context .pt；或每个 latent 内提供 text_embedding")
    parser.add_argument("--output-dir", required=True, help="新运行目录；非空目录必须显式 resume")
    parser.add_argument("--resume", help="本工具完整 checkpoint 目录，要求相同 GPU 数和训练配置")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--epoch-axis-policy", choices=["one_per_image", "all_axes"], default="one_per_image",
                        help="One cached axis per MRI per epoch; three epochs cover all axes (default)")
    parser.add_argument("--loader-workers", type=int, default=8, help="CPU workers per GPU/rank (default 8)")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Queued micro-batches per worker (default 2)")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-fraction", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.)
    parser.add_argument("--sigma-distribution", choices=["logit_normal", "uniform"], default="logit_normal")
    parser.add_argument("--sigma-shift", type=float, default=1.)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--validation-seed", type=int, default=271828)
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="优化步间隔；0 表示仅每个 epoch 保存")
    parser.add_argument("--keep-last", type=int, default=2, help="只保留最近若干完整 checkpoint")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    if args.loader_workers < 0 or args.prefetch_factor < 1:
        parser.error('loader-workers must be nonnegative and prefetch-factor positive')
    if (min(args.epochs, args.micro_batch, args.gradient_accumulation, args.keep_last, args.log_every) < 1 or
            args.learning_rate <= 0 or args.weight_decay < 0 or args.sigma_shift <= 0 or
            args.max_grad_norm <= 0 or not 0 <= args.warmup_fraction < 1 or args.checkpoint_every < 0):
        parser.error("数值参数超出有效范围")
    run(args)


if __name__ == "__main__":
    main()
