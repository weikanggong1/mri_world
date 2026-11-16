"""FP32 clean-latent, timestep-zero, valid-token mean DiT representations.

Paths and image identities are supplied by the caller. Importing this module
loads no model, data, tensor backend, or device.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

PROTOCOL = "clean_latent_t0_mean_valid_v1"


def mean_valid_tokens(hidden, token_mask):
    """Return FP32 [batch, channels] means over each record's valid tokens."""
    import torch
    if hidden.ndim != 3 or hidden.dtype != torch.float32 or hidden.shape[0] < 1 or hidden.shape[2] < 1:
        raise ValueError("hidden must be FP32 [batch, tokens, channels]")
    mask = token_mask.flatten(1)
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(hidden.shape[:2]):
        raise ValueError("token_mask must be boolean and match hidden tokens")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("every record must contain a valid token")
    return torch.stack([hidden[i, mask[i]].mean(0) for i in range(len(hidden))])


def forward_clean(model, clean, token_mask, embeddings, *, block_index=14):
    """One deterministic representation forward: direct latent, t=0, no noise.

    The caller supplies an evaluation-mode FP32 model and its validated inputs.
    One record per call preserves the reference feature extraction operation.
    """
    import torch
    if clean.dtype != torch.float32 or len(clean) != 1:
        raise ValueError("clean must be one FP32 latent record")
    if type(block_index) is not int or not 0 <= block_index < len(model.blocks):
        raise ValueError("block_index is a zero-based model block index")
    if model.training or any(p.requires_grad or p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("model must be frozen FP32 in eval mode")
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.dtype != torch.float32:
            raise ValueError("intermediate block output must be FP32 [B,L,D]")
        captured["hidden"] = hidden.detach()

    handle = model.blocks[block_index].register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast(clean.device.type, enabled=False):
            model(hidden_states=clean,
                  timestep=torch.full((1,), 0.0, device=clean.device, dtype=torch.float32),
                  encoder_hidden_states=embeddings, return_dict=False)
        if "hidden" not in captured:
            raise RuntimeError("selected model block did not run")
        vector = mean_valid_tokens(captured["hidden"], token_mask)[0].cpu()
        if vector.ndim != 1 or not bool(torch.isfinite(vector).all()):
            raise FloatingPointError("feature vector must be finite")
        return vector
    finally:
        handle.remove()


def extract_clean(*, manifest, model_dir, checkpoint, output_dir, context_path=None,
                  pet_code_root=None, splits=("train", "val", "test"), block_index=14,
                  shard_size=48, device="cuda:0", max_gpu_gib=20.0):
    """Extract complete, new per-axis feature shards from a caller-owned manifest.

    Manifest loading/latent validation are provided by the project's generic
    Wan interface. No cohort, target name, participant, or server is embedded.
    A COMPLETE marker is written only after every selected record succeeds.
    """
    import torch
    from . import wan
    if type(shard_size) is not int or shard_size < 1:
        raise ValueError("shard_size must be a positive integer")
    if not splits or not set(splits) <= {"train", "val", "test"}:
        raise ValueError("splits must select train, val, and/or test")
    selected_device = torch.device(device)
    if selected_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if not 0 < max_gpu_gib <= 20:
            raise ValueError("max_gpu_gib must be in (0,20]")
        cap = int(max_gpu_gib * 1024**3)
        total = int(torch.cuda.get_device_properties(selected_device).total_memory)
        if total < cap:
            raise RuntimeError("device memory is smaller than the requested cap")
        torch.cuda.set_per_process_memory_fraction(cap / total, selected_device)
        torch.cuda.reset_peak_memory_stats(selected_device)
    else:
        cap = None
    wan.configure_tf32(True)
    records, manifest_sha = wan.load_manifest(manifest)
    selected = [r for r in records if r["split"] in splits]
    if not selected:
        raise ValueError("selected manifest scope is empty")
    for record in selected:
        if not isinstance(record.get("image_id"), str) or not record["image_id"].strip():
            raise ValueError("an explicit nonempty image_id is required")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError("output_dir must be new; partial runs are preserved")
    destination.mkdir(parents=True)
    started = time.perf_counter()
    context = wan.load_context(context_path)
    model = wan.load_backbone(model_dir, False, pet_code_root=pet_code_root)
    checkpoint_path = Path(checkpoint).resolve()
    if checkpoint_path.is_dir():
        if not (checkpoint_path / "COMPLETE").is_file():
            raise ValueError("checkpoint directory is incomplete")
        checkpoint_path /= "model.pt"
    checkpoint_sha = wan.file_sha256(checkpoint_path)
    # The public interface accepts a caller-owned state-dict file. It never
    # includes or downloads model weights.
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    del state
    model.to(selected_device).eval().requires_grad_(False)
    if len(model.blocks) != 30 or any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("expected the FP32 30-block backbone")
    if selected_device.type == "cuda":
        torch.cuda.synchronize(selected_device)
    model_load_seconds = time.perf_counter() - started
    provenance = dict(protocol=PROTOCOL, manifest_sha256=manifest_sha,
        checkpoint_sha256=checkpoint_sha, backbone=wan.backbone_provenance(model_dir, pet_code_root),
        context_sha256=wan.file_sha256(context_path) if context_path else None,
        layers=[block_index], pooling="mean_valid_tokens", parameter_dtype="float32", tf32=True,
        sigma=0.0, timestep=0, noise_seed=None, added_noise=False, latent_scale=1.0,
        max_gpu_bytes=cap, image_identity="explicit_manifest_image_id")
    (destination / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    features, rows = [], []
    shard_count = processed = 0
    first_axis_seconds = None
    subsequent_axis_seconds = 0.0
    index = (destination / "index.jsonl").open("x")

    def flush():
        nonlocal shard_count
        if not features:
            return
        filename = f"features-{shard_count:06d}.pt"
        temporary = destination / (filename + ".tmp")
        torch.save(dict(features=torch.stack(features), records=rows, provenance=provenance), temporary)
        temporary.replace(destination / filename)
        for number, row in enumerate(rows):
            index.write(json.dumps(dict(row, feature_path=filename, feature_row=number), allow_nan=False) + "\n")
        index.flush()
        features.clear(); rows.clear(); shard_count += 1

    try:
        for record in selected:
            begin = time.perf_counter()
            clean, token_mask, embeddings = wan.load_batch([record], context, selected_device)
            vector = forward_clean(model, clean, token_mask, embeddings, block_index=block_index)
            elapsed = time.perf_counter() - begin
            if first_axis_seconds is None:
                first_axis_seconds = elapsed
            else:
                subsequent_axis_seconds += elapsed
            if cap is not None and (torch.cuda.max_memory_allocated(selected_device) > cap
                    or torch.cuda.max_memory_reserved(selected_device) > cap):
                raise RuntimeError("configured GPU allocation/reservation cap exceeded")
            features.append(vector)
            rows.append({key: record[key] for key in ("dataset", "subject_id", "image_id", "modality", "axis", "split")}
                        | dict(visit=record.get("visit"), valid_tokens=int(token_mask.sum()),
                               representation=wan.representation_of(record), space=wan.space_for_representation(record)))
            processed += 1
            if len(features) >= shard_size:
                flush()
            del clean, token_mask, embeddings, vector
        flush()
        complete = dict(protocol=PROTOCOL, records=processed, shards=shard_count,
            feature_dim=int(model.config.num_attention_heads * model.config.attention_head_dim),
            model_load_count=1, model_load_seconds=model_load_seconds,
            first_axis_seconds=first_axis_seconds, subsequent_axis_count=max(processed - 1, 0),
            subsequent_axis_seconds_sum=subsequent_axis_seconds, all_features_finite=True,
            max_memory_allocated=torch.cuda.max_memory_allocated(selected_device) if cap else None,
            max_memory_reserved=torch.cuda.max_memory_reserved(selected_device) if cap else None)
        (destination / "COMPLETE").write_text(json.dumps(complete, indent=2, allow_nan=False) + "\n")
        return complete
    finally:
        index.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "model-dir", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--context-path")
    parser.add_argument("--pet-code-root")
    parser.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    parser.add_argument("--block-index", type=int, default=14)
    parser.add_argument("--shard-size", type=int, default=48)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-gpu-gib", type=float, default=20.0)
    args = parser.parse_args(argv)
    result = extract_clean(**vars(args))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
