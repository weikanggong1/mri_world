"""Memory-map owned torch.save ZIP checkpoints; never silently allocate legacy full copies."""
from pathlib import Path
import zipfile
import torch

def _load_owned_zip(path, *, weights_only):
    path=Path(path)
    if not zipfile.is_zipfile(path):
        raise ValueError('mmap resume requires the existing torch.save ZIP checkpoint format; legacy pickle is not silently loaded into full per-rank CPU copies')
    return torch.load(path,map_location='cpu',weights_only=weights_only,mmap=True)

def load_model_state_mmap(path):
    """Full FP32 model state remains file-backed; caller keeps assign=False/parameter identity."""
    return _load_owned_zip(path,weights_only=True)

def load_rank_payload_mmap(path):
    """Only a trusted private per-rank optimizer/scheduler/RNG payload, as in the original API."""
    return _load_owned_zip(path,weights_only=False)
