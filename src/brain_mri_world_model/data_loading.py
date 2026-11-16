"""CPU worker prefetch and asynchronous pinned-memory CUDA transfer."""
from __future__ import annotations

import hashlib
import json
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler


def axis_for_image(dataset, image_id, seed, epoch):
    """Same deterministic aligned three-epoch permutation for list and SQLite inputs."""
    if type(epoch) is not int or epoch < 0:
        raise ValueError('Epoch must be nonnegative')
    digest = hashlib.sha256(json.dumps([seed, dataset, image_id, epoch // 3], separators=(',', ':')).encode()).digest()
    axes = [0, 1, 2]
    random.Random(int.from_bytes(digest[:8], 'big')).shuffle(axes)
    return axes[epoch % 3]


def epoch_axis_indices(records, seed, epoch):
    """One view per MRI/EP; volumes rotate XYZ over3EP, projection keeps its angular video0."""
    from .wan import representation_of
    if epoch < 0:
        raise ValueError('Epoch must be nonnegative')
    if hasattr(records, 'epoch_view_metadata'):
        return [index for index, _ in records.epoch_view_metadata(seed, epoch, 'one_per_image')]
    images = {}
    for index, row in enumerate(records):
        key = (row['dataset'], row['image_id'])
        identity=tuple(row[name] for name in ('subject_id','modality','split'))+(representation_of(row),)
        group = images.setdefault(key, {'identity': identity, 'axes': {}})
        if group['identity'] != identity:
            raise ValueError('Axes of one MRI disagree about identity, modality or split')
        axis = row['axis']
        if type(axis) is not int or axis not in (0, 1, 2) or axis in group['axes']:
            raise ValueError('One MRI has a duplicate or invalid video axis')
        group['axes'][axis] = index
    selected = []
    for key, group in images.items():
        if group['identity'][-1]!='volume_xyz':
            if set(group['axes'])!={0}:raise ValueError('Projection requires exactly one angular video axis0')
            selected.append(group['axes'][0]);continue
        if set(group['axes']) != {0, 1, 2}:
            raise ValueError('Random-axis training requires all three cached axes for each MRI')
        selected.append(group['axes'][axis_for_image(*key, seed, epoch)])
    return selected


class RecordKeyDataset(Dataset):
    """Workers receive only queued record keys, not the full multi-million-row list.

    PyTorch map datasets accept noninteger keys from a custom sampler. The sampler
    stays in the parent; only each small batch's records enter worker queues.
    """
    def __init__(self, length, context_path=None):
        self.length = length
        self.context_path = str(context_path) if context_path else None
        self._context = None
        self._context_loaded = False

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        from .wan import load_context, load_latent
        record, weight = key
        from .runtime_eligibility import required_worker_runtime
        required_worker_runtime().assert_allowed(record)
        if not self._context_loaded:
            self._context = load_context(self.context_path)
            self._context_loaded = True
        latent, mask, context = load_latent(record, self._context)
        return {'clean': latent, 'mask': mask, 'context': context,
                'weight': weight, 'record': record}


class RankBatchSampler(Sampler):
    def __init__(self, records, rank, micro_batch):
        self.records, self.rank, self.micro_batch = records, rank, micro_batch
        self.batches = []

    def set_batches(self, batches):
        from .runtime_eligibility import required_worker_runtime
        self.runtime=getattr(getattr(self.records,'handle',None),'runtime',None) or getattr(self.records,'runtime',None) or required_worker_runtime()
        self.runtime.assert_current()
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        start = self.rank * self.micro_batch
        for batch in self.batches:
            local = batch[start:start + self.micro_batch]
            if len(local) != self.micro_batch:
                raise ValueError('Global batch does not contain this rank micro-batch')
            selected=[(self.records[index],weight) for index,weight in local]
            for record,_ in selected:self.runtime.assert_allowed(record)
            yield selected


def collate_records(samples):
    return {'clean': torch.stack([sample['clean'] for sample in samples]),
            'mask': torch.stack([sample['mask'] for sample in samples]),
            'context': torch.stack([sample['context'] for sample in samples]),
            'weights': torch.tensor([sample['weight'] for sample in samples], dtype=torch.float32),
            'records': [sample['record'] for sample in samples]}


def initialize_worker(worker_id):
    # Eight CPU workers per GPU, each with one numerical thread.
    torch.set_num_threads(1)


def make_loader(records, context_path, rank, micro_batch, workers=8, prefetch_factor=2, seed=20261106):
    if workers < 0 or prefetch_factor < 1:
        raise ValueError('workers must be nonnegative; prefetch_factor must be positive')
    sampler = RankBatchSampler(records, rank, micro_batch)
    dataset = RecordKeyDataset(len(records), context_path)
    options = dict(dataset=dataset, batch_sampler=sampler, num_workers=workers,
                   collate_fn=collate_records, pin_memory=True,
                   generator=torch.Generator().manual_seed(seed + rank * 100003))
    if workers:
        options.update(persistent_workers=True, prefetch_factor=prefetch_factor,
                       multiprocessing_context='spawn', worker_init_fn=initialize_worker)
    return DataLoader(**options), sampler


class CudaPrefetcher:
    """Queue nonblocking H2D on a separate stream while the current batch computes."""
    def __init__(self, loader, device):
        self.iterator = iter(loader)
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('Asynchronous transfer requires CUDA')
        self.stream = torch.cuda.Stream(device=self.device)
        self.pending = None
        self._preload()

    def _preload(self):
        try:
            cpu_batch = next(self.iterator)
        except StopIteration:
            self.pending = None
            return
        with torch.cuda.stream(self.stream):
            self.pending = {name: value.to(self.device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                            for name, value in cpu_batch.items()}

    def __iter__(self):
        return self

    def __next__(self):
        current = torch.cuda.current_stream(self.device)
        current.wait_stream(self.stream)
        if self.pending is None:
            raise StopIteration
        batch = self.pending
        for value in batch.values():
            if isinstance(value, torch.Tensor):
                value.record_stream(current)
        self._preload()
        return batch
