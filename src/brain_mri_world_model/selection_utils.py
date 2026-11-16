"""Stable cohort identities and subject-level splits, independent of image axes."""
from __future__ import annotations

import hashlib
import json


CANONICAL_DATASETS = frozenset(('ukb', 'mrrate', 'tiantan', 'fomo'))
DEFAULT_SPLIT_SEED = 20261106


def split_for_subject(dataset, subject_id, seed=DEFAULT_SPLIT_SEED,
                      train_fraction=0.8, validation_fraction=0.1):
    """Hash known subject identity once, before visits/modalities/axes expand."""
    if dataset not in CANONICAL_DATASETS or not isinstance(subject_id, str) or not subject_id:
        raise ValueError('Expected a canonical dataset and nonempty stable subject identity')
    if not (0 < train_fraction < 1 and 0 <= validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError('Invalid train/validation fractions')
    key = json.dumps([int(seed), dataset, subject_id], ensure_ascii=False,
                     separators=(',', ':')).encode()
    value = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
    return ('train' if value < train_fraction else
            'val' if value < train_fraction + validation_fraction else 'test')


def stable_image_id(dataset, subject_id, scan_key):
    if dataset not in CANONICAL_DATASETS:
        raise ValueError('Unknown cohort')
    if any(not isinstance(value, str) or not value for value in (subject_id, scan_key)):
        raise ValueError('Subject and source scan key must be nonempty strings')
    key = json.dumps([dataset, subject_id, scan_key], ensure_ascii=False,
                     separators=(',', ':')).encode()
    return hashlib.sha256(key).hexdigest()
