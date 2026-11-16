"""Reconstruct one scalar map using an audited original voxel-index mapping."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections import OrderedDict

_MATRIX_CACHE = OrderedDict()
_MAPPING_CACHE = OrderedDict()


def _matrix_once(path, mapping):
    """Audit one shared 15-column source once, then reuse its FP32 data for each map."""
    import numpy as np
    from .integrity import fingerprint
    path = Path(path).resolve()
    key = (str(path), fingerprint(path), mapping['maps_axis'], mapping['n_maps'],
           mapping['n_voxels'], mapping.get('value_type'))
    if key in _MATRIX_CACHE:
        _MATRIX_CACHE.move_to_end(key)
        return _MATRIX_CACHE[key]
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    maps_axis = mapping['maps_axis']
    expected = [mapping['n_maps'], mapping['n_voxels']] if maps_axis == 0 else [mapping['n_voxels'], mapping['n_maps']]
    if list(array.shape) != expected or array.dtype.kind not in ('f', 'i', 'u'):
        raise ValueError('Numeric source array shape does not match the audited mapping')
    raw = array.T if maps_axis == 0 else array
    finite = np.isfinite(raw)
    if np.isinf(raw).any():
        raise ValueError('Infinite FC values are invalid; zero fill applies only to NaN')
    if mapping.get('value_type') == 'Pearson_r' and np.any(np.abs(raw[finite]) > 1 + 1e-6):
        raise ValueError('FC values conflict with audited Pearson-r units')
    with np.errstate(over='ignore', invalid='ignore'):
        values = np.array(raw, dtype=np.float32, copy=True)
    if np.isinf(values).any():
        raise ValueError('FC values overflow float32; only source NaN may be filled with zero')
    matrix_nan = np.isnan(values)
    result = (values, matrix_nan, int(matrix_nan.sum()), int(matrix_nan.any(axis=1).sum()))
    if fingerprint(path) != key[1]:
        raise ValueError('FC matrix changed while being read')
    _MATRIX_CACHE[key] = result
    while len(_MATRIX_CACHE) > 2:
        _MATRIX_CACHE.popitem(last=False)
    return result


def file_sha256(path):
    from .integrity import file_sha256 as cached_sha256
    return cached_sha256(path)


def read_mapping(path):
    import nibabel as nib
    import numpy as np
    path = Path(path).resolve()
    from .integrity import fingerprint
    cached = _MAPPING_CACHE.get(str(path))
    if cached and all(fingerprint(name) == signature for name, signature in cached['fingerprints']):
        return dict(cached['mapping']), cached['image'], cached['mask']
    mapping = json.loads(path.read_text())
    if mapping.get('verified') is not True or mapping.get('verification_basis') != 'original_generation_code_and_exact_mask':
        raise ValueError('Masked-vector mapping must be verified against the original generating code and mask')
    if mapping.get('voxel_order') not in ('C', 'F') or mapping.get('maps_axis') not in (0, 1):
        raise ValueError('Explicit voxel order and maps axis are required')
    mask_path = Path(mapping['mask_path'])
    if not mask_path.is_absolute():
        mask_path = path.parent / mask_path
    if file_sha256(mask_path) != mapping['mask_sha256']:
        raise ValueError('Original vector mask SHA-256 mismatch')
    generation_code = Path(mapping['generation_code_path'])
    if not generation_code.is_absolute():
        generation_code = path.parent / generation_code
    if file_sha256(generation_code) != mapping['generation_code_sha256']:
        raise ValueError('Original generation code SHA-256 mismatch')
    image = nib.load(mask_path)
    mask = np.asarray(image.dataobj) > 0
    if mask.ndim != 3 or tuple(image.shape) != tuple(mapping['spatial_shape']):
        raise ValueError('Mask shape and audited mapping disagree')
    if not np.allclose(image.affine, mapping['affine'], atol=1e-5):
        raise ValueError('Mask affine and audited mapping disagree')
    if int(mask.sum()) != mapping['n_voxels']:
        raise ValueError('Mask voxel count and audited mapping disagree')
    mapping['mask_path'] = str(mask_path.resolve())
    mask.setflags(write=False)
    _MAPPING_CACHE[str(path)] = {'mapping': dict(mapping), 'image': image, 'mask': mask,
        'fingerprints': [(str(name), fingerprint(name)) for name in (path, mask_path, generation_code)]}
    while len(_MAPPING_CACHE) > 8:
        _MAPPING_CACHE.popitem(last=False)
    return mapping, image, mask


def load_component(record, return_details=False):
    import nibabel as nib
    import numpy as np
    mapping, mask_image, mask = read_mapping(record['source_mapping_path'])
    if not record.get('source_mask_path') or Path(record['source_mask_path']).resolve() != Path(mapping['mask_path']):
        raise ValueError('Selected source mask must be the exact audited vector mask')
    index = record.get('source_component_index')
    if type(index) is not int or not 0 <= index < mapping['n_maps']:
        raise ValueError('Invalid scalar component index')
    voxel_by_map, matrix_nan, matrix_nan_count, matrix_nan_rows = _matrix_once(record['source_path'], mapping)
    policy = record.get('nonfinite_policy', 'reject')
    if policy not in ('reject', 'replace_nan_with_zero'):
        raise ValueError('Unknown FC non-finite policy')
    if matrix_nan_count and policy != 'replace_nan_with_zero':
        raise ValueError('FC contains NaN without an explicit zero-fill policy')
    original_values = voxel_by_map[:, index]
    finite_component = np.isfinite(original_values)
    nan_count = int(np.isnan(original_values).sum())
    values = original_values.copy()
    if not finite_component.any() or float(values[finite_component].min()) == float(values[finite_component].max()):
        raise ValueError('Scalar component has no finite varying brain signal')
    values[np.isnan(values)] = 0  # User-authorized fill; retain the original anatomical brain mask.
    order = mapping['voxel_order']
    flat = np.zeros(mask.size, dtype=np.float32)
    flat[np.flatnonzero(mask.ravel(order=order))] = values
    volume = flat.reshape(mask.shape, order=order)
    valid_mask = mask.astype(np.uint8)
    # An exact vector -> volume -> vector round trip is mandatory.
    if not np.array_equal(volume.ravel(order=order)[np.flatnonzero(mask.ravel(order=order))], values):
        raise RuntimeError('Masked vector reconstruction failed exact round trip')
    header = mask_image.header.copy()
    header.set_data_dtype(np.float32)
    image = nib.Nifti1Image(volume, mask_image.affine, header=header)
    if not return_details:
        return image
    mask_header = mask_image.header.copy()
    mask_header.set_data_dtype(np.uint8)
    output_mask = nib.Nifti1Image(valid_mask, mask_image.affine, header=mask_header)
    return image, output_mask, {
        'source_nonfinite_policy': policy,
        'source_nan_replaced_with_zero': nan_count,
        'source_nan_fraction': nan_count / mapping['n_voxels'],
        'source_matrix_nan_values': matrix_nan_count,
        'source_matrix_rows_with_nan': matrix_nan_rows,
        'source_valid_voxels': int(mask.sum()),
        'source_nan_fill_retains_anatomical_mask': True,
        'source_mapping_sha256': file_sha256(record['source_mapping_path']),
        'source_voxel_order': order,
        'source_value_type': mapping.get('value_type'),
    }
