"""Publish and verify the exact skull-stripped MNI volume used by the VAE."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from collections import OrderedDict

_FILE_PROOFS = OrderedDict()


def _proof(path, digest, shape, affine, is_mask):
    from .integrity import fingerprint
    import numpy as np
    return (str(Path(path).resolve()), fingerprint(path), digest, tuple(shape),
            tuple(np.asarray(affine).ravel()), is_mask)


def _remember(proof):
    _FILE_PROOFS[proof] = True
    _FILE_PROOFS.move_to_end(proof)
    while len(_FILE_PROOFS) > 1024:
        _FILE_PROOFS.popitem(last=False)


def voxel_sha256(array, mask=False):
    import numpy as np
    values = np.ascontiguousarray(array, dtype=np.uint8 if mask else '<f4')
    return hashlib.sha256(values.tobytes()).hexdigest()


def _verify(path, reference_array, affine, is_mask=False):
    import nibabel as nib
    import numpy as np
    expected_digest = voxel_sha256(reference_array, is_mask)
    proof = _proof(path, expected_digest, reference_array.shape, affine, is_mask)
    if proof in _FILE_PROOFS:
        return
    image = nib.load(path)
    if image.shape != reference_array.shape or not np.allclose(image.affine, affine, atol=1e-4):
        raise ValueError('Existing standardized file has a different MNI grid')
    values = np.asarray(image.dataobj) > 0 if is_mask else image.get_fdata(dtype=np.float32)
    if voxel_sha256(values, is_mask) != expected_digest:
        raise ValueError('Standardized file voxel digest differs; refusing to overwrite')
    _remember(proof)


def verify_persisted(metadata):
    """Read back all encoded input files, including during latent-only resume."""
    import nibabel as nib
    import numpy as np
    expected_shape = tuple(metadata['spatial_shape'])
    expected_affine = np.asarray(metadata['affine'])
    for path_key, hash_key, is_mask in (
        ('registered_image_path', 'preprocessed_volume_sha256', False),
        ('registered_brain_mask_path', 'preprocessed_brain_mask_sha256', True),
        ('registered_fov_mask_path', 'preprocessed_fov_sha256', True),
    ):
        proof = _proof(metadata[path_key], metadata[hash_key], expected_shape, expected_affine, is_mask)
        if proof in _FILE_PROOFS:
            continue
        image = nib.load(metadata[path_key])
        if image.shape != expected_shape or not np.allclose(image.affine, expected_affine, atol=1e-4):
            raise ValueError(f'Saved standardized input has a different grid: {path_key}')
        values = np.asarray(image.dataobj)
        if not np.isfinite(values).all() or (is_mask and not np.isin(values, [0, 1]).all()):
            raise ValueError(f'Saved standardized input has invalid values: {path_key}')
        if voxel_sha256(values > 0 if is_mask else values, is_mask) != metadata[hash_key]:
            raise ValueError(f'Saved standardized input changed after encoding: {path_key}')
        _remember(proof)


def _publish_nifti(path, array, affine, is_mask=False):
    import nibabel as nib
    import numpy as np
    path = Path(path)
    if path.exists() or path.is_symlink():
        _verify(path, array, affine, is_mask)
        return
    descriptor, name = tempfile.mkstemp(prefix='.standardized-', suffix='.nii.gz', dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        image = nib.Nifti1Image(array.astype(np.uint8 if is_mask else np.float32), affine)
        image.header.set_xyzt_units('mm')
        image.set_qform(affine, 4)
        image.set_sform(affine, 4)
        nib.save(image, temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _verify(path, array, affine, is_mask)
    finally:
        temporary.unlink(missing_ok=True)
    _verify(path, array, affine, is_mask)


def publish_registered(root, record, array, brain_mask, metadata, image_identifier, field_of_view=None):
    """Reuse only already matching brain data; raw source images are never copied."""
    import numpy as np
    root = Path(root).resolve()
    destination = root / record['dataset']
    mask_pool = root / 'brain_masks'
    fov_pool = root / 'field_of_view_masks'
    destination.mkdir(parents=True, exist_ok=True)
    mask_pool.mkdir(parents=True, exist_ok=True)
    fov_pool.mkdir(parents=True, exist_ok=True)
    target = destination / f'{image_identifier}_brain_MNI152_2mm.nii.gz'
    mask_path = mask_pool / f'{voxel_sha256(brain_mask, True)}.nii.gz'
    affine = np.asarray(metadata['affine'])
    _publish_nifti(mask_path, brain_mask, affine, True)
    if field_of_view is None:
        field_of_view = np.ones(array.shape, dtype=bool)
    fov_path = fov_pool / f'{voxel_sha256(field_of_view, True)}.nii.gz'
    _publish_nifti(fov_path, field_of_view, affine, True)
    reuse = (record.get('source_format', 'nifti') == 'nifti'
             and metadata.get('source_grid_reused') is True
             and metadata.get('mask_changed_voxels') == 0)
    if reuse and not target.exists() and not target.is_symlink():
        source = Path(record['source_path']).resolve()
        _verify(source, array, affine)
        try:
            # A symlink preserves the original inode/ctime. Creating another
            # hard link would look like a source modification to later records
            # that share this T1 image as their anatomical mask reference.
            target.symlink_to(source)
        except FileExistsError:
            pass
    if target.exists() or target.is_symlink():
        _verify(target, array, affine)
    else:
        _publish_nifti(target, array, affine)
    return {'registered_image_path': str(target.absolute()),
            'registered_brain_mask_path': str(mask_path),
            'registered_fov_mask_path': str(fov_path),
            'preprocessed_volume_sha256': voxel_sha256(array),
            'preprocessed_brain_mask_sha256': voxel_sha256(brain_mask, True),
            'preprocessed_fov_sha256': voxel_sha256(field_of_view, True),
            'registered_storage': 'reuse_existing_verified_standardized_brain' if reuse else 'new_standardized_brain',
            'encoded_input_representation': 'persisted_skull_stripped_MNI152_2mm_volume'}
