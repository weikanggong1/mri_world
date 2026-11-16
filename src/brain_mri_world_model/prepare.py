"""Reviewed MRI list -> persisted skull-stripped MNI152 2mm data -> frozen VAE.

Existing originals are read in place. The VAE reads the verified registered file.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time


MNI_SHAPE = (91, 109, 91)
MNI_AFFINE = [[-2., 0., 0., 90.], [0., 2., 0., -126.],
              [0., 0., 2., -72.], [0., 0., 0., 1.]]
ENCODING_VERSION = 'brain_mri_encoding_v2'
CANONICAL_DATASETS = ('ukb', 'mrrate', 'tiantan', 'fomo')
NORMALIZATION = 'within_brain_robust_quantile_to_minus1_plus1'
FIXED_NORMALIZATION = 'fixed_physical_range_to_minus1_plus1'
QUANTILES = (0.005, 0.995)
MASK_DEFINITION = ('nominal causal temporal bin and 32x32 viewing-raster bin with >0 valid '
                   'acquisition-field samples; excludes bins outside source FOV or padding; not a brain mask')
PADDING_NOTE = ('Token mask does not remove convolutional or global-attention influence '
                'of boundary padding')


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


@contextmanager
def encoding_lock(output_dir):
    """Kernel lock is released after a crash; a stale file does not block resume."""
    import fcntl
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / '.encoding.lock').open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another encoder holds this output directory') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def image_identifier(row):
    identity = json.dumps([row['dataset'], row['image_id']], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(identity.encode()).hexdigest()


def video_geometry(shape, axis, spatial_scale=1):
    if type(axis) is not int or axis not in (0, 1, 2) or len(shape) != 3:
        raise ValueError('Video geometry requires a 3D shape and axis 0/1/2')
    if type(spatial_scale) is not int or spatial_scale not in (1, 2):
        raise ValueError('Video raster sampling scale must be 1 or 2')
    plane_axes = [dimension for dimension in range(3) if dimension != axis]
    frames, height, width = (int(shape[index]) for index in [axis, *plane_axes])
    if min(frames, height, width) < 2:
        raise ValueError('Video dimensions must be at least two')
    padded_frames = 1 + math.ceil((frames - 1) / 4) * 4
    padded_height, padded_width = math.ceil(height / 32) * 32, math.ceil(width / 32) * 32
    left, top = (padded_width - width) // 2, (padded_height - height) // 2
    padding = (left, padded_width - width - left, top, padded_height - height - top,
               0, padded_frames - frames)
    return {'original_video_shape': [frames, height, width],
            'padded_video_shape': [padded_frames, padded_height * spatial_scale, padded_width * spatial_scale],
            'padding_torch_order': [value * spatial_scale if index < 4 else value for index, value in enumerate(padding)],
            'base_padding_torch_order': list(padding), 'plane_axes': plane_axes,
            'video_spatial_scale': spatial_scale,
            'video_raster_policy': 'pad_then_bilinear_spatial_scale'}


def nominal_token_coverage(geometry, field_of_view=None):
    import torch
    import torch.nn.functional as F
    frames, height, width = geometry['original_video_shape']
    if field_of_view is None:
        valid = torch.ones((frames, height, width), dtype=torch.float32)
    else:
        axis = next(dimension for dimension in range(3) if dimension not in geometry['plane_axes'])
        valid = torch.as_tensor(field_of_view.copy(), dtype=torch.float32).permute(axis, *geometry['plane_axes'])
        if list(valid.shape) != [frames, height, width] or not bool(torch.isfinite(valid).all()):
            raise ValueError('Field-of-view mask and image geometry differ')
        if float(valid.min()) < 0 or float(valid.max()) > 1:
            raise ValueError('Field-of-view mask must be bounded by 0 and 1')
    coverage = F.pad(valid[None, None],
                     geometry.get('base_padding_torch_order', geometry['padding_torch_order']), value=0)
    scale = geometry.get('video_spatial_scale', 1)
    coverage = coverage.repeat_interleave(scale, -2).repeat_interleave(scale, -1)
    first = F.avg_pool3d(coverage[:, :, :1], (1, 32, 32))
    later = F.avg_pool3d(coverage[:, :, 1:], (4, 32, 32))
    return torch.cat((first, later), dim=2)[0, 0]


def validate_video(video):
    import torch
    if (not isinstance(video, torch.Tensor) or video.dtype != torch.float32 or
            video.ndim != 5 or tuple(video.shape[:2]) != (1, 3)):
        raise ValueError('VAE input must be FP32 [1,3,T,H,W]')
    frames, height, width = video.shape[2:]
    if frames % 4 != 1 or height % 32 or width % 32:
        raise ValueError('Wan2.2 VAE/DiT input requires T=4k+1 and H/W divisible by 32')
    if not bool(torch.isfinite(video).all()) or float(video.min()) < -1 or float(video.max()) > 1:
        raise ValueError('VAE input must contain finite values in [-1,1]')


def sha256_file(path):
    from .integrity import file_sha256
    return file_sha256(path)


def verify_resources(manifest_path):
    """A private manifest binds local resources to verified upstream files."""
    manifest_path = Path(manifest_path).resolve()
    resources = json.loads(manifest_path.read_text())
    verified = {}
    for key in ('template', 'brain_mask', 'synthstrip', 'synthmorph_affine', 'synthmorph_deform'):
        record = resources[key]
        if not all(record.get(field) for field in ('path', 'size', 'sha256', 'source_url', 'license')):
            raise ValueError(f'Incomplete resource provenance: {key}')
        path = Path(record['path'])
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.stat().st_size != int(record['size']) or sha256_file(path) != record['sha256']:
            raise ValueError(f'Resource size/SHA-256 mismatch: {key}')
        verified[key] = dict(record, path=str(path.resolve()))
    return verified


def read_selected_manifest(path):
    """Read only explicitly selected records; no dataset discovery here."""
    path = Path(path).resolve()
    rows, identities, images = [], {}, set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ('source_path', 'dataset', 'subject_id', 'image_id', 'modality',
                    'split', 'preprocess_mode', 'source_skull_stripped')
        if any(row.get(key) is None or row.get(key) == '' for key in required):
            raise ValueError(f'Missing input fields on line {line_number}')
        if row.get('approved') is not True:
            raise ValueError(f'Input list is not approved: line {line_number}')
        if type(row['source_skull_stripped']) is not bool:
            raise ValueError(f'source_skull_stripped must be a JSON boolean: line {line_number}')
        for key in ('dataset', 'subject_id', 'image_id', 'modality', 'source_path'):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError(f'{key} must be a non-empty string: line {line_number}')
        if row['dataset'] not in CANONICAL_DATASETS:
            raise ValueError('dataset must be the canonical cohort ukb/mrrate/tiantan/fomo, not a source or registration entry name')
        if row['split'] not in ('train', 'val', 'test'):
            raise ValueError(f'Invalid split: line {line_number}')
        if row['preprocess_mode'] not in ('already_mni', 'fnit_joint'):
            raise ValueError(f'Invalid preprocessing route: line {line_number}')
        if row.get('source_format', 'nifti') not in ('nifti', 'npy_masked_maps'):
            raise ValueError('Unsupported source_format')
        if row.get('source_format') == 'npy_masked_maps':
            if type(row.get('source_component_index')) is not int or not row.get('source_mapping_path'):
                raise ValueError('Masked-array source requires an audited mapping and scalar component index')
        normalization_spec(row)
        identity = (row['dataset'], row['subject_id'])
        if identity in identities and identities[identity] != row['split']:
            raise ValueError('A subject occurs in more than one split')
        identities[identity] = row['split']
        image_key = (row['dataset'], row['image_id'])
        if image_key in images:
            raise ValueError('Duplicate image ID in selected list')
        images.add(image_key)
        source_path = Path(row['source_path'])
        row['source_path'] = str((source_path if source_path.is_absolute() else path.parent / source_path).resolve())
        for auxiliary in ('source_mask_path', 'source_mask_reference_path', 'source_mapping_path', 'source_fov_mask_path'):
            if row.get(auxiliary):
                auxiliary_path = Path(row[auxiliary])
                row[auxiliary] = str((auxiliary_path if auxiliary_path.is_absolute() else path.parent / auxiliary_path).resolve())
        rows.append(row)
    if not rows:
        raise ValueError('Selected manifest is empty')
    return rows


def normalization_spec(record):
    """Freeze a per-modality scaling policy rather than remove quantitative units silently."""
    mode = record.get('normalization_mode', 'robust_quantile')
    if mode == 'robust_quantile':
        return {'normalization': NORMALIZATION, 'quantiles': list(QUANTILES),
                'fixed_range': None, 'intensity_unit': record.get('intensity_unit', 'arbitrary')}
    if mode != 'fixed_range':
        raise ValueError('normalization_mode must be robust_quantile or fixed_range')
    bounds = record.get('intensity_range')
    if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
            or any(type(value) not in (float, int) or not math.isfinite(value) for value in bounds)
            or bounds[1] <= bounds[0]):
        raise ValueError('fixed_range requires finite intensity_range=[low,high] with high>low')
    unit = record.get('intensity_unit')
    if not isinstance(unit, str) or not unit.strip():
        raise ValueError('fixed_range requires an explicit intensity_unit')
    return {'normalization': FIXED_NORMALIZATION, 'quantiles': None,
            'fixed_range': [float(value) for value in bounds], 'intensity_unit': unit}


def load_volume(path):
    import nibabel as nib
    import numpy as np
    image = nib.load(str(path))
    if isinstance(image, nib.Cifti2Image):
        raise ValueError('CIFTI is not a 3D MRI volume')
    array = image.get_fdata(dtype=np.float32, caching='unchanged')
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3 or min(array.shape) < 2:
        raise ValueError(f'Expected one 3D volume, got {array.shape}')
    if not np.isfinite(array).all() or float(array.min()) == float(array.max()):
        raise ValueError('MRI contains non-finite values or is constant')
    affine = np.asarray(image.affine)
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-12:
        raise ValueError('Invalid MRI affine')
    # SynthStrip uses the acquisition header's pixdim when conforming. Rebuilding
    # a fresh header from rounded sform values can change ceil(shape * zoom) by
    # one voxel and discard spatial units, even though the world affine matches.
    header = image.header.copy()
    header.set_data_dtype(np.float32)
    return nib.Nifti1Image(array, affine, header=header)


def same_grid(image, reference):
    import numpy as np
    return image.shape == reference.shape and np.allclose(image.affine, reference.affine, atol=1e-4)


def resample_in_same_world(image, reference, device, method='linear'):
    """Only for a proven common world space; this does not perform registration."""
    import numpy as np
    if same_grid(image, reference):
        return image
    from fnit import apply_transform
    from fnit.synthmorph import WorldTransformChain
    chain = WorldTransformChain(reference=reference, reference_to_source_world=np.eye(4))
    return apply_transform(image=image, transformation=chain, method=method,
                           device=device, dtype='float32', frame_chunk_size=1)


class Preprocessor:
    def __init__(self, resources, device='cuda:0', extent=192, minimum_mask_dice=0.65, *, require_review=True):
        import nibabel as nib
        import numpy as np
        self.resources, self.device = resources, device
        self.extent, self.minimum_mask_dice = extent, minimum_mask_dice
        self.require_review = require_review
        self.template = load_volume(resources['template']['path'])
        self.template_mask_image = nib.load(resources['brain_mask']['path'])
        if self.template.shape != MNI_SHAPE or not np.allclose(self.template.affine, MNI_AFFINE, atol=1e-4):
            raise ValueError('Template is not the required FSL MNI152 2mm grid')
        if not same_grid(self.template_mask_image, self.template):
            raise ValueError('Template mask and template grids differ')
        self.template_mask = np.asarray(self.template_mask_image.dataobj) > 0
        self.fixed = nib.Nifti1Image(self.template.get_fdata(dtype=np.float32) * self.template_mask,
                                    self.template.affine)
        self.strip = self.morph = None
        self.last_fov_mask = None
        self.mask_reference_cache = {}

    def __call__(self, record):
        import nibabel as nib
        import numpy as np
        from fnit import FNIT_SynthStrip, FNIT_SynthMorph, apply_transform
        started = time.monotonic()
        source_mask_override, source_details = None, {}
        if record.get('source_format') == 'npy_masked_maps':
            from .masked_volume import load_component
            image, source_mask_override, source_details = load_component(record, return_details=True)
        else:
            image = load_volume(record['source_path'])
        native_shape, native_affine = image.shape, image.affine.tolist()
        source_fov = np.ones(image.shape, dtype=np.uint8)
        if record.get('source_fov_mask_path'):
            fov_image = nib.load(record['source_fov_mask_path'])
            if not same_grid(fov_image, image):
                raise ValueError('Source FOV mask and MRI grids differ')
            source_fov = (np.asarray(fov_image.dataobj) > 0.999).astype(np.uint8)
        fov_image = nib.Nifti1Image(source_fov, image.affine)
        if record['source_skull_stripped'] is True:
            brain = image
            if source_mask_override is not None:
                mask_array = np.asarray(source_mask_override.dataobj) > 0
            elif record.get('source_mask_path'):
                mask = nib.load(record['source_mask_path'])
                if not same_grid(mask, image):
                    raise ValueError('Source brain and source mask grids differ')
                mask_array = np.asarray(mask.dataobj) > 0
            elif record.get('source_mask_reference_path'):
                from .integrity import fingerprint
                reference_path = record['source_mask_reference_path']
                cache_key = (reference_path, fingerprint(reference_path), tuple(image.shape), tuple(image.affine.ravel()))
                mask_array = self.mask_reference_cache.get(cache_key)
                if mask_array is None:
                    reference = load_volume(reference_path)
                    reference_values = np.asarray(reference.dataobj)
                    reference_mask = nib.Nifti1Image(
                        (np.isfinite(reference_values) & (reference_values > 0)).astype(np.uint8), reference.affine)
                    if not same_grid(reference_mask, image):
                        if record.get('source_space') != 'FSL_MNI152' or record.get('registration_qc_passed') is not True:
                            raise ValueError('Different-grid mask reference requires verified common MNI space')
                        reference_mask = resample_in_same_world(reference_mask, image, self.device, method='nearest')
                    mask_array = np.asarray(reference_mask.dataobj) > 0
                    if len(self.mask_reference_cache) >= 8:
                        self.mask_reference_cache.pop(next(iter(self.mask_reference_cache)))
                    self.mask_reference_cache[cache_key] = mask_array
                mask_array = mask_array.copy()
            else:
                mask_array = np.asarray(image.dataobj) != 0
            mask = nib.Nifti1Image(mask_array.astype(np.uint8), image.affine)
        else:
            if self.strip is None:
                self.strip = FNIT_SynthStrip(weights=self.resources['synthstrip']['path'],
                                            device=self.device, threads=4)
            stripped = self.strip(image=image, border=1, fill=0)
            brain, mask = stripped.image, stripped.mask
        strip_finished = time.monotonic()
        if record['preprocess_mode'] == 'already_mni':
            if record.get('registration_qc_passed') is not True or record.get('source_space') != 'FSL_MNI152':
                raise ValueError('Existing MNI registration requires explicit provenance and reviewed QC')
            brain_mni = resample_in_same_world(brain, self.template, self.device)
            mask_mni = resample_in_same_world(mask, self.template, self.device, method='nearest')
            fov_mni = resample_in_same_world(fov_image, self.template, self.device, method='nearest')
        else:
            if self.morph is None:
                self.morph = FNIT_SynthMorph(
                    weights={'affine': self.resources['synthmorph_affine']['path'],
                             'deform': self.resources['synthmorph_deform']['path']},
                    device=self.device, model='joint', extent=self.extent, hyper=0.5)
            registration = self.morph(moving=brain, fixed=self.fixed,
                                      transform_only=True, compute_inverse=False)
            brain_mni = apply_transform(image=brain, transformation=registration.transform,
                                        method='linear', device=self.device)
            mask_mni = apply_transform(image=mask, transformation=registration.transform,
                                       method='nearest', device=self.device, dtype='uint8')
            fov_mni = apply_transform(image=fov_image, transformation=registration.transform,
                                      method='nearest', device=self.device, dtype='uint8')
        registration_finished = time.monotonic()
        if not same_grid(brain_mni, self.template) or not same_grid(mask_mni, self.template):
            raise ValueError('Preprocessing returned an unexpected grid')
        array = brain_mni.get_fdata(dtype=np.float32)
        mask_array = np.asarray(mask_mni.dataobj) > 0
        fov_array = np.asarray(fov_mni.dataobj) > 0
        if not same_grid(fov_mni, self.template):
            raise ValueError('Registered FOV grid differs from the fixed template')
        mask_array &= fov_array
        fov_coverage = float((fov_array & self.template_mask).sum() / self.template_mask.sum())
        minimum_coverage = record.get('minimum_brain_fov_coverage', 0.95)
        if not 0 < minimum_coverage <= 1 or fov_coverage < minimum_coverage:
            raise ValueError(f'Brain acquisition coverage {fov_coverage:.4f} is below declared {minimum_coverage}')
        self.last_fov_mask = fov_array
        if not np.isfinite(array).all() or not mask_array.any():
            raise ValueError('Empty/non-finite registered brain')
        dice = float(2 * (mask_array & self.template_mask).sum() /
                     (mask_array.sum() + self.template_mask.sum()))
        if dice < self.minimum_mask_dice:
            raise ValueError(f'Registered brain mask fails screening: Dice={dice:.4f}')
        # A quantitative gate supplements, rather than replaces, reviewed registration QC.
        if self.require_review and record.get('preprocess_qc_reviewed') is not True:
            raise ValueError('Preprocessing QC review is required before latent production')
        source_grid_reused = record['preprocess_mode'] == 'already_mni' and same_grid(image, self.template)
        mask_changed_voxels = int(np.count_nonzero(array[~mask_array]))
        array[~mask_array] = 0
        # 在本次必需的RAM读取中保存少量强度统计，后续只用train记录
        # 拟合跨受试者固定范围，不为归一化再遍历、解压原MRI。
        intensity_started = time.monotonic()
        brain_values = array[mask_array]
        intensity_quantiles = np.quantile(brain_values, QUANTILES[:1] + (0.5,) + QUANTILES[1:])
        if not float(brain_values.max()) > float(brain_values.min()):
            raise ValueError('Registered brain is constant inside its anatomical mask')
        intensity_statistics = {'space': 'preprocessed_brain', 'count': int(brain_values.size),
            'min': float(brain_values.min()), 'max': float(brain_values.max()),
            'quantile_probabilities': [0.005, 0.5, 0.995],
            'quantiles': [float(value) for value in intensity_quantiles], 'finite': True}
        intensity_seconds = time.monotonic() - intensity_started
        processed_digest = hashlib.sha256(np.ascontiguousarray(array, dtype='<f4').tobytes()).hexdigest()
        mask_digest = hashlib.sha256(np.ascontiguousarray(mask_array, dtype=np.uint8).tobytes()).hexdigest()
        phase1_proof = record.get('phase1_provenance')
        if phase1_proof is not None:
            # Reuse the already decoded input; never approve modified standard
            # voxels merely by binding a new compressed-file byte digest.
            current = {'preprocessed_volume_sha256': processed_digest,
                       'preprocessed_brain_mask_sha256': mask_digest,
                       'preprocessed_fov_sha256': hashlib.sha256(np.ascontiguousarray(fov_array, dtype=np.uint8).tobytes()).hexdigest()}
            if not isinstance(phase1_proof, dict) or any(phase1_proof.get(key) != value for key, value in current.items()):
                raise ValueError('Persisted brain/mask/FOV voxels differ from their phase1 producer provenance')
        return array, mask_array, {
            **source_details,
            'space': 'MNI152_2mm', 'spatial_shape': list(MNI_SHAPE), 'affine': MNI_AFFINE,
            'source_shape': list(native_shape), 'source_affine': native_affine,
            'skull_stripped': True, 'preprocess_qc_passed': record.get('preprocess_qc_reviewed') is True,
            'mask_vs_template_dice': dice, 'preprocess_mode': record['preprocess_mode'],
            'preprocessed_volume_sha256': processed_digest,
            'preprocessed_volume_hash_format': 'float32 little-endian C-order XYZ voxels; shape and affine stored separately',
            'preprocessed_brain_mask_sha256': mask_digest,
            'source_format': record.get('source_format', 'nifti'),
            'source_component_index': record.get('source_component_index'),
            'source_grid_reused': source_grid_reused,
            'mask_changed_voxels': mask_changed_voxels,
            'intensity_statistics': intensity_statistics,
            'brain_fov_coverage': fov_coverage,
            'fov_mask_kind': record.get('source_fov_mask_kind',
                'provided_input_field_mask' if record.get('source_fov_mask_path') else 'input_grid_domain'),
            'coverage_type': ('complete_brain' if fov_coverage >= 0.95 else 'partial_brain_slab')
                if (record.get('source_grid_is_acquisition_grid') is True or
                    record.get('source_fov_mask_kind') == 'native_acquisition_fov') else 'nominal_input_field_coverage',
            'seconds_strip_including_read': strip_finished - started,
            'seconds_registration_including_resample': registration_finished - strip_finished,
            'seconds_intensity_statistics': intensity_seconds,
        }


def make_video(array, brain_mask, axis, lower_quantile=0.005, upper_quantile=0.995,
               *, fixed_range=None, intensity_unit='arbitrary', field_of_view=None, spatial_scale=1):
    """Preserve MNI data; pad, optionally sample the viewing raster at 2x, repeat RGB."""
    import torch
    import torch.nn.functional as F
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError('Intensity quantiles must satisfy 0 <= lower < upper <= 1')
    if tuple(array.shape) != tuple(brain_mask.shape) or len(array.shape) != 3:
        raise ValueError('MRI and brain mask must have the same 3D shape')
    geometry = video_geometry(array.shape, axis, spatial_scale)
    volume = torch.as_tensor(array.copy(), dtype=torch.float32)
    mask = torch.as_tensor(brain_mask.copy(), dtype=torch.bool)
    if not bool(mask.any()) or not bool(torch.isfinite(volume).all()):
        raise ValueError('MRI must be finite with a non-empty brain mask')
    if fixed_range is None:
        low, high = torch.quantile(volume[mask], torch.tensor([lower_quantile, upper_quantile]))
    else:
        spec = normalization_spec({'normalization_mode': 'fixed_range',
                                   'intensity_range': fixed_range, 'intensity_unit': intensity_unit})
        low, high = torch.tensor(spec['fixed_range'], dtype=torch.float32)
    if not bool(torch.isfinite(low) & torch.isfinite(high)) or not bool(high > low):
        raise ValueError('Degenerate within-brain intensity range')
    fraction_below = float((volume[mask] < low).float().mean())
    fraction_above = float((volume[mask] > high).float().mean())
    volume = ((volume.clamp(low, high) - low) / (high - low) * 2 - 1)
    if not bool(volume[mask].amax() > volume[mask].amin()):
        raise ValueError('Normalized brain is constant; check physical units and fixed intensity range')
    volume[~mask] = -1
    video = volume.permute(axis, *geometry['plane_axes']).contiguous()
    video = F.pad(video[None, None], geometry['base_padding_torch_order'], value=-1)
    if spatial_scale != 1:
        raster = F.interpolate(video[0].permute(1, 0, 2, 3), scale_factor=spatial_scale,
                               mode='bilinear', align_corners=False)
        video = raster.permute(1, 0, 2, 3).unsqueeze(0)
    token_coverage = nominal_token_coverage(geometry, field_of_view)
    valid_token_mask = token_coverage > 0
    if not bool(valid_token_mask.any()):
        raise ValueError('No valid tokens after geometric padding')
    metadata = {
        **geometry,
        'normalization': NORMALIZATION if fixed_range is None else FIXED_NORMALIZATION,
        'intensity_low': float(low), 'intensity_high': float(high),
        'quantiles': [lower_quantile, upper_quantile] if fixed_range is None else None,
        'intensity_unit': intensity_unit,
        'brain_fraction_clipped_below': fraction_below,
        'brain_fraction_clipped_above': fraction_above,
        'coverage_source': 'registered_fov_mask' if field_of_view is not None else 'nominal_full_grid',
        'valid_token_mask_definition': MASK_DEFINITION,
    }
    video = video.repeat(1, 3, 1, 1, 1)
    validate_video(video)
    return video, valid_token_mask, metadata


def encoding_source_record(row, stable_hash):
    """编码器与批调度器共用源合同，防止FOV/溯源字段各写一套而漂移。"""
    result = {'source_format': row.get('source_format', 'nifti'),
              'source_component_index': row.get('source_component_index'),
              'source_fov_mask_kind': row.get('source_fov_mask_kind'),
              'source_grid_is_acquisition_grid': row.get('source_grid_is_acquisition_grid'),
              'minimum_brain_fov_coverage': row.get('minimum_brain_fov_coverage', .95),
              'nonfinite_policy': row.get('nonfinite_policy', 'reject'),
              'normalization': normalization_spec(row)}
    for path_key in ('source_path', 'source_mask_path', 'source_mask_reference_path',
                     'source_mapping_path', 'source_fov_mask_path'):
        path = Path(row[path_key]) if row.get(path_key) else None
        result[path_key] = str(path) if path is not None else None
        digest_key = 'source_sha256' if path_key == 'source_path' else path_key.removesuffix('_path') + '_sha256'
        result[digest_key] = stable_hash(path) if path is not None else None
    if row.get('phase1_provenance') is not None:
        result['phase1_provenance'] = row['phase1_provenance']
    return result


def build_encoding_contract(args, rows, resources):
    """Hash selected inputs before opening an existing output run."""
    import importlib.util
    from .wan import WAN22_VAE_SHA256, WAN22_VAE_SIZE
    checkpoint = Path(args.vae_checkpoint).resolve()
    if checkpoint.stat().st_size != WAN22_VAE_SIZE or sha256_file(checkpoint) != WAN22_VAE_SHA256:
        raise ValueError('VAE checkpoint differs from the verified original Wan2.2 5B file')
    source_records, fingerprints, digest_cache = {}, {}, {}
    def stable_hash(path):
        before = path.stat()
        fingerprint = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
        if digest_cache.get(str(path), (None, None))[0] == fingerprint:
            fingerprints[str(path)] = fingerprint
            return digest_cache[str(path)][1]
        digest = sha256_file(path)
        after = path.stat()
        if fingerprint != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
            raise ValueError('A selected MRI or mask changed while being hashed')
        fingerprints[str(path)] = fingerprint
        digest_cache[str(path)] = (fingerprint, digest)
        return digest
    for row in rows:
        identifier = image_identifier(row)
        source = Path(row['source_path'])
        source_hash = stable_hash(source)
        if row.get('source_sha256') and row['source_sha256'] != source_hash:
            raise ValueError('Selected source SHA-256 does not match its current contents')
        source_records[identifier] = encoding_source_record(row, stable_hash)
    fnit_sources = {}
    for module in ('fnit.synthstrip.pipeline', 'fnit.synthmorph.pipeline', 'fnit._world_resampling'):
        specification = importlib.util.find_spec(module)
        if specification is None or not specification.origin:
            raise ImportError(f'Required mature FNIT implementation is unavailable: {module}')
        fnit_sources[module] = sha256_file(specification.origin)
    package_specification = importlib.util.find_spec('fnit')
    package_root = Path(next(iter(package_specification.submodule_search_locations)))
    fnit_runtime_sources = {str(path.relative_to(package_root)): sha256_file(path)
                            for path in sorted(package_root.rglob('*.py'))}
    import importlib.metadata
    import torch
    contract = {
        'schema_version': ENCODING_VERSION,
        'manifest_sha256': sha256_file(args.manifest),
        'resource_manifest_sha256': sha256_file(args.resource_manifest),
        'resource_sha256': {key: value['sha256'] for key, value in resources.items()},
        'prepare_code_sha256': sha256_file(__file__),
        'wan_adapter_code_sha256': sha256_file(Path(__file__).with_name('wan.py')),
        'masked_volume_code_sha256': sha256_file(Path(__file__).with_name('masked_volume.py')),
        'registered_publisher_code_sha256': sha256_file(Path(__file__).with_name('registered.py')),
        'integrity_cache_code_sha256': sha256_file(Path(__file__).with_name('integrity.py')),
        'pet_loader_code_sha256': sha256_file(Path(args.pet_code_root) / 'pet_wan/model.py'),
        'official_vae_code_sha256': sha256_file(Path(args.wan_source) / 'wan/modules/vae2_2.py'),
        'fnit_code_sha256': fnit_sources,
        'fnit_runtime_sources_sha256': fnit_runtime_sources,
        'software': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                     'cudnn': torch.backends.cudnn.version(),
                     **{name: importlib.metadata.version(name) for name in ('numpy', 'nibabel', 'scipy', 'h5py')}},
        'vae_checkpoint_sha256': WAN22_VAE_SHA256,
        'axes': sorted(set(args.axes)), 'sources': source_records,
        'extent': args.extent, 'minimum_mask_dice': args.minimum_mask_dice,
        'space': 'MNI152_2mm', 'spatial_shape': list(MNI_SHAPE), 'affine': MNI_AFFINE,
        'dtype': 'float32', 'tf32': True, 'normalization': NORMALIZATION,
        'quantiles': list(QUANTILES), 'padding': 'pad_only_T4k_plus1_HW32',
        'video_spatial_scale': args.video_spatial_scale,
        'video_raster_policy': 'pad_then_bilinear_spatial_scale',
        'normalization_policy': 'per_record_robust_quantile_or_fixed_physical_range',
        'image_audit_required': True,
        'registered_output_dir': str(Path(args.registered_output_dir).resolve()),
        'valid_token_mask_definition': MASK_DEFINITION,
    }
    return contract, fingerprints


def assert_source_unchanged(source_info, fingerprints):
    for key in ('source_path', 'source_mask_path', 'source_mask_reference_path', 'source_mapping_path', 'source_fov_mask_path'):
        if source_info.get(key):
            path = Path(source_info[key])
            info = path.stat()
            current = (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino)
            if current != fingerprints[str(path)]:
                raise ValueError('A selected MRI or mask changed after startup hashing; restart from a reviewed list')


def initialize_journal(output_dir, contract, expected_names):
    output_dir = Path(output_dir)
    contract_path = output_dir / 'encoding_contract.json'
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError('Existing output belongs to a different list, axes, source, code, resource or normalization contract')
    else:
        other_files = [path for path in output_dir.iterdir() if path.name != '.encoding.lock']
        if other_files:
            raise FileExistsError('Non-empty output has no matching encoding contract; existing artifacts will not be adopted or overwritten')
        atomic_json(contract_path, contract)
    unexpected = {path.name for path in output_dir.glob('*.pt')} - set(expected_names)
    if unexpected:
        raise FileExistsError('Output contains latent files outside this frozen list/axes contract')
    contract_hash = sha256_file(contract_path)
    journal_path = output_dir / 'encoding_journal.json'
    if journal_path.exists():
        journal = json.loads(journal_path.read_text())
        if journal.get('contract_sha256') != contract_hash or not isinstance(journal.get('completed'), dict):
            raise ValueError('Encoding journal does not match the current contract')
        if journal.get('expected_records') != len(expected_names):
            raise ValueError('Encoding journal has a different expected record count')
        if set(journal['completed']) - set(expected_names):
            raise ValueError('Encoding journal contains records outside this list')
    else:
        journal = {'schema_version': ENCODING_VERSION, 'contract_sha256': contract_hash,
                   'expected_records': len(expected_names), 'completed': {}}
    # During recovery only the journal is consumable. Final manifests are republished
    # after every expected latent has been validated; no partial list is called complete.
    journal['status'] = 'incomplete'
    atomic_json(journal_path, journal)
    (output_dir / 'COMPLETE').unlink(missing_ok=True)
    (output_dir / 'latents.jsonl').unlink(missing_ok=True)
    for temporary in output_dir.glob('.latent-write-*.tmp'):
        temporary.unlink()
    return contract_hash, journal


def validate_latent_payload(payload, row, axis, source_info, contract, contract_hash):
    import torch
    if not isinstance(payload, dict) or not isinstance(payload.get('metadata'), dict):
        raise ValueError('Existing latent payload lacks structured metadata')
    metadata = payload['metadata']
    if metadata.get('phase1_provenance') != source_info.get('phase1_provenance'):
        raise ValueError('Latent phase1 input provenance differs from the frozen encoding source')
    geometry = video_geometry(MNI_SHAPE, axis, contract.get('video_spatial_scale', 1))
    expected = {key: row[key] for key in ('dataset', 'subject_id', 'image_id', 'modality', 'split', 'preprocess_mode')}
    expected.update(geometry)
    expected.update(axis=axis, encoding_contract_sha256=contract_hash,
                    source_sha256=source_info['source_sha256'],
                    source_mask_sha256=source_info['source_mask_sha256'],
                    source_mask_reference_sha256=source_info.get('source_mask_reference_sha256'),
                    source_mapping_sha256=source_info.get('source_mapping_sha256'),
                    source_fov_mask_sha256=source_info.get('source_fov_mask_sha256'),
                    source_component_index=source_info.get('source_component_index'),
                    source_format=source_info.get('source_format', 'nifti'),
                    resources_sha256=contract['resource_sha256'],
                    vae_checkpoint_sha256=contract['vae_checkpoint_sha256'],
                    vae_family='Wan2.2_TI2V_5B_48ch', dtype='float32',
                    posterior='normalized_posterior_mean', space='MNI152_2mm',
                    spatial_shape=list(MNI_SHAPE), affine=MNI_AFFINE,
                    skull_stripped=True, preprocess_qc_passed=True,
                    normalization=normalization_spec(row)['normalization'],
                    quantiles=normalization_spec(row)['quantiles'],
                    intensity_unit=normalization_spec(row)['intensity_unit'],
                    valid_token_mask_definition=MASK_DEFINITION,
                    padding_convolution_note=PADDING_NOTE)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'Existing latent metadata conflicts with the frozen contract: {key}')
    if contract.get('image_audit_required'):
        for key in ('preprocessed_volume_sha256', 'preprocessed_brain_mask_sha256'):
            value = metadata.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Latent lacks an actual standardized image/mask digest')
        if not metadata.get('registered_image_path') or not metadata.get('registered_brain_mask_path'):
            raise ValueError('Latent lacks its actual registered image and brain-mask paths')
        from .registered import verify_persisted
        verify_persisted(metadata)
    low, high = metadata.get('intensity_low'), metadata.get('intensity_high')
    if (type(low) not in (float, int) or type(high) not in (float, int) or
            not math.isfinite(low) or not math.isfinite(high) or high <= low):
        raise ValueError('Existing latent has invalid intensity normalization bounds')
    fixed_bounds = normalization_spec(row)['fixed_range']
    if fixed_bounds is not None:
        import numpy as np
        if not np.allclose([low, high], fixed_bounds, rtol=1e-6, atol=0):
            raise ValueError('Existing latent has different fixed physical intensity bounds')
    dice = metadata.get('mask_vs_template_dice')
    if type(dice) not in (float, int) or not math.isfinite(dice) or not contract['minimum_mask_dice'] <= dice <= 1:
        raise ValueError('Existing latent fails the frozen registration-mask screening threshold')
    frames, height, width = geometry['padded_video_shape']
    expected_shape = [48, (frames - 1) // 4 + 1, height // 16, width // 16]
    latent = payload.get('latent')
    if (not isinstance(latent, torch.Tensor) or latent.dtype != torch.float32 or
            list(latent.shape) != expected_shape or not bool(torch.isfinite(latent).all())):
        raise ValueError('Existing latent shape, FP32 dtype or finite-value check failed')
    expected_fov = None
    if metadata.get('coverage_source') == 'registered_fov_mask':
        import nibabel as nib
        import numpy as np
        from .registered import voxel_sha256
        fov_image = nib.load(metadata['registered_fov_mask_path'])
        if tuple(fov_image.shape) != MNI_SHAPE or not np.allclose(fov_image.affine, MNI_AFFINE, atol=1e-4):
            raise ValueError('Saved FOV mask differs from the fixed MNI grid')
        expected_fov = np.asarray(fov_image.dataobj) > 0
        if voxel_sha256(expected_fov, True) != metadata.get('preprocessed_fov_sha256'):
            raise ValueError('Saved FOV mask content changed after encoding')
    expected_coverage = nominal_token_coverage(geometry, expected_fov)
    mask, coverage = payload.get('valid_token_mask'), payload.get('token_coverage')
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or
            not torch.equal(mask, expected_coverage > 0) or not bool(mask.any())):
        raise ValueError('Existing latent has a different nominal padding-token mask')
    if (not isinstance(coverage, torch.Tensor) or coverage.dtype != torch.float32 or
            not torch.equal(coverage, expected_coverage)):
        raise ValueError('Existing latent has missing or inconsistent nominal token coverage')
    return expected_shape


def latent_manifest_row(row, axis, target, shape, contract_hash, latent_hash):
    result = {key: row[key] for key in ('dataset', 'subject_id', 'image_id', 'modality', 'split')}
    result.update(latent_path=str(Path(target).resolve()), latent_shape=shape, axis=axis,
                  approved=True, qc_passed=True, encoding_contract_sha256=contract_hash,
                  latent_sha256=latent_hash)
    return result


def recover_latent(target, row, axis, source_info, contract, contract_hash, journal):
    import torch
    target = Path(target)
    if not target.exists():
        journal['completed'].pop(target.name, None)
        return None
    digest = sha256_file(target)
    recorded = journal['completed'].get(target.name)
    if recorded and recorded.get('latent_sha256') != digest:
        raise ValueError('Existing latent differs from its completed journal checksum; it will not be overwritten')
    payload = torch.load(target, map_location='cpu', weights_only=True)
    shape = validate_latent_payload(payload, row, axis, source_info, contract, contract_hash)
    output_row = latent_manifest_row(row, axis, target, shape, contract_hash, digest)
    journal['completed'][target.name] = {'latent_sha256': digest, 'output_row': output_row,
                                        'image_audit': build_image_audit(row, source_info, payload['metadata']),
                                        'recovered_from_existing_file': True}
    return output_row


def build_image_audit(row, source_info, metadata):
    result = {key: row[key] for key in ('dataset', 'subject_id', 'image_id', 'modality', 'split')}
    result.update({key: source_info.get(key) for key in (
        'source_path', 'source_sha256', 'source_format', 'source_component_index',
        'source_mask_path', 'source_mask_sha256', 'source_mask_reference_path',
        'source_mask_reference_sha256', 'source_mapping_path', 'source_mapping_sha256',
        'source_fov_mask_path', 'source_fov_mask_sha256')})
    result.update({key: metadata.get(key) for key in (
        'registered_image_path', 'registered_brain_mask_path', 'registered_fov_mask_path',
        'preprocessed_volume_sha256', 'preprocessed_brain_mask_sha256', 'preprocessed_fov_sha256',
        'space', 'spatial_shape', 'affine', 'skull_stripped', 'preprocess_qc_passed',
        'preprocess_mode', 'mask_vs_template_dice', 'brain_fov_coverage', 'coverage_type', 'fov_mask_kind',
        'source_nonfinite_policy', 'source_nan_replaced_with_zero', 'source_nan_fraction',
        'source_matrix_nan_values', 'source_matrix_rows_with_nan',
        'source_nan_fill_retains_anatomical_mask',
        'normalization', 'intensity_unit', 'intensity_low', 'intensity_high')})
    if source_info.get('phase1_provenance') is not None:
        result['phase1_provenance'] = source_info['phase1_provenance']
    return result


def save_new_latent_exclusively(payload, target):
    """Publish a complete new file with atomic no-overwrite hard-link creation."""
    import torch
    target = Path(target)
    descriptor, temporary_name = tempfile.mkstemp(prefix='.latent-write-', suffix='.tmp', dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Unlike replace(), link() fails if target already exists, including a
        # competing writer that did not respect our directory lock.
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def publish_completed_manifest(output_dir, output_rows, contract_hash, journal):
    output_dir = Path(output_dir)
    if len(output_rows) != journal['expected_records'] or len(journal['completed']) != journal['expected_records']:
        raise ValueError('Cannot mark an incomplete latent list as complete')
    manifest = output_dir / 'latents.jsonl'
    temporary = manifest.with_name(manifest.name + '.tmp')
    with temporary.open('w') as stream:
        for row in output_rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(manifest)
    audits = {}
    latent_rows = {}
    for item in journal['completed'].values():
        audit = item.get('image_audit')
        if audit is None:
            if journal.get('image_audit_required'):
                raise ValueError('Completed encoding lacks an actual processed-image audit')
            continue
        key = (audit['dataset'], audit['image_id'])
        if key in audits and audits[key] != audit:
            raise ValueError('Different axes disagree about their processed input volume')
        audits[key] = audit
        latent_rows.setdefault(key, []).append(item['output_row'])
    used_images = output_dir / 'used_images.jsonl'
    temporary_audit = used_images.with_suffix('.jsonl.tmp')
    with temporary_audit.open('w') as stream:
        for key in sorted(audits):
            record = dict(audits[key], latent_records=sorted(latent_rows[key], key=lambda row: row['axis']))
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary_audit.replace(used_images)
    journal['status'] = 'complete'
    journal['manifest_sha256'] = sha256_file(manifest)
    atomic_json(output_dir / 'encoding_journal.json', journal)
    atomic_json(output_dir / 'COMPLETE', {'schema_version': 'brain_mri_encoding_complete_v2',
                'contract_sha256': contract_hash, 'manifest_sha256': journal['manifest_sha256'],
                'records': len(output_rows), 'used_images': len(audits),
                'used_images_sha256': sha256_file(used_images)})
    return manifest


def main(argv=None, *, runtime=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='Explicit approved selected MRI JSONL')
    parser.add_argument('--resource-manifest', type=Path, required=True, help='Verified resource paths/size/SHA-256/license JSON')
    parser.add_argument('--output-dir', type=Path, required=True, help='Persistent latent directory; never receives MRI volumes')
    parser.add_argument('--registered-output-dir', type=Path, required=True, help='Persistent skull-stripped MNI152 2mm brain images and masks')
    parser.add_argument('--wan-source', type=Path, required=True, help='Existing official Wan2.2 source checkout')
    parser.add_argument('--vae-checkpoint', type=Path, required=True, help='Original verified Wan2.2_VAE.pth for 5B')
    parser.add_argument('--pet-code-root', type=Path, required=True, help='Existing read-only PET package root with frozen VAE adapter')
    parser.add_argument('--preprocess-device', default='cuda:0', help='FNIT device, default cuda:0')
    parser.add_argument('--vae-device', default='cuda:1', help='Frozen VAE device, default cuda:1')
    parser.add_argument('--axes', nargs='+', type=int, choices=[0, 1, 2], default=[0, 1, 2], help='Video axes; each original volume can yield three latents')
    parser.add_argument('--extent', type=int, choices=[192, 256], default=192, help='FNIT SynthMorph extent, default 192')
    parser.add_argument('--minimum-mask-dice', type=float, default=0.65, help='Screening threshold, default 0.65; does not substitute visual QC')
    parser.add_argument('--gpu-budget-gb', type=float, default=20., help='Per-device PyTorch allocation cap in decimal GB')
    parser.add_argument('--video-spatial-scale', type=int, choices=[1, 2], default=2,
                        help='After padding, sample each 2D viewing raster at 1x or 2x; MNI files unchanged (default 2)')
    parser.add_argument('--cpu-threads', type=int, default=8, help='CPU numerical threads for this GPU worker (default 8)')
    args = parser.parse_args(argv)
    rows = read_selected_manifest(args.manifest)
    resources = verify_resources(args.resource_manifest)
    import torch
    from .wan import load_vae
    if args.cpu_threads < 1:
        parser.error('cpu-threads must be positive')
    torch.set_num_threads(args.cpu_threads)
    if (not math.isfinite(args.minimum_mask_dice) or not 0 < args.minimum_mask_dice <= 1 or
            not math.isfinite(args.gpu_budget_gb) or args.gpu_budget_gb <= 0):
        parser.error('Dice threshold and GPU budget must be positive')
    args.output_dir = args.output_dir.resolve()
    contract, fingerprints = build_encoding_contract(args, rows, resources)
    axes = contract['axes']
    expected_names = [f'{image_identifier(row)}_axis{axis}.pt' for row in rows for axis in axes]
    with encoding_lock(args.output_dir):
        contract_hash, journal = initialize_journal(args.output_dir, contract, expected_names)
        journal['image_audit_required'] = contract.get('image_audit_required', False)
        completed = {}
        for row in rows:
            source_info = contract['sources'][image_identifier(row)]
            for axis in axes:
                target = args.output_dir / f'{image_identifier(row)}_axis{axis}.pt'
                recovered = recover_latent(target, row, axis, source_info, contract, contract_hash, journal)
                if recovered:
                    completed[target.name] = recovered
        atomic_json(args.output_dir / 'encoding_journal.json', journal)
        if len(completed) < len(expected_names):
            for device_name in set((args.preprocess_device, args.vae_device)):
                device = torch.device(device_name)
                if device.type == 'cuda':
                    total = torch.cuda.get_device_properties(device).total_memory
                    torch.cuda.set_per_process_memory_fraction(min(1., args.gpu_budget_gb * 1e9 / total), device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            runtime_signature = {key: value for key, value in contract.items()
                                 if key not in ('manifest_sha256', 'sources', 'registered_output_dir')}
            runtime_signature.update(preprocess_device=args.preprocess_device, vae_device=args.vae_device)
            if runtime is not None and runtime.get('signature') == runtime_signature:
                preprocessor, vae = runtime['preprocessor'], runtime['vae']
            else:
                preprocessor = Preprocessor(resources, args.preprocess_device, args.extent, args.minimum_mask_dice)
                vae = load_vae(args.wan_source, args.vae_checkpoint, args.vae_device, args.pet_code_root)
                if runtime is not None:
                    runtime.clear()
                    runtime.update(signature=runtime_signature, preprocessor=preprocessor, vae=vae)
            receipt = args.output_dir / 'encoding_receipts.jsonl'
            for row in rows:
                identifier = image_identifier(row)
                pending_axes = [axis for axis in axes if f'{identifier}_axis{axis}.pt' not in completed]
                if not pending_axes:
                    continue
                source_info = contract['sources'][identifier]
                assert_source_unchanged(source_info, fingerprints)
                start = time.monotonic()
                array, mask, preprocessing_metadata = preprocessor(row)
                from .registered import publish_registered
                stored = publish_registered(args.registered_output_dir, row, array, mask,
                                            preprocessing_metadata, identifier, preprocessor.last_fov_mask)
                preprocessing_metadata.update(stored)
                # Encoding reads the actual published standardized image, whose
                # voxel digest has just been independently verified after saving.
                import nibabel as nib
                import numpy as np
                array = load_volume(stored['registered_image_path']).get_fdata(dtype=np.float32)
                mask = np.asarray(nib.load(stored['registered_brain_mask_path']).dataobj) > 0
                for axis in pending_axes:
                    scaling = normalization_spec(row)
                    video, valid_token_mask, video_metadata = make_video(
                        array, mask, axis, *QUANTILES, fixed_range=scaling['fixed_range'],
                        intensity_unit=scaling['intensity_unit'], field_of_view=preprocessor.last_fov_mask,
                        spatial_scale=args.video_spatial_scale)
                    validate_video(video)
                    with torch.inference_mode(), torch.autocast(device_type=torch.device(args.vae_device).type, enabled=False):
                        latent = vae.encode(video.to(args.vae_device))[0].detach().float().cpu()
                    metadata = {key: row[key] for key in ('dataset', 'subject_id', 'image_id', 'modality', 'split')}
                    metadata.update(preprocessing_metadata)
                    metadata.update(video_metadata)
                    metadata.update(axis=axis, source_sha256=source_info['source_sha256'],
                                    source_mask_sha256=source_info['source_mask_sha256'],
                                    source_mask_reference_sha256=source_info.get('source_mask_reference_sha256'),
                                    source_mapping_sha256=source_info.get('source_mapping_sha256'),
                                    source_fov_mask_sha256=source_info.get('source_fov_mask_sha256'),
                                    encoding_contract_sha256=contract_hash,
                                    vae_family='Wan2.2_TI2V_5B_48ch',
                                    vae_checkpoint_sha256=contract['vae_checkpoint_sha256'],
                                    resources_sha256=contract['resource_sha256'],
                                    dtype='float32', posterior='normalized_posterior_mean',
                                    padding_convolution_note=PADDING_NOTE)
                    if source_info.get('phase1_provenance') is not None:
                        metadata['phase1_provenance'] = source_info['phase1_provenance']
                    payload = {'latent': latent, 'valid_token_mask': valid_token_mask,
                               'token_coverage': nominal_token_coverage(video_metadata, preprocessor.last_fov_mask), 'metadata': metadata}
                    shape = validate_latent_payload(payload, row, axis, source_info, contract, contract_hash)
                    target = args.output_dir / f'{identifier}_axis{axis}.pt'
                    save_new_latent_exclusively(payload, target)
                    digest = sha256_file(target)
                    output_row = latent_manifest_row(row, axis, target, shape, contract_hash, digest)
                    completed[target.name] = output_row
                    journal['completed'][target.name] = {'latent_sha256': digest, 'output_row': output_row,
                                                         'image_audit': build_image_audit(row, source_info, metadata),
                                                         'recovered_from_existing_file': False}
                    atomic_json(args.output_dir / 'encoding_journal.json', journal)
                    with receipt.open('a') as stream:
                        stream.write(json.dumps(dict(output_row, seconds_since_image_start=time.monotonic() - start)) + '\n')
                    del payload, latent, video
                del array, mask
        output_rows = [completed[name] for name in expected_names]
        output_manifest = publish_completed_manifest(args.output_dir, output_rows, contract_hash, journal)
        print(json.dumps({'manifest': str(output_manifest), 'encoded_axis_records': len(output_rows),
                          'encoding_contract_sha256': contract_hash, 'status': 'complete'}))


if __name__ == '__main__':
    main()
