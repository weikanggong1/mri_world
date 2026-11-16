"""原生2D MRA视角集合→Wan2.2 latent；不做3D配准，也不保存原图副本。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import time

from .integrity import fingerprint
from .prepare import (ENCODING_VERSION, MASK_DEFINITION, PADDING_NOTE, QUANTILES,
    atomic_json, encoding_lock, image_identifier, initialize_journal, make_video,
    nominal_token_coverage, publish_completed_manifest, save_new_latent_exclusively,
    sha256_file, video_geometry)

REPRESENTATION = 'mra_native_2d'
SPACE = 'native_2D_MRA_views'
SOURCE_FORMAT = 'mra_2d_collection'
RASTER_POLICY = 'native_2d_aspect_resize_then_pad'
NORMALIZATION = 'within_native_foreground_robust_quantile_to_minus1_plus1'
IDENTITY = ('dataset', 'subject_id', 'image_id', 'modality', 'split')
DEFERRED_QC = 'deferred_to_encoder_pixel_checks'


class NativePixelQCError(ValueError):
    """实际像素为空、非有限或强度退化；可另行复核，不是永久排除。"""


def input_review_status(row):
    if row.get('native_qc_status') == DEFERRED_QC:
        if (any(row.get(flag) is not True for flag in ('approved','selected_by_user','native_protocol_reviewed'))
                or row.get('native_qc_reviewed') is True or row.get('preprocess_qc_reviewed') is True):
            raise ValueError('Deferred native input needs reviewed selection/protocol and must not claim completed pixel QA')
        return DEFERRED_QC
    for flag in ('approved','preprocess_qc_reviewed','native_qc_reviewed'):
        if row.get(flag) is not True:raise ValueError('Native MRA video review is incomplete: '+flag)
    return 'legacy_reviewed'


def validate_selected_record(row,base):
    """仅校验选图/协议身份；不打开descriptor或原图，不把协议审核当像素QA。"""
    from .selection_utils import split_for_subject
    from .wan import CANONICAL_DATASETS
    if not isinstance(row,dict) or any(not isinstance(row.get(name),str) or not row[name] for name in IDENTITY):
        raise ValueError('Native MRA requires complete canonical identity')
    if row['dataset'] not in CANONICAL_DATASETS or row['split']!=split_for_subject(row['dataset'],row['subject_id']):
        raise ValueError('Native MRA must retain the common subject split')
    if row.get('representation')!=REPRESENTATION or row.get('source_format')!=SOURCE_FORMAT:
        raise ValueError('Explicit native2D collection representation is required')
    input_review_status(row)
    source=row.get('source_path')
    if not isinstance(source,str) or Path(source).suffix.lower()!='.json':
        raise ValueError('Native source is a small JSON descriptor; raw frame copies are not accepted')
    result=dict(row);result['source_path']=os.path.abspath(os.path.join(str(base),source))
    return result


def frame_geometry(header):
    """以原始像素物理宽高比生成不超过512的行列尺寸；不会估计配准。"""
    shape, spacing = header.get('shape'), header.get('voxel_sizes')
    if (not isinstance(shape, list) or len(shape) != 3 or
            any(type(size) is not int or size < 1 for size in shape) or shape.count(1) != 1 or
            not isinstance(spacing, list) or len(spacing) < 3 or
            any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in spacing[:3])):
        raise ValueError('Native2D requires an explicit singleton-plane NIfTI shape and positive voxel sizes')
    axes = [axis for axis, size in enumerate(shape) if size > 1]
    # NIfTI第一有效轴是显示列，第二有效轴是显示行；不按世界坐标重定向。
    width, height = (shape[axis] for axis in axes)
    sx, sy = (float(spacing[axis]) for axis in axes)
    physical_width, physical_height = width * sx, height * sy
    longest_pixels = min(512, max(width, height))
    pitch = max(physical_width, physical_height) / longest_pixels
    output = [max(2, round(physical_height / pitch)), max(2, round(physical_width / pitch))]
    return {'input_shape': shape, 'plane_voxel_axes': axes,
            'output_hw': output, 'display_rule': 'native_second_voxel_axis_rows_first_axis_columns',
            'physical_pixel_aspect_preserved': True, 'estimated_registration': False}


def checked_descriptor(path, row):
    descriptor = json.loads(Path(path).read_text())
    if descriptor.get('schema_version') != 'native_mra_collection_v1':
        raise ValueError('Unsupported native MRA collection descriptor')
    identity = ('dataset', 'subject_id', 'exam_id', 'image_id', 'split')
    if (any(not isinstance(row.get(key), str) or not row[key] or descriptor.get(key) != row[key] for key in identity)
            or descriptor.get('collection_id') != row.get('image_id')):
        raise ValueError('Native descriptor and selected manifest identities disagree')
    for key, value in {'representation': REPRESENTATION, 'source_format': SOURCE_FORMAT,
                       'space': SPACE, 'frame_geometry': SPACE,
                       'video_spatial_scale': 1, 'maximum_frame_side': 512}.items():
        if descriptor.get(key) != value or row.get(key) != value:
            raise ValueError('Native descriptor viewing protocol differs from its selected row: ' + key)
    if descriptor.get('frame_axis_kind') not in ('ordered_views', 'projection_angle'):
        raise ValueError('Native descriptor must declare ordered_views or documented projection_angle')
    if not isinstance(descriptor.get('ordering_basis'), str) or not descriptor['ordering_basis']:
        raise ValueError('Native view order needs a provenance label')
    if not isinstance(descriptor.get('source_brain_extraction_status'), str) or not descriptor['source_brain_extraction_status']:
        raise ValueError('Source brain extraction status must be retained without inference')
    frames = descriptor.get('frames')
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError('A native view video requires at least two frames')
    keys, geometries = set(), []
    for frame in frames:
        source = frame.get('source_path')
        if not isinstance(source, str) or not Path(source).is_absolute() or source in keys:
            raise ValueError('Frame references must be unique absolute original NIfTI paths')
        keys.add(source)
        geometries.append(frame_geometry(frame.get('source_header', {})))
    shape = [len(frames), max(g['output_hw'][0] for g in geometries), max(g['output_hw'][1] for g in geometries)]
    return descriptor, geometries, shape


def read_selected(path):
    rows, keys = [], set()
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row=validate_selected_record(json.loads(line),Path(path).parent)
        key = (row['dataset'], row['image_id'])
        if key in keys:
            raise ValueError('Duplicate native video identity')
        keys.add(key)
        rows.append(row)
    if not rows:
        raise ValueError('Selected native MRA manifest is empty')
    return rows


def decode_frame(task):
    """原始NIfTI只解码一次；前后文件指纹及已有头信息保持一致。"""
    import nibabel as nib
    import numpy as np
    frame, geometry = task
    path = Path(frame['source_path'])
    before = fingerprint(path)
    prior = frame.get('source_prior_identity', {}).get('image', {})
    if prior and (prior.get('size') != before[2] or prior.get('mtime_ns') != before[3]):
        raise ValueError('Native source changed since its inventory record')
    image = nib.load(str(path))
    if list(image.shape) != geometry['input_shape']:
        raise ValueError('Native source shape differs from its descriptor')
    expected_spacing = frame['source_header']['voxel_sizes'][:3]
    if not np.allclose(image.header.get_zooms()[:3], expected_spacing, rtol=1e-4, atol=1e-6):
        raise ValueError('Native source pixel spacing differs from its descriptor')
    array = np.squeeze(image.get_fdata(dtype=np.float32, caching='unchanged')).T.copy()
    if (array.ndim != 2 or not np.isfinite(array).all() or
            not np.any(array != 0) or float(array.min()) == float(array.max())):
        raise NativePixelQCError('Native MRA frame is not a finite nonconstant 2D image')
    if fingerprint(path) != before:
        raise ValueError('Native source changed during its single decode')
    return array, {'source_path': str(path), 'filesystem_fingerprint': list(before),
                   'original_shape': list(image.shape), 'original_dtype': str(image.get_data_dtype())}


def assemble_frames(descriptor, geometries, shape, cpu_workers=8):
    """并行读取原图，在RAM中等比缩放/补边；矩形FOV不是脑掩膜。"""
    import numpy as np
    import torch
    import torch.nn.functional as F
    arrays = np.zeros(shape, dtype=np.float32)
    fov = np.zeros(shape, dtype=np.bool_)
    receipts = []
    with ThreadPoolExecutor(max_workers=cpu_workers) as pool:
        results = pool.map(decode_frame, zip(descriptor['frames'], geometries))
        for index, ((array, receipt), geometry) in enumerate(zip(results, geometries)):
            height, width = geometry['output_hw']
            tensor = torch.from_numpy(array)[None, None]
            if list(array.shape) != [height, width]:
                tensor = F.interpolate(tensor, size=(height, width), mode='bilinear', align_corners=False)
            top, left = (shape[1] - height) // 2, (shape[2] - width) // 2
            arrays[index, top:top+height, left:left+width] = tensor[0, 0].numpy()
            fov[index, top:top+height, left:left+width] = True
            receipts.append(dict(receipt, **geometry, rectangle_top_left_hw=[top, left, height, width]))
    return arrays, fov, receipts


def make_native_video(array, fov):
    import numpy as np
    support = (array != 0) & fov
    if any(np.count_nonzero(frame) < 2 for frame in support):
        raise NativePixelQCError('Native video contains an empty frame')
    try:
        video, mask, metadata = make_video(array, support, 0, field_of_view=fov, spatial_scale=1)
    except ValueError as error:
        if any(token in str(error) for token in ('Degenerate within-brain','Normalized brain is constant','MRI must be finite')):
            raise NativePixelQCError(str(error)) from error
        raise
    # Reuse arithmetic only; this support is not anatomical skull stripping.
    metadata['normalization'] = NORMALIZATION
    metadata['foreground_fraction_clipped_below'] = metadata.pop('brain_fraction_clipped_below')
    metadata['foreground_fraction_clipped_above'] = metadata.pop('brain_fraction_clipped_above')
    metadata.update(video_raster_policy=RASTER_POLICY, resize_max_side=512,
                    coverage_source='native_frame_rectangles', intensity_support='finite_nonzero_native_pixels_not_a_brain_mask')
    return video, mask, metadata


def build_contract(args, rows):
    from .wan import WAN22_VAE_SHA256, WAN22_VAE_SIZE
    if args.vae_checkpoint.stat().st_size != WAN22_VAE_SIZE or sha256_file(args.vae_checkpoint) != WAN22_VAE_SHA256:
        raise ValueError('Expected the verified original Wan2.2 5B VAE checkpoint')
    sources, details = {}, {}
    axis_kinds = set()
    for row in rows:
        descriptor, geometries, shape = checked_descriptor(row['source_path'], row)
        source = dict(source_path=row['source_path'], source_sha256=sha256_file(row['source_path']),
            source_format=SOURCE_FORMAT, source_component_index=None, representation=REPRESENTATION,
            frame_shape=shape, frame_axis_kind=descriptor['frame_axis_kind'],
            source_brain_extraction_status=descriptor['source_brain_extraction_status'],
            normalization={'normalization': NORMALIZATION, 'quantiles': list(QUANTILES),
                           'fixed_range': None, 'intensity_unit': 'arbitrary'})
        if row.get('source_sha256') and row['source_sha256'] != source['source_sha256']:
            raise ValueError('Native collection descriptor changed after review')
        identifier = image_identifier(row)
        sources[identifier] = source
        details[identifier] = (descriptor, geometries, shape)
        axis_kinds.add(descriptor['frame_axis_kind'])
    if len(axis_kinds) != 1:
        raise ValueError('Split differently documented view-axis kinds into separate child batches')
    contract = dict(schema_version=ENCODING_VERSION, representation=REPRESENTATION,
        manifest_sha256=sha256_file(args.manifest), sources=sources, axes=[0], space=SPACE,
        frame_geometry=SPACE, frame_axis_kind=next(iter(axis_kinds)), dtype='float32', tf32=True,
        vae_checkpoint_sha256=WAN22_VAE_SHA256, padding='pad_only_T4k_plus1_HW32',
        video_spatial_scale=1, video_raster_policy=RASTER_POLICY, resize_max_side=512,
        valid_token_mask_definition=MASK_DEFINITION, image_audit_required=True,
        native_input_review={image_identifier(row):{'status':input_review_status(row),
            'native_protocol_sha256':row.get('native_protocol_sha256')} for row in rows},
        native_encoder_code_sha256=sha256_file(__file__),
        prepare_helper_code_sha256=sha256_file(Path(__file__).with_name('prepare.py')),
        registered_voxel_digest_code_sha256=sha256_file(Path(__file__).with_name('registered.py')),
        wan_adapter_code_sha256=sha256_file(Path(__file__).with_name('wan.py')),
        official_vae_code_sha256=sha256_file(args.wan_source / 'wan/modules/vae2_2.py'),
        pet_loader_code_sha256=sha256_file(args.pet_code_root / 'pet_wan/model.py'))
    return contract, details


def validate_payload(payload, row, source, contract, contract_sha):
    import torch
    from .wan import expected_record_latent_shape
    metadata = payload['metadata']
    shape = expected_record_latent_shape(contract, source, dict(row, axis=0))
    expected = dict(**{key: row[key] for key in IDENTITY}, representation=REPRESENTATION,
        axis=0, space=SPACE, frame_geometry=SPACE, frame_axis_kind=source['frame_axis_kind'],
        frame_shape=source['frame_shape'], registration_performed=False,
        qc_scope='native_2d_video', preprocess_qc_passed=True,
        encoded_input_descriptor_path=source['source_path'], source_sha256=source['source_sha256'],
        encoding_contract_sha256=contract_sha, vae_checkpoint_sha256=contract['vae_checkpoint_sha256'])
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Native latent metadata conflicts with its reviewed source contract')
    if input_review_status(row)==DEFERRED_QC and (metadata.get('source_native_qc_status')!=DEFERRED_QC or
            metadata.get('actual_native_pixel_checks_passed') is not True):
        raise ValueError('Deferred native input lacks actual encoder pixel-QA evidence')
    latent, mask = payload['latent'], payload['valid_token_mask']
    if latent.dtype != torch.float32 or list(latent.shape) != shape or not torch.isfinite(latent).all():
        raise ValueError('Native latent dtype/shape/values are invalid')
    if mask.dtype != torch.bool or list(mask.shape) != [shape[1], shape[2]//2, shape[3]//2] or not mask.any():
        raise ValueError('Native token FOV mask is invalid')
    return shape


def main(argv=None, *, runtime=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'output-dir', 'wan-source', 'vae-checkpoint', 'pet-code-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--gpu-budget-gb', type=float, default=20.)
    parser.add_argument('--cpu-workers', type=int, default=8)
    args = parser.parse_args(argv)
    if args.cpu_workers < 1 or not math.isfinite(args.gpu_budget_gb) or not 0 < args.gpu_budget_gb <= 20:
        parser.error('CPU workers must be positive; GPU memory budget must be >0 and <=20GB')
    args.output_dir = args.output_dir.resolve()
    with encoding_lock(args.output_dir):
        if (args.output_dir / 'COMPLETE').exists():
            previous = json.loads((args.output_dir / 'encoding_contract.json').read_text())
            if previous['manifest_sha256'] != sha256_file(args.manifest):
                raise ValueError('Completed native batch belongs to a different selected list')
            from .wan import load_manifest
            rows, _ = load_manifest(args.output_dir / 'latents.jsonl')
            print(json.dumps({'status': 'complete_metadata_reused', 'videos': len(rows)}))
            return
        rows = read_selected(args.manifest)
        contract, details = build_contract(args, rows)
        import torch
        from .registered import voxel_sha256
        from .wan import load_vae
        torch.set_num_threads(args.cpu_workers)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        device = torch.device(args.device)
        if device.type == 'cuda':
            total = torch.cuda.get_device_properties(device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(1., args.gpu_budget_gb * 1e9 / total), device)
            torch.cuda.reset_peak_memory_stats(device)
        names = [image_identifier(row) + '_axis0.pt' for row in rows]
        contract_sha, journal = initialize_journal(args.output_dir, contract, names)
        journal['image_audit_required'] = True
        output_rows, vae = [], None
        for row, name in zip(rows, names):
            if name in journal['completed']:
                output_rows.append(journal['completed'][name]['output_row'])
                continue
            started = time.monotonic()
            source = contract['sources'][image_identifier(row)]
            target = args.output_dir / name
            recovered = target.exists()
            if recovered:
                payload = torch.load(target, map_location='cpu', weights_only=True)
                shape = validate_payload(payload, row, source, contract, contract_sha)
                metadata = payload['metadata']
            else:
                descriptor, geometries, frame_shape = details[image_identifier(row)]
                arrays, fov, receipts = assemble_frames(descriptor, geometries, frame_shape, args.cpu_workers)
                video, mask, geometry = make_native_video(arrays, fov)
                if vae is None:
                    signature = (str(args.vae_checkpoint.resolve()), str(args.wan_source.resolve()), str(device))
                    if runtime is not None and runtime.get('signature') == signature:
                        vae = runtime['vae']
                    else:
                        vae = load_vae(args.wan_source, args.vae_checkpoint, device, args.pet_code_root)
                        if runtime is not None:
                            runtime.clear(); runtime.update(signature=signature, vae=vae)
                with torch.inference_mode(), torch.autocast(device.type, enabled=False):
                    latent = vae.encode(video.to(device))[0].detach().float().cpu()
                metadata = dict(**{key: row[key] for key in IDENTITY}, **geometry,
                    representation=REPRESENTATION, axis=0, space=SPACE, frame_geometry=SPACE,
                    frame_axis_kind=source['frame_axis_kind'], frame_shape=frame_shape,
                    has_single_3D_spatial_affine=False, registration_performed=False,
                    preprocess_mode='native_2d_no_registration', qc_scope='native_2d_video',
                    skull_stripped=False, source_brain_extraction_status=source['source_brain_extraction_status'],
                    source_native_qc_status=input_review_status(row),actual_native_pixel_checks_passed=True,
                    preprocess_qc_passed=True, encoded_input_descriptor_path=source['source_path'],
                    frame_sequence_sha256=voxel_sha256(arrays), preprocessed_fov_sha256=voxel_sha256(fov, True),
                    source_sha256=source['source_sha256'], source_frame_receipts=receipts,
                    ordering_basis=descriptor['ordering_basis'],
                    encoding_contract_sha256=contract_sha, vae_checkpoint_sha256=contract['vae_checkpoint_sha256'],
                    vae_family='Wan2.2_TI2V_5B_48ch', dtype='float32', posterior='normalized_posterior_mean',
                    padding_convolution_note=PADDING_NOTE)
                payload = {'latent': latent, 'valid_token_mask': mask,
                           'token_coverage': nominal_token_coverage(geometry, fov), 'metadata': metadata}
                shape = validate_payload(payload, row, source, contract, contract_sha)
                save_new_latent_exclusively(payload, target)
                del arrays, fov, video, latent
            digest = sha256_file(target)
            record = {key: row[key] for key in IDENTITY}
            record.update(axis=0, representation=REPRESENTATION, latent_path=str(target), latent_shape=shape,
                approved=True, qc_passed=True, encoding_contract_sha256=contract_sha, latent_sha256=digest)
            audit = dict(source)
            audit.update({key: row[key] for key in IDENTITY})
            audit.update({key: metadata[key] for key in ('representation', 'space', 'frame_geometry', 'frame_axis_kind',
                'frame_shape', 'registration_performed', 'qc_scope', 'preprocess_qc_passed',
                'source_brain_extraction_status', 'encoded_input_descriptor_path', 'frame_sequence_sha256',
                'preprocessed_fov_sha256', 'skull_stripped', 'normalization', 'intensity_unit', 'intensity_low', 'intensity_high')})
            audit.update({key:metadata.get(key) for key in ('source_native_qc_status','actual_native_pixel_checks_passed')})
            journal['completed'][name] = dict(output_row=record, latent_sha256=digest,
                image_audit=audit, recovered_from_existing_file=recovered)
            atomic_json(args.output_dir / 'encoding_journal.json', journal)
            output_rows.append(record)
            print(json.dumps({'status': 'video_encoded', 'image_id': row['image_id'], 'frames': source['frame_shape'][0],
                'latent_shape': shape, 'seconds': time.monotonic()-started,
                'gpu_peak_allocated_gb': torch.cuda.max_memory_allocated(device)/1e9 if device.type == 'cuda' else None}), flush=True)
        publish_completed_manifest(args.output_dir, output_rows, contract_sha, journal)
        print(json.dumps({'status': 'complete', 'native_videos': len(output_rows), 'registration_calls': 0}))


if __name__ == '__main__':
    main()
