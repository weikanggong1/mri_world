"""Wan2.2 5B 的 FP32 薄适配、已审核 latent 清单和文本条件。

不复制官方或现有 PET 项目的模型实现；可直接复用 PET 基础 loader。
导入本模块不加载权重、不下载资源、不启动训练。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import torch


WAN22_VAE_SIZE = 2818839170
WAN22_VAE_SHA256 = "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
CANONICAL_DATASETS = frozenset({"ukb", "mrrate", "tiantan", "fomo"})
REPRESENTATIONS = frozenset({'volume_xyz','mra_projection','mra_native_2d'})
NATIVE_VIEW_KINDS = frozenset({'projection_angle','ordered_views'})
MNI_REFERENCE_SHAPE = [91,109,91]
MNI_REFERENCE_AFFINE = [[-2.,0.,0.,90.],[0.,2.,0.,-126.],[0.,0.,2.,-72.],[0.,0.,0.,1.]]
_LATENT_HASH_CACHE: dict[str, tuple[tuple[int, int, int, int], str]] = {}


def configure_tf32(enabled: bool = True) -> None:
    """保留 FP32 参数/输入/损失，允许 Ampere Tensor Core 使用 TF32。"""
    torch.backends.cuda.matmul.allow_tf32 = enabled
    torch.backends.cudnn.allow_tf32 = enabled


def file_sha256(path: str | Path) -> str:
    from .integrity import file_sha256 as cached_sha256
    return cached_sha256(path)


def representation_of(record):
    """Absent representation is the legacy three-axis volume protocol."""
    value=record.get('representation','volume_xyz')
    if not isinstance(value,str) or value not in REPRESENTATIONS:raise ValueError('Unknown volume_xyz/mra_projection/mra_native_2d representation')
    return value


def space_for_representation(record):
    return {'volume_xyz':'MNI152_2mm','mra_projection':'MNI152_2mm_projection',
            'mra_native_2d':'native_2D_MRA_views'}[representation_of(record)]


def encoding_reference_geometry(contract):
    """Normalize only reference geometry; an angular video has no3D array affine."""
    representation=representation_of(contract)
    if representation=='mra_native_2d':
        if (contract.get('space')!='native_2D_MRA_views' or contract.get('frame_geometry')!='native_2D_MRA_views'
                or contract.get('frame_axis_kind') not in NATIVE_VIEW_KINDS
                or any(contract.get(key) is not None for key in ('spatial_shape','affine','reference_space','reference_spatial_shape','reference_affine'))):
            raise ValueError('Native2D views have native frame geometry, no MNI reference or3D affine')
        return {'space':'native_2D_MRA_views'}
    if representation=='mra_projection':
        if (contract.get('space')!='MNI152_2mm_projection' or contract.get('spatial_shape') is not None
                or contract.get('affine') is not None):
            raise ValueError('Projection uses MNI152_2mm_projection and reference_*; never a single3D affine/shape')
        space,shape,affine=(contract.get(name) for name in ('reference_space','reference_spatial_shape','reference_affine'))
    else:space,shape,affine=(contract.get(name) for name in ('space','spatial_shape','affine'))
    if (space!='MNI152_2mm' or not isinstance(shape,list) or shape!=MNI_REFERENCE_SHAPE
            or any(type(value) is not int for value in shape) or not isinstance(affine,list) or len(affine)!=4
            or any(not isinstance(row,list) or len(row)!=4 for row in affine)
            or any(type(value) not in (int,float) or not math.isfinite(value) for row in affine for value in row)
            or affine!=MNI_REFERENCE_AFFINE):
        raise ValueError('Encoding reference must be the fixed FSL MNI1522mm space/shape/affine')
    return {'space':space,'spatial_shape':shape,'affine':affine}


def latent_shape_for_video(shape,axis,scale=1):
    if (not isinstance(shape,list) or len(shape)!=3 or any(type(value) is not int or value<1 for value in shape)
            or type(axis) is not int or axis not in (0,1,2) or type(scale) is not int or scale not in (1,2)):
        raise ValueError('Expected explicit three-dimensional video shape, axis0/1/2 and scale1/2')
    other=[index for index in range(3) if index!=axis]
    if shape[other[0]]<2 or shape[other[1]]<2:raise ValueError('Video plane dimensions must be at least2')
    return [48,1+math.ceil((shape[axis]-1)/4),
            math.ceil(shape[other[0]]/32)*2*scale,math.ceil(shape[other[1]]/32)*2*scale]


def expected_record_latent_shape(contract,source,record):
    representation=representation_of(contract)
    if representation_of(record)!=representation:raise ValueError('Manifest representation differs from its child contract')
    if representation in ('mra_projection','mra_native_2d'):
        if contract.get('axes')!=[0] or record.get('axis')!=0 or type(record.get('axis')) is not int:
            raise ValueError('Projection is one angular video with exactly axes=[0]')
        expected_format='mra_projection_frames' if representation=='mra_projection' else 'mra_2d_collection'
        if source.get('source_format')!=expected_format or source.get('source_component_index') is not None:
            raise ValueError('Single-view-sequence representation/source format or component index differs')
        shape=source.get('frame_shape')
        if representation=='mra_native_2d':
            if (contract.get('video_spatial_scale')!=1 or contract.get('video_raster_policy')!='native_2d_aspect_resize_then_pad'
                    or contract.get('resize_max_side')!=512):raise ValueError('Native2D requires aspect resize<=512, scale1, then padding')
            if not isinstance(source.get('source_path'),str) or Path(source['source_path']).suffix.lower()!='.json':
                raise ValueError('Native2D primary source must be a small JSON collection descriptor, never copied raw-frame NPY')
            if not isinstance(shape,list) or len(shape)!=3 or any(type(value) is not int for value in shape) or max(shape[1:])>512:
                raise ValueError('Native2D frame_shape must be prepad F/H/W after aspect resize with side<=512')
    else:
        if source.get('source_format') in ('mra_projection_frames','mra_2d_collection'):raise ValueError('View sequences cannot masquerade as volume_xyz')
        shape=contract.get('spatial_shape')
    return latent_shape_for_video(shape,record['axis'],contract.get('video_spatial_scale',1))


def verify_used_images(manifest_path, records, contract, complete):
    """核验实际用图账本与冻结来源、轴清单逐项一致；不打开原MRI或模型。"""
    if 'image_audit_required' in contract and type(contract['image_audit_required']) is not bool:
        raise ValueError('image_audit_required 必须为JSON布尔值')
    representation=representation_of(contract)
    if representation!='volume_xyz' and contract.get('image_audit_required') is not True:
        raise ValueError('Projection encoding requires a complete used-images audit ledger')
    if contract.get('image_audit_required') is not True:
        return None
    reference=encoding_reference_geometry(contract)
    manifest_path = Path(manifest_path).resolve()
    audit_path = manifest_path.parent / 'used_images.jsonl'
    if not audit_path.is_file():
        raise ValueError('完整编码缺少必需的 used_images.jsonl')
    expected_sha = complete.get('used_images_sha256')
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64 or
            any(character not in '0123456789abcdef' for character in expected_sha)):
        raise ValueError('COMPLETE 缺少有效 used_images_sha256')
    before = audit_path.stat()
    fingerprint = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
    if file_sha256(audit_path) != expected_sha:
        raise ValueError('used_images.jsonl 实际SHA-256与COMPLETE不一致')
    axes = contract.get('axes')
    if (not isinstance(axes, list) or not axes or
            any(type(axis) is not int or axis not in (0, 1, 2) for axis in axes) or len(set(axes)) != len(axes)):
        raise ValueError('实际用图账本需要明确、不重复的合同axes')
    scale=contract.get('video_spatial_scale',1)
    raster_policy=contract.get('video_raster_policy','native')
    if type(scale) is not int or scale not in (1,2):
        raise ValueError('实际用图账本的video_spatial_scale必须为1或2')
    if representation=='mra_native_2d':
        if scale!=1 or raster_policy!='native_2d_aspect_resize_then_pad':raise ValueError('Native2D has its own scale1 aspect-resize raster protocol')
    elif raster_policy not in ('native','pad_then_bilinear_spatial_scale') or (scale==2 and raster_policy!='pad_then_bilinear_spatial_scale'):
        raise ValueError('实际用图账本的video raster policy无效')
    sources = contract.get('sources')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('实际用图账本缺少完整合同sources')
    if representation!='volume_xyz' and axes!=[0]:
        raise ValueError('Projection child must declare only angular axis0')
    identity_fields = ('dataset', 'subject_id', 'image_id', 'modality', 'split')
    expected = {}
    for record in records:
        if any(not isinstance(record.get(key), str) or not record[key] for key in identity_fields):
            raise ValueError('启用实际用图审计时每条latent必须有完整影像身份')
        key = (record['dataset'], record['image_id'])
        common = {name: record[name] for name in identity_fields}
        group = expected.setdefault(key, {'identity': common, 'axes': {}})
        if group['identity'] != common:
            raise ValueError('同一image_id的latent身份或split不一致')
        axis = record.get('axis')
        if type(axis) is not int or axis not in axes or axis in group['axes']:
            raise ValueError('实际用图账本对应的latent轴重复或超出合同')
        source_key=hashlib.sha256(json.dumps(list(key),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
        source=sources.get(source_key)
        if not isinstance(source,dict):raise ValueError('Latent image lacks a matching frozen source')
        expected_shape=expected_record_latent_shape(contract,source,record)
        if record.get('latent_shape')!=expected_shape:
            raise ValueError('latent_shape与合同空间scale/视频轴不一致')
        group['axes'][axis] = record
    if any(set(group['axes']) != set(axes) for group in expected.values()):
        raise ValueError('至少一幅实际用图缺少合同声明的latent轴')
    source_keys = {
        hashlib.sha256(json.dumps(list(key), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        for key in expected
    }
    if source_keys != set(sources):
        raise ValueError('latent影像集合没有完整覆盖合同sources')
    if type(complete.get('used_images')) is not int or complete['used_images'] != len(expected):
        raise ValueError('COMPLETE中的实际用图数量与latent影像集合不一致')

    def same_json(left, right):
        return json.dumps(left, ensure_ascii=False, sort_keys=True, separators=(',', ':')) == json.dumps(
            right, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

    def valid_sha(value):
        return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

    source_fields = ('source_path', 'source_sha256', 'source_format', 'source_component_index',
        'source_mask_path', 'source_mask_sha256', 'source_mask_reference_path', 'source_mask_reference_sha256',
        'source_mapping_path', 'source_mapping_sha256', 'source_fov_mask_path', 'source_fov_mask_sha256',
        'projection_metadata_path','projection_metadata_sha256','phase1_provenance')
    seen = set()
    volume_hashes = {}
    volume_splits = {}
    with audit_path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            audit = json.loads(line)
            if not isinstance(audit, dict):
                raise ValueError('used_images每行必须为对象')
            key = (audit.get('dataset'), audit.get('image_id'))
            if key not in expected or key in seen:
                raise ValueError('used_images含未知或重复影像，不能标记完整')
            seen.add(key)
            group = expected[key]
            if any(audit.get(name) != value for name, value in group['identity'].items()):
                raise ValueError('used_images身份、模态或split与latent不一致')
            if representation_of(audit)!=representation:raise ValueError('Audit representation differs from its encoding contract')
            source_key = hashlib.sha256(json.dumps(list(key), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
            source = sources[source_key]
            for name in source_fields:
                expected_value = source.get(name, 'nifti' if name == 'source_format' else None)
                if not same_json(audit.get(name), expected_value):
                    raise ValueError('used_images与冻结源合同不一致: ' + name)
            if not valid_sha(audit.get('source_sha256')):
                raise ValueError('used_images缺少实际源摘要')
            if representation=='mra_native_2d':
                if (audit.get('encoded_input_descriptor_path')!=source.get('source_path')
                        or any(audit.get(key) is not None for key in ('registered_image_path','registered_brain_mask_path','registered_fov_mask_path'))):
                    raise ValueError('Native2D ledger references the collection descriptor, never registered image/masks')
                volume_sha=audit.get('frame_sequence_sha256',audit.get('preprocessed_volume_sha256'))
                if not valid_sha(volume_sha) or not valid_sha(audit.get('preprocessed_fov_sha256')):
                    raise ValueError('Native2D ledger requires in-memory frame-sequence and rectangle-FOV digests')
                if audit.get('preprocessed_volume_sha256') is not None and audit['preprocessed_volume_sha256']!=volume_sha:
                    raise ValueError('Native frame-sequence digest alias differs')
            else:
                for name in ('registered_image_path', 'registered_brain_mask_path', 'registered_fov_mask_path'):
                    if not isinstance(audit.get(name), str) or not audit[name]:
                        raise ValueError('used_images缺少实际标准图/brain/FOV路径')
                for name in ('preprocessed_volume_sha256', 'preprocessed_brain_mask_sha256', 'preprocessed_fov_sha256'):
                    if not valid_sha(audit.get(name)):
                        raise ValueError('used_images缺少有效标准图/brain/FOV体素摘要')
                volume_sha=audit['preprocessed_volume_sha256']
            volume_identity=(representation,volume_sha)
            split = group['identity']['split']
            if volume_identity in volume_splits and volume_splits[volume_identity] != split:
                raise ValueError('相同标准图体素摘要跨train/val/test，拒绝精确影像重复泄露')
            volume_splits[volume_identity] = split
            volume_hashes[key] = volume_sha
            if audit.get('preprocess_qc_passed') is not True:raise ValueError('used_images lacks the required preprocessing QA')
            if representation=='mra_native_2d':
                if (not same_json(encoding_reference_geometry(audit),reference)
                        or audit.get('registration_performed') is not False or audit.get('qc_scope')!='native_2d_video'
                        or audit.get('frame_axis_kind')!=source.get('frame_axis_kind',contract.get('frame_axis_kind'))
                        or not same_json(audit.get('frame_shape'),source.get('frame_shape'))
                        or audit.get('mask_vs_template_dice') is not None or audit.get('brain_fov_coverage') is not None):
                    raise ValueError('Native2D QA must retain native space/ordered views, no registration or3D brain coverage claims')
                status=audit.get('source_brain_extraction_status')
                if not isinstance(status,str) or not status:raise ValueError('Native2D must honestly retain source brain-extraction status')
                if source.get('source_brain_extraction_status') is not None and status!=source['source_brain_extraction_status']:
                    raise ValueError('Native source brain-extraction status differs from descriptor provenance')
            elif audit.get('skull_stripped') is not True:
                raise ValueError('Volume/MNI projection lacks required skull-stripping evidence')
            if representation=='mra_projection':
                if (audit.get('space')!='MNI152_2mm_projection' or audit.get('frame_geometry')!='MNI152_2mm_projection'
                        or audit.get('frame_axis_kind')!='projection_angle' or audit.get('affine') is not None
                        or audit.get('spatial_shape') is not None or not same_json(audit.get('frame_shape'),source.get('frame_shape'))
                        or not same_json(encoding_reference_geometry(audit),reference)):
                    raise ValueError('Projection audit requires frame shape/angle geometry and fixed reference_*; no3D movie affine')
                for alias in ('preprocessed_frame_sequence_sha256','frame_sequence_sha256'):
                    if alias in audit and audit[alias]!=volume_sha:raise ValueError('Frame-sequence voxel digest alias differs')
                for path_field,sha_field in (('source_mask_path','source_mask_sha256'),('source_fov_mask_path','source_fov_mask_sha256'),
                        ('source_mapping_path','source_mapping_sha256'),('projection_metadata_path','projection_metadata_sha256')):
                    if not isinstance(source.get(path_field),str) or not source[path_field] or not valid_sha(source.get(sha_field)):
                        raise ValueError('Projection source lacks frozen mask/FOV/mapping/metadata provenance')
            elif representation=='volume_xyz' and (audit.get('space')!=contract.get('space') or not same_json(audit.get('spatial_shape'),contract.get('spatial_shape'))
                    or not same_json(audit.get('affine'),contract.get('affine'))):
                raise ValueError('Volume audit space/shape/affine differ from its contract')
            if representation!='mra_native_2d':
                dice, coverage = audit.get('mask_vs_template_dice'), audit.get('brain_fov_coverage')
                if (type(dice) not in (float, int) or not math.isfinite(dice) or
                        not contract.get('minimum_mask_dice', 0) <= dice <= 1):
                    raise ValueError('used_images配准mask筛查结果无效')
                if (type(coverage) not in (float, int) or not math.isfinite(coverage) or
                        not source.get('minimum_brain_fov_coverage', .95) <= coverage <= 1):
                    raise ValueError('used_images脑部采集覆盖不足或无效')
            normalization = source.get('normalization')
            if not isinstance(normalization, dict):
                raise ValueError('合同缺少逐影像归一化定义')
            if (audit.get('normalization') != normalization.get('normalization') or
                    audit.get('intensity_unit') != normalization.get('intensity_unit')):
                raise ValueError('used_images归一化模式或单位与合同不一致')
            bounds = [audit.get('intensity_low'), audit.get('intensity_high')]
            if (any(type(value) not in (float, int) or not math.isfinite(value) for value in bounds)
                    or bounds[1] <= bounds[0]):
                raise ValueError('used_images强度范围无效')
            fixed = normalization.get('fixed_range')
            if fixed is not None:
                if (not isinstance(fixed, list) or len(fixed) != 2 or
                        any(type(value) not in (int,float) or not math.isfinite(value) for value in fixed) or
                        any(not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=0) for a,b in zip(bounds,fixed))):
                    raise ValueError('used_images固定范围与源合同不一致')
            if source.get('source_format') == 'npy_masked_maps':
                policy = source.get('nonfinite_policy', 'reject')
                if audit.get('source_nonfinite_policy') != policy:
                    raise ValueError('used_images的FC非有限值政策与合同不一致')
                if policy == 'replace_nan_with_zero':
                    replaced = audit.get('source_nan_replaced_with_zero')
                    if type(replaced) is not int or replaced < 0 or audit.get('source_nan_fill_retains_anatomical_mask') is not True:
                        raise ValueError('used_images缺少逐component NaN填0数或完整原brain-mask声明')
            listed = audit.get('latent_records')
            if not isinstance(listed, list) or len(listed) != len(group['axes']):
                raise ValueError('used_images中的latent_records轴数量不完整')
            audit_axes = set()
            for listed_record in listed:
                if not isinstance(listed_record, dict):
                    raise ValueError('used_images latent_records必须为对象')
                axis = listed_record.get('axis')
                if type(axis) is not int or axis not in group['axes'] or axis in audit_axes:
                    raise ValueError('used_images latent_records含未知/重复轴')
                audit_axes.add(axis)
                normalized = dict(listed_record)
                expected_record = group['axes'][axis]
                if normalized.get('latent_path') != expected_record.get('latent_path'):
                    value = normalized.get('latent_path')
                    if not isinstance(value, str) or not value:
                        raise ValueError('used_images latent_path缺失')
                    path = Path(value)
                    normalized['latent_path'] = os.path.abspath(path if path.is_absolute() else manifest_path.parent/path)
                if not same_json(normalized, expected_record):
                    normalized.setdefault('representation','volume_xyz')
                    comparable=dict(expected_record);comparable.setdefault('representation','volume_xyz')
                    if not same_json(normalized,comparable):
                        raise ValueError('used_images latent_records与实际latent清单不一致')
    if seen != set(expected):
        raise ValueError('used_images没有逐source完整覆盖所有实际影像')
    after = audit_path.stat()
    if fingerprint != (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino):
        raise ValueError('used_images在校验过程中发生变化')
    return {'path': str(audit_path), 'sha256': expected_sha, 'images': len(seen),
            'axis_records': len(records), 'image_voxel_sha256': volume_hashes}


def _pet_loaders(pet_code_root: str | Path):
    source = Path(pet_code_root).resolve() / "pet_wan/model.py"
    if not source.is_file():
        raise FileNotFoundError(f"PET 基础 loader 不存在：{source}")
    module_name = "_brain_mri_existing_pet_model_" + file_sha256(source)[:16]
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, source)
        if spec is None or spec.loader is None:
            raise ImportError(str(source))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def load_vae(wan_source: str | Path, vae_checkpoint: str | Path,
             device: str | torch.device, pet_code_root: str | Path):
    """复用已核查的 PET FP32 冻结 VAE loader；返回 encode/decode 接口。"""
    configure_tf32()
    checkpoint = Path(vae_checkpoint)
    if checkpoint.stat().st_size != WAN22_VAE_SIZE or file_sha256(checkpoint) != WAN22_VAE_SHA256:
        raise ValueError("Wan2.2_VAE.pth 大小或 SHA-256 与固定官方 5B 权重不一致")
    return _pet_loaders(pet_code_root).load_vae(wan_source, vae_checkpoint, device)


def load_backbone(model_dir: str | Path, gradient_checkpointing: bool = True,
                  device: str | torch.device | None = None,
                  pet_code_root: str | Path | None = None):
    """加载本地 Diffusers 5B transformer；不要求存在完整 pipeline。"""
    configure_tf32()
    model_dir = Path(model_dir).resolve()
    configuration = json.loads((model_dir / "transformer/config.json").read_text())
    expected = {"in_channels": 48, "out_channels": 48,
                "num_layers": 30, "num_attention_heads": 24,
                "attention_head_dim": 128, "text_dim": 4096}
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"要求 Wan2.2 TI2V-5B：{key} 应为 {value}")
    if tuple(configuration.get("patch_size", [])) != (1, 2, 2):
        raise ValueError("不支持此 DiT patch_size")
    if pet_code_root:
        model = _pet_loaders(pet_code_root).load_backbone(
            model_dir, gradient_checkpointing, device)
    else:
        from diffusers import WanTransformer3DModel
        model = WanTransformer3DModel.from_pretrained(
            str(model_dir), subfolder="transformer", local_files_only=True,
            torch_dtype=torch.float32, low_cpu_mem_usage=True)
        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
        if device is not None:
            model.to(device)
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise TypeError("DiT 参数必须全部为 float32")
    # Diffusers 0.35.2 原生处理器使用 PyTorch SDPA，不走原生 Wan 的 BF16 FA 转换。
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor
    model.set_attn_processor(WanAttnProcessor())
    return model


def backbone_provenance(model_dir: str | Path,
                        pet_code_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(model_dir).resolve()
    result = {"model_dir": str(root), "parameter_dtype": "float32", "tf32": True,
              "attention": "diffusers.WanAttnProcessor/PyTorch_SDPA",
              "config_sha256": file_sha256(root / "transformer/config.json"),
              "index_sha256": file_sha256(root / "transformer/diffusion_pytorch_model.safetensors.index.json")}
    if pet_code_root:
        result["pet_loader_sha256"] = file_sha256(Path(pet_code_root) / "pet_wan/model.py")
    return result


def load_manifest(manifest_path: str | Path, *, audit_latents: bool = False) -> tuple[list[dict[str, Any]], str]:
    """默认核对完整元数据；audit_latents显式逐latent强制读摘要。"""
    path = Path(manifest_path).resolve()
    if (path.parent / "CORPUS_COMPLETE").exists() or (path.parent / "corpus_contract.json").exists():
        from .corpus import load_corpus_manifest
        return load_corpus_manifest(path, single_loader=_load_single_manifest, audit_latents=audit_latents)
    return _load_single_manifest(path, audit_latents=audit_latents)


def _audit_latent_file(path, expected_sha):
    """显式审计才读取完整latent；实际训练reader另按需处理。"""
    from .integrity import file_sha256 as cached_sha256
    if (not isinstance(expected_sha,str) or len(expected_sha)!=64 or
            any(character not in '0123456789abcdef' for character in expected_sha)):
        raise ValueError('显式latent审计需要已记录的小写SHA-256')
    if cached_sha256(path,force=True)!=expected_sha:
        raise ValueError('latent文件实际SHA-256与编码完成清单不一致')


def _load_single_manifest(manifest_path: str | Path, *, audit_latents: bool = False) -> tuple[list[dict[str, Any]], str]:
    """只接受显式选定且 QC 通过的清单；同一受试者不得跨 split。"""
    manifest_path = Path(manifest_path).resolve()
    from .integrity import fingerprint
    manifest_fingerprint = fingerprint(manifest_path)
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = file_sha256(manifest_path)
    records, subject_splits, seen_paths = [], {}, set()
    contract_hashes = set()
    records_with_contract = 0
    for line_number, line in enumerate(manifest_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "source_encoding_manifest" in record:
            raise ValueError("合并清单必须有同目录 corpus_contract.json 和 CORPUS_COMPLETE，且不能作为子批次嵌套")
        if "encoding_contract_sha256" in record:
            value = record["encoding_contract_sha256"]
            if (not isinstance(value, str) or len(value) != 64 or
                    any(character not in "0123456789abcdef" for character in value)):
                raise ValueError("encoding_contract_sha256 必须为小写 SHA-256")
            contract_hashes.add(value)
            records_with_contract += 1
        for key in ("latent_path", "subject_id", "dataset", "modality", "axis", "split"):
            if key not in record:
                raise ValueError(f"第 {line_number} 行缺少 {key}")
        if record.get("approved") is not True or record.get("qc_passed") is not True:
            raise ValueError(f"第 {line_number} 行未经选定审核或未通过 QC")
        if type(record["axis"]) is not int or record["axis"] not in (0, 1, 2):
            raise ValueError("axis 必须为固定 MNI 网格的 0/1/2")
        representation=representation_of(record)
        if representation!='volume_xyz' and (record['axis']!=0 or 'encoding_contract_sha256' not in record):
            raise ValueError('Projection manifest requires its complete encoding contract and only angular axis0')
        if record["split"] not in {"train", "val", "test"}:
            raise ValueError("split 必须为 train/val/test")
        for key in ("subject_id", "dataset", "modality"):
            if not isinstance(record[key], str) or not record[key].strip():
                raise ValueError(f"{key} 必须为非空字符串")
        if record["dataset"] not in CANONICAL_DATASETS:
            raise ValueError("dataset 只能是 ukb/mrrate/tiantan/fomo；扫描入口名不能当作队列名")
        subject = (record["dataset"], record["subject_id"])
        if subject in subject_splits and subject_splits[subject] != record["split"]:
            raise ValueError("发现同一 subject_id 跨 train/val/test；请重新划分清单")
        subject_splits[subject] = record["split"]
        source = Path(record["latent_path"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        # 这里只做词法绝对化，不对百万条latent路径resolve/stat。
        source = Path(os.path.abspath(source))
        if source in seen_paths:
            raise ValueError("同一 latent_path 在清单中重复")
        seen_paths.add(source)
        record["latent_path"] = str(source)
        if "latent_shape" not in record:
            raise ValueError('清单必须显式带latent_shape；日常加载不解码latent推断shape')
        shape = record["latent_shape"]
        if (not isinstance(shape, (list, tuple)) or len(shape) != 4 or
                any(type(size) is not int or size <= 0 for size in shape) or
                shape[0] != 48 or shape[2] % 2 or shape[3] % 2):
            raise ValueError("latent_shape 必须为 [48,T,偶数H,偶数W]")
        record["latent_shape"] = list(shape)
        records.append(record)
    if not records:
        raise ValueError("清单为空")
    if contract_hashes:
        if len(contract_hashes) != 1 or records_with_contract != len(records):
            raise ValueError("清单不能混合不同编码契约或无契约记录")
        complete_path = manifest_path.parent / "COMPLETE"
        contract_path = manifest_path.parent / "encoding_contract.json"
        if not complete_path.is_file() or not contract_path.is_file():
            raise ValueError("编码尚未完成：缺少同目录 COMPLETE 或 encoding_contract.json")
        complete = json.loads(complete_path.read_text())
        contract_sha = next(iter(contract_hashes))
        if (complete.get("schema_version") != "brain_mri_encoding_complete_v2" or
                complete.get("manifest_sha256") != manifest_sha or
                complete.get("contract_sha256") != contract_sha or
                type(complete.get("records")) is not int or complete["records"] != len(records) or
                file_sha256(contract_path) != contract_sha):
            raise ValueError("编码完成标记、实际清单或编码契约的 SHA-256/记录数量不一致")
        contract = json.loads(contract_path.read_text())
        verify_used_images(manifest_path, records, contract, complete)
    if fingerprint(manifest_path)!=manifest_fingerprint:
        raise ValueError('清单在读取核验期间发生变化')
    if audit_latents:
        for record in records:
            _audit_latent_file(record['latent_path'],record.get('latent_sha256'))
    return records, manifest_sha


def _verify_latent_hash(path: str | Path, expected_sha: str) -> None:
    """每 worker 首次及文件属性变化后校验 latent；不反复读完冻结文件。"""
    if (not isinstance(expected_sha, str) or len(expected_sha) != 64 or
            any(character not in "0123456789abcdef" for character in expected_sha)):
        raise ValueError("latent_sha256 必须为小写 SHA-256")
    path = Path(path)
    before = path.stat()
    signature = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino)
    cached = _LATENT_HASH_CACHE.get(str(path))
    if cached == (signature, expected_sha):
        return
    actual = file_sha256(path)
    after = path.stat()
    after_signature = (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino)
    if after_signature != signature:
        raise ValueError("latent 文件在 SHA-256 校验过程中发生变化")
    if actual != expected_sha:
        raise ValueError("latent 文件 SHA-256 与已审核清单不一致")
    _LATENT_HASH_CACHE[str(path)] = (signature, actual)


def validate_text_embedding(embedding: torch.Tensor) -> torch.Tensor:
    if (not isinstance(embedding, torch.Tensor) or embedding.dtype != torch.float32 or
            embedding.ndim != 2 or not 1 <= embedding.shape[0] <= 512 or
            embedding.shape[1] != 4096 or not torch.isfinite(embedding).all()):
        raise ValueError("text_embedding 必须是有限 float32 [1..512,4096] 的真实 T5 输出")
    # 与官方 pipeline 相同，以零补足文本序列；不制造文本 embedding。
    return torch.nn.functional.pad(embedding, (0, 0, 0, 512 - embedding.shape[0]))


def load_context(context_path: str | Path | None) -> torch.Tensor | None:
    if context_path is None:
        return None
    payload = torch.load(context_path, map_location="cpu", weights_only=True)
    embedding = payload["text_embedding"] if isinstance(payload, dict) else payload
    return validate_text_embedding(embedding)


def load_latent(record: dict[str, Any], fallback_context: torch.Tensor | None = None, *, audit=False):
    """返回 CPU FP32 latent、bool token mask 和真实文本条件。"""
    if audit and "latent_sha256" in record:
        _verify_latent_hash(record["latent_path"], record["latent_sha256"])
    payload = torch.load(record["latent_path"], map_location="cpu", weights_only=True)
    latent, metadata = payload["latent"], payload["metadata"]
    if (not isinstance(latent, torch.Tensor) or latent.dtype != torch.float32 or
            list(latent.shape) != record["latent_shape"] or not torch.isfinite(latent).all()):
        raise ValueError("latent 的形状、精度或有限值检查失败")
    for key in ("subject_id", "dataset", "modality", "axis", "split"):
        if metadata.get(key) != record[key]:
            raise ValueError(f"latent metadata 与清单的 {key} 不一致")
    if record.get('image_id') is not None and metadata.get('image_id') != record['image_id']:
        raise ValueError('latent metadata 与清单的 image_id 不一致')
    representation=representation_of(record)
    if representation_of(metadata)!=representation:
        raise ValueError('latent metadata representation differs from its manifest')
    if metadata.get('preprocess_qc_passed') is not True:raise ValueError('latent缺少预处理QA证据')
    if representation=='mra_native_2d':
        encoding_reference_geometry(metadata)
        if (record['axis']!=0 or metadata.get('registration_performed') is not False
                or metadata.get('qc_scope')!='native_2d_video'
                or not isinstance(metadata.get('source_brain_extraction_status'),str)
                or not metadata['source_brain_extraction_status']
                or not isinstance(metadata.get('encoded_input_descriptor_path'),str)
                or not metadata['encoded_input_descriptor_path']
                or metadata.get('video_spatial_scale',1)!=1
                or metadata.get('video_raster_policy','native_2d_aspect_resize_then_pad')!='native_2d_aspect_resize_then_pad'):
            raise ValueError('Native2D payload must retain descriptor/native-video QA and no registration claim')
        shape=metadata.get('frame_shape')
        expected=latent_shape_for_video(shape,0,1)
        if max(shape[1:])>512 or expected!=list(latent.shape):
            raise ValueError('Native2D latent differs from its aspect-resized prepad frame_shape')
    elif metadata.get('skull_stripped') is not True:
        raise ValueError('MNI volume/projection latent缺少去颅骨证据')
    if representation=='mra_projection':
        encoding_reference_geometry(metadata)
        if (record['axis']!=0 or metadata.get('frame_geometry')!='MNI152_2mm_projection'
                or metadata.get('frame_axis_kind')!='projection_angle'):
            raise ValueError('Projection latent must retain angular-video geometry, never3D XYZ slices')
        expected=latent_shape_for_video(metadata.get('frame_shape'),0,metadata.get('video_spatial_scale',1))
        if 'video_spatial_scale' in metadata and expected!=list(latent.shape):
            raise ValueError('Projection latent shape differs from its frame_shape/raster scale')
    elif representation=='volume_xyz' and metadata.get('space')!='MNI152_2mm':
        raise ValueError('Volume latent must retain MNI1522mm space')
    token_shape = (latent.shape[1], latent.shape[2] // 2, latent.shape[3] // 2)
    mask = payload.get("valid_token_mask")
    if mask is None:
        if metadata.get("all_latent_tokens_valid") is not True:
            raise ValueError("缺少 valid_token_mask，且未显式证明所有 latent token 均有效")
        if any(value for pair in metadata.get("padding_voxels", []) for value in pair):
            raise ValueError("存在 padding 时必须提供 valid_token_mask")
        mask = torch.ones(token_shape, dtype=torch.bool)
    if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or
            tuple(mask.shape) != token_shape or not mask.any()):
        raise ValueError("valid_token_mask 必须是非空 bool [T,H/2,W/2]")
    context = payload.get("text_embedding")
    if context is not None:
        context = validate_text_embedding(context)
    elif fallback_context is not None:
        context = fallback_context
    else:
        raise ValueError("没有真实 T5 text_embedding；请先缓存空提示或模态/轴提示")
    return latent, mask, context


def load_batch(records: list[dict[str, Any]], context: torch.Tensor | None,
               device: torch.device):
    samples = [load_latent(record, context) for record in records]
    return tuple(torch.stack([sample[index] for sample in samples]).to(device)
                 for index in range(3))


def fixed_seed(record: dict[str, Any], seed: int, purpose: str = "noise") -> int:
    identity = {key: record.get(key) for key in
                ("subject_id", "dataset", "modality", "axis", "image_id")}
    if not identity["image_id"]:
        identity["image_id"] = Path(record["latent_path"]).name
    message = json.dumps([identity, int(seed), purpose], sort_keys=True).encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8], "little") % (2**63 - 1)


def cache_text_context(model_dir: str | Path, output_path: str | Path,
                       prompt: str = "", device: str = "cpu") -> None:
    """离线运行本地 umT5，缓存真实空提示/描述条件；不下载文件。"""
    from transformers import AutoTokenizer, UMT5EncoderModel
    root = Path(model_dir).resolve()
    configure_tf32()
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    encoder = UMT5EncoderModel.from_pretrained(
        root / "text_encoder", local_files_only=True,
        torch_dtype=torch.float32, low_cpu_mem_usage=True).to(device).eval().requires_grad_(False)
    tokens = tokenizer([prompt], padding="max_length", truncation=True,
                       max_length=512, add_special_tokens=True, return_tensors="pt")
    tokens = {key: value.to(device) for key, value in tokens.items()}
    with torch.no_grad(), torch.autocast(torch.device(device).type, enabled=False):
        embedding = encoder(**tokens).last_hidden_state[0]
    length = int(tokens["attention_mask"][0].sum())
    embedding = embedding[:length].float().cpu()
    validate_text_embedding(embedding)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".tmp")
    torch.save({"text_embedding": embedding,
                "metadata": {"source": "local_UMT5EncoderModel", "prompt": prompt,
                             "text_encoder_config_sha256": file_sha256(root / "text_encoder/config.json"),
                             "parameter_dtype": "float32", "tokens": length}}, temporary)
    temporary.replace(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="本地 5B Diffusers 目录，含 tokenizer/text_encoder")
    parser.add_argument("--output-path", required=True, help="新建 T5 context .pt 文件")
    parser.add_argument("--prompt", default="", help="空提示或无诊断标签的模态/轴描述")
    parser.add_argument("--device", default="cpu", help="T5 编码设备，默认 CPU FP32")
    args = parser.parse_args()
    cache_text_context(args.model_dir, args.output_path, args.prompt, args.device)


if __name__ == "__main__":
    main()
