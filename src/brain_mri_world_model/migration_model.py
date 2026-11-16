"""Initialize identical architecture; strict full checkpoint restore must follow.

No pretrained tensor download/read. Original loader is the same Diffusers class;
this only avoids loading weights that would immediately be overwritten.
"""
import hashlib,json,pathlib
def initialize_from_config(model_dir,pet_code_root=None):
    import torch,importlib.metadata
    from .wan import configure_tf32
    configure_tf32()
    from diffusers import WanTransformer3DModel
    from diffusers.models.transformers.transformer_wan import WanAttnProcessor
    if torch.__version__!='2.6.0+cu124'or importlib.metadata.version('diffusers')!='0.35.2':raise ValueError('ORIGINAL_TORCH_DIFFUSERS_REQUIRED')
    root=pathlib.Path(model_dir);config=json.loads((root/'transformer/config.json').read_text())
    # bind_contract checks the original loader digest before this function runs.
    expected={'in_channels':48,'out_channels':48,'num_layers':30,'num_attention_heads':24,'attention_head_dim':128,'text_dim':4096}
    if any(config.get(k)!=v for k,v in expected.items())or tuple(config.get('patch_size',[]))!=(1,2,2):raise ValueError('EXACT_ORIGINAL_MODEL_ARCHITECTURE_REQUIRED')
    model=WanTransformer3DModel.from_config(config).float();model.enable_gradient_checkpointing();model.set_attn_processor(WanAttnProcessor());return model
