# 训练与检查点

`train.py` 对已审核 latent 进行 Wan2.2 5B DiT 微调；冻结 VAE 不参与优化。输入可以是完整清单或不可变 SQLite 索引，且必须提供当前合格性登记。默认训练 50 轮、FP32 配合 TF32，不自行切换 float16。

## 运行接口

```bash
torchrun --standalone --nproc_per_node=6   -m brain_mri_world_model.train   --input-index "$IMMUTABLE_INPUT_INDEX"   --runtime-eligibility-registry "$APPLIED_ELIGIBILITY_REGISTRY"   --model-dir "$VERIFIED_WAN_MODEL_DIR"   --context-path "$FIXED_TEXT_CONTEXT"   --output-dir "$NEW_TRAINING_DIR"   --epochs 50 --micro-batch 1 --gradient-accumulation 4
```

- `--manifest` 与 `--input-index` 二选一：前者是已完成 latent JSONL，后者是完整输入索引目录。
- `--runtime-eligibility-registry`：当前 APPLIED 登记，排除未批准来源。
- `--model-dir`：核验过的本地 Diffusers 模型目录；`--context-path`：固定文本条件张量。
- `--output-dir`：新的输出目录；非空目录必须显式 `--resume`。
- `--epochs`：目标总轮数；`--micro-batch`：每 rank 每次输入量；`--gradient-accumulation`：每次优化前累积次数。
- `--epoch-axis-policy one_per_image`：每轮每影像选一轴；`all_axes`：使用所有轴。
- `--loader-workers` 默认 8；`--prefetch-factor` 默认 2。
- 学习率默认 `1e-5`，weight decay `1e-3`，warmup fraction `0.03`，梯度裁剪 `1.0`。
- `--checkpoint-every` 默认 1,000 优化步；0 表示仅轮末保存。`--keep-last` 默认保留最近两个完整检查点。

其余噪声分布、seed 和日志参数参见 `python -m brain_mri_world_model.train --help`。训练目标包含扩散噪声；下游无噪特征提取直接使用 latent，两者不能混为同一个设置。

## 恢复与 GPU 数变化

完整检查点包含模型、各 rank optimizer、RNG、训练元数据和 COMPLETE。仅模型归档用于提取表征，不能作为完整训练恢复。中途检查点不能标记为该轮完成。

当前原生 `--resume` 要求相同 world size 和训练配置。由 6 卡迁到 8 卡必须先转换完整优化器状态，保留 Adam step、动量、参数组、scheduler，并重建当前轮的逻辑 batch 游标；不能复制或补零 optimizer shard，也不能静默重置。6 个 RNG 流无法构成 8 卡逐位相同恢复。

本仓库尚未提供完成验证的跨 world-size 恢复转换器。直接把 6 卡 checkpoint 传给 8 卡 `--resume` 会被拒绝。八卡训练启动前必须完成这项转换及真实输入验证，不能把加载模型权重称为实验续训。

## 依赖来源

模型薄适配器使用 [Diffusers](https://github.com/huggingface/diffusers) 和 [Wan2.2](https://github.com/Wan-Video/Wan2.2)。外部权重遵循原始来源条款；本仓库不包含这些权重或原实现副本。
