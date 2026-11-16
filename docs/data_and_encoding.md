# 数据、预处理与 VAE 编码

三维影像保留去颅骨并完成配准的 2 mm NIfTI；脑 mask 与有效视野 FOV 分别保存。三轴切片视频各产生一个冻结 VAE latent。原生二维序列单独标记，不能称为 2 mm 三维影像。

建议的可迁移结构为：

```text
data/<dataset>/<subject_id>/<modality>/
  images/<image_id>.nii.gz
  masks/<image_id>.brainmask.nii.gz
  masks/<image_id>.fov.nii.gz
  latents/<image_id>/axis0.pt
  latents/<image_id>/axis1.pt
  latents/<image_id>/axis2.pt
```

`image_id` 区分同一受试者的访视与采集。迁移前后保留 dataset、subject_id、image_id、modality、split 及源文件 SHA-256；变更路径应生成新 path map，不覆盖原 provenance。目标影像和 latent 必须为可独立读取的文件，不能保留指向源服务器的绝对链接。

## 输入与输出

`preprocess_runner` 输入 JSONL recipes、已审核资源清单、冻结的外部 FNIT PyTorch 运行时，输出注册影像、mask/FOV、SQLite 状态和逐图回执。默认仅建立计划；`--execute` 才执行 GPU。计算使用 PyTorch，影像读写使用 nibabel；本包不启动 FSL、FreeSurfer、SPM、MRtrix3 或 AFNI。

`finalize_encoding_list` 输入通过预处理的记录及已审归一化策略，输出可编码清单，不重新运行数值预处理。`prepare` 接收这份显式批准清单及已核资源，输出 latent、used_images、encoding contract、journal 和完成标志。已处理、已编码和批准进入训练是三个不同状态。

```bash
python -m brain_mri_world_model.preprocess_runner --help
python -m brain_mri_world_model.finalize_encoding_list --help
python -m brain_mri_world_model.prepare --help
```

示例以实际配置文件变量传参，不提供会自动放宽审核的示例合同：

```bash
python -m brain_mri_world_model.preprocess_runner   --recipes "$REVIEWED_RECIPES_JSONL"   --resource-manifest "$VERIFIED_RESOURCES_JSON"   --fnit-runtime "$FROZEN_FNIT_RUNTIME"   --output-dir "$PREPROCESS_RUN_DIR"   --registered-output-dir "$REGISTERED_IMAGE_DIR"   --gpu-indices 0 --gpu-budget-gb 20 --execute
```

`--extent` 默认 192，`--minimum-mask-dice` 默认 0.65，`--max-images 0` 遍历全部显式 recipe；这些参数不能替代质量审核。资源不足时等待或停止，不能自动降精度或降低质量门槛。全部参数含义可由上述 `--help` 查看。

不重复终验已接受批次。迁移发送时沿用原始完整性记录，接收端在写入过程中核对 SHA-256；新文件系统的指纹独立记录，原指纹保留为来源证据。只有完整且合格的编码批次才进入训练索引。
