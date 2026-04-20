# SAM Integration

这个目录存放 Clip-UIE 接入 SAM 的新增文件。

## 包含内容

- `generate_sam_masks.py`: 为 UIEB 数据集离线生成 `mask_sam/*.npy` 掩码。
- `configs/clipuie_uieb_sam.yaml`: 启用 SAM 掩码训练的示例配置。

## 推荐流程

1. 先运行 `generate_sam_masks.py` 为 `train/val/test` 生成掩码。
2. 再使用 `python train.py --config sam_integration/configs/clipuie_uieb_sam.yaml` 训练。
3. 测试时使用相同配置，或额外传入 `--checkpoint`。

## 掩码约定

- 掩码保存在每个 split 下的 `mask_sam/` 目录。
- 每张图对应一个 `.npy` 文件，文件名与输入图像同名。
- 掩码值范围为 `[0, 1]`，值越大表示越偏向前景。
