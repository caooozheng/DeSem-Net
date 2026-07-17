# DeSem-Net
<img width="1571" height="696" alt="image" src="https://github.com/user-attachments/assets/253dc958-5110-4b82-8a9f-044ac11f1fb0" />

## 项目结构
```
Clip-UIE/
  train.py                     # 训练入口
  test.py                      # 有参考测试入口
  no_ref_test.py               # 无参考图像增强与指标测试入口
  pyproject.toml               # Python 包与依赖配置

  clipuie/
    config.py                  # YAML 配置解析
    data/                      # 数据集、DataLoader、prompt 构造
    models/                    # ClipUIE 网络结构与模块
    losses/                    # 训练损失
    engine/                    # Trainer、Evaluator、checkpoint
    utils/                     # 指标、设备、随机种子等工具

  configs/
    uieb.yaml
    euvp.yaml
    euvp_dark.yaml
    euvp_scene.yaml
    lsui.yaml
    u45.yaml
    ch60.yaml
    val.yaml
    ablations/                 # 消融实验配置

  sam_integration/
    generate_sam_masks.py      # 生成 SAM 掩码
    uranker/                   # URanker 相关代码
```

##  环境配置
```
conda create -n DeSem-Net python=3.10 -y
conda activate DeSem-Net
安装项目依赖：
pip install -e .
如果使用多模态配置，需要准备本地 Hugging Face 模型目录。当前 YAML 默认路径为：
configs/models/clip-vit-base-patch32
configs/models/Qwen2.5-1.5B-Instruct
```
##  数据集划分
```
UnderwaterDatasets/
  UIEB/
    train/
      input/
      target/
      mask_sam/
    val/
      input/
      target/
      mask_sam/
    test/
      input/
      target/
      mask_sam/
```
##  生成SAM掩码
准备 SAM 权重，例如：
```
sam_integration/weight/sam_vit_h_4b8939.pth
```
生成掩码
```
python sam_integration/generate_sam_masks.py \
  --dataset-root UnderwaterDatasets/UIEB \
  --sam-checkpoint sam_integration/weight/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --gpu 0 \
  --splits train val test
```

##  模型权重
```
artifacts/{experiment.name}/checkpoints/latest.pth
artifacts/{experiment.name}/checkpoints/best.pth
artifacts/{experiment.name}/checkpoints/runs/{run_id}/
```

##  Train
```
python train.py --config configs/uieb.yaml --gpu 0
```
训练过程中会在 artifacts/ 下生成实验目录：
```
artifacts/
  uieb/
    checkpoints/
      best.pth
      latest.pth
      runs/
    predictions/
    logs/
```
##  Test
```
python test.py \
  --config configs/uieb.yaml \
  --checkpoint artifacts/uieb/checkpoints/best.pth \
  --split test \
  --gpu 0 \
  --save-images
```
##  常用参数
```
--config          YAML 配置文件
--checkpoint      指定模型权重，优先级高于 YAML 中的 evaluation.checkpoint
--split           val 或 test
--gpu             GPU 编号
--device          指定 cpu / cuda / cuda:0
--save-images     保存增强结果和 input-output-target 对比图
--self-ensemble   使用翻转 / 转置测试时增强
--soft-route      使用软路由
--output-branch   固定使用某个分支输出，例如 1
--route-output    使用模型路由输出
```
对于没有 GT 的图像，可以使用 no_ref_test.py。

##  注意事项
train.py 建议始终显式传入 --config，例如 python train.py --config configs/uieb.yaml。
使用多模态模型时，configs/models/ 下需要提前准备 CLIP 和 LLM 模型文件。
使用 SAM mask 时，确保每张输入图像都有同名 .npy 掩码；没有掩码时会自动使用全 1 mask。
测试时如果 YAML 中的 evaluation.checkpoint 不存在，请使用 --checkpoint 手动指定。
默认输入尺寸为 256x256，可在 YAML 的 dataset.image_size 中修改。
##  引用
如果该项目对你的研究有帮助，请引用本项目或相关论文。
