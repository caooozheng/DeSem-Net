# DeSem-Net
<img width="1571" height="696" alt="image" src="https://github.com/user-attachments/assets/253dc958-5110-4b82-8a9f-044ac11f1fb0" />

## 项目结构

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

