# DeSem-Net
<img width="1571" height="696" alt="image" src="https://github.com/user-attachments/assets/253dc958-5110-4b82-8a9f-044ac11f1fb0" />

## Project Structure
```
Clip-UIE/
  train.py                     # Training entry point
  test.py                      # Full-reference evaluation entry point
  no_ref_test.py               # No-reference image enhancement and evaluation entry point
  pyproject.toml               # Python package and dependency configuration

  clipuie/
    config.py                  # YAML configuration parser
    data/                      # Dataset, DataLoader, and prompt construction
    models/                    # ClipUIE network architecture and modules
    losses/                    # Training losses
    engine/                    # Trainer, Evaluator, and checkpoint utilities
    utils/                     # Metrics, device utilities, random seed helpers, etc.

  configs/
    uieb.yaml
    euvp.yaml
    euvp_dark.yaml
    euvp_scene.yaml
    lsui.yaml
    u45.yaml
    ch60.yaml
    val.yaml
    ablations/                 # Ablation experiment configurations

  sam_integration/
    generate_sam_masks.py      # Generate SAM masks
    uranker/                   # URanker-related code
```
## Environment Setup
```
conda create -n DeSem-Net python=3.10 -y
conda activate DeSem-Net

Install project dependencies
pip install -e .
```
If multimodal configuration is enabled, local Hugging Face model directories are required. 
The default paths in the YAML files are:
```
configs/models/clip-vit-base-patch32
configs/models/Qwen2.5-1.5B-Instruct
```
## Dataset Split
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
## Generate SAM Masks
```
Prepare the SAM checkpoint, for example:
sam_integration/weight/sam_vit_h_4b8939.pth
Generate masks:
python sam_integration/generate_sam_masks.py \
  --dataset-root UnderwaterDatasets/UIEB \
  --sam-checkpoint sam_integration/weight/sam_vit_h_4b8939.pth \
  --model-type vit_h \
  --gpu 0 \
  --splits train val test
```
## Model Weights
```
artifacts/{experiment.name}/checkpoints/latest.pth
artifacts/{experiment.name}/checkpoints/best.pth
artifacts/{experiment.name}/checkpoints/runs/{run_id}/
```
## Train
```
python train.py --config configs/uieb.yaml --gpu 0
During training, experiment files will be saved under artifacts/:
artifacts/
  uieb/
    checkpoints/
      best.pth
      latest.pth
      runs/
    predictions/
    logs/
```
## Test
```
python test.py \
  --config configs/uieb.yaml \
  --checkpoint artifacts/uieb/checkpoints/best.pth \
  --split test \
  --gpu 0 \
  --save-images
```
## Common Arguments
```
--config          YAML configuration file
--checkpoint      Specify model weights; this overrides evaluation.checkpoint in the YAML file
--split           val or test
--gpu             GPU index
--device          Specify cpu / cuda / cuda:0
--save-images     Save enhanced results and input-output-target comparison images
--self-ensemble   Use flip/transpose test-time self-ensemble
--soft-route      Use soft routing
--output-branch   Use a fixed output branch, for example 1
--route-output    Use the model routing output
```
For images without ground truth, use no_ref_test.py.
## Notes
```
* It is recommended to explicitly pass --config when running train.py, for example: python train.py --config configs/uieb.yaml.
* When using the multimodal model, prepare the CLIP and LLM model files under configs/models/.
* When using SAM masks, make sure each input image has a corresponding .npy mask with the same filename stem. If no mask is found, an all-one mask will be used by default.
* If evaluation.checkpoint in the YAML file does not exist, specify the checkpoint manually with --checkpoint.
* The default input size is 256x256, which can be changed via dataset.image_size in the YAML file.
```
## Citation
If this project is helpful for your research, please cite this repository or the related paper.
