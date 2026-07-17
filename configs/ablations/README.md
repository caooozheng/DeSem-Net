# UIEB Ablation Configs

These files are isolated experiment entries for the ablation table:

| No. | Config | Baseline | TPG | RAE | MCR | RD |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `uieb_01_baseline.yaml` | yes | no | no | no | no |
| 2 | `uieb_02_tpg.yaml` | yes | yes | no | no | no |
| 3 | `uieb_03_tpg_rae.yaml` | yes | yes | yes | no | no |
| 4 | `uieb_04_tpg_rae_mcr.yaml` | yes | yes | yes | yes | no |
| 5 | `uieb_05_full_rd.yaml` | yes | yes | yes | yes | yes |

Module mapping:

- TPG: `multimodal.enabled: true`.
- RAE: `dataset.mask_dir_name: mask_sam`, `model.use_sam_mask: true`, and `model.use_dual_region_branch: true`.
- MCR: `model.num_branch: 3` with `training.route_start_epoch: 0`, matching the current UIEB training setup.
- RD: `model.use_fg_bg_decoder: true`. The UIEB base config does not enable extra output refinement heads, so these ablation configs keep them disabled to avoid changing the original UIEB model setting.

Train each row from scratch:

```powershell
python train.py --config configs/ablations/uieb_01_baseline.yaml --gpu 0
python train.py --config configs/ablations/uieb_02_tpg.yaml --gpu 0
python train.py --config configs/ablations/uieb_03_tpg_rae.yaml --gpu 0
python train.py --config configs/ablations/uieb_04_tpg_rae_mcr.yaml --gpu 0
python train.py --config configs/ablations/uieb_05_full_rd.yaml --gpu 0
```

After training, evaluate the best checkpoint for each row:

```powershell
python test.py --config configs/ablations/uieb_01_baseline.yaml --split test --gpu 0
python test.py --config configs/ablations/uieb_02_tpg.yaml --split test --gpu 0
python test.py --config configs/ablations/uieb_03_tpg_rae.yaml --split test --gpu 0
python test.py --config configs/ablations/uieb_04_tpg_rae_mcr.yaml --split test --gpu 0
python test.py --config configs/ablations/uieb_05_full_rd.yaml --split test --gpu 0
```

The evaluation command automatically uses the corresponding `best.pth` under `artifacts/ablations/<experiment.name>/checkpoints/` if `evaluation.checkpoint` is left as `null`.

To run the same ablation on another dataset, copy these five files and only change:

- `experiment.name`
- `dataset.train_root`
- `dataset.val_root`
- `dataset.test_root`
- optional RD strengths, if the original dataset config uses different values
