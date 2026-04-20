# ClipUIe

`ClipUIe` is an underwater image enhancement project with a cleaner engineering layout.

## Goals

- Keep a reproducible ClipUIE baseline.
- Separate data, models, losses, metrics, training, and evaluation.
- Use YAML configs instead of hard-coded paths.
- Make later model changes local and explicit.

## Layout

- `train.py` / `test.py`: normal project entry points.
- `clipuie/models/`: model architecture and reusable blocks.
- `clipuie/data/`: dataset and dataloader code.
- `clipuie/losses/`: training losses.
- `clipuie/engine/`: trainer and evaluator.
- `clipuie/utils/`: runtime helpers and metric computation.

## Quick Start

```bash
pip install -e .
python train.py
python test.py
```

## Dataset Layout

The project now expects three explicit splits:

```text
UnderwaterDatasets/UIEB-new/
  train/
    input/
    target/
  val/
    input/
    target/
  test/
    input/
    target/
```

## Notes

- The baseline config still uses the original `256x256` evaluation protocol for comparability.
- Training now validates on `val` by default, and `test` is reserved for final evaluation.
