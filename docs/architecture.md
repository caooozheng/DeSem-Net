# Architecture

## Design Principles

- One module, one responsibility.
- Config-first experiments.
- Baseline fidelity first, then extensibility.
- No training logic inside model definitions.

## Main Components

- `clipuie.data.datasets.PairedImageDataset`
- `clipuie.models.architectures.ClipUIENet`
- `clipuie.losses.CompositeReconstructionLoss`
- `clipuie.engine.ClipUIETrainer`
- `clipuie.engine.Evaluator`
