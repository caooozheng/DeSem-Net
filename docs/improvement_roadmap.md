# Improvement Roadmap

## Immediate Fixes Over the Original Project

- Add a proper validation split instead of reusing test during training.
- Report both `256x256` and original-resolution metrics.
- Remove ambiguous names like `PSNR2`; report branch metrics explicitly.
- Save experiment metadata next to checkpoints.

## Likely Research Improvement Points

- Replace soft routing with hard routing or top-1 gating and compare.
- Decouple branch supervision from final-output supervision more cleanly.
- Add branch diversity regularization.
- Replace the legacy VGG perceptual loss with a lighter frozen encoder.
