# SSV2 x SlowFast frozen-linear-probe sensitivity run

This directory contains the Phase 4 SSV2 C50 configs for the frozen SlowFast
cell. The formal run ID is
`ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe`.

The minimal adapter loads `facebookresearch/pytorchvideo:slowfast_r50` through
`torch.hub` and uses a 32-frame fast pathway, alpha 4, and 256 center-crop
preprocessing. Confirm this backend on a compute node before full extraction;
do not mix these artifacts with VideoMAE artifacts.

Smoke commands:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_smoke_train.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_smoke_heldout_original.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_smoke_temporal_shuffle_mid.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_smoke_color_mid.json \
  --limit 1
```

Full extraction:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_train_original.json
uv run python -m src.pipeline.extract --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_original.json

for config in \
  configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_temporal_shuffle_mid.json \
  configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_freeze_tail_low.json \
  configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_freeze_tail_mid.json \
  configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_freeze_tail_high.json \
  configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_color_low.json \
  configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_color_mid.json \
  configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_color_high.json \
  configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done

uv run python -m src.pipeline.evaluate \
  --config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_linear_probe_evaluation.json
```

## RGB quantization and solarization

Run these only after the train-only pixel audit freezes the strengths:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_rgb_quantization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_rgb_quantization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_rgb_quantization_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_solarization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_solarization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/ssv2_slowfast_linear_probe/ssv2_slowfast_c50_heldout_solarization_high.json
```
