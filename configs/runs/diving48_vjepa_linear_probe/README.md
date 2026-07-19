# Diving48 x V-JEPA2 frozen-linear-probe sensitivity run

This directory contains the Diving48 C32 train50 / heldout15 configs for the frozen V-JEPA2 ViT-L (fpc64, 256px).
The formal run ID is `diving48-c32-train50-heldout15-vjepa2-vitl-fpc64-256-frozen-linear-probe`.

Includes the full 14-artifact perturbation set (temporal_shuffle, freeze_tail x3, color_transform x3,
spatial_blur, rgb_quantization x3, solarization x3) — this cell was built after the perturbation matrix
was expanded, so unlike the earlier HMDB51/Kinetics cells it does not need a separate catch-up pass.

Smoke commands:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_smoke_train.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_smoke_heldout_original.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_smoke_temporal_shuffle_mid.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_smoke_color_mid.json --limit 1
```

Full extraction and evaluation:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_train_original.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_original.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_temporal_shuffle_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_freeze_tail_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_freeze_tail_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_freeze_tail_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_color_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_color_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_color_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_blur_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_rgb_quantization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_rgb_quantization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_rgb_quantization_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_solarization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_solarization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_heldout_solarization_high.json
uv run python -m src.pipeline.evaluate --config configs/runs/diving48_vjepa_linear_probe/diving48_vjepa_c32_linear_probe_evaluation.json
```
