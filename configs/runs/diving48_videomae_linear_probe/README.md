# Diving48 x VideoMAE frozen-linear-probe sensitivity run

This directory contains the Diving48 C32 train50 / heldout15 configs for the frozen VideoMAE base encoder.
The formal run ID is `diving48-c32-train50-heldout15-videomae-base-frozen-linear-probe`.

Diving48 is used here as a class-balanced fine-grained motion / pose contrast, not as a full Diving48 benchmark.

Smoke commands:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_smoke_train.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_smoke_heldout_original.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_smoke_temporal_shuffle_mid.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_smoke_color_mid.json --limit 1
```

Full extraction and evaluation:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_train_original.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_original.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_temporal_shuffle_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_freeze_tail_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_freeze_tail_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_freeze_tail_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_color_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_color_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_color_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_blur_mid.json
uv run python -m src.pipeline.evaluate --config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_linear_probe_evaluation.json
```

## RGB quantization and solarization

Run these only after the train-only pixel audit freezes the strengths:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_rgb_quantization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_rgb_quantization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_rgb_quantization_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_solarization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_solarization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/diving48_videomae_linear_probe/diving48_videomae_c32_heldout_solarization_high.json
```
