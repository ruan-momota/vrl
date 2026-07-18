# UCF101 x DINOv2 frozen-linear-probe sensitivity run

This directory contains the UCF101 C50 configs for the frozen DINOv2 frame-mean
baseline. The formal run ID is
`ucf101-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe`.

The adapter loads `facebook/dinov2-base`, encodes 16 deterministic center-clip
frames independently, and averages frame CLS embeddings into one clip embedding.
DINOv2 is a frame-wise image baseline, not a temporal video encoder.

Smoke commands:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_smoke_train.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_smoke_heldout_original.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_smoke_temporal_shuffle_mid.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_smoke_color_mid.json \
  --limit 1
```

Full extraction:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_train_original.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_original.json

for config in \
  configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_temporal_shuffle_mid.json \
  configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_freeze_tail_low.json \
  configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_freeze_tail_mid.json \
  configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_freeze_tail_high.json \
  configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_color_low.json \
  configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_color_mid.json \
  configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_color_high.json \
  configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done

uv run python -m src.pipeline.evaluate \
  --config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_linear_probe_evaluation.json
```

## RGB quantization and solarization

Run these only after the train-only pixel audit freezes the strengths:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_rgb_quantization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_rgb_quantization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_rgb_quantization_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_solarization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_solarization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/ucf101_dinov2_linear_probe/ucf101_dinov2_c50_heldout_solarization_high.json
```
