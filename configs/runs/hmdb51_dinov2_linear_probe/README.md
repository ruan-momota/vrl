# HMDB51 x DINOv2 frame-mean frozen-linear-probe sensitivity run

This directory contains the HMDB51 full split1 configs for the frozen frame-wise DINOv2 image baseline with mean pooling. Run on the LMU IfI CIP-pool SLURM cluster (`NvidiaAll` partition, RTX 2060/2060 SUPER, 8GB VRAM) rather than RunPod.
The formal run ID is `hmdb51-full-split1-dinov2-base-frame-mean-frozen-linear-probe`.

Includes the full 14-artifact perturbation set (temporal_shuffle, freeze_tail x3, color_transform x3,
spatial_blur, rgb_quantization x3, solarization x3) — this cell was built after the perturbation matrix
was expanded, so unlike the earlier HMDB51/Kinetics cells it does not need a separate catch-up pass.

Smoke commands:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_smoke_train.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_smoke_heldout_original.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_smoke_temporal_shuffle_mid.json --limit 1
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_smoke_color_mid.json --limit 1
```

Full extraction and evaluation:

```bash
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_train_original.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_original.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_temporal_shuffle_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_freeze_tail_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_freeze_tail_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_freeze_tail_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_color_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_color_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_color_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_blur_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_rgb_quantization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_rgb_quantization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_rgb_quantization_high.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_solarization_low.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_solarization_mid.json
uv run python -m src.pipeline.extract --run-config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_heldout_solarization_high.json
uv run python -m src.pipeline.evaluate --config configs/runs/hmdb51_dinov2_linear_probe/hmdb51_dinov2_full_split1_linear_probe_evaluation.json
```
