# HMDB51 × V-JEPA 2 frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`hmdb51-full-split1-vjepa2-vitl-fpc64-256-frozen-linear-probe`. They produce one
original train artifact, one original held-out artifact, and eight held-out
perturbation artifacts under the same `outputs/runs/{run_id}/` directory.

Unlike the UCF101/SSV2 cells (a hand-picked 50-class, 100-train/30-heldout
subset), HMDB51 uses its **full official split 1** as-is: all 51 classes,
70 train / 30 test clips each, since HMDB51 doesn't have 100 train clips per
class to begin with. This is a dataset-driven scale difference, not a
protocol difference — the extraction, perturbation, linear-probe, bootstrap,
and KNN code is unmodified and shared with every other cell in this repo.

The HMDB51 subset index is expected at
`data/hmdb51/subsets/full_split1/`. It was built from the official
`testTrainMulti_7030_splits` fold 1 via `scripts/organize_hmdb51_split.py`
followed by:

```bash
uv run python -m src.data.hmdb51_index \
  --video-root data/hmdb51/videos_split1 \
  --output-dir data/hmdb51/subsets/full_split1 \
  --subset-id full_split1 \
  --probe-decode
```

25 of 6,766 raw clips (0.5%) fail to decode due to a known PyAV/legacy-AVI
metadata quirk unrelated to this pipeline; those are excluded from
`train.jsonl`/`heldout.jsonl` (final counts: 3,551 train / 1,524 heldout).

Run smoke extraction on a compute node before full extraction:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_smoke_train.json \
  --limit 1 --local-files-only

uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_smoke_heldout_original.json \
  --limit 1 --local-files-only

uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_smoke_temporal_shuffle_mid.json \
  --limit 1 --local-files-only

uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_smoke_color_mid.json \
  --limit 1 --local-files-only
```

The perturbation strengths match every other cell's SSV2 × VideoMAE anchor:

| Intervention | low | mid | high |
| --- | ---: | ---: | ---: |
| `freeze_tail` start fraction | 0.75 | 0.50 | 0.25 |
| `color_transform` strength | 0.15 | 0.35 | 0.60 |

`spatial_blur` uses a fixed 5x5 separable box kernel; `temporal_shuffle` uses
deterministic seed 42.

V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`) uses 64 frames at 256x256, versus
VideoMAE's 16 frames at 224x224 — each model's own native input spec, matching
how the SlowFast cells also use a different frame count/size from VideoMAE.
Measured throughput on a single RTX A40: ~1.5-1.9 sec/clip at batch size 1.
Full extraction (train + heldout original + 8 heldout perturbations, ~17,267
clip-passes) is estimated at roughly 7-9 GPU-hours.

After full extraction, run:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_train_original.json

uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_original.json

for config in \
  configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_temporal_shuffle_mid.json \
  configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_freeze_tail_low.json \
  configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_freeze_tail_mid.json \
  configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_freeze_tail_high.json \
  configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_color_low.json \
  configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_color_mid.json \
  configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_color_high.json \
  configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_heldout_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done

uv run python -m src.pipeline.evaluate \
  --config configs/runs/hmdb51_vjepa_linear_probe/hmdb51_vjepa_full_split1_linear_probe_evaluation.json
```

## Combining with the SSV2/UCF101 x VideoMAE/SlowFast/DINOv2 matrix

The `videomae-slowfast-ssv2-ucf101` branch aggregates cells via
`scripts/build_dinov2_3x2_summary.py`, which reads each cell's
`outputs/runs/<run_id>/reports/*` artifacts (produced by the same, unmodified
`src.evaluation` pipeline used here) into one `Cell(dataset, dataset_role,
model, checkpoint, run_id)` table. This cell's `run_id` is
`hmdb51-full-split1-vjepa2-vitl-fpc64-256-frozen-linear-probe`, ready to be
added as another `Cell` entry once a full (non-smoke) run completes.
