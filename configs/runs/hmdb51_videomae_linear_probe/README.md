# HMDB51 × VideoMAE frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`hmdb51-full-split1-videomae-base-frozen-linear-probe`. They produce one
original train artifact, one original held-out artifact, and eight held-out
perturbation artifacts under the same `outputs/runs/{run_id}/` directory.

This cell mirrors `hmdb51_vjepa_linear_probe`, using HMDB51's full official
split 1 as-is (all 51 classes, 3,551 train / 1,524 heldout clips after
excluding 25 undecodable clips — see that cell's README for the dataset
scale rationale). The subset index is shared with the V-JEPA2 cell; no need
to rebuild it, it's already at `data/hmdb51/subsets/full_split1/` (raw
video also already present locally under `data/hmdb51/videos_split1/`).

VideoMAE (`MCG-NJU/videomae-base`) uses 16 frames at 224x224 — its own
native input spec, versus V-JEPA2's 64 frames at 256x256. `batch_size=8`,
`num_workers=8` (with `ulimit -n 65536` on the compute node — see
`scripts/run_kinetics_extraction_pipeline.sh`, which is dataset-agnostic
despite the name and works for any cell given `<config_dir> <prefix>
<pod_id>`).

The perturbation strengths match every other cell:

| Intervention | low | mid | high |
| --- | ---: | ---: | ---: |
| `freeze_tail` start fraction | 0.75 | 0.50 | 0.25 |
| `color_transform` strength | 0.15 | 0.35 | 0.60 |

`spatial_blur` uses a fixed 5x5 separable box kernel; `temporal_shuffle`
uses deterministic seed 42.

## Extract and evaluate

Run smoke extraction on a compute node before full extraction:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_smoke_train.json \
  --limit 1 --local-files-only
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_smoke_heldout_original.json \
  --limit 1 --local-files-only
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_smoke_temporal_shuffle_mid.json \
  --limit 1 --local-files-only
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_smoke_color_mid.json \
  --limit 1 --local-files-only
```

After smoke checks pass, run the full extraction:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_train_original.json

uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_original.json

for config in \
  configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_temporal_shuffle_mid.json \
  configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_freeze_tail_low.json \
  configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_freeze_tail_mid.json \
  configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_freeze_tail_high.json \
  configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_color_low.json \
  configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_color_mid.json \
  configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_color_high.json \
  configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_heldout_spatial_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done

uv run python -m src.pipeline.evaluate \
  --config configs/runs/hmdb51_videomae_linear_probe/hmdb51_videomae_full_split1_linear_probe_evaluation.json
```

## Combining with the rest of the matrix

Add a `Cell(dataset="HMDB51", ..., model="VideoMAE",
run_id="hmdb51-full-split1-videomae-base-frozen-linear-probe")` entry to
`scripts/build_full_matrix_summary.py` once a full (non-smoke) run
completes.
