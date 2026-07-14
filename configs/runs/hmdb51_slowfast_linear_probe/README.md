# HMDB51 × SlowFast R50 8x8 frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`hmdb51-full-split1-slowfast-r50-8x8-frozen-linear-probe`. They produce one
original train artifact, one original held-out artifact, and eight
held-out perturbation artifacts under the same `outputs/runs/{run_id}/`
directory.

This cell mirrors `hmdb51_videomae_linear_probe`/`hmdb51_vjepa_linear_probe`
(same `full_split1` subset index, full 8-perturbation matrix). SlowFast
R50 8x8 (`facebookresearch/pytorchvideo:slowfast_r50`, loaded via
`torch.hub`) uses 32 frames at 256x256 with `slowfast_alpha=4` — its own
native input spec, already used unmodified from the shared-ancestor commit
for the SSV2/UCF101/Diving48 SlowFast cells. Picked as the next cell to
fill in because it needs no new dataset fetch (HMDB51 is already local)
and no new model integration code (already registered and proven on the
colleague's branch's cells).

`batch_size=8`, `num_workers=8` (with `ulimit -n 65536` on the compute
node — see `scripts/run_kinetics_extraction_pipeline.sh`, dataset-agnostic
despite the name).

## Extract and evaluate

Run smoke extraction on a compute node before full extraction:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_smoke_train.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_smoke_heldout_original.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_smoke_temporal_shuffle_mid.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_smoke_color_mid.json \
  --limit 1
```

After smoke checks pass, run the full extraction:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_train_original.json

uv run python -m src.pipeline.extract \
  --run-config configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_original.json

for config in \
  configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_temporal_shuffle_mid.json \
  configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_freeze_tail_low.json \
  configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_freeze_tail_mid.json \
  configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_freeze_tail_high.json \
  configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_color_low.json \
  configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_color_mid.json \
  configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_color_high.json \
  configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_heldout_spatial_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done

uv run python -m src.pipeline.evaluate \
  --config configs/runs/hmdb51_slowfast_linear_probe/hmdb51_slowfast_full_split1_linear_probe_evaluation.json
```

## Combining with the rest of the matrix

Add a `Cell(dataset="HMDB51", ..., model="SlowFast R50 8x8",
run_id="hmdb51-full-split1-slowfast-r50-8x8-frozen-linear-probe")` entry
to `scripts/build_full_matrix_summary.py` once a full (non-smoke) run
completes.
