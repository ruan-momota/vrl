# SSV2 × VideoMAE frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe`. They produce one original
train artifact, one original held-out artifact, and eight held-out perturbation
artifacts under the same `outputs/runs/{run_id}/` directory.

The strength parameters are fixed before extraction:

| Intervention | low | mid | high |
| --- | ---: | ---: | ---: |
| `freeze_tail` start fraction | 0.75 | 0.50 | 0.25 |
| `color_transform` strength | 0.15 | 0.35 | 0.60 |

Smaller `freeze_tail` start fractions preserve less motion and are therefore
stronger temporal disruptions. `spatial_blur` uses a fixed 5×5 separable box
kernel; `temporal_shuffle` uses deterministic seed 42.

Run each extraction only after a smoke run has succeeded:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_videomae_linear_probe/ssv2_videomae_c50_smoke_train.json \
  --limit 1 --local-files-only

# Optional integration checks for the new appearance probes
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_videomae_linear_probe/ssv2_videomae_c50_smoke_heldout_original.json \
  --limit 1 --local-files-only
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_videomae_linear_probe/ssv2_videomae_c50_smoke_color_mid.json \
  --limit 1 --local-files-only
uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_videomae_linear_probe/ssv2_videomae_c50_smoke_blur_mid.json \
  --limit 1 --local-files-only

uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_videomae_linear_probe/ssv2_videomae_c50_train_original.json

uv run python -m src.pipeline.extract \
  --run-config configs/runs/ssv2_videomae_linear_probe/ssv2_videomae_c50_heldout_original.json
```

Run the eight held-out perturbation configs next, then evaluate only the
complete run:

```bash
uv run python -m src.pipeline.evaluate \
  --config configs/runs/ssv2_videomae_linear_probe/ssv2_videomae_c50_linear_probe_evaluation.json
```

`legacy_anchor` artifacts and the legacy perturbation matrix must not be mixed
with this run.
