# Kinetics × V-JEPA 2 frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`kinetics-c50-train100-heldout30-vjepa2-vitl-fpc64-256-frozen-linear-probe`.
They produce one original train artifact, one original held-out artifact,
and eight held-out perturbation artifacts under the same
`outputs/runs/{run_id}/` directory. This mirrors
`configs/runs/hmdb51_vjepa_linear_probe/` — same model, same perturbation
matrix, same extraction/evaluation code — applied to the Kinetics subset
instead of HMDB51.

## Subset scope

The Kinetics subset index is shared with the VideoMAE/DisMo cells — build it
once (see `configs/runs/kinetics_videomae_linear_probe/README.md`). It's
`c50_train100_heldout30` (50 classes, 100 train / 30 held-out per class),
matching the UCF101/SSV2 hand-picked-subset scope.

The perturbation strengths match every other cell's SSV2 × VideoMAE anchor:

| Intervention | low | mid | high |
| --- | ---: | ---: | ---: |
| `freeze_tail` start fraction | 0.75 | 0.50 | 0.25 |
| `color_transform` strength | 0.15 | 0.35 | 0.60 |

`spatial_blur` uses a fixed 5x5 separable box kernel; `temporal_shuffle` uses
deterministic seed 42.

V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`) uses 64 frames at 256x256, versus
VideoMAE's 16 frames at 224x224 and DisMo's 16 frames at 256x256 — each
model's own native input spec, matching how every other cell in this repo
also lets each model use its own frame count/size. Measured throughput on a
single RTX A40 (HMDB51 run): ~1.5-1.9 sec/clip at batch size 1. This cell's
clip-pass count (5,000 train + 1,500 held-out original + 1,500 × 8 held-out
perturbations = 18,500 clip-passes) is comparable to the HMDB51 run's
~17,267, so expect a similar ballpark of GPU-hours (roughly 7.5-9.5).

## Extract and evaluate

Run smoke extraction on a compute node before full extraction:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_smoke_train.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_smoke_heldout_original.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_smoke_temporal_shuffle_mid.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_smoke_color_mid.json \
  --limit 1
```

After smoke checks pass (and confirm `torch.cuda.is_available()` before
trusting the timing), run the full extraction:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_train_original.json

uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_original.json

for config in \
  configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_temporal_shuffle_mid.json \
  configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_freeze_tail_low.json \
  configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_freeze_tail_mid.json \
  configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_freeze_tail_high.json \
  configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_color_low.json \
  configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_color_mid.json \
  configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_color_high.json \
  configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_heldout_spatial_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done

uv run python -m src.pipeline.evaluate \
  --config configs/runs/kinetics_vjepa_linear_probe/kinetics_vjepa_c50_linear_probe_evaluation.json
```

## Combining with the rest of the matrix

Same aggregation pattern as `hmdb51_vjepa_linear_probe`: once a full
(non-smoke) run completes, add a `Cell(dataset="kinetics", ...,
run_id="kinetics-c50-train100-heldout30-vjepa2-vitl-fpc64-256-frozen-linear-probe")`
entry to the summary-building script that aggregates
`outputs/runs/<run_id>/reports/*` across cells.
