# Kinetics × DisMo frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`kinetics-c50-train100-heldout30-dismo-motion-extractor-large-frozen-linear-probe`.
They produce one original train artifact, one original held-out artifact,
and eight held-out perturbation artifacts under the same
`outputs/runs/{run_id}/` directory.

This cell ports the Kinetics + DisMo half of the `kinetics-videomae-dismo`
notebook into the repo pipeline. DisMo (`CompVis/DisMo` `motion_extractor_large`)
is a pure representation model with no classification head, so only the frozen
linear-probe and cosine sensitivity metrics are reported — there are no
backbone logits. Frames are resized to 256×256 and scaled to `[-1, 1]` (THWC),
mean-pooled over the sliding-window motion features, matching the notebook's
`DisMoBackbone`.

It reuses the same predefined, deterministic perturbations as the other cells
(`src/video/perturbations.py`): the full eight-perturbation matrix
(`freeze_tail` low/mid/high, `color_transform` low/mid/high, fixed-mid
`spatial_blur`, fixed-mid `temporal_shuffle`), matching every other dataset
cell.

## Subset scope

The Kinetics subset index is shared with the VideoMAE cell — build it once
(see `configs/runs/kinetics_videomae_linear_probe/README.md`). It's
`c50_train100_heldout30` (50 classes, 100 train / 30 held-out per class),
matching the UCF101/SSV2 hand-picked-subset scope rather than the earlier
`c20_train16_heldout4` placeholder.

## Extract and evaluate

DisMo is loaded from `torch.hub` and needs network access on first run
(`--local-files-only` is not supported). Run a smoke extraction first:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_smoke_train.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_smoke_heldout_original.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_smoke_temporal_shuffle_mid.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_smoke_color_mid.json \
  --limit 1

uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_train_original.json
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_original.json

for config in \
  configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_temporal_shuffle_mid.json \
  configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_freeze_tail_low.json \
  configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_freeze_tail_mid.json \
  configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_freeze_tail_high.json \
  configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_color_low.json \
  configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_color_mid.json \
  configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_color_high.json \
  configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_heldout_spatial_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done
```

Then evaluate only the complete run:

```bash
uv run python -m src.pipeline.evaluate \
  --config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c50_linear_probe_evaluation.json
```

Use a GPU compute node for full embedding extraction.
