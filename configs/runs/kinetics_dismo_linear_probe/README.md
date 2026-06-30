# Kinetics × DisMo frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`kinetics-c20-train16-heldout4-dismo-motion-extractor-large-frozen-linear-probe`.
They produce one original train artifact, one original held-out artifact, and
the held-out perturbation artifacts under the same `outputs/runs/{run_id}/`
directory.

This cell ports the Kinetics + DisMo half of the `kinetics-videomae-dismo`
notebook into the repo pipeline. DisMo (`CompVis/DisMo` `motion_extractor_large`)
is a pure representation model with no classification head, so only the frozen
linear-probe and cosine sensitivity metrics are reported — there are no
backbone logits. Frames are resized to 256×256 and scaled to `[-1, 1]` (THWC),
mean-pooled over the sliding-window motion features, matching the notebook's
`DisMoBackbone`.

It reuses the same predefined, deterministic perturbations as the other cells
(`src/video/perturbations.py`). This is a minimal cell shipping the two
fixed-mid probes (`temporal_shuffle` for motion, `spatial_blur` for
appearance); `freeze_tail` and `color_transform` strength curves can be added
later to complete the eight-perturbation matrix.

The Kinetics subset index is shared with the VideoMAE cell — build it once (see
`configs/runs/kinetics_videomae_linear_probe/README.md`).

## Extract and evaluate

DisMo is loaded from `torch.hub` and needs network access on first run
(`--local-files-only` is not supported). Run a smoke extraction first:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c20_train_original.json \
  --limit 1

uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c20_train_original.json
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c20_heldout_original.json
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c20_heldout_temporal_shuffle_mid.json
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c20_heldout_spatial_blur_mid.json
```

Then evaluate only the complete run:

```bash
uv run python -m src.pipeline.evaluate \
  --config configs/runs/kinetics_dismo_linear_probe/kinetics_dismo_c20_linear_probe_evaluation.json
```

Use a GPU compute node for full embedding extraction.
