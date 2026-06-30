# Kinetics × VideoMAE frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`kinetics-c20-train16-heldout4-videomae-base-frozen-linear-probe`. They produce
one original train artifact, one original held-out artifact, and the held-out
perturbation artifacts under the same `outputs/runs/{run_id}/` directory.

This cell ports the Kinetics + VideoMAE half of the
`kinetics-videomae-dismo` notebook into the repo pipeline. It reuses the same
predefined, deterministic perturbations as the SSV2/UCF101 cells (defined in
`src/video/perturbations.py`). This is a minimal cell: it ships the two
fixed-mid probes (`temporal_shuffle` for motion, `spatial_blur` for
appearance). The `freeze_tail` and `color_transform` strength curves can be
added later to match the full eight-perturbation matrix.

The perturbation parameters match the other cells: `spatial_blur` uses a fixed
5×5 separable box kernel and `temporal_shuffle` uses deterministic seed 42.

## Build the subset index

Kinetics is indexed by parent-folder class name (the same convention as the
notebook's `index_videos`). Point the indexer at a `<class>/<video>` or
`<split>/<class>/<video>` tree:

```bash
uv run python -m src.data.kinetics_index \
  --video-root /path/to/kinetics400_subset/train \
  --output-dir data/kinetics/subsets/c20_train16_heldout4 \
  --subset-id c20_train16_heldout4 \
  --max-classes 20 --train-per-class 16 --heldout-per-class 4 --seed 42
```

## Extract and evaluate

Run a smoke extraction first, then the full train/held-out artifacts:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c20_train_original.json \
  --limit 1

uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c20_train_original.json
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c20_heldout_original.json
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c20_heldout_temporal_shuffle_mid.json
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c20_heldout_spatial_blur_mid.json
```

Then evaluate only the complete run:

```bash
uv run python -m src.pipeline.evaluate \
  --config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c20_linear_probe_evaluation.json
```

Use a GPU compute node for full embedding extraction.
