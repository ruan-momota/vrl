# Kinetics × VideoMAE frozen-linear-probe sensitivity run

All extraction configs in this directory share the run ID
`kinetics-c50-train100-heldout30-videomae-base-frozen-linear-probe`. They
produce one original train artifact, one original held-out artifact, and
eight held-out perturbation artifacts under the same
`outputs/runs/{run_id}/` directory.

This cell ports the Kinetics + VideoMAE half of the
`kinetics-videomae-dismo` notebook into the repo pipeline, using the same
predefined, deterministic perturbations as the SSV2/UCF101/HMDB51 cells
(defined in `src/video/perturbations.py`): `freeze_tail` and
`color_transform` strength curves plus the fixed-mid `spatial_blur`/
`temporal_shuffle` probes — the full eight-perturbation matrix, matching
every other dataset cell.

The perturbation parameters match the other cells: `spatial_blur` uses a
fixed 5×5 separable box kernel and `temporal_shuffle` uses deterministic
seed 42.

## Subset scope

Kinetics is indexed by parent-folder class name (the same convention as the
notebook's `index_videos`). The subset is `c50_train100_heldout30` — 50
classes, 100 train / 30 held-out clips per class — chosen to match the
UCF101/SSV2 hand-picked-subset scope (`c50_train100_heldout30` there too),
rather than the earlier `c20_train16_heldout4` smoke-scale placeholder. This
is a scope choice, not a dataset constraint: Kinetics itself has far more
classes/clips available than HMDB51's full dataset, so "full Kinetics" was
never a realistic target for this pipeline; the goal here is comparability
with the other hand-picked subsets, not exhaustiveness.

Raw clips aren't bundled anywhere — pull them straight onto the compute
node with `scripts/fetch_kinetics_cvdf_subset.py`, which downloads only as
many of CVDF's 242 official Kinetics-400 train shards as needed to cover
the target classes (see that script's module docstring for why a partial,
adaptive download is necessary — shards are class-mixed, not sorted):

```bash
uv run python -m scripts.fetch_kinetics_cvdf_subset --plan-only  # cheap preview
uv run python -m scripts.fetch_kinetics_cvdf_subset \
  --output-dir data/kinetics/raw/c50 --work-dir /workspace/kinetics_cvdf
```

Then build the normalized index from the fetched clips, exactly like every
other dataset cell:

```bash
uv run python -m src.data.kinetics_index \
  --video-root data/kinetics/raw/c50 \
  --output-dir data/kinetics/subsets/c50_train100_heldout30 \
  --subset-id c50_train100_heldout30 \
  --max-classes 50 --train-per-class 100 --heldout-per-class 30 --seed 0
```

If this subset is later widened, only growing `--train-per-class` while
keeping `--heldout-per-class` fixed guarantees the current train/held-out
split stays a strict subset of the wider one (the indexer takes the
held-out slice first from a seeded per-class shuffle, then the train slice
right after it — moving the held-out boundary reshuffles which clips land
in which split). Any other resize should be built as a new `subset_id`
directory, same as this one was.

## Extract and evaluate

Run a smoke extraction first, then the full train/held-out artifacts:

```bash
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_smoke_train.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_smoke_heldout_original.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_smoke_temporal_shuffle_mid.json \
  --limit 1
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_smoke_color_mid.json \
  --limit 1

uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_train_original.json
uv run python -m src.pipeline.extract \
  --run-config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_original.json

for config in \
  configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_temporal_shuffle_mid.json \
  configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_freeze_tail_low.json \
  configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_freeze_tail_mid.json \
  configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_freeze_tail_high.json \
  configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_color_low.json \
  configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_color_mid.json \
  configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_color_high.json \
  configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_heldout_spatial_blur_mid.json
do
  uv run python -m src.pipeline.extract --run-config "$config"
done
```

Then evaluate only the complete run:

```bash
uv run python -m src.pipeline.evaluate \
  --config configs/runs/kinetics_videomae_linear_probe/kinetics_videomae_c50_linear_probe_evaluation.json
```

Use a GPU compute node for full embedding extraction.
