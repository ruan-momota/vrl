# VRL

Motion/appearance sensitivity experiments for frozen pretrained video models.
The project is intentionally scoped around a small course experiment matrix, not
a general video research platform.

## Question

The experiments ask how much frozen video representations change when held-out
clips are perturbed along two broad information sources:

- motion / temporal information: frame order, temporal continuity, action
  progression;
- appearance / spatial information: color, texture, objects, people, and scene
  context.

Results are interpreted at the level of `model x dataset x intervention`.
Perturbation sensitivity is not treated as proof of human-like action
understanding.

## Current Status

Completed:

| Cell | Status | Run directory |
| --- | --- | --- |
| VideoMAE x SSV2 C50 | Complete | `outputs/runs/ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe/` |
| VideoMAE x UCF101 C50 | Complete | `outputs/runs/ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe/` |

Next:

- add SlowFast `4x16 R50`;
- run SlowFast x SSV2 and SlowFast x UCF101 with the same perturbation and
  evaluation protocol.

## Data

Both completed cells use 50 classes with balanced splits:

| Dataset | Train | Held-out | Notes |
| --- | ---: | ---: | --- |
| SSV2 C50 | 5,000 videos | 1,500 videos | 100/30 per class |
| UCF101 C50 | 5,000 videos | 1,500 videos | 100/30 per class; UCF101 `test` is reported as held-out |

UCF101 subset manifests are stored in
`data/ucf101/subsets/c50_train100_heldout30/`:

- `label_mapping.json`
- `train.jsonl`
- `heldout.jsonl`
- `selected_samples.jsonl`
- `summary.json`
- `decode_failures.jsonl`

The UCF101 decode audit attempted 6,500 videos with 0 failures.

Raw videos and large embedding/prediction artifacts are not meant to be
committed.

## Experimental Protocol

For each completed cell:

- encoder: frozen `MCG-NJU/videomae-base`;
- input: deterministic 16-frame center clip, image size 224;
- train artifact: original train embeddings only;
- held-out artifacts: original held-out plus eight perturbations;
- classifier: train-only frozen linear probe with stratified train/probe-val
  split for L2 selection, then full-train refit;
- statistics: video-level paired bootstrap with 1,000 resamples and fixed seed;
- diagnostic: auxiliary cosine KNN with `k=5`.

Held-out perturbations:

| Group | Perturbation | Variants |
| --- | --- | --- |
| motion | `temporal_shuffle` | fixed mid |
| motion | `freeze_tail` | low, mid, high |
| appearance | `color_transform` | low, mid, high |
| appearance | `spatial_blur` | fixed mid |

Original and perturbed held-out artifacts must share video IDs, labels, sample
order, and sampled frame indices.

## Results Snapshot

Primary label-related metric: linear-probe accuracy drop from original held-out
to perturbed held-out. KNN is auxiliary.

| Cell | Original LP acc. | Original KNN k=5 acc. | Largest fixed-mid drops |
| --- | ---: | ---: | --- |
| VideoMAE x SSV2 | 0.2507 | 0.0993 | `temporal_shuffle`: 0.1767; `spatial_blur`: 0.0273 |
| VideoMAE x UCF101 | 0.8533 | 0.8360 | `spatial_blur`: 0.2920; `temporal_shuffle`: 0.2493 |

Both `freeze_tail` and `color_transform` show increasing effects from low to
high strength in the completed runs. The two VideoMAE cells can support a
dataset-interaction discussion, but they are not cross-model evidence.

Main reports:

- `outputs/runs/ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`

## Repository Layout

```text
src/
  data/         normalized indexes and dataset adapters
  video/        decoding, sampling, perturbations
  models/       encoder adapters
  pipeline/     run-scoped extraction and evaluation entry points
  evaluation/   sensitivity, linear probe, bootstrap, KNN, reporting
configs/runs/   per-artifact extraction configs and evaluation configs
data/            tracked subset manifests; raw videos are local data
outputs/runs/    tracked manifests/reports/plots; large artifacts are ignored
```

## Reproduction

Install dependencies and run tests:

```bash
uv sync --dev
uv run pytest
```

Build or refresh the UCF101 subset index and decode audit:

```bash
uv run python -m src.data.ucf101_index --decode-audit
```

Run configs are grouped by cell:

- `configs/runs/ssv2_videomae_linear_probe/`
- `configs/runs/ucf101_videomae_linear_probe/`

Each directory has a `README.md` with smoke extraction, full extraction, and
evaluation commands. The common entry points are:

```bash
uv run python -m src.pipeline.extract --run-config <extraction-config.json>
uv run python -m src.pipeline.evaluate --config <evaluation-config.json>
```

Use a GPU compute node for full embedding extraction. Evaluation reads existing
artifacts and writes run-scoped metrics, reports, and plots.
