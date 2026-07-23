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

The full **5 models x 5 datasets = 25 cell** matrix is complete.

Models: VideoMAE, SlowFast R50 8x8, DINOv2 frame-mean, V-JEPA2, DisMo.
Datasets: SSV2, UCF101, Diving48, HMDB51, Kinetics.

Every cell has the full 14-perturbation matrix (temporal shuffle; freeze tail
low/mid/high; color transform low/mid/high; spatial blur; RGB quantization
low/mid/high; solarization low/mid/high), a frozen linear probe, and an
auxiliary cosine-KNN baseline. Quality audits report 0 failed samples across
all 25 cells.

Known comparability limitation: models sample different numbers of native
frames (VideoMAE/DINOv2/DisMo 16, SlowFast 32, V-JEPA2 64), which by default
means a lower-`num_frames` model implicitly covers a shorter slice of
real-world video time -- confounding cross-model motion-perturbation-magnitude
comparisons. An opt-in `window_frames` field (`src/video/io.py`) normalizes
this by spreading a model's frames evenly across a shared reference window
instead of packing them densely; it is applied to the SSV2 x DisMo and
SSV2 x V-JEPA2 cells (making the SSV2 row the one fully frame-count-consistent
row in the matrix) and to UCF101 x V-JEPA2. Every other cell/row still carries
the original confound and should be read as a documented limitation, not
assumed comparable -- the per-cell grid figure labels each column with its
native frame count as a standing reminder.

Next: the course deliverable is the slide presentation (`VRL.pptx`). A
possible follow-up (not yet done) is benchmarking whether `torch.compile`
gives a measurable speedup for the heavier backbones (DisMo, V-JEPA2) beyond
the TF32/bf16-autocast speedups already enabled for extraction on CUDA.

## Data

| Dataset | Train | Held-out | Notes |
| --- | ---: | ---: | --- |
| SSV2 C50 | 5,000 videos | 1,500 videos | 100/30 per class |
| UCF101 C50 | 5,000 videos | 1,500 videos | 100/30 per class; UCF101 `test` is reported as held-out |
| Diving48 C32 | 1,600 videos | 480 videos | 50/15 per class; Diving48 `test` is reported as held-out |
| HMDB51 full_split1 | 3,551 videos | 1,524 videos | official split 1, all 51 classes |
| Kinetics C50 | 5,000 videos | 1,500 videos | 100/30 per class, subset of Kinetics-400 |

Subset manifests (`label_mapping.json`, `train.jsonl`, `heldout.jsonl`,
`selected_samples.jsonl`, `summary.json`, `decode_failures.jsonl`) are stored
under `data/<dataset>/subsets/<subset_id>/` and tracked in git via a
`.gitignore` punch-through, e.g. `data/ucf101/subsets/c50_train100_heldout30/`.

**Known gap:** `data/kinetics/` has no such punch-through in `.gitignore`
(every other dataset does), so the Kinetics subset manifests were never
committed -- worth fixing separately, since it means Kinetics cells cannot
currently be reproduced from a fresh clone the way the other four datasets
can.

Raw videos and large embedding/prediction artifacts are not meant to be
committed.

## Experimental Protocol

For each completed cell:

- encoder: frozen `MCG-NJU/videomae-base` (VideoMAE), PyTorchVideo SlowFast R50
  `8x8`, `facebook/dinov2-base` frame-mean baseline (DINOv2), Meta's
  `facebook/vjepa2-vitl-fpc64-256` (V-JEPA2), or the `CompVis/DisMo`
  `motion_extractor_large` motion encoder (DisMo, via `torch.hub`);
- input: deterministic center clip, model-specific preprocessing --
  VideoMAE/DisMo use 16 frames (224px / 256px), SlowFast uses 32 fast-pathway
  frames at 256px and alpha 4, DINOv2 encodes 16 frames independently at 224px
  before averaging frame CLS embeddings, and V-JEPA2 uses 64 frames at 256px;
- train artifact: original train embeddings only;
- held-out artifacts: original held-out plus fourteen perturbations;
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
| appearance | `rgb_quantization` | low, mid, high; RGB levels 16, 8, 4 |
| appearance | `solarization` | low, mid, high; thresholds 192, 128, 64 |

Original and perturbed held-out artifacts must share video IDs, labels, sample
order, and sampled frame indices.

Extraction on CUDA uses TF32 matmul and bf16 autocast by default (see
`src/pipeline/extract.py` / `extraction.py`), plus `torch.compile` for
DisMo/V-JEPA2 specifically (measured 1.18-1.37x on an A40/RTX 3090). Set
`VRL_FORCE_FP32=1` to disable all three for a run that must stay numerically
consistent with an existing fp32 artifact (e.g. backfilling new perturbations
onto a cell whose original embedding was extracted before these were added).

## Results Snapshot

Primary label-related metric: linear-probe accuracy drop / correct-to-incorrect
flip rate from original held-out to perturbed held-out. KNN is auxiliary.
Full per-cell numbers (baselines, fixed-mid drops, and a motion-vs-appearance
bias decomposition that corrects for each model's own accuracy ceiling and
embedding-dimensionality scale) live in
`outputs/reports/full_matrix/full_matrix_summary.md` -- the 25-cell table is
too large to duplicate here.

Headline pattern (see that report for the full reasoning): DisMo shows the
strongest motion bias of any model (mean behavioral bias +0.19 across its 5
datasets), SlowFast and VideoMAE are moderately motion-biased, V-JEPA2 is
close to neutral, and DINOv2 frame-mean is exactly appearance-only for
`temporal_shuffle` by construction (mean-pooling is invariant to frame order,
not evidence about its frame-level features). Representational shift (cosine
distance) and behavioral/label sensitivity (flip rate) disagree in sign for
several cells (e.g. V-JEPA2 x HMDB51/Kinetics/SSV2) -- a bigger embedding move
does not always mean a more sensitive downstream decision.

Main reports: every cell has its own
`outputs/runs/<run_id>/reports/linear_probe_sensitivity_report.md`; run IDs
are listed in `outputs/reports/full_matrix/full_matrix_summary.md`.

## Repository Layout

```text
src/
  data/         normalized indexes and dataset adapters
  video/        decoding, sampling, perturbations
  models/       encoder adapters
  pipeline/     run-scoped extraction and evaluation entry points
  evaluation/   sensitivity, linear probe, bootstrap, KNN, reporting
scripts/         matrix aggregation, plotting, and compute-orchestration helpers
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

Build or refresh a dataset's subset index and decode audit, e.g.:

```bash
uv run python -m src.data.ucf101_index --decode-audit
uv run python -m src.data.diving48_index --decode-audit
```

Run configs are grouped by cell, one directory per `<dataset>_<model>_linear_probe/`
under `configs/runs/` (25 total, e.g. `ssv2_videomae_linear_probe/`,
`kinetics_dismo_linear_probe/`). Each directory has extraction configs (train
original, heldout original, 14 perturbations), an evaluation config, smoke
variants, and its own `README.md`. The common entry points are:

```bash
uv run python -m src.pipeline.extract --run-config <extraction-config.json>
uv run python -m src.pipeline.evaluate --config <evaluation-config.json>
```

Use a GPU compute node for full embedding extraction (DisMo/V-JEPA2 need
24-48GB VRAM; the lighter models fit in 8GB). Evaluation reads existing
artifacts and writes run-scoped metrics, reports, and plots.

After adding or backfilling a cell, regenerate the cross-cell aggregate view:

```bash
uv run python scripts/build_full_matrix_summary.py
uv run python scripts/render_full_matrix_pngs.py
```
