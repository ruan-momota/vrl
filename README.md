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
| SlowFast R50 8x8 x SSV2 C50 | Complete | `outputs/runs/ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe/` |
| SlowFast R50 8x8 x UCF101 C50 | Complete | `outputs/runs/ucf101-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe/` |
| DINOv2 frame-mean x SSV2 C50 | Complete | `outputs/runs/ssv2-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe/` |
| DINOv2 frame-mean x UCF101 C50 | Complete | `outputs/runs/ucf101-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe/` |
| VideoMAE x Diving48 C32 | Complete | `outputs/runs/diving48-c32-train50-heldout15-videomae-base-frozen-linear-probe/` |
| SlowFast R50 8x8 x Diving48 C32 | Complete | `outputs/runs/diving48-c32-train50-heldout15-slowfast-r50-8x8-frozen-linear-probe/` |
| DINOv2 frame-mean x Diving48 C32 | Complete | `outputs/runs/diving48-c32-train50-heldout15-dinov2-base-frame-mean-frozen-linear-probe/` |

Matrix and appearance-extension verification, 2026-07-18:

- the primary `3 models x 3 datasets` matrix is complete;
- all nine cells contain 2 original artifacts and 14 held-out perturbation
  artifacts; the six RGB-quantization/solarization artifacts are complete in
  every cell;
- all artifacts passed paired alignment checks; quality audits report 0 failed
  samples in every cell;
- the train-only pixel audit passed for both 16- and 32-frame profiles and froze
  RGB levels at 16/8/4 and solarization thresholds at 192/128/64;
- the matrix summary contains 9 baselines, 126 perturbation rows, and 144
  artifact-quality rows;
- matrix summary Markdown files in `outputs/reports/diving48_3x3/` are now written in English;
- current lightweight verification passed: `123 passed`.

Next:

- review the completed course report and figures; no extraction or evaluation
  remains for the current matrix.

## Data

Completed cells use balanced controlled subsets:

| Dataset | Train | Held-out | Notes |
| --- | ---: | ---: | --- |
| SSV2 C50 | 5,000 videos | 1,500 videos | 100/30 per class |
| UCF101 C50 | 5,000 videos | 1,500 videos | 100/30 per class; UCF101 `test` is reported as held-out |
| Diving48 C32 | 1,600 videos | 480 videos | 50/15 per class; Diving48 `test` is reported as held-out |

UCF101 and Diving48 subset manifests are stored in:

- `data/ucf101/subsets/c50_train100_heldout30/`
- `data/diving48/subsets/c32_train50_heldout15/`

- `label_mapping.json`
- `train.jsonl`
- `heldout.jsonl`
- `selected_samples.jsonl`
- `summary.json`
- `decode_failures.jsonl`

The UCF101 decode audit attempted 6,500 videos with 0 failures. The Diving48
decode audit attempted 2,080 videos with 0 failures.

Raw videos and large embedding/prediction artifacts are not meant to be
committed.

## Experimental Protocol

For each completed cell:

- encoder: frozen `MCG-NJU/videomae-base`, PyTorchVideo SlowFast R50 `8x8`,
  or DINOv2 frame-mean baseline;
- input: deterministic center clip, model-specific preprocessing;
  VideoMAE uses 16 frames at image size 224, SlowFast uses 32 fast-pathway
  frames at image size 256 and alpha 4, and DINOv2 encodes 16 frames
  independently at image size 224 before averaging frame CLS embeddings;
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

## Results Snapshot

Primary label-related metric: linear-probe accuracy drop from original held-out
to perturbed held-out. KNN is auxiliary.

| Cell | Original LP acc. | Original KNN k=5 acc. | Selected fixed-mid LP drops |
| --- | ---: | ---: | --- |
| VideoMAE x SSV2 | 0.2460 | 0.0993 | shuffle 0.1800; quant. 0.1007; solar. 0.0507 |
| VideoMAE x UCF101 | 0.8533 | 0.8360 | shuffle 0.2493; quant. 0.3413; solar. 0.2540 |
| SlowFast x SSV2 | 0.3393 | 0.2033 | shuffle 0.2053; quant. 0.0920; solar. 0.0627 |
| SlowFast x UCF101 | 0.9940 | 0.9933 | shuffle 0.0460; quant. 0.0387; solar. 0.0753 |
| DINOv2 frame-mean x SSV2 | 0.2973 | 0.1940 | shuffle 0.0000; quant. 0.0373; solar. 0.0133 |
| DINOv2 frame-mean x UCF101 | 0.9900 | 0.9773 | shuffle 0.0000; quant. 0.0113; solar. 0.0173 |
| VideoMAE x Diving48 | 0.0729 | 0.0375 | shuffle 0.0354; quant. 0.0063; solar. 0.0083 |
| SlowFast x Diving48 | 0.0938 | 0.0729 | shuffle 0.0229; quant. 0.0229; solar. 0.0021 |
| DINOv2 frame-mean x Diving48 | 0.0958 | 0.0896 | shuffle 0.0000; quant. 0.0188; solar. 0.0083 |

RGB quantization and solarization produce substantially larger representation
and label effects than the original color control in most SSV2/UCF101 cells.
Their pixel strength is monotonic, although LP-drop curves need not be strictly
monotonic because they measure a downstream decision boundary. DINOv2
`temporal_shuffle` is a sanity check:
frame-mean embeddings are effectively invariant to frame order. Diving48
baselines are low across all three models, which is treated as a fine-grained,
small-sample model-dataset interaction rather than a reason to change the
frozen subset.

Main reports:

- `outputs/runs/ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/ucf101-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/ssv2-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/ucf101-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/diving48-c32-train50-heldout15-videomae-base-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/diving48-c32-train50-heldout15-slowfast-r50-8x8-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- `outputs/runs/diving48-c32-train50-heldout15-dinov2-base-frame-mean-frozen-linear-probe/reports/linear_probe_sensitivity_report.md`
- English 3x2 summary: `outputs/reports/dinov2_3x2/dinov2_3x2_summary.md`
- English 3x3 summary: `outputs/reports/diving48_3x3/diving48_3x3_summary.md`

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

Build or refresh the Diving48 subset index and decode audit:

```bash
uv run python -m src.data.diving48_index --decode-audit
```

Run configs are grouped by cell:

- `configs/runs/ssv2_videomae_linear_probe/`
- `configs/runs/ucf101_videomae_linear_probe/`
- `configs/runs/ssv2_slowfast_linear_probe/`
- `configs/runs/ucf101_slowfast_linear_probe/`
- `configs/runs/ssv2_dinov2_linear_probe/`
- `configs/runs/ucf101_dinov2_linear_probe/`
- `configs/runs/diving48_videomae_linear_probe/`
- `configs/runs/diving48_slowfast_linear_probe/`
- `configs/runs/diving48_dinov2_linear_probe/`

Each directory has a `README.md` with smoke extraction, full extraction, and
evaluation commands. The common entry points are:

```bash
uv run python -m src.pipeline.extract --run-config <extraction-config.json>
uv run python -m src.pipeline.evaluate --config <evaluation-config.json>
```

Use a GPU compute node for full embedding extraction. Evaluation reads existing
artifacts and writes run-scoped metrics, reports, and plots.
