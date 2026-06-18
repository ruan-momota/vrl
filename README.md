# VRL

VideoMAE + Something-Something-V2 sensitivity experiment over a small local SSV2 sample.

The current experiment uses frozen `MCG-NJU/videomae-base` embeddings to compare original validation videos against deterministic motion and appearance perturbations. The main outputs are KNN baseline reports, embedding sensitivity reports, and perturbation KNN accuracy drop reports.

## Experiment Setup

| Field | Value |
| --- | --- |
| Dataset | Local Something-Something-V2 sample |
| Model | `MCG-NJU/videomae-base` |
| Model usage | frozen encoder, no training or fine-tuning |
| Input clip | 16 deterministic center frames |
| Embedding | mean pool over `last_hidden_state` |
| Embedding dim | 768 |
| KNN reference | original train embeddings |
| Evaluation split | original and perturbed validation embeddings |
| Main config | `configs/ssv2_videomae_perturbation_matrix.json` |

Generated embedding artifacts are stored under `outputs/embeddings/`; JSON metrics and reports are stored under `outputs/logs/`.

## Data Scope

| split | local videos | labeled | class count | role |
| --- | ---: | --- | ---: | --- |
| train | 100 | yes | 66 | KNN reference |
| validation | 100 | yes | 67 | evaluation |
| test | 100 | no by default | n/a | not used for current metrics |

Train / validation label overlap is limited:

| metric | value |
| --- | ---: |
| common classes | 28 |
| train-only classes | 38 |
| validation-only classes | 39 |
| validation samples with train-seen label | 46 / 100 |
| validation samples with train-unseen label | 54 / 100 |

Because 54 validation samples have labels absent from the 100-sample train reference, KNN accuracy is primarily a pipeline and trend diagnostic, not a model performance claim.

## Original KNN Baseline

Train reference: `outputs/embeddings/ssv2_train100_videomae_base_16f_mean_original.pt`

Validation query: `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_original.pt`

| metric | k | all correct / total | all accuracy | train-seen correct / total | train-seen accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| cosine | 1 | 2 / 100 | 0.0200 | 2 / 46 | 0.0435 |
| cosine | 5 | 1 / 100 | 0.0100 | 1 / 46 | 0.0217 |
| cosine | 10 | 1 / 100 | 0.0100 | 1 / 46 | 0.0217 |
| L2 | 1 | 2 / 100 | 0.0200 | 2 / 46 | 0.0435 |
| L2 | 5 | 1 / 100 | 0.0100 | 1 / 46 | 0.0217 |
| L2 | 10 | 1 / 100 | 0.0100 | 1 / 46 | 0.0217 |

Reports:

- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_original_knn_cosine.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_original_knn_l2.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_original_baseline_interpretability.json`

## Perturbations

First-round perturbations are applied only to validation videos. Train reference embeddings remain original.

| perturbation | group | default setting |
| --- | --- | --- |
| `temporal_reverse` | motion | reverse sampled frame order |
| `temporal_shuffle` | motion | deterministic per-video shuffle, seed `0` |
| `freeze_tail` | motion | `freeze_start_fraction = 0.5` |
| `single_frame` | motion | repeat center sampled frame |
| `grayscale` | appearance | RGB to grayscale RGB |
| `center_occlusion` | appearance | center box, `occlusion_size_fraction = 0.25`, fill `0` |

All six perturbation artifacts have 100 successful samples, 0 failed samples, shape `[100, 768]`, and row alignment with the original validation artifact.

## Embedding Sensitivity Results

Metric: row-wise original-vs-perturbed embedding shift on validation100.

| perturbation | group | mean cosine distance | median cosine distance | mean L2 distance | mean relative L2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `single_frame` | motion | 0.024987 | 0.020776 | 2.8343 | 0.2086 |
| `temporal_shuffle` | motion | 0.021562 | 0.014308 | 2.4716 | 0.1819 |
| `temporal_reverse` | motion | 0.009373 | 0.002881 | 1.3898 | 0.1022 |
| `freeze_tail` | motion | 0.006110 | 0.003259 | 1.2669 | 0.0934 |
| `center_occlusion` | appearance | 0.002787 | 0.002641 | 1.0174 | 0.0745 |
| `grayscale` | appearance | 0.001747 | 0.001434 | 0.7844 | 0.0574 |

Group comparison:

| group | mean cosine distance across perturbation means |
| --- | ---: |
| motion | 0.015508 |
| appearance | 0.002267 |
| motion - appearance | 0.013241 |

Current first-round result: motion perturbations produce substantially larger embedding shifts than appearance perturbations on this local validation100 split. The largest shifts come from `single_frame` and `temporal_shuffle`.

## Perturbation KNN Drop Results

Metric: cosine KNN with original train100 embeddings as the reference set.

Primary readout below uses `k=1`, because the original baseline is strongest at `k=1` on this local sample.

| perturbation | group | perturbed all accuracy | all accuracy drop | perturbed train-seen accuracy | train-seen accuracy drop | prediction change rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `temporal_reverse` | motion | 0.0000 | 0.0200 | 0.0000 | 0.0435 | 0.31 |
| `temporal_shuffle` | motion | 0.0000 | 0.0200 | 0.0000 | 0.0435 | 0.62 |
| `freeze_tail` | motion | 0.0000 | 0.0200 | 0.0000 | 0.0435 | 0.44 |
| `grayscale` | appearance | 0.0100 | 0.0100 | 0.0217 | 0.0217 | 0.12 |
| `center_occlusion` | appearance | 0.0100 | 0.0100 | 0.0217 | 0.0217 | 0.34 |
| `single_frame` | motion | 0.0200 | 0.0000 | 0.0435 | 0.0000 | 0.77 |

Group-level `k=1` means:

| group | mean all accuracy drop | mean train-seen accuracy drop | mean prediction change rate |
| --- | ---: | ---: | ---: |
| motion | 0.0150 | 0.0326 | 0.5350 |
| appearance | 0.0100 | 0.0217 | 0.2300 |

Current first-round KNN result: temporal reverse, temporal shuffle, and freeze-tail remove the two original `k=1` correct predictions. `single_frame` causes the highest prediction change rate, but its accuracy drop is zero at `k=1` because two correct-to-incorrect changes are offset by two incorrect-to-correct changes. The absolute accuracies are very small, so the train-seen and prediction-change columns are more informative than all-validation accuracy alone.

## Sweep Results

Sweep metric rows report mean cosine distance plus cosine KNN `k=1` all-validation accuracy drop.

| sweep | case | mean cosine distance | k=1 all accuracy drop | prediction change rate | trend note |
| --- | --- | ---: | ---: | ---: | --- |
| `freeze_tail_start_fraction` | 0.25 | 0.014284 | 0.0200 | 0.59 | smaller start freezes more and shifts more |
| `freeze_tail_start_fraction` | 0.5 | 0.006110 | 0.0200 | 0.44 |  |
| `freeze_tail_start_fraction` | 0.75 | 0.001373 | 0.0200 | 0.17 |  |
| `center_occlusion_size_fraction` | 0.15 | 0.000881 | 0.0100 | 0.21 | larger occlusion shifts more |
| `center_occlusion_size_fraction` | 0.25 | 0.002787 | 0.0100 | 0.34 |  |
| `center_occlusion_size_fraction` | 0.4 | 0.012902 | 0.0100 | 0.75 |  |
| `single_frame_position` | first | 0.027557 | 0.0200 | 0.78 | first and last shift slightly more than center |
| `single_frame_position` | center | 0.024987 | 0.0000 | 0.77 |  |
| `single_frame_position` | last | 0.027101 | 0.0200 | 0.73 |  |
| `temporal_shuffle_seed_repeat` | seed 0 | 0.021562 | 0.0200 | 0.62 | seed repeat is stable for k=1 drop |
| `temporal_shuffle_seed_repeat` | seed 1 | 0.022341 | 0.0200 | 0.61 |  |
| `temporal_shuffle_seed_repeat` | seed 2 | 0.022779 | 0.0200 | 0.68 |  |

Sweep observations:

- `freeze_tail` embedding shift decreases monotonically as `freeze_start_fraction` increases from 0.25 to 0.75.
- `center_occlusion` embedding shift increases monotonically with occlusion size.
- `temporal_shuffle` seed repeats are stable for KNN drop: all three seeds have `k=1` all accuracy drop `0.0200`.
- KNN drop is often flat across sweep strengths because the baseline has only 2 correct `k=1` predictions.

## Class-Level Sensitivity Candidates

These are qualitative inspection candidates, not stable class-level conclusions. Many validation classes have only one local sample.

| label id | label name | mean cosine distance across perturbations | strongest perturbation |
| ---: | --- | ---: | --- |
| 75 | Pretending to put something onto something | 0.063082 | `single_frame` |
| 166 | Turning the camera left while filming something | 0.057534 | `temporal_reverse` |
| 154 | Throwing something in the air and letting it fall | 0.052393 | `temporal_shuffle` |
| 151 | Throwing something | 0.030586 | `single_frame` |
| 173 | Wiping something off of something | 0.028445 | `single_frame` |

## Reporting Outputs

Phase 8 consolidates the JSON reports into CSV tables, a Markdown report draft, qualitative sample lists, and SVG plots.

Tables and qualitative samples:

- `outputs/reports/baseline_table.csv`
- `outputs/reports/perturbation_summary_table.csv`
- `outputs/reports/class_sensitivity_table.csv`
- `outputs/reports/sweep_summary_table.csv`
- `outputs/reports/qualitative_samples.json`
- `outputs/reports/ssv2_videomae_sensitivity_report.md`

Plots:

- `outputs/plots/perturbation_mean_cosine_distance.svg`
- `outputs/plots/perturbation_k1_accuracy_drop.svg`
- `outputs/plots/sweep_mean_cosine_distance.svg`

Qualitative sample categories include largest embedding shift, smallest embedding shift, correct-to-incorrect predictions, incorrect-to-correct predictions, high-shift unchanged predictions, and low-shift changed predictions.

## Result Files

First-round sensitivity reports:

- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_temporal_reverse_sensitivity.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_temporal_shuffle_sensitivity.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_freeze_tail_sensitivity.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_single_frame_sensitivity.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_grayscale_sensitivity.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_center_occlusion_sensitivity.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_all_perturbations_sensitivity_summary.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_class_sensitivity.json`

First-round KNN drop reports:

- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_temporal_reverse_knn_cosine.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_temporal_shuffle_knn_cosine.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_freeze_tail_knn_cosine.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_single_frame_knn_cosine.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_grayscale_knn_cosine.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_center_occlusion_knn_cosine.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_all_perturbations_knn_drop_cosine.json`

Sweep reports:

- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_freeze_tail_start_fraction_sweep_summary.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_center_occlusion_size_fraction_sweep_summary.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_single_frame_position_sweep_summary.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_temporal_shuffle_seed_repeat_sweep_summary.json`
- `outputs/logs/ssv2_validation100_videomae_base_16f_mean_all_sweeps_summary.json`

Core embedding artifacts:

- `outputs/embeddings/ssv2_train100_videomae_base_16f_mean_original.pt`
- `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_original.pt`
- `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_temporal_reverse.pt`
- `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_temporal_shuffle.pt`
- `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_freeze_tail.pt`
- `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_single_frame.pt`
- `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_grayscale.pt`
- `outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_center_occlusion.pt`

## Reproduce Reports

Create the environment:

```bash
uv sync --dev
```

Run the test suite:

```bash
uv run python -m pytest
```

Current verified result:

```text
57 passed
```

Regenerate the first-round sensitivity reports from existing artifacts:

```bash
uv run python -m src.embedding_sensitivity \
  --matrix configs/ssv2_videomae_perturbation_matrix.json \
  --output-dir outputs/logs \
  --overwrite
```

Regenerate the original cosine KNN baseline:

```bash
uv run python -m src.knn_baseline \
  --train-artifact outputs/embeddings/ssv2_train100_videomae_base_16f_mean_original.pt \
  --validation-artifact outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_original.pt \
  --k 1 5 10 \
  --metric cosine \
  --output-path outputs/logs/ssv2_validation100_videomae_base_16f_mean_original_knn_cosine.json \
  --overwrite
```

Regenerate the first-round perturbation KNN drop reports:

```bash
uv run python -m src.knn_perturbation_analysis \
  --matrix configs/ssv2_videomae_perturbation_matrix.json \
  --output-dir outputs/logs \
  --overwrite
```

Regenerate missing sweep artifacts and sweep reports:

```bash
uv run python -m src.perturbation_sweeps \
  --matrix configs/ssv2_videomae_perturbation_matrix.json \
  --config configs/ssv2_videomae_smoke.json \
  --output-dir outputs/logs \
  --generate-missing-artifacts \
  --overwrite-reports \
  --no-progress
```

Regenerate tables, qualitative sample lists, and SVG plots:

```bash
uv run python -m src.reporting \
  --log-dir outputs/logs \
  --report-dir outputs/reports \
  --plot-dir outputs/plots
```

## Current Missing Results

The first-round local experiment now has original baseline, embedding sensitivity, class-level sensitivity, KNN drop, selected sweep reports, consolidated tables, SVG plots, and qualitative sample candidates. Remaining result work is a written conclusion-focused experiment summary.
