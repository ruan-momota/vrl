# VRL

VideoMAE + Something-Something-V2 sensitivity experiment over a small local SSV2 sample.

The current experiment uses frozen `MCG-NJU/videomae-base` embeddings to compare original validation videos against deterministic motion and appearance perturbations. The main outputs are KNN baseline reports and original-vs-perturbed embedding sensitivity reports.

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

## Class-Level Sensitivity Candidates

These are qualitative inspection candidates, not stable class-level conclusions. Many validation classes have only one local sample.

| label id | label name | mean cosine distance across perturbations | strongest perturbation |
| ---: | --- | ---: | --- |
| 75 | Pretending to put something onto something | 0.063082 | `single_frame` |
| 166 | Turning the camera left while filming something | 0.057534 | `temporal_reverse` |
| 154 | Throwing something in the air and letting it fall | 0.052393 | `temporal_shuffle` |
| 151 | Throwing something | 0.030586 | `single_frame` |
| 173 | Wiping something off of something | 0.028445 | `single_frame` |

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
45 passed
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

## Current Missing Result

The remaining first-round metric is KNN accuracy drop for each perturbation, reported both on all validation samples and on the 46 train-seen validation samples.
