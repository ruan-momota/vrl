# diving48 × frozen VideoMAE linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.080182 | 0.074795 | 0.085359 | 0.037500 | 0.014531 | 0.064583 | 0.058333 | -0.008333 |
| freeze-tail-low | motion | curve | low | 0.003937 | 0.003654 | 0.004238 | 0.002083 | -0.014583 | 0.018750 | 0.018750 | -0.012500 |
| freeze-tail-mid | motion | curve | mid | 0.023550 | 0.021803 | 0.025509 | 0.008333 | -0.012500 | 0.031250 | 0.033333 | -0.006250 |
| freeze-tail-high | motion | curve | high | 0.056811 | 0.052614 | 0.061090 | 0.039583 | 0.012500 | 0.066667 | 0.062500 | -0.002083 |
| color-low | appearance | curve | low | 0.000001 | 0.000001 | 0.000001 | -0.002083 | -0.006250 | 0.000000 | 0.000000 | -0.002083 |
| color-mid | appearance | curve | mid | 0.000024 | 0.000023 | 0.000025 | 0.006250 | -0.004219 | 0.018750 | 0.012500 | 0.000000 |
| color-high | appearance | curve | high | 0.000224 | 0.000214 | 0.000233 | 0.000000 | -0.016667 | 0.016667 | 0.016667 | 0.000000 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.001066 | 0.001004 | 0.001133 | 0.018750 | 0.002083 | 0.035417 | 0.027083 | 0.002083 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | heldout_08_qD3IMtAanSg_00101 | Back_35som_NoTwis_PIKE | 0.325244 | 4 | 19 | False |
| temporal-shuffle-mid | heldout_11_Le6xdQ2OO8w_00080 | Back_Dive_NoTwis_TUCK | 0.240125 | 11 | 19 | True |
| temporal-shuffle-mid | heldout_28_bSsVWVfYU4w_00168 | Reverse_25som_NoTwis_TUCK | 0.227877 | 14 | 19 | False |
| temporal-shuffle-mid | heldout_27_Le6xdQ2OO8w_00045 | Inward_Dive_NoTwis_PIKE | 0.227673 | 11 | 19 | False |
| temporal-shuffle-mid | heldout_13_bSsVWVfYU4w_00000 | Forward_15som_NoTwis_PIKE | 0.222500 | 23 | 19 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | heldout_11_Le6xdQ2OO8w_00080 | Back_Dive_NoTwis_TUCK | 0.240125 | 11 | 19 |
| temporal-shuffle-mid | heldout_04_qD3IMtAanSg_00067 | Back_25som_25Twis_PIKE | 0.209366 | 4 | 19 |
| temporal-shuffle-mid | heldout_03_zBXtMcI41z8_00034 | Back_25som_15Twis_PIKE | 0.197989 | 3 | 1 |
| temporal-shuffle-mid | heldout_08_zBXtMcI41z8_00058 | Back_35som_NoTwis_PIKE | 0.169977 | 8 | 26 |
| temporal-shuffle-mid | heldout_08_qD3IMtAanSg_00056 | Back_35som_NoTwis_PIKE | 0.165180 | 8 | 27 |

## Data quality and failures

Fail-fast extraction was used. All extraction artifacts succeeded: True. 
All sampled frame-index and sampling-strategy checks passed: True.

| artifact_label | dataset_size | successful_samples | failed_samples |
| --- | --- | --- | --- |
| train_original | 1600 | 1600 | 0 |
| heldout_original | 480 | 480 | 0 |
| temporal-shuffle-mid | 480 | 480 | 0 |
| freeze-tail-low | 480 | 480 | 0 |
| freeze-tail-mid | 480 | 480 | 0 |
| freeze-tail-high | 480 | 480 | 0 |
| color-low | 480 | 480 | 0 |
| color-mid | 480 | 480 | 0 |
| color-high | 480 | 480 | 0 |
| spatial-blur-mid | 480 | 480 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
