# diving48 × frozen dinov2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| freeze-tail-low | motion | curve | low | 0.003740 | 0.003485 | 0.003985 | 0.000000 | -0.010417 | 0.010417 | 0.008333 | 0.008333 |
| freeze-tail-mid | motion | curve | mid | 0.024579 | 0.022923 | 0.026265 | -0.014583 | -0.033333 | 0.002135 | 0.012500 | 0.033333 |
| freeze-tail-high | motion | curve | high | 0.062827 | 0.059134 | 0.066584 | -0.012500 | -0.035417 | 0.008333 | 0.025000 | 0.004167 |
| color-low | appearance | curve | low | 0.000041 | 0.000038 | 0.000044 | -0.004167 | -0.010417 | 0.000000 | 0.000000 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000648 | 0.000622 | 0.000674 | -0.006250 | -0.014583 | 0.000000 | 0.000000 | 0.000000 |
| color-high | appearance | curve | high | 0.006583 | 0.006339 | 0.006845 | -0.006250 | -0.018750 | 0.006250 | 0.006250 | 0.010417 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.003329 | 0.003111 | 0.003574 | 0.002083 | -0.010417 | 0.016667 | 0.010417 | 0.004167 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | heldout_06_jMEYIEYkpY0_00088 | Back_25som_NoTwis_TUCK | 0.227093 | 31 | 20 | False |
| freeze-tail-high | heldout_07_qD3IMtAanSg_00078 | Back_2som_25Twis_FREE | 0.215198 | 26 | 3 | False |
| freeze-tail-high | heldout_01_TMJ1XIZBRW4_00055 | Back_15som_15Twis_FREE | 0.206218 | 13 | 13 | False |
| freeze-tail-high | heldout_31_MbzIpx8kAD0_00003 | Reverse_Dive_NoTwis_TUCK | 0.200364 | 31 | 31 | False |
| freeze-tail-high | heldout_07_qD3IMtAanSg_00115 | Back_2som_25Twis_FREE | 0.198074 | 20 | 20 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | heldout_03_zBXtMcI41z8_00074 | Back_25som_15Twis_PIKE | 0.171941 | 3 | 8 |
| freeze-tail-high | heldout_27_zBXtMcI41z8_00012 | Inward_Dive_NoTwis_PIKE | 0.103133 | 27 | 3 |
| freeze-tail-high | heldout_31_bSsVWVfYU4w_00067 | Reverse_Dive_NoTwis_TUCK | 0.101065 | 31 | 11 |
| freeze-tail-high | heldout_19_zmacW6f2Tdk_00059 | Forward_35som_NoTwis_TUCK | 0.089872 | 19 | 23 |
| freeze-tail-high | heldout_08_qD3IMtAanSg_00056 | Back_35som_NoTwis_PIKE | 0.088465 | 8 | 9 |

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
