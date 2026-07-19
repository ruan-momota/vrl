# diving48 × frozen dismo linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.709405 | 0.689249 | 0.731151 | 0.045833 | 0.016667 | 0.079167 | 0.083333 | 0.029167 |
| freeze-tail-low | motion | curve | low | 0.004202 | 0.004025 | 0.004390 | -0.002083 | -0.014583 | 0.010417 | 0.008333 | 0.004167 |
| freeze-tail-mid | motion | curve | mid | 0.106442 | 0.103052 | 0.110196 | 0.002083 | -0.025052 | 0.031250 | 0.050000 | 0.004167 |
| freeze-tail-high | motion | curve | high | 0.387781 | 0.374404 | 0.401325 | 0.045833 | 0.016667 | 0.077083 | 0.083333 | 0.020833 |
| color-low | appearance | curve | low | 0.000012 | 0.000011 | 0.000013 | 0.004167 | 0.000000 | 0.010417 | 0.004167 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000399 | 0.000378 | 0.000421 | 0.006250 | 0.000000 | 0.014583 | 0.006250 | -0.002083 |
| color-high | appearance | curve | high | 0.004250 | 0.004041 | 0.004472 | 0.010417 | -0.002083 | 0.022917 | 0.016667 | 0.000000 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.002589 | 0.002390 | 0.002787 | 0.008333 | -0.004167 | 0.020833 | 0.012500 | 0.010417 |
| rgb-quantization-low | appearance | curve | low | 0.005486 | 0.005169 | 0.005802 | 0.006250 | -0.006250 | 0.018802 | 0.014583 | 0.004167 |
| rgb-quantization-mid | appearance | curve | mid | 0.013326 | 0.012733 | 0.013946 | 0.010417 | -0.006250 | 0.027083 | 0.025000 | 0.000000 |
| rgb-quantization-high | appearance | curve | high | 0.040186 | 0.038521 | 0.041831 | 0.006250 | -0.014583 | 0.027083 | 0.033333 | 0.000000 |
| solarization-low | appearance | curve | low | 0.021830 | 0.019887 | 0.024029 | 0.006250 | -0.010469 | 0.025000 | 0.025000 | -0.002083 |
| solarization-mid | appearance | curve | mid | 0.023946 | 0.022018 | 0.026020 | 0.008333 | -0.010417 | 0.029167 | 0.033333 | 0.010417 |
| solarization-high | appearance | curve | high | 0.058692 | 0.056609 | 0.060848 | 0.000000 | -0.020885 | 0.022917 | 0.035417 | 0.008333 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | heldout_16_k71Cc-Sm-Mg_00075 | Forward_25som_2Twis_PIKE | 1.200280 | 16 | 19 | True |
| temporal-shuffle-mid | heldout_10_Le6xdQ2OO8w_00053 | Back_Dive_NoTwis_PIKE | 1.131587 | 31 | 19 | False |
| temporal-shuffle-mid | heldout_13_Le6xdQ2OO8w_00006 | Forward_15som_NoTwis_PIKE | 1.130917 | 20 | 19 | False |
| temporal-shuffle-mid | heldout_09_sFO6XlfgxNQ_00101 | Back_35som_NoTwis_TUCK | 1.128939 | 4 | 19 | False |
| temporal-shuffle-mid | heldout_00_Le6xdQ2OO8w_00150 | Back_15som_05Twis_FREE | 1.124298 | 16 | 19 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | heldout_16_k71Cc-Sm-Mg_00075 | Forward_25som_2Twis_PIKE | 1.200280 | 16 | 19 |
| temporal-shuffle-mid | heldout_05_XfCaMLlyj9I_00114 | Back_25som_NoTwis_PIKE | 1.111275 | 5 | 25 |
| temporal-shuffle-mid | heldout_08_qD3IMtAanSg_00102 | Back_35som_NoTwis_PIKE | 1.091112 | 8 | 14 |
| temporal-shuffle-mid | heldout_26_XfCaMLlyj9I_00129 | Inward_35som_NoTwis_TUCK | 1.072518 | 26 | 25 |
| temporal-shuffle-mid | heldout_31_Le6xdQ2OO8w_00059 | Reverse_Dive_NoTwis_TUCK | 1.065495 | 31 | 19 |

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
| rgb-quantization-low | 480 | 480 | 0 |
| rgb-quantization-mid | 480 | 480 | 0 |
| rgb-quantization-high | 480 | 480 | 0 |
| solarization-low | 480 | 480 | 0 |
| solarization-mid | 480 | 480 | 0 |
| solarization-high | 480 | 480 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
