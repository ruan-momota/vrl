# diving48 × frozen dinov2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| freeze-tail-low | motion | curve | low | 0.003740 | 0.003485 | 0.003985 | -0.002083 | -0.014583 | 0.012500 | 0.010417 | 0.008333 |
| freeze-tail-mid | motion | curve | mid | 0.024579 | 0.022923 | 0.026265 | 0.006250 | -0.014583 | 0.027083 | 0.031250 | 0.033333 |
| freeze-tail-high | motion | curve | high | 0.062827 | 0.059134 | 0.066584 | 0.014583 | -0.010469 | 0.041667 | 0.056250 | 0.004167 |
| color-low | appearance | curve | low | 0.000041 | 0.000038 | 0.000044 | 0.000000 | -0.006250 | 0.006250 | 0.002083 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000648 | 0.000622 | 0.000674 | 0.000000 | -0.008333 | 0.008333 | 0.004167 | 0.000000 |
| color-high | appearance | curve | high | 0.006583 | 0.006339 | 0.006845 | -0.002083 | -0.016667 | 0.012500 | 0.012500 | 0.010417 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.003329 | 0.003111 | 0.003574 | 0.004167 | -0.006250 | 0.016667 | 0.010417 | 0.004167 |
| rgb-quantization-low | appearance | curve | low | 0.036489 | 0.033922 | 0.039173 | 0.000000 | -0.025000 | 0.022917 | 0.031250 | -0.006250 |
| rgb-quantization-mid | appearance | curve | mid | 0.159336 | 0.149528 | 0.168877 | 0.018750 | -0.010417 | 0.045833 | 0.058333 | 0.031250 |
| rgb-quantization-high | appearance | curve | high | 0.384938 | 0.372608 | 0.397459 | 0.045833 | 0.014583 | 0.075000 | 0.079167 | 0.033333 |
| solarization-low | appearance | curve | low | 0.078616 | 0.070473 | 0.086963 | 0.008333 | -0.014583 | 0.033333 | 0.041667 | 0.016667 |
| solarization-mid | appearance | curve | mid | 0.118649 | 0.110532 | 0.126308 | 0.008333 | -0.020833 | 0.035417 | 0.054167 | 0.025000 |
| solarization-high | appearance | curve | high | 0.362977 | 0.353795 | 0.371474 | 0.052083 | 0.020833 | 0.085417 | 0.091667 | 0.037500 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | heldout_28_jMEYIEYkpY0_00055 | Reverse_25som_NoTwis_TUCK | 0.782548 | 23 | 24 | False |
| rgb-quantization-high | heldout_26_jMEYIEYkpY0_00052 | Inward_35som_NoTwis_TUCK | 0.764235 | 14 | 22 | False |
| rgb-quantization-high | heldout_01_SbzcCzXjYSU_00052 | Back_15som_15Twis_FREE | 0.756486 | 0 | 22 | False |
| rgb-quantization-high | heldout_11_CVAfPfVFulQ_00015 | Back_Dive_NoTwis_TUCK | 0.750232 | 11 | 1 | True |
| rgb-quantization-high | heldout_11_zBXtMcI41z8_00020 | Back_Dive_NoTwis_TUCK | 0.749875 | 26 | 15 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | heldout_11_CVAfPfVFulQ_00015 | Back_Dive_NoTwis_TUCK | 0.750232 | 11 | 1 |
| rgb-quantization-high | heldout_29_zBXtMcI41z8_00062 | Reverse_35som_NoTwis_TUCK | 0.614820 | 29 | 15 |
| rgb-quantization-high | heldout_03_roFeEJPgJD8_00015 | Back_25som_15Twis_PIKE | 0.581980 | 3 | 22 |
| solarization-high | heldout_03_sFO6XlfgxNQ_00119 | Back_25som_15Twis_PIKE | 0.572265 | 3 | 1 |
| rgb-quantization-high | heldout_17_MbzIpx8kAD0_00050 | Forward_25som_NoTwis_PIKE | 0.565202 | 17 | 24 |

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
