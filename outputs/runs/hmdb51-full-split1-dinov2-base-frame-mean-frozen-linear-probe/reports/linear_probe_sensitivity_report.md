# HMDB51 × frozen dinov2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| freeze-tail-low | motion | curve | low | 0.001674 | 0.001572 | 0.001790 | 0.010499 | 0.005889 | 0.015748 | 0.011155 | 0.001312 |
| freeze-tail-mid | motion | curve | mid | 0.011105 | 0.010515 | 0.011708 | 0.013123 | 0.003937 | 0.022966 | 0.024934 | 0.000656 |
| freeze-tail-high | motion | curve | high | 0.032117 | 0.030166 | 0.034289 | 0.028215 | 0.015748 | 0.041995 | 0.049213 | 0.006562 |
| color-low | appearance | curve | low | 0.000050 | 0.000047 | 0.000052 | 0.001312 | 0.000000 | 0.003281 | 0.001312 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000758 | 0.000727 | 0.000791 | 0.000656 | -0.003937 | 0.005906 | 0.004593 | 0.004593 |
| color-high | appearance | curve | high | 0.007313 | 0.007069 | 0.007552 | 0.007874 | 0.000000 | 0.016404 | 0.016404 | 0.017717 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.037913 | 0.036597 | 0.039227 | 0.019029 | 0.005906 | 0.033465 | 0.045932 | 0.005249 |
| rgb-quantization-low | appearance | curve | low | 0.026620 | 0.025181 | 0.028134 | 0.013123 | 0.000656 | 0.025607 | 0.037402 | 0.009843 |
| rgb-quantization-mid | appearance | curve | mid | 0.107462 | 0.103433 | 0.111540 | 0.056430 | 0.038058 | 0.076132 | 0.094488 | 0.043963 |
| rgb-quantization-high | appearance | curve | high | 0.280983 | 0.274206 | 0.287705 | 0.152887 | 0.126624 | 0.178494 | 0.206693 | 0.085958 |
| solarization-low | appearance | curve | low | 0.089897 | 0.085356 | 0.094636 | 0.038058 | 0.021654 | 0.055118 | 0.071522 | 0.016404 |
| solarization-mid | appearance | curve | mid | 0.103790 | 0.099765 | 0.108334 | 0.044619 | 0.027559 | 0.063009 | 0.084646 | 0.011155 |
| solarization-high | appearance | curve | high | 0.296903 | 0.290969 | 0.303116 | 0.165354 | 0.139764 | 0.190289 | 0.214567 | 0.091864 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | Moviekissmontage_kiss_h_cm_np2_fr_med_8 | kiss | 0.898478 | 22 | 19 | True |
| rgb-quantization-high | Finding_Forrester_3_drink_h_nm_np1_fr_med_16 | drink | 0.881181 | 10 | 32 | True |
| rgb-quantization-high | FC_Venus_-_Ausschnitt_Teil_3_kick_ball_l_cm_np1_fr_med_0 | kick_ball | 0.863719 | 21 | 32 | True |
| rgb-quantization-high | TVs_Best_Kisses_Top_50_(52_to_41)_kiss_h_nm_np2_le_goo_3 | kiss | 0.843048 | 22 | 19 | True |
| rgb-quantization-high | kick__Baddest_Fight_Scenes_EVER!_-_Kickboxer_-_Part_1_of_2_kick_f_cm_np1_fr_med_11 | kick | 0.837907 | 27 | 12 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | Moviekissmontage_kiss_h_cm_np2_fr_med_8 | kiss | 0.898478 | 22 | 19 |
| rgb-quantization-high | Finding_Forrester_3_drink_h_nm_np1_fr_med_16 | drink | 0.881181 | 10 | 32 |
| rgb-quantization-high | FC_Venus_-_Ausschnitt_Teil_3_kick_ball_l_cm_np1_fr_med_0 | kick_ball | 0.863719 | 21 | 32 |
| rgb-quantization-high | TVs_Best_Kisses_Top_50_(52_to_41)_kiss_h_nm_np2_le_goo_3 | kiss | 0.843048 | 22 | 19 |
| rgb-quantization-high | Vorschlaghammer_hit_f_cm_np1_fr_bad_1 | hit | 0.798251 | 17 | 12 |

## Data quality and failures

Fail-fast extraction was used. All extraction artifacts succeeded: True. 
All sampled frame-index and sampling-strategy checks passed: True.

| artifact_label | dataset_size | successful_samples | failed_samples |
| --- | --- | --- | --- |
| train_original | 3551 | 3551 | 0 |
| heldout_original | 1524 | 1524 | 0 |
| temporal-shuffle-mid | 1524 | 1524 | 0 |
| freeze-tail-low | 1524 | 1524 | 0 |
| freeze-tail-mid | 1524 | 1524 | 0 |
| freeze-tail-high | 1524 | 1524 | 0 |
| color-low | 1524 | 1524 | 0 |
| color-mid | 1524 | 1524 | 0 |
| color-high | 1524 | 1524 | 0 |
| spatial-blur-mid | 1524 | 1524 | 0 |
| rgb-quantization-low | 1524 | 1524 | 0 |
| rgb-quantization-mid | 1524 | 1524 | 0 |
| rgb-quantization-high | 1524 | 1524 | 0 |
| solarization-low | 1524 | 1524 | 0 |
| solarization-mid | 1524 | 1524 | 0 |
| solarization-high | 1524 | 1524 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one action-recognition HMDB51 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
