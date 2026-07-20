# HMDB51 × frozen dismo linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.362236 | 0.351485 | 0.374048 | 0.188320 | 0.162073 | 0.213927 | 0.261811 | 0.175197 |
| freeze-tail-low | motion | curve | low | 0.003304 | 0.003157 | 0.003459 | 0.007874 | -0.003297 | 0.019029 | 0.028871 | 0.009186 |
| freeze-tail-mid | motion | curve | mid | 0.055699 | 0.053693 | 0.057809 | 0.099738 | 0.078084 | 0.120095 | 0.151575 | 0.152231 |
| freeze-tail-high | motion | curve | high | 0.171859 | 0.166116 | 0.177872 | 0.283465 | 0.258514 | 0.309711 | 0.329396 | 0.297900 |
| color-low | appearance | curve | low | 0.000026 | 0.000023 | 0.000029 | 0.000656 | -0.002625 | 0.004593 | 0.002625 | -0.002625 |
| color-mid | appearance | curve | mid | 0.000510 | 0.000491 | 0.000528 | 0.002625 | -0.005266 | 0.010499 | 0.013123 | -0.009186 |
| color-high | appearance | curve | high | 0.005609 | 0.005424 | 0.005799 | 0.011155 | -0.001969 | 0.024278 | 0.041339 | -0.009186 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.017677 | 0.017187 | 0.018166 | 0.029528 | 0.013123 | 0.044636 | 0.068898 | 0.022966 |
| rgb-quantization-low | appearance | curve | low | 0.008552 | 0.008260 | 0.008852 | 0.007874 | -0.006562 | 0.021654 | 0.042651 | -0.002625 |
| rgb-quantization-mid | appearance | curve | mid | 0.019624 | 0.018939 | 0.020271 | 0.022966 | 0.005233 | 0.040026 | 0.076772 | 0.020997 |
| rgb-quantization-high | appearance | curve | high | 0.051608 | 0.049607 | 0.053344 | 0.072178 | 0.049869 | 0.094488 | 0.137139 | 0.049869 |
| solarization-low | appearance | curve | low | 0.029860 | 0.028446 | 0.031364 | 0.057743 | 0.039370 | 0.075459 | 0.093176 | 0.019685 |
| solarization-mid | appearance | curve | mid | 0.019850 | 0.018896 | 0.020910 | 0.028215 | 0.011811 | 0.045276 | 0.070866 | 0.011155 |
| solarization-high | appearance | curve | high | 0.067966 | 0.066257 | 0.069568 | 0.134514 | 0.110876 | 0.157497 | 0.192913 | 0.061680 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | Bodenturnen_im_sportunterricht_flic_flac_f_cm_np1_le_med_3 | flic_flac | 1.198595 | 19 | 14 | False |
| temporal-shuffle-mid | How_to_Improve_Your_Basketball_Skills_-_How_to_Dribble_a_Basketball_Between_the_Legs_dribble_f_cm_np1_fr_med_0 | dribble | 1.145500 | 31 | 30 | False |
| temporal-shuffle-mid | How_to_Shoot_Penalty_Kicks_kick_ball_f_cm_np1_ba_bad_4 | kick_ball | 1.134413 | 19 | 32 | False |
| temporal-shuffle-mid | How_to_Shoot_Penalty_Kicks_kick_ball_f_cm_np1_ba_bad_1 | kick_ball | 1.134059 | 9 | 32 | False |
| temporal-shuffle-mid | hechtsprung_2_somersault_f_cm_np1_le_bad_0 | somersault | 1.116623 | 7 | 19 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 10YearOldYouthBasketballStarBaller_dribble_f_cm_np1_ri_med_4 | dribble | 1.103820 | 9 | 1 |
| temporal-shuffle-mid | LONGESTYARD_fall_floor_f_cm_np1_le_bad_2 | fall_floor | 0.953642 | 12 | 19 |
| temporal-shuffle-mid | Braune_Stiefel_brown_boots_hot_high_heels_Treppen_steigen_Down-_and_upstairs_climb_stairs_l_cm_np1_ri_med_4 | climb_stairs | 0.941468 | 6 | 32 |
| temporal-shuffle-mid | 10YearOldYouthBasketballStarBaller_dribble_f_cm_np1_fr_med_8 | dribble | 0.911122 | 9 | 32 |
| temporal-shuffle-mid | funny_soccer_commercial_kick_ball_f_cm_np1_fr_med_0 | kick_ball | 0.872885 | 21 | 41 |

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
