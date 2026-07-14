# HMDB51 × frozen VideoMAE linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.000466 | 0.000426 | 0.000505 | 0.011155 | 0.000656 | 0.021654 | 0.028871 | 0.013780 |
| freeze-tail-mid | motion | curve | mid | 0.002481 | 0.002304 | 0.002662 | 0.039370 | 0.023622 | 0.057087 | 0.069554 | 0.031496 |
| freeze-tail-high | motion | curve | high | 0.006871 | 0.006430 | 0.007340 | 0.106299 | 0.087270 | 0.126640 | 0.141076 | 0.046588 |
| color-low | appearance | curve | low | 0.000002 | 0.000002 | 0.000002 | 0.001312 | -0.001312 | 0.003937 | 0.001969 | -0.000656 |
| color-mid | appearance | curve | mid | 0.000051 | 0.000049 | 0.000052 | 0.007874 | 0.000000 | 0.015748 | 0.015748 | 0.000656 |
| color-high | appearance | curve | high | 0.000487 | 0.000476 | 0.000498 | 0.036089 | 0.020997 | 0.051854 | 0.066929 | -0.001312 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.009106 | 0.008958 | 0.009249 | 0.143701 | 0.121391 | 0.167323 | 0.187008 | 0.068241 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.014283 | 0.013334 | 0.015346 | 0.149606 | 0.125984 | 0.175197 | 0.191601 | 0.039370 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | RATRACE_stand_f_cm_np1_fr_med_36 | stand | 0.148843 | 50 | 36 | False |
| temporal-shuffle-mid | bungee_jumping_compilation_dive_f_cm_np1_ri_bad_3 | dive | 0.122894 | 7 | 36 | True |
| temporal-shuffle-mid | How_to_Shoot_Penalty_Kicks_kick_ball_f_cm_np1_ba_bad_1 | kick_ball | 0.114684 | 9 | 41 | False |
| temporal-shuffle-mid | OldSchool_dive_f_cm_np2_ri_bad_15 | dive | 0.108820 | 12 | 41 | False |
| temporal-shuffle-mid | LONGESTYARD_fall_floor_f_cm_np1_fr_med_19 | fall_floor | 0.106816 | 7 | 41 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | bungee_jumping_compilation_dive_f_cm_np1_ri_bad_3 | dive | 0.122894 | 7 | 36 |
| temporal-shuffle-mid | 10YearOldYouthBasketballStarBaller_dribble_f_cm_np1_ri_med_4 | dribble | 0.102791 | 9 | 41 |
| temporal-shuffle-mid | funny_soccer_commercial_kick_ball_f_cm_np1_fr_med_0 | kick_ball | 0.102104 | 21 | 41 |
| freeze-tail-high | bungee_jumping_compilation_dive_f_cm_np1_ri_bad_3 | dive | 0.096313 | 7 | 29 |
| temporal-shuffle-mid | bungee_jumping_compilation_dive_f_cm_np1_le_bad_4 | dive | 0.091091 | 7 | 41 |

## Data quality and failures

Fail-fast extraction was used. All extraction artifacts succeeded: True. 
All sampled frame-index and sampling-strategy checks passed: True.

| artifact_label | dataset_size | successful_samples | failed_samples |
| --- | --- | --- | --- |
| train_original | 3551 | 3551 | 0 |
| heldout_original | 1524 | 1524 | 0 |
| freeze-tail-low | 1524 | 1524 | 0 |
| freeze-tail-mid | 1524 | 1524 | 0 |
| freeze-tail-high | 1524 | 1524 | 0 |
| color-low | 1524 | 1524 | 0 |
| color-mid | 1524 | 1524 | 0 |
| color-high | 1524 | 1524 | 0 |
| spatial-blur-mid | 1524 | 1524 | 0 |
| temporal-shuffle-mid | 1524 | 1524 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one action-recognition HMDB51 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
