# HMDB51 × frozen V-JEPA2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.090657 | 0.088979 | 0.092460 | 0.184383 | 0.160761 | 0.210630 | 0.232283 | 0.051837 |
| freeze-tail-low | motion | curve | low | 0.017075 | 0.016172 | 0.018020 | 0.024278 | 0.009170 | 0.040682 | 0.062992 | 0.003937 |
| freeze-tail-mid | motion | curve | mid | 0.039817 | 0.038308 | 0.041365 | 0.073491 | 0.054446 | 0.093832 | 0.118110 | 0.053150 |
| freeze-tail-high | motion | curve | high | 0.088777 | 0.086515 | 0.091268 | 0.175197 | 0.152231 | 0.200787 | 0.211286 | 0.104331 |
| color-low | appearance | curve | low | 0.000037 | 0.000036 | 0.000039 | -0.000656 | -0.003937 | 0.001969 | 0.001312 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000990 | 0.000967 | 0.001014 | -0.003937 | -0.011811 | 0.003281 | 0.009843 | 0.005906 |
| color-high | appearance | curve | high | 0.012508 | 0.012240 | 0.012782 | 0.022310 | 0.006562 | 0.038074 | 0.058399 | 0.005906 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.147036 | 0.145138 | 0.148986 | 0.140420 | 0.115469 | 0.163386 | 0.187664 | 0.184383 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | FC_Venus_-_Ausschnitt_Teil_3_kick_ball_l_cm_np1_fr_med_0 | kick_ball | 0.330586 | 34 | 2 | False |
| spatial-blur-mid | BASE_Jumping_Compilation_-_Brilliant_dive_u_cm_np1_ba_bad_7 | dive | 0.308362 | 7 | 23 | True |
| spatial-blur-mid | Brushing_my_long_hair_brush_hair_u_nm_np1_ba_goo_2 | brush_hair | 0.308289 | 0 | 0 | False |
| freeze-tail-high | RATRACE_fall_floor_u_cm_np1_fr_bad_23 | fall_floor | 0.295811 | 12 | 2 | True |
| freeze-tail-high | Oberstrick_-_K_pfer_-_3m_dive_f_cm_np1_le_bad_0 | dive | 0.295056 | 7 | 17 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| spatial-blur-mid | BASE_Jumping_Compilation_-_Brilliant_dive_u_cm_np1_ba_bad_7 | dive | 0.308362 | 7 | 23 |
| freeze-tail-high | RATRACE_fall_floor_u_cm_np1_fr_bad_23 | fall_floor | 0.295811 | 12 | 2 |
| freeze-tail-high | Oberstrick_-_K_pfer_-_3m_dive_f_cm_np1_le_bad_0 | dive | 0.295056 | 7 | 17 |
| spatial-blur-mid | The_Pomegranate_Martini_pour_u_nm_np1_fr_med_0 | pour | 0.274035 | 25 | 23 |
| freeze-tail-high | bungee_jumping_compilation_dive_f_cm_np1_le_bad_2 | dive | 0.266822 | 7 | 19 |

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

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one action-recognition HMDB51 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
