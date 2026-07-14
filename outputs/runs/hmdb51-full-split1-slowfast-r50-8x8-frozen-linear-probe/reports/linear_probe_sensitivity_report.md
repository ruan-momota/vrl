# HMDB51 × frozen SlowFast linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.039365 | 0.038260 | 0.040520 | 0.026903 | 0.013107 | 0.041339 | 0.059055 | 0.007218 |
| freeze-tail-mid | motion | curve | mid | 0.115164 | 0.112784 | 0.117427 | 0.093176 | 0.072818 | 0.112877 | 0.127953 | 0.062992 |
| freeze-tail-high | motion | curve | high | 0.230272 | 0.226331 | 0.234792 | 0.220472 | 0.196194 | 0.244767 | 0.251312 | 0.240157 |
| color-low | appearance | curve | low | 0.000072 | 0.000068 | 0.000075 | 0.001312 | -0.001969 | 0.004593 | 0.002625 | -0.001312 |
| color-mid | appearance | curve | mid | 0.001617 | 0.001563 | 0.001671 | 0.003281 | -0.001969 | 0.009186 | 0.007874 | -0.001969 |
| color-high | appearance | curve | high | 0.014192 | 0.013717 | 0.014663 | 0.003281 | -0.007218 | 0.012467 | 0.023622 | 0.002625 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.066634 | 0.064733 | 0.068680 | 0.043307 | 0.026247 | 0.059072 | 0.083333 | 0.022966 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.240719 | 0.235241 | 0.246183 | 0.173228 | 0.151558 | 0.197507 | 0.211286 | 0.095801 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | TheBoondockSaints_fall_floor_f_nm_np1_fr_med_65 | fall_floor | 0.721963 | 42 | 36 | False |
| temporal-shuffle-mid | Britney_Spears_-_Comercial_Pepsi_soccer_kick_ball_f_nm_np1_ba_med_0 | kick_ball | 0.669666 | 21 | 36 | True |
| temporal-shuffle-mid | TrumanShow_wave_f_nm_np1_fr_med_26 | wave | 0.639341 | 50 | 36 | True |
| temporal-shuffle-mid | veoh_harold_and_kumar_run_f_nm_np2_fr_med_6 | run | 0.630905 | 32 | 27 | True |
| temporal-shuffle-mid | Prelinger_HabitPat1954_walk_f_nm_np6_fr_med_20 | walk | 0.608015 | 27 | 27 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | Britney_Spears_-_Comercial_Pepsi_soccer_kick_ball_f_nm_np1_ba_med_0 | kick_ball | 0.669666 | 21 | 36 |
| temporal-shuffle-mid | TrumanShow_wave_f_nm_np1_fr_med_26 | wave | 0.639341 | 50 | 36 |
| temporal-shuffle-mid | veoh_harold_and_kumar_run_f_nm_np2_fr_med_6 | run | 0.630905 | 32 | 27 |
| temporal-shuffle-mid | A_Beginners_Guide_to_Smoking_smoke_u_cm_np1_ri_bad_0 | smoke | 0.564962 | 40 | 36 |
| temporal-shuffle-mid | Prelinger_LetYours1940_walk_f_nm_np1_ba_med_5 | walk | 0.556860 | 49 | 27 |

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
