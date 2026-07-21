# HMDB51 × frozen VideoMAE linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.000466 | 0.000425 | 0.000505 | 0.007874 | -0.002625 | 0.018373 | 0.026903 | 0.011811 |
| freeze-tail-mid | motion | curve | mid | 0.002478 | 0.002301 | 0.002660 | 0.033465 | 0.018356 | 0.049229 | 0.066273 | 0.032152 |
| freeze-tail-high | motion | curve | high | 0.006862 | 0.006420 | 0.007330 | 0.103675 | 0.083990 | 0.124672 | 0.141076 | 0.046588 |
| color-low | appearance | curve | low | 0.000002 | 0.000002 | 0.000002 | 0.000000 | -0.001969 | 0.001969 | 0.000656 | -0.001969 |
| color-mid | appearance | curve | mid | 0.000051 | 0.000049 | 0.000052 | 0.007218 | -0.000656 | 0.015092 | 0.015748 | -0.001312 |
| color-high | appearance | curve | high | 0.000487 | 0.000475 | 0.000498 | 0.031496 | 0.017044 | 0.047244 | 0.063648 | 0.000656 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.009095 | 0.008948 | 0.009238 | 0.141732 | 0.119423 | 0.166027 | 0.184383 | 0.067585 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.014272 | 0.013324 | 0.015335 | 0.146982 | 0.122703 | 0.171276 | 0.191601 | 0.038714 |
| rgb-quantization-low | appearance | curve | low | 0.003149 | 0.003053 | 0.003259 | 0.105643 | 0.082677 | 0.130577 | 0.159449 | 0.013123 |
| rgb-quantization-mid | appearance | curve | mid | 0.009404 | 0.009161 | 0.009647 | 0.206037 | 0.182415 | 0.231644 | 0.242782 | 0.041995 |
| rgb-quantization-high | appearance | curve | high | 0.020450 | 0.019999 | 0.020915 | 0.267717 | 0.242126 | 0.295276 | 0.299213 | 0.079396 |
| solarization-low | appearance | curve | low | 0.004467 | 0.004165 | 0.004820 | 0.095144 | 0.076099 | 0.118110 | 0.136483 | 0.013123 |
| solarization-mid | appearance | curve | mid | 0.005501 | 0.005201 | 0.005829 | 0.123360 | 0.102346 | 0.146342 | 0.169948 | 0.015092 |
| solarization-high | appearance | curve | high | 0.013200 | 0.012818 | 0.013604 | 0.208005 | 0.183727 | 0.234908 | 0.255906 | 0.060367 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | RATRACE_stand_f_cm_np1_fr_med_36 | stand | 0.148734 | 50 | 36 | False |
| temporal-shuffle-mid | bungee_jumping_compilation_dive_f_cm_np1_ri_bad_3 | dive | 0.123193 | 7 | 19 | True |
| temporal-shuffle-mid | How_to_Shoot_Penalty_Kicks_kick_ball_f_cm_np1_ba_bad_1 | kick_ball | 0.114665 | 9 | 41 | False |
| temporal-shuffle-mid | OldSchool_dive_f_cm_np2_ri_bad_15 | dive | 0.108768 | 7 | 41 | True |
| temporal-shuffle-mid | LONGESTYARD_fall_floor_f_cm_np1_fr_med_19 | fall_floor | 0.106833 | 7 | 41 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | bungee_jumping_compilation_dive_f_cm_np1_ri_bad_3 | dive | 0.123193 | 7 | 19 |
| temporal-shuffle-mid | OldSchool_dive_f_cm_np2_ri_bad_15 | dive | 0.108768 | 7 | 41 |
| temporal-shuffle-mid | 10YearOldYouthBasketballStarBaller_dribble_f_cm_np1_ri_med_4 | dribble | 0.102765 | 9 | 41 |
| temporal-shuffle-mid | funny_soccer_commercial_kick_ball_f_cm_np1_fr_med_0 | kick_ball | 0.102004 | 21 | 41 |
| freeze-tail-high | bungee_jumping_compilation_dive_f_cm_np1_ri_bad_3 | dive | 0.096428 | 7 | 29 |

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
