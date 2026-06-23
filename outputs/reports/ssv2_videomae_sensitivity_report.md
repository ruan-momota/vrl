# SSV2 VideoMAE Sensitivity Report

## Original KNN Baseline

| metric | k | all_accuracy | train_seen_accuracy | all_correct | train_seen_correct |
| --- | --- | --- | --- | --- | --- |
| cosine | 1 | 0.084000 | 0.084000 | 126 | 126 |
| cosine | 5 | 0.099333 | 0.099333 | 149 | 149 |
| cosine | 10 | 0.110000 | 0.110000 | 165 | 165 |
| l2 | 1 | 0.085333 | 0.085333 | 128 | 128 |
| l2 | 5 | 0.101333 | 0.101333 | 152 | 152 |
| l2 | 10 | 0.106667 | 0.106667 | 160 | 160 |

## Perturbation Summary

| perturbation | group | mean_cosine_distance | all_accuracy_drop | train_seen_accuracy_drop | prediction_change_rate |
| --- | --- | --- | --- | --- | --- |
| single_frame | motion | 0.024057 | 0.058667 | 0.058667 | 0.897333 |
| temporal_shuffle | motion | 0.022874 | 0.046667 | 0.046667 | 0.820667 |
| temporal_reverse | motion | 0.010014 | 0.029333 | 0.029333 | 0.576000 |
| freeze_tail | motion | 0.004984 | 0.024667 | 0.024667 | 0.636667 |
| center_occlusion | appearance | 0.002773 | -0.001333 | -0.001333 | 0.556667 |
| grayscale | appearance | 0.001916 | -0.001333 | -0.001333 | 0.356000 |

## Sweep Summary

| sweep_name | case_label | mean_cosine_distance | k1_all_accuracy_drop | k1_prediction_change_rate |
| --- | --- | --- | --- | --- |
| center_occlusion_size_fraction | 0.15 | 0.000889 | -0.002000 | 0.333333 |
| center_occlusion_size_fraction | 0.25 | 0.002773 | -0.001333 | 0.556667 |
| center_occlusion_size_fraction | 0.4 | 0.012875 | 0.024667 | 0.870667 |
| freeze_tail_start_fraction | 0.25 | 0.012561 | 0.047333 | 0.824667 |
| freeze_tail_start_fraction | 0.5 | 0.004984 | 0.024667 | 0.636667 |
| freeze_tail_start_fraction | 0.75 | 0.001039 | 0.014000 | 0.317333 |
| single_frame_position | first | 0.025708 | 0.069333 | 0.904667 |
| single_frame_position | center | 0.024057 | 0.058667 | 0.897333 |
| single_frame_position | last | 0.025748 | 0.056667 | 0.909333 |
| temporal_shuffle_seed_repeat | 0 | 0.022874 | 0.046667 | 0.820667 |
| temporal_shuffle_seed_repeat | 1 | 0.022305 | 0.042000 | 0.816000 |
| temporal_shuffle_seed_repeat | 2 | 0.022700 | 0.040667 | 0.817333 |

## Class-Level Candidates

| label_id | label_name | sample_count | strongest_perturbation | mean_cosine_distance_across_perturbations |
| --- | --- | --- | --- | --- |
| 48 | Turning the camera right while filming something | 30 | temporal_shuffle | 0.044558 |
| 47 | Turning the camera left while filming something | 30 | temporal_shuffle | 0.038754 |
| 42 | Throwing something in the air and letting it fall | 30 | single_frame | 0.026943 |
| 10 | Lifting something up completely without letting it drop down | 30 | temporal_shuffle | 0.015515 |
| 2 | Covering something with something | 30 | single_frame | 0.014965 |
| 16 | Moving something up | 30 | temporal_shuffle | 0.014961 |
| 8 | Letting something roll along a flat surface | 30 | single_frame | 0.014727 |
| 12 | Moving something across a surface until it falls down | 30 | temporal_shuffle | 0.014556 |
| 31 | Pushing something onto something | 30 | temporal_shuffle | 0.013622 |
| 36 | Putting something underneath something | 30 | single_frame | 0.013137 |

## Qualitative Sample Candidates

Largest embedding shift:

| perturbation | video_id | label_name | cosine_distance | prediction_changed |
| --- | --- | --- | --- | --- |
| temporal_shuffle | 166929 | Turning the camera right while filming something | 0.274345 | True |
