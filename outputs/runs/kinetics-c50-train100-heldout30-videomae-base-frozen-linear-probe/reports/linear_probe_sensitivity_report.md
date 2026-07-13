# Kinetics × frozen VideoMAE linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.000657 | 0.000597 | 0.000722 | -0.010000 | -0.020017 | -0.001333 | 0.012000 | 0.001333 |
| freeze-tail-mid | motion | curve | mid | 0.003293 | 0.003030 | 0.003561 | 0.000667 | -0.014000 | 0.015333 | 0.046000 | 0.002000 |
| freeze-tail-high | motion | curve | high | 0.008685 | 0.008113 | 0.009308 | 0.028000 | 0.008667 | 0.046683 | 0.083333 | 0.026667 |
| color-low | appearance | curve | low | 0.000002 | 0.000001 | 0.000002 | -0.000667 | -0.004000 | 0.002667 | 0.002000 | 0.000667 |
| color-mid | appearance | curve | mid | 0.000040 | 0.000037 | 0.000042 | 0.001333 | -0.006667 | 0.010000 | 0.012000 | 0.002000 |
| color-high | appearance | curve | high | 0.000357 | 0.000344 | 0.000370 | 0.010667 | -0.003333 | 0.025333 | 0.044000 | 0.011333 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.002634 | 0.002472 | 0.002788 | 0.025333 | 0.010667 | 0.040667 | 0.057333 | 0.015333 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.018077 | 0.016911 | 0.019280 | 0.148000 | 0.126650 | 0.172017 | 0.191333 | 0.017333 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | -QJz_YH0TMA_000335_000345 | barbequing | 0.200273 | 31 | 48 | False |
| temporal-shuffle-mid | 1UqN3hbK7x4_000003_000013 | applauding | 0.167358 | 3 | 3 | False |
| temporal-shuffle-mid | 1CeODw5s6Ho_000018_000028 | auctioning | 0.140671 | 3 | 48 | False |
| temporal-shuffle-mid | AUpo8x-nXBs_000017_000027 | applauding | 0.139953 | 40 | 16 | False |
| temporal-shuffle-mid | 2F-2vf-LdUw_000056_000066 | bungee jumping | 0.138205 | 47 | 1 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | UyWbZRdnXKc_000086_000096 | bee keeping | 0.132915 | 17 | 0 |
| temporal-shuffle-mid | 5H1VnOA4wfE_000029_000039 | bungee jumping | 0.124389 | 40 | 39 |
| temporal-shuffle-mid | 29ZTZQtOTsg_000003_000013 | bungee jumping | 0.115719 | 40 | 17 |
| freeze-tail-high | 6wmzxqaEBd4_000018_000028 | capoeira | 0.112270 | 43 | 3 |
| temporal-shuffle-mid | TPUzY64CdZk_000359_000369 | bee keeping | 0.106364 | 17 | 39 |

## Data quality and failures

Fail-fast extraction was used. All extraction artifacts succeeded: True. 
All sampled frame-index and sampling-strategy checks passed: True.

| artifact_label | dataset_size | successful_samples | failed_samples |
| --- | --- | --- | --- |
| train_original | 5000 | 5000 | 0 |
| heldout_original | 1500 | 1500 | 0 |
| freeze-tail-low | 1500 | 1500 | 0 |
| freeze-tail-mid | 1500 | 1500 | 0 |
| freeze-tail-high | 1500 | 1500 | 0 |
| color-low | 1500 | 1500 | 0 |
| color-mid | 1500 | 1500 | 0 |
| color-high | 1500 | 1500 | 0 |
| spatial-blur-mid | 1500 | 1500 | 0 |
| temporal-shuffle-mid | 1500 | 1500 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one large-scale web-video Kinetics dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
