# SSV2 × frozen dinov2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| freeze-tail-low | motion | curve | low | 0.002897 | 0.002715 | 0.003086 | -0.002000 | -0.008000 | 0.004000 | 0.006667 | 0.000667 |
| freeze-tail-mid | motion | curve | mid | 0.020854 | 0.019792 | 0.022039 | 0.005333 | -0.005333 | 0.018000 | 0.028667 | -0.001333 |
| freeze-tail-high | motion | curve | high | 0.055140 | 0.052384 | 0.058008 | 0.020667 | 0.007983 | 0.034667 | 0.050000 | 0.007333 |
| color-low | appearance | curve | low | 0.000062 | 0.000059 | 0.000065 | -0.000667 | -0.002000 | 0.000000 | 0.000000 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000836 | 0.000806 | 0.000867 | -0.001333 | -0.004667 | 0.002000 | 0.002000 | -0.000667 |
| color-high | appearance | curve | high | 0.007709 | 0.007456 | 0.007961 | -0.004667 | -0.012667 | 0.003333 | 0.011333 | 0.000000 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.048301 | 0.046690 | 0.049982 | 0.006667 | -0.006000 | 0.019350 | 0.037333 | 0.012000 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 208338 | Pretending or failing to wipe something off of something | 0.417915 | 30 | 24 | False |
| freeze-tail-high | 37955 | Spinning something that quickly stops spinning | 0.376904 | 30 | 30 | False |
| freeze-tail-high | 7675 | Pretending to put something onto something | 0.375259 | 20 | 43 | False |
| freeze-tail-high | 60004 | Putting something underneath something | 0.363095 | 3 | 43 | False |
| freeze-tail-high | 143160 | Moving something and something away from each other | 0.348994 | 35 | 19 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 62508 | Dropping something onto something | 0.257454 | 6 | 4 |
| freeze-tail-high | 145252 | Pretending or failing to wipe something off of something | 0.251576 | 24 | 30 |
| freeze-tail-high | 23681 | Putting something onto something else that cannot support it so it falls down | 0.183612 | 34 | 22 |
| freeze-tail-high | 109320 | Tipping something with something in it over, so something in it falls out | 0.176050 | 43 | 5 |
| freeze-tail-high | 129049 | Throwing something in the air and letting it fall | 0.173114 | 42 | 37 |

## Data quality and failures

Fail-fast extraction was used. All extraction artifacts succeeded: True. 
All sampled frame-index and sampling-strategy checks passed: True.

| artifact_label | dataset_size | successful_samples | failed_samples |
| --- | --- | --- | --- |
| train_original | 5000 | 5000 | 0 |
| heldout_original | 1500 | 1500 | 0 |
| temporal-shuffle-mid | 1500 | 1500 | 0 |
| freeze-tail-low | 1500 | 1500 | 0 |
| freeze-tail-mid | 1500 | 1500 | 0 |
| freeze-tail-high | 1500 | 1500 | 0 |
| color-low | 1500 | 1500 | 0 |
| color-mid | 1500 | 1500 | 0 |
| color-high | 1500 | 1500 | 0 |
| spatial-blur-mid | 1500 | 1500 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one motion-oriented SSV2 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
