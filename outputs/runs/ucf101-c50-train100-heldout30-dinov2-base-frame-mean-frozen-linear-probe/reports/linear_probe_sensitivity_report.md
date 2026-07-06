# UCF101 × frozen dinov2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| freeze-tail-low | motion | curve | low | 0.001216 | 0.001150 | 0.001292 | 0.001333 | 0.000000 | 0.003333 | 0.001333 | 0.000667 |
| freeze-tail-mid | motion | curve | mid | 0.008164 | 0.007723 | 0.008622 | 0.000000 | -0.002000 | 0.002000 | 0.000667 | -0.001333 |
| freeze-tail-high | motion | curve | high | 0.021402 | 0.020399 | 0.022467 | 0.002667 | 0.000000 | 0.006000 | 0.003333 | 0.001333 |
| color-low | appearance | curve | low | 0.000052 | 0.000049 | 0.000056 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000667 |
| color-mid | appearance | curve | mid | 0.000745 | 0.000707 | 0.000781 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| color-high | appearance | curve | high | 0.007478 | 0.007120 | 0.007851 | 0.002000 | 0.000000 | 0.004667 | 0.002000 | -0.000667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.039373 | 0.037947 | 0.040932 | 0.002000 | -0.001333 | 0.006000 | 0.004000 | 0.004667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| spatial-blur-mid | v_SalsaSpin_g08_c01 | SalsaSpin | 0.335810 | 39 | 39 | False |
| spatial-blur-mid | v_JumpRope_g04_c01 | JumpRope | 0.225290 | 24 | 24 | False |
| spatial-blur-mid | v_Archery_g14_c04 | Archery | 0.213725 | 1 | 1 | False |
| freeze-tail-high | v_CliffDiving_g02_c01 | CliffDiving | 0.195767 | 14 | 14 | False |
| spatial-blur-mid | v_Archery_g11_c06 | Archery | 0.191828 | 1 | 1 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | v_Basketball_g06_c04 | Basketball | 0.119759 | 5 | 35 |
| freeze-tail-high | v_Shotput_g21_c01 | Shotput | 0.110130 | 41 | 26 |
| spatial-blur-mid | v_HammerThrow_g03_c01 | HammerThrow | 0.087475 | 21 | 42 |
| spatial-blur-mid | v_PlayingFlute_g18_c02 | PlayingFlute | 0.083627 | 32 | 24 |
| freeze-tail-high | v_FrontCrawl_g23_c03 | FrontCrawl | 0.079259 | 19 | 17 |

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
- This report covers one frozen model × one appearance-rich/context-correlated UCF101 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
