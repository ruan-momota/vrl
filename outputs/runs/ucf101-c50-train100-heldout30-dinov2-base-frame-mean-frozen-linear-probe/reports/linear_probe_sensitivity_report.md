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
| rgb-quantization-low | appearance | curve | low | 0.021192 | 0.019987 | 0.022414 | 0.000667 | -0.002667 | 0.004000 | 0.002667 | 0.004000 |
| rgb-quantization-mid | appearance | curve | mid | 0.081287 | 0.077985 | 0.084563 | 0.011333 | 0.004667 | 0.018000 | 0.014000 | 0.008000 |
| rgb-quantization-high | appearance | curve | high | 0.243048 | 0.236219 | 0.249647 | 0.064667 | 0.051983 | 0.077333 | 0.066667 | 0.034000 |
| solarization-low | appearance | curve | low | 0.110982 | 0.106594 | 0.115630 | 0.012667 | 0.005333 | 0.020000 | 0.016667 | 0.008000 |
| solarization-mid | appearance | curve | mid | 0.123449 | 0.118873 | 0.128201 | 0.017333 | 0.009333 | 0.024667 | 0.020000 | 0.007333 |
| solarization-high | appearance | curve | high | 0.313513 | 0.306568 | 0.320614 | 0.116000 | 0.100667 | 0.133333 | 0.118000 | 0.079333 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| solarization-high | v_FrontCrawl_g14_c02 | FrontCrawl | 0.840537 | 19 | 14 | True |
| solarization-high | v_Mixing_g01_c01 | Mixing | 0.824508 | 27 | 18 | True |
| solarization-high | v_FrontCrawl_g13_c01 | FrontCrawl | 0.816223 | 19 | 14 | True |
| solarization-high | v_FrontCrawl_g14_c05 | FrontCrawl | 0.801174 | 19 | 17 | True |
| solarization-high | v_CliffDiving_g19_c06 | CliffDiving | 0.794752 | 14 | 14 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| solarization-high | v_FrontCrawl_g14_c02 | FrontCrawl | 0.840537 | 19 | 14 |
| solarization-high | v_Mixing_g01_c01 | Mixing | 0.824508 | 27 | 18 |
| solarization-high | v_FrontCrawl_g13_c01 | FrontCrawl | 0.816223 | 19 | 14 |
| solarization-high | v_FrontCrawl_g14_c05 | FrontCrawl | 0.801174 | 19 | 17 |
| solarization-high | v_BenchPress_g23_c03 | BenchPress | 0.793141 | 7 | 18 |

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
| rgb-quantization-low | 1500 | 1500 | 0 |
| rgb-quantization-mid | 1500 | 1500 | 0 |
| rgb-quantization-high | 1500 | 1500 | 0 |
| solarization-low | 1500 | 1500 | 0 |
| solarization-mid | 1500 | 1500 | 0 |
| solarization-high | 1500 | 1500 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one appearance-rich/context-correlated UCF101 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
