# UCF101 × frozen SlowFast linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.161367 | 0.156806 | 0.166179 | 0.046000 | 0.035333 | 0.057333 | 0.048667 | 0.045333 |
| freeze-tail-low | motion | curve | low | 0.028714 | 0.027742 | 0.029901 | 0.003333 | 0.000000 | 0.006667 | 0.004000 | 0.006667 |
| freeze-tail-mid | motion | curve | mid | 0.086512 | 0.084278 | 0.088736 | 0.009333 | 0.003333 | 0.015333 | 0.011333 | 0.014000 |
| freeze-tail-high | motion | curve | high | 0.176553 | 0.172811 | 0.180520 | 0.038667 | 0.029317 | 0.048667 | 0.040667 | 0.052667 |
| color-low | appearance | curve | low | 0.000060 | 0.000056 | 0.000065 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | -0.000667 |
| color-mid | appearance | curve | mid | 0.001180 | 0.001127 | 0.001240 | 0.000000 | -0.002000 | 0.002000 | 0.000667 | -0.000667 |
| color-high | appearance | curve | high | 0.010949 | 0.010504 | 0.011422 | 0.000000 | -0.003333 | 0.003333 | 0.002000 | 0.002000 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.060820 | 0.058867 | 0.062900 | 0.014000 | 0.008000 | 0.021333 | 0.016000 | 0.013333 |
| rgb-quantization-low | appearance | curve | low | 0.025029 | 0.023647 | 0.026601 | 0.008000 | 0.003333 | 0.013333 | 0.008667 | 0.005333 |
| rgb-quantization-mid | appearance | curve | mid | 0.095569 | 0.091490 | 0.099596 | 0.038667 | 0.028000 | 0.050000 | 0.042000 | 0.029333 |
| rgb-quantization-high | appearance | curve | high | 0.255998 | 0.248759 | 0.263048 | 0.178000 | 0.159333 | 0.198667 | 0.180667 | 0.177333 |
| solarization-low | appearance | curve | low | 0.142013 | 0.136145 | 0.147651 | 0.062667 | 0.049983 | 0.075333 | 0.065333 | 0.050667 |
| solarization-mid | appearance | curve | mid | 0.155623 | 0.149761 | 0.160762 | 0.075333 | 0.062000 | 0.088667 | 0.078000 | 0.055333 |
| solarization-high | appearance | curve | high | 0.316851 | 0.310213 | 0.323796 | 0.264667 | 0.244000 | 0.287333 | 0.266000 | 0.266667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | v_Kayaking_g24_c02 | Kayaking | 0.730678 | 25 | 19 | True |
| solarization-high | v_SoccerPenalty_g17_c03 | SoccerPenalty | 0.725260 | 44 | 19 | True |
| rgb-quantization-high | v_Kayaking_g08_c01 | Kayaking | 0.698647 | 25 | 19 | True |
| rgb-quantization-high | v_Skiing_g07_c04 | Skiing | 0.694392 | 42 | 19 | True |
| rgb-quantization-high | v_Kayaking_g15_c03 | Kayaking | 0.691227 | 25 | 37 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | v_Kayaking_g24_c02 | Kayaking | 0.730678 | 25 | 19 |
| solarization-high | v_SoccerPenalty_g17_c03 | SoccerPenalty | 0.725260 | 44 | 19 |
| rgb-quantization-high | v_Kayaking_g08_c01 | Kayaking | 0.698647 | 25 | 19 |
| rgb-quantization-high | v_Skiing_g07_c04 | Skiing | 0.694392 | 42 | 19 |
| rgb-quantization-high | v_Kayaking_g15_c03 | Kayaking | 0.691227 | 25 | 37 |

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
