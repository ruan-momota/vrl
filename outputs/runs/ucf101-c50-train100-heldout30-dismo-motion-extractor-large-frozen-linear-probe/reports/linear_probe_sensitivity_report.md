# UCF101 × frozen dismo linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.340658 | 0.329243 | 0.352060 | 0.365333 | 0.338650 | 0.393333 | 0.398000 | 0.371333 |
| freeze-tail-low | motion | curve | low | 0.003217 | 0.003089 | 0.003348 | 0.006667 | -0.000667 | 0.014000 | 0.014000 | 0.010000 |
| freeze-tail-mid | motion | curve | mid | 0.051890 | 0.050225 | 0.053839 | 0.117333 | 0.096000 | 0.139333 | 0.155333 | 0.258667 |
| freeze-tail-high | motion | curve | high | 0.163702 | 0.158096 | 0.169668 | 0.497333 | 0.468000 | 0.525350 | 0.525333 | 0.654667 |
| color-low | appearance | curve | low | 0.000045 | 0.000040 | 0.000052 | 0.002000 | -0.002000 | 0.006000 | 0.004000 | 0.000667 |
| color-mid | appearance | curve | mid | 0.000618 | 0.000593 | 0.000645 | 0.004000 | -0.003333 | 0.011333 | 0.012000 | 0.004000 |
| color-high | appearance | curve | high | 0.006081 | 0.005882 | 0.006280 | 0.012667 | 0.000667 | 0.024667 | 0.034000 | 0.014667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.021412 | 0.020851 | 0.022013 | 0.050000 | 0.033317 | 0.066683 | 0.079333 | 0.035333 |
| rgb-quantization-low | appearance | curve | low | 0.008215 | 0.007978 | 0.008464 | 0.012667 | 0.001333 | 0.024017 | 0.033333 | 0.002000 |
| rgb-quantization-mid | appearance | curve | mid | 0.017310 | 0.016665 | 0.017981 | 0.024000 | 0.010667 | 0.037333 | 0.054000 | 0.014667 |
| rgb-quantization-high | appearance | curve | high | 0.042704 | 0.041539 | 0.043725 | 0.072667 | 0.053333 | 0.092017 | 0.114667 | 0.060667 |
| solarization-low | appearance | curve | low | 0.036909 | 0.035388 | 0.038584 | 0.073333 | 0.054000 | 0.092000 | 0.111333 | 0.024667 |
| solarization-mid | appearance | curve | mid | 0.025647 | 0.024438 | 0.026825 | 0.033333 | 0.018000 | 0.049350 | 0.066000 | 0.017333 |
| solarization-high | appearance | curve | high | 0.074731 | 0.072950 | 0.076545 | 0.195333 | 0.174000 | 0.220000 | 0.223333 | 0.112000 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | v_BasketballDunk_g01_c04 | BasketballDunk | 1.124464 | 6 | 26 | True |
| temporal-shuffle-mid | v_IceDancing_g20_c02 | IceDancing | 1.117706 | 23 | 26 | True |
| temporal-shuffle-mid | v_Skiing_g02_c05 | Skiing | 1.109032 | 42 | 42 | False |
| temporal-shuffle-mid | v_Bowling_g06_c06 | Bowling | 1.041157 | 10 | 26 | True |
| temporal-shuffle-mid | v_PoleVault_g20_c03 | PoleVault | 1.034327 | 35 | 35 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | v_BasketballDunk_g01_c04 | BasketballDunk | 1.124464 | 6 | 26 |
| temporal-shuffle-mid | v_IceDancing_g20_c02 | IceDancing | 1.117706 | 23 | 26 |
| temporal-shuffle-mid | v_Bowling_g06_c06 | Bowling | 1.041157 | 10 | 26 |
| temporal-shuffle-mid | v_Skiing_g05_c02 | Skiing | 1.015618 | 42 | 26 |
| temporal-shuffle-mid | v_PoleVault_g08_c03 | PoleVault | 1.010298 | 35 | 45 |

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
