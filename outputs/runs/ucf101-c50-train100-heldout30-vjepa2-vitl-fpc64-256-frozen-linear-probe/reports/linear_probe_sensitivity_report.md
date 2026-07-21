# UCF101 × frozen V-JEPA2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.072891 | 0.070973 | 0.074715 | 0.076000 | 0.062667 | 0.090667 | 0.077333 | 0.022000 |
| freeze-tail-low | motion | curve | low | 0.011832 | 0.011313 | 0.012349 | 0.003333 | 0.000000 | 0.007333 | 0.004000 | 0.001333 |
| freeze-tail-mid | motion | curve | mid | 0.030335 | 0.029292 | 0.031449 | 0.014667 | 0.008000 | 0.022000 | 0.017333 | 0.007333 |
| freeze-tail-high | motion | curve | high | 0.073860 | 0.071825 | 0.075881 | 0.049333 | 0.038000 | 0.061333 | 0.053333 | 0.029333 |
| color-low | appearance | curve | low | 0.000036 | 0.000034 | 0.000039 | 0.000667 | 0.000000 | 0.002000 | 0.000667 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000769 | 0.000747 | 0.000792 | -0.000667 | -0.003333 | 0.001333 | 0.000667 | 0.003333 |
| color-high | appearance | curve | high | 0.009892 | 0.009598 | 0.010178 | 0.005333 | 0.000667 | 0.010017 | 0.006667 | 0.003333 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.151218 | 0.149087 | 0.153396 | 0.164000 | 0.145333 | 0.183333 | 0.165333 | 0.184000 |
| rgb-quantization-low | appearance | curve | low | 0.017179 | 0.016461 | 0.017872 | 0.005333 | 0.000000 | 0.011333 | 0.010000 | 0.005333 |
| rgb-quantization-mid | appearance | curve | mid | 0.047813 | 0.046328 | 0.049269 | 0.025333 | 0.016000 | 0.034667 | 0.030000 | 0.023333 |
| rgb-quantization-high | appearance | curve | high | 0.110252 | 0.107766 | 0.112452 | 0.103333 | 0.088650 | 0.118667 | 0.106667 | 0.110000 |
| solarization-low | appearance | curve | low | 0.046410 | 0.044803 | 0.048112 | 0.022000 | 0.013983 | 0.031333 | 0.026000 | 0.026000 |
| solarization-mid | appearance | curve | mid | 0.046650 | 0.045146 | 0.048090 | 0.029333 | 0.020667 | 0.039333 | 0.033333 | 0.028667 |
| solarization-high | appearance | curve | high | 0.102558 | 0.100743 | 0.104393 | 0.102000 | 0.087317 | 0.118667 | 0.105333 | 0.086000 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | v_Kayaking_g24_c02 | Kayaking | 0.363197 | 25 | 14 | True |
| rgb-quantization-high | v_Kayaking_g20_c06 | Kayaking | 0.362824 | 25 | 23 | True |
| rgb-quantization-high | v_Kayaking_g23_c03 | Kayaking | 0.360466 | 25 | 14 | True |
| rgb-quantization-high | v_Kayaking_g03_c02 | Kayaking | 0.349176 | 25 | 16 | True |
| rgb-quantization-high | v_Kayaking_g24_c03 | Kayaking | 0.333973 | 25 | 14 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | v_Kayaking_g24_c02 | Kayaking | 0.363197 | 25 | 14 |
| rgb-quantization-high | v_Kayaking_g20_c06 | Kayaking | 0.362824 | 25 | 23 |
| rgb-quantization-high | v_Kayaking_g23_c03 | Kayaking | 0.360466 | 25 | 14 |
| rgb-quantization-high | v_Kayaking_g03_c02 | Kayaking | 0.349176 | 25 | 16 |
| rgb-quantization-high | v_Kayaking_g24_c03 | Kayaking | 0.333973 | 25 | 14 |

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
