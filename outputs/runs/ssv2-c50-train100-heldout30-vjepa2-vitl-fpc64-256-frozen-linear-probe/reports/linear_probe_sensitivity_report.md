# SSV2 × frozen V-JEPA2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.076698 | 0.075100 | 0.078346 | 0.240000 | 0.213333 | 0.266667 | 0.274667 | 0.060000 |
| freeze-tail-low | motion | curve | low | 0.014287 | 0.013549 | 0.015092 | 0.005333 | -0.009333 | 0.020000 | 0.045333 | 0.002000 |
| freeze-tail-mid | motion | curve | mid | 0.047764 | 0.046154 | 0.049563 | 0.094000 | 0.072000 | 0.116667 | 0.141333 | 0.038000 |
| freeze-tail-high | motion | curve | high | 0.104644 | 0.102345 | 0.107212 | 0.235333 | 0.209333 | 0.262667 | 0.284667 | 0.090000 |
| color-low | appearance | curve | low | 0.000042 | 0.000040 | 0.000044 | 0.000667 | -0.001333 | 0.002667 | 0.001333 | -0.002000 |
| color-mid | appearance | curve | mid | 0.000878 | 0.000856 | 0.000902 | 0.000000 | -0.006667 | 0.006667 | 0.008667 | -0.002000 |
| color-high | appearance | curve | high | 0.009058 | 0.008827 | 0.009305 | -0.006667 | -0.019333 | 0.006000 | 0.026000 | -0.001333 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.104611 | 0.103070 | 0.106172 | 0.050000 | 0.029983 | 0.071333 | 0.107333 | 0.055333 |
| rgb-quantization-low | appearance | curve | low | 0.042195 | 0.040802 | 0.043608 | 0.019333 | 0.002667 | 0.036000 | 0.064667 | 0.026000 |
| rgb-quantization-mid | appearance | curve | mid | 0.079838 | 0.077701 | 0.082027 | 0.071333 | 0.050667 | 0.091333 | 0.119333 | 0.040000 |
| rgb-quantization-high | appearance | curve | high | 0.135049 | 0.132277 | 0.137777 | 0.133333 | 0.108650 | 0.157350 | 0.190667 | 0.085333 |
| solarization-low | appearance | curve | low | 0.052573 | 0.050877 | 0.054300 | 0.042000 | 0.022667 | 0.063333 | 0.100000 | 0.022000 |
| solarization-mid | appearance | curve | mid | 0.042078 | 0.040889 | 0.043314 | 0.031333 | 0.014000 | 0.049333 | 0.076000 | 0.020000 |
| solarization-high | appearance | curve | high | 0.093180 | 0.091413 | 0.095073 | 0.097333 | 0.074000 | 0.120667 | 0.159333 | 0.044667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 84921 | Putting something into something | 0.403661 | 32 | 4 | False |
| rgb-quantization-high | 18455 | Putting number of something onto something | 0.399678 | 6 | 17 | False |
| solarization-high | 131038 | Trying to pour something into something, but missing so it spills next to it | 0.373209 | 21 | 22 | False |
| freeze-tail-high | 178445 | Turning the camera right while filming something | 0.371763 | 48 | 4 | True |
| freeze-tail-mid | 178445 | Turning the camera right while filming something | 0.350363 | 48 | 37 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 178445 | Turning the camera right while filming something | 0.371763 | 48 | 4 |
| freeze-tail-mid | 178445 | Turning the camera right while filming something | 0.350363 | 48 | 37 |
| freeze-tail-high | 214066 | Turning the camera left while filming something | 0.321972 | 47 | 7 |
| freeze-tail-high | 195700 | Throwing something in the air and letting it fall | 0.316217 | 42 | 7 |
| rgb-quantization-high | 92635 | Pouring something out of something | 0.311666 | 23 | 4 |

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
- This report covers one frozen model × one motion-oriented SSV2 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
