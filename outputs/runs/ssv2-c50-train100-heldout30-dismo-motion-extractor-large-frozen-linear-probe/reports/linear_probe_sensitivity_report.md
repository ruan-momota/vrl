# SSV2 × frozen dismo linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.409785 | 0.399946 | 0.419568 | 0.249333 | 0.224000 | 0.276000 | 0.284667 | 0.198667 |
| freeze-tail-low | motion | curve | low | 0.006977 | 0.006719 | 0.007265 | 0.018667 | 0.006000 | 0.030667 | 0.038000 | 0.013333 |
| freeze-tail-mid | motion | curve | mid | 0.145168 | 0.141277 | 0.149090 | 0.113333 | 0.089333 | 0.138017 | 0.162667 | 0.158667 |
| freeze-tail-high | motion | curve | high | 0.396311 | 0.387057 | 0.406455 | 0.307333 | 0.279317 | 0.334000 | 0.330667 | 0.274000 |
| color-low | appearance | curve | low | 0.000046 | 0.000043 | 0.000049 | 0.000667 | 0.000000 | 0.002000 | 0.000667 | 0.001333 |
| color-mid | appearance | curve | mid | 0.000983 | 0.000942 | 0.001027 | 0.000667 | -0.005350 | 0.006667 | 0.008000 | 0.000000 |
| color-high | appearance | curve | high | 0.008839 | 0.008533 | 0.009150 | 0.012000 | 0.002000 | 0.021350 | 0.024667 | 0.004000 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.021687 | 0.021175 | 0.022187 | 0.011333 | -0.003333 | 0.024667 | 0.044000 | 0.018000 |
| rgb-quantization-low | appearance | curve | low | 0.021910 | 0.021223 | 0.022641 | 0.018000 | 0.002667 | 0.032000 | 0.048667 | 0.010000 |
| rgb-quantization-mid | appearance | curve | mid | 0.044115 | 0.042644 | 0.045519 | 0.032667 | 0.014667 | 0.048683 | 0.073333 | 0.032000 |
| rgb-quantization-high | appearance | curve | high | 0.090083 | 0.087330 | 0.092691 | 0.079333 | 0.058667 | 0.100000 | 0.122000 | 0.046000 |
| solarization-low | appearance | curve | low | 0.080405 | 0.077458 | 0.083544 | 0.098667 | 0.078667 | 0.120000 | 0.136667 | 0.037333 |
| solarization-mid | appearance | curve | mid | 0.032851 | 0.031524 | 0.034155 | 0.030667 | 0.013983 | 0.049350 | 0.070000 | 0.009333 |
| solarization-high | appearance | curve | high | 0.112800 | 0.110241 | 0.115428 | 0.096667 | 0.074667 | 0.119333 | 0.148000 | 0.044000 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 129049 | Throwing something in the air and letting it fall | 1.145330 | 42 | 37 | True |
| freeze-tail-high | 27569 | Throwing something in the air and letting it fall | 1.135402 | 42 | 9 | True |
| temporal-shuffle-mid | 219864 | Moving something across a surface until it falls down | 1.131960 | 31 | 24 | False |
| temporal-shuffle-mid | 171614 | Closing something | 1.084451 | 31 | 49 | False |
| temporal-shuffle-mid | 98494 | Turning the camera right while filming something | 1.083011 | 48 | 24 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 129049 | Throwing something in the air and letting it fall | 1.145330 | 42 | 37 |
| freeze-tail-high | 27569 | Throwing something in the air and letting it fall | 1.135402 | 42 | 9 |
| temporal-shuffle-mid | 98494 | Turning the camera right while filming something | 1.083011 | 48 | 24 |
| temporal-shuffle-mid | 164567 | Moving something across a surface until it falls down | 1.074289 | 12 | 32 |
| temporal-shuffle-mid | 163769 | Turning the camera left while filming something | 1.072824 | 47 | 24 |

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
