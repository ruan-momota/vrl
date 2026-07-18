# SSV2 × frozen VideoMAE linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.023021 | 0.021700 | 0.024457 | 0.180000 | 0.158000 | 0.202000 | 0.202667 | 0.046000 |
| freeze-tail-low | motion | curve | low | 0.001039 | 0.000958 | 0.001120 | 0.021333 | 0.008667 | 0.034000 | 0.042000 | 0.012667 |
| freeze-tail-mid | motion | curve | mid | 0.004984 | 0.004658 | 0.005332 | 0.056667 | 0.038667 | 0.075350 | 0.092667 | 0.036000 |
| freeze-tail-high | motion | curve | high | 0.012561 | 0.011857 | 0.013305 | 0.126000 | 0.106650 | 0.147367 | 0.156667 | 0.062667 |
| color-low | appearance | curve | low | 0.000001 | 0.000001 | 0.000001 | 0.002000 | -0.000667 | 0.005333 | 0.002667 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000030 | 0.000030 | 0.000031 | 0.000667 | -0.005333 | 0.006667 | 0.008000 | 0.000667 |
| color-high | appearance | curve | high | 0.000288 | 0.000280 | 0.000296 | 0.010000 | 0.000000 | 0.020667 | 0.026667 | -0.004667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.006473 | 0.006321 | 0.006640 | 0.033333 | 0.014650 | 0.053333 | 0.084000 | 0.006667 |
| rgb-quantization-low | appearance | curve | low | 0.006635 | 0.006346 | 0.006913 | 0.068667 | 0.050667 | 0.088667 | 0.106000 | 0.016000 |
| rgb-quantization-mid | appearance | curve | mid | 0.016110 | 0.015573 | 0.016607 | 0.100667 | 0.080667 | 0.121350 | 0.139333 | 0.022667 |
| rgb-quantization-high | appearance | curve | high | 0.029840 | 0.029104 | 0.030603 | 0.121333 | 0.101333 | 0.144017 | 0.159333 | 0.035333 |
| solarization-low | appearance | curve | low | 0.007164 | 0.006767 | 0.007593 | 0.060667 | 0.043317 | 0.079333 | 0.093333 | 0.010667 |
| solarization-mid | appearance | curve | mid | 0.009520 | 0.008938 | 0.010127 | 0.050667 | 0.031983 | 0.070000 | 0.092000 | 0.008000 |
| solarization-high | appearance | curve | high | 0.018787 | 0.018026 | 0.019570 | 0.075333 | 0.056650 | 0.095350 | 0.124000 | 0.017333 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| solarization-high | 30163 | Plugging something into something | 0.186752 | 9 | 19 | False |
| freeze-tail-high | 129049 | Throwing something in the air and letting it fall | 0.174796 | 42 | 42 | False |
| temporal-shuffle-mid | 46375 | Pouring something into something until it overflows | 0.168495 | 22 | 24 | False |
| temporal-shuffle-mid | 166665 | Turning the camera right while filming something | 0.167460 | 48 | 24 | True |
| freeze-tail-high | 48157 | Throwing something in the air and letting it fall | 0.161370 | 9 | 9 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 166665 | Turning the camera right while filming something | 0.167460 | 48 | 24 |
| temporal-shuffle-mid | 47530 | Turning the camera right while filming something | 0.160047 | 48 | 24 |
| temporal-shuffle-mid | 158563 | Turning the camera left while filming something | 0.157502 | 47 | 24 |
| temporal-shuffle-mid | 164973 | Throwing something in the air and letting it fall | 0.156913 | 42 | 24 |
| temporal-shuffle-mid | 198540 | Throwing something in the air and letting it fall | 0.154912 | 42 | 24 |

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
