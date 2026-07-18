# SSV2 × frozen SlowFast linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.221834 | 0.216572 | 0.227351 | 0.205333 | 0.181317 | 0.228000 | 0.233333 | 0.085333 |
| freeze-tail-low | motion | curve | low | 0.042602 | 0.041313 | 0.043829 | 0.034000 | 0.017333 | 0.050000 | 0.069333 | 0.009333 |
| freeze-tail-mid | motion | curve | mid | 0.120433 | 0.117870 | 0.123141 | 0.096000 | 0.074000 | 0.116683 | 0.140000 | 0.036000 |
| freeze-tail-high | motion | curve | high | 0.229698 | 0.225756 | 0.234211 | 0.194000 | 0.167983 | 0.216000 | 0.230000 | 0.089333 |
| color-low | appearance | curve | low | 0.000095 | 0.000091 | 0.000099 | -0.002000 | -0.005333 | 0.001333 | 0.001333 | -0.002667 |
| color-mid | appearance | curve | mid | 0.002014 | 0.001944 | 0.002079 | 0.000667 | -0.006667 | 0.007350 | 0.010000 | -0.001333 |
| color-high | appearance | curve | high | 0.018532 | 0.017896 | 0.019138 | 0.005333 | -0.006017 | 0.017333 | 0.028667 | -0.010667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.076139 | 0.074207 | 0.078135 | 0.000667 | -0.016000 | 0.020000 | 0.064667 | -0.011333 |
| rgb-quantization-low | appearance | curve | low | 0.084670 | 0.080852 | 0.088786 | 0.034000 | 0.016000 | 0.050017 | 0.079333 | 0.001333 |
| rgb-quantization-mid | appearance | curve | mid | 0.224328 | 0.217839 | 0.230738 | 0.092000 | 0.067333 | 0.113350 | 0.154000 | 0.056667 |
| rgb-quantization-high | appearance | curve | high | 0.408799 | 0.402293 | 0.415660 | 0.169333 | 0.141333 | 0.193333 | 0.230000 | 0.117333 |
| solarization-low | appearance | curve | low | 0.197609 | 0.190982 | 0.204070 | 0.082000 | 0.060000 | 0.102000 | 0.138000 | 0.062667 |
| solarization-mid | appearance | curve | mid | 0.184459 | 0.178971 | 0.189910 | 0.062667 | 0.040650 | 0.084000 | 0.133333 | 0.035333 |
| solarization-high | appearance | curve | high | 0.346081 | 0.340686 | 0.351873 | 0.158000 | 0.132000 | 0.182017 | 0.219333 | 0.110667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 32943 | Closing something | 0.760801 | 17 | 22 | False |
| freeze-tail-high | 18886 | Letting something roll along a flat surface | 0.755089 | 47 | 47 | False |
| rgb-quantization-high | 70264 | Moving something across a surface until it falls down | 0.740832 | 27 | 24 | False |
| rgb-quantization-high | 220728 | Trying to bend something unbendable so nothing happens | 0.714352 | 45 | 24 | True |
| rgb-quantization-high | 11178 | Poking a hole into something soft | 0.710094 | 7 | 7 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | 220728 | Trying to bend something unbendable so nothing happens | 0.714352 | 45 | 24 |
| rgb-quantization-high | 37062 | Digging something out of something | 0.705753 | 3 | 27 |
| rgb-quantization-high | 97298 | Pretending to sprinkle air onto something | 0.685413 | 28 | 10 |
| rgb-quantization-high | 114127 | Unfolding something | 0.682472 | 49 | 45 |
| rgb-quantization-high | 81747 | Digging something out of something | 0.682180 | 3 | 19 |

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
