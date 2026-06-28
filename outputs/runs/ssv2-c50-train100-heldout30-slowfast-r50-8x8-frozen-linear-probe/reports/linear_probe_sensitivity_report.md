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

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 32943 | Closing something | 0.760801 | 17 | 22 | False |
| freeze-tail-high | 18886 | Letting something roll along a flat surface | 0.755089 | 47 | 47 | False |
| temporal-shuffle-mid | 108191 | Turning the camera left while filming something | 0.710005 | 48 | 22 | False |
| temporal-shuffle-mid | 69778 | Turning the camera left while filming something | 0.680679 | 47 | 22 | True |
| temporal-shuffle-mid | 49938 | Pretending or failing to wipe something off of something | 0.659309 | 38 | 22 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 69778 | Turning the camera left while filming something | 0.680679 | 47 | 22 |
| freeze-tail-high | 162861 | Tearing something into two pieces | 0.606190 | 40 | 0 |
| temporal-shuffle-mid | 448 | Opening something | 0.586172 | 17 | 38 |
| temporal-shuffle-mid | 45627 | Trying to bend something unbendable so nothing happens | 0.535223 | 45 | 38 |
| temporal-shuffle-mid | 176988 | Putting something into something | 0.515736 | 33 | 7 |

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
