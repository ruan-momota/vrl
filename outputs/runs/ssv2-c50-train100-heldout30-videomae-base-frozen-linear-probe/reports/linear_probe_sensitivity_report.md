# SSV2 × frozen VideoMAE linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.023021 | 0.021700 | 0.024457 | 0.176667 | 0.155333 | 0.198683 | 0.197333 | 0.046000 |
| freeze-tail-low | motion | curve | low | 0.001039 | 0.000958 | 0.001120 | 0.020667 | 0.008000 | 0.033333 | 0.040000 | 0.012667 |
| freeze-tail-mid | motion | curve | mid | 0.004984 | 0.004658 | 0.005332 | 0.050000 | 0.032000 | 0.068017 | 0.087333 | 0.036000 |
| freeze-tail-high | motion | curve | high | 0.012561 | 0.011857 | 0.013305 | 0.117333 | 0.097317 | 0.139333 | 0.152667 | 0.062667 |
| color-low | appearance | curve | low | 0.000001 | 0.000001 | 0.000001 | 0.000667 | -0.002667 | 0.004000 | 0.002667 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000030 | 0.000030 | 0.000031 | 0.001333 | -0.004667 | 0.007333 | 0.007333 | 0.000667 |
| color-high | appearance | curve | high | 0.000288 | 0.000280 | 0.000296 | 0.007333 | -0.004667 | 0.018000 | 0.030667 | -0.004667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.006473 | 0.006321 | 0.006640 | 0.027333 | 0.007333 | 0.046000 | 0.083333 | 0.006667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 129049 | Throwing something in the air and letting it fall | 0.174796 | 42 | 42 | False |
| temporal-shuffle-mid | 46375 | Pouring something into something until it overflows | 0.168495 | 20 | 42 | False |
| temporal-shuffle-mid | 166665 | Turning the camera right while filming something | 0.167460 | 48 | 24 | True |
| freeze-tail-high | 48157 | Throwing something in the air and letting it fall | 0.161370 | 42 | 42 | False |
| temporal-shuffle-mid | 47530 | Turning the camera right while filming something | 0.160047 | 48 | 24 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 166665 | Turning the camera right while filming something | 0.167460 | 48 | 24 |
| temporal-shuffle-mid | 47530 | Turning the camera right while filming something | 0.160047 | 48 | 24 |
| temporal-shuffle-mid | 158563 | Turning the camera left while filming something | 0.157502 | 47 | 24 |
| temporal-shuffle-mid | 204496 | Turning the camera right while filming something | 0.151919 | 48 | 24 |
| temporal-shuffle-mid | 131421 | Moving something across a surface until it falls down | 0.150130 | 12 | 24 |

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
- This report covers one frozen model × one motion-oriented dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.

## Legacy-anchor comparison

Legacy KNN k=1 and current KNN k=5 are not directly comparable. Only the explicitly declared common perturbations compare representation shift directly.

| legacy_perturbation | current_artifact_label | legacy_mean_cosine_distance | current_mean_cosine_distance | mean_cosine_distance_delta_current_minus_legacy | comparability_note |
| --- | --- | --- | --- | --- | --- |
| temporal_shuffle | temporal-shuffle-mid | 0.022874 | 0.023021 | 0.000148 | The sampled-frame set and temporal-shuffle family match, but the deterministic shuffle seed changed from legacy seed 0 to current seed 42; compare the representation-shift magnitude as a stability check, not an exact repeat. |
| freeze_tail | freeze-tail-mid | 0.004984 | 0.004984 | 0.000000 | Both runs use freeze_start_fraction=0.5 and the same frozen VideoMAE/SSV2 setup; KNN drops differ in k (legacy k=1, current k=5) and are not directly comparable. |
