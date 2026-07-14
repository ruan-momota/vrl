# Kinetics × frozen dismo linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.003362 | 0.003224 | 0.003518 | 0.018667 | 0.009317 | 0.028667 | 0.030667 | 0.005333 |
| freeze-tail-mid | motion | curve | mid | 0.054439 | 0.052450 | 0.056466 | 0.038000 | 0.019333 | 0.056667 | 0.091333 | 0.077333 |
| freeze-tail-high | motion | curve | high | 0.171181 | 0.165083 | 0.177371 | 0.132000 | 0.106650 | 0.157333 | 0.196667 | 0.115333 |
| color-low | appearance | curve | low | 0.000030 | 0.000027 | 0.000034 | 0.000667 | -0.002667 | 0.004017 | 0.002667 | -0.001333 |
| color-mid | appearance | curve | mid | 0.000572 | 0.000546 | 0.000601 | 0.001333 | -0.006667 | 0.010000 | 0.014000 | -0.003333 |
| color-high | appearance | curve | high | 0.005477 | 0.005275 | 0.005674 | 0.017333 | 0.003333 | 0.032667 | 0.051333 | 0.000667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.008973 | 0.008503 | 0.009452 | 0.016667 | 0.004000 | 0.028667 | 0.039333 | -0.006667 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.377278 | 0.365686 | 0.389341 | 0.107333 | 0.085317 | 0.131350 | 0.168000 | 0.094667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 1oS16G2sB44_000022_000032 | bowling | 1.103464 | 45 | 31 | False |
| temporal-shuffle-mid | 04ab6rh2Mrs_000104_000114 | bungee jumping | 1.077841 | 40 | 40 | False |
| temporal-shuffle-mid | 1kUrSDihAsI_000059_000069 | bungee jumping | 1.066106 | 42 | 28 | False |
| temporal-shuffle-mid | 0pnFoPG0dXM_000001_000011 | catching or throwing frisbee | 1.052381 | 22 | 22 | False |
| temporal-shuffle-mid | -Hn5WqDh9hI_000104_000114 | archery | 1.050627 | 48 | 2 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 6OmEdqNyha8_000126_000136 | abseiling | 1.022875 | 0 | 28 |
| temporal-shuffle-mid | -MomTQ4exNk_000072_000082 | bungee jumping | 1.011309 | 40 | 35 |
| temporal-shuffle-mid | 8x9rty5TpC8_000025_000035 | bouncing on trampoline | 0.996727 | 30 | 22 |
| temporal-shuffle-mid | 8YF0t08eQ6U_000000_000010 | catching or throwing frisbee | 0.991136 | 49 | 28 |
| temporal-shuffle-mid | 7lTuuRfE7UU_000308_000318 | catching fish | 0.973466 | 47 | 28 |

## Data quality and failures

Fail-fast extraction was used. All extraction artifacts succeeded: True. 
All sampled frame-index and sampling-strategy checks passed: True.

| artifact_label | dataset_size | successful_samples | failed_samples |
| --- | --- | --- | --- |
| train_original | 5000 | 5000 | 0 |
| heldout_original | 1500 | 1500 | 0 |
| freeze-tail-low | 1500 | 1500 | 0 |
| freeze-tail-mid | 1500 | 1500 | 0 |
| freeze-tail-high | 1500 | 1500 | 0 |
| color-low | 1500 | 1500 | 0 |
| color-mid | 1500 | 1500 | 0 |
| color-high | 1500 | 1500 | 0 |
| spatial-blur-mid | 1500 | 1500 | 0 |
| temporal-shuffle-mid | 1500 | 1500 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one large-scale web-video Kinetics dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
