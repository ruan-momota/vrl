# Kinetics × frozen dismo linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.003368 | 0.003224 | 0.003520 | 0.012667 | 0.004000 | 0.021333 | 0.023333 | 0.002000 |
| freeze-tail-mid | motion | curve | mid | 0.054776 | 0.052792 | 0.056839 | 0.037333 | 0.018667 | 0.057333 | 0.096000 | 0.082000 |
| freeze-tail-high | motion | curve | high | 0.172150 | 0.165851 | 0.178208 | 0.139333 | 0.114667 | 0.164667 | 0.204667 | 0.120667 |
| color-low | appearance | curve | low | 0.000046 | 0.000042 | 0.000050 | -0.004000 | -0.008667 | 0.000000 | 0.001333 | 0.001333 |
| color-mid | appearance | curve | mid | 0.000603 | 0.000572 | 0.000637 | 0.000667 | -0.008000 | 0.009333 | 0.013333 | -0.000667 |
| color-high | appearance | curve | high | 0.005522 | 0.005318 | 0.005730 | 0.018667 | 0.004650 | 0.034000 | 0.050667 | 0.000667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.009170 | 0.008696 | 0.009665 | 0.018667 | 0.005333 | 0.030017 | 0.040000 | 0.002000 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.379100 | 0.367460 | 0.390841 | 0.103333 | 0.080667 | 0.127333 | 0.166000 | 0.100667 |
| rgb-quantization-low | appearance | curve | low | 0.011217 | 0.010757 | 0.011721 | 0.024000 | 0.008667 | 0.037333 | 0.054000 | 0.002000 |
| rgb-quantization-mid | appearance | curve | mid | 0.022871 | 0.021900 | 0.023833 | 0.028667 | 0.011333 | 0.046000 | 0.074000 | 0.013333 |
| rgb-quantization-high | appearance | curve | high | 0.048433 | 0.046681 | 0.050159 | 0.065333 | 0.044667 | 0.085350 | 0.120000 | 0.023333 |
| solarization-low | appearance | curve | low | 0.038462 | 0.036782 | 0.040017 | 0.050667 | 0.032667 | 0.070667 | 0.099333 | 0.020667 |
| solarization-mid | appearance | curve | mid | 0.023444 | 0.022369 | 0.024626 | 0.035333 | 0.018000 | 0.053333 | 0.075333 | 0.008667 |
| solarization-high | appearance | curve | high | 0.069230 | 0.067619 | 0.071088 | 0.097333 | 0.075333 | 0.121333 | 0.152667 | 0.030000 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 1oS16G2sB44_000022_000032 | bowling | 1.102499 | 45 | 31 | False |
| temporal-shuffle-mid | 04ab6rh2Mrs_000104_000114 | bungee jumping | 1.077067 | 40 | 40 | False |
| temporal-shuffle-mid | 1kUrSDihAsI_000059_000069 | bungee jumping | 1.066676 | 42 | 28 | False |
| temporal-shuffle-mid | 0pnFoPG0dXM_000001_000011 | catching or throwing frisbee | 1.049914 | 22 | 22 | False |
| temporal-shuffle-mid | -Hn5WqDh9hI_000104_000114 | archery | 1.049143 | 48 | 2 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | 6OmEdqNyha8_000126_000136 | abseiling | 1.023465 | 0 | 5 |
| temporal-shuffle-mid | -MomTQ4exNk_000072_000082 | bungee jumping | 1.013028 | 40 | 2 |
| temporal-shuffle-mid | 8x9rty5TpC8_000025_000035 | bouncing on trampoline | 0.996518 | 30 | 22 |
| temporal-shuffle-mid | 8YF0t08eQ6U_000000_000010 | catching or throwing frisbee | 0.991553 | 49 | 28 |
| temporal-shuffle-mid | 7lTuuRfE7UU_000308_000318 | catching fish | 0.971105 | 47 | 28 |

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
- This report covers one frozen model × one large-scale web-video Kinetics dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
