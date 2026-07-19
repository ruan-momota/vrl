# Kinetics × frozen dinov2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| freeze-tail-low | motion | curve | low | 0.001513 | 0.001398 | 0.001642 | 0.000000 | -0.005333 | 0.005333 | 0.005333 | -0.001333 |
| freeze-tail-mid | motion | curve | mid | 0.009908 | 0.009252 | 0.010608 | -0.001333 | -0.008667 | 0.006000 | 0.010667 | -0.002000 |
| freeze-tail-high | motion | curve | high | 0.028392 | 0.026642 | 0.030350 | 0.010000 | 0.000000 | 0.020667 | 0.026667 | 0.012000 |
| color-low | appearance | curve | low | 0.000054 | 0.000051 | 0.000058 | -0.001333 | -0.004000 | 0.001333 | 0.000667 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000756 | 0.000713 | 0.000803 | 0.000000 | -0.003333 | 0.004000 | 0.002667 | 0.000667 |
| color-high | appearance | curve | high | 0.007009 | 0.006740 | 0.007313 | 0.000667 | -0.004667 | 0.006667 | 0.006667 | 0.002000 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.014942 | 0.013829 | 0.016219 | 0.005333 | -0.002000 | 0.012667 | 0.012000 | 0.009333 |
| rgb-quantization-low | appearance | curve | low | 0.034416 | 0.032520 | 0.036361 | 0.012000 | 0.003333 | 0.022667 | 0.024000 | 0.012667 |
| rgb-quantization-mid | appearance | curve | mid | 0.121673 | 0.117021 | 0.126776 | 0.032667 | 0.018667 | 0.048000 | 0.056667 | 0.041333 |
| rgb-quantization-high | appearance | curve | high | 0.301642 | 0.293422 | 0.310417 | 0.145333 | 0.124667 | 0.167333 | 0.173333 | 0.140667 |
| solarization-low | appearance | curve | low | 0.138246 | 0.132910 | 0.143612 | 0.060000 | 0.043333 | 0.076667 | 0.083333 | 0.058000 |
| solarization-mid | appearance | curve | mid | 0.122621 | 0.118092 | 0.127275 | 0.046000 | 0.030000 | 0.062017 | 0.068667 | 0.047333 |
| solarization-high | appearance | curve | high | 0.341431 | 0.333992 | 0.348565 | 0.206667 | 0.184650 | 0.230000 | 0.231333 | 0.200667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| solarization-high | 9zE-AjQdSnM_000074_000084 | catching or throwing baseball | 0.868311 | 48 | 34 | True |
| rgb-quantization-high | 6ihXDAP82lw_000099_000109 | canoeing or kayaking | 0.864342 | 42 | 35 | True |
| rgb-quantization-high | 4ksWvsO4p08_000044_000054 | bandaging | 0.854084 | 37 | 46 | False |
| solarization-high | 4ksWvsO4p08_000044_000054 | bandaging | 0.841120 | 37 | 21 | False |
| rgb-quantization-high | JAZwLCYuVDc_000022_000032 | baby waking up | 0.829208 | 10 | 2 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| solarization-high | 9zE-AjQdSnM_000074_000084 | catching or throwing baseball | 0.868311 | 48 | 34 |
| rgb-quantization-high | 6ihXDAP82lw_000099_000109 | canoeing or kayaking | 0.864342 | 42 | 35 |
| rgb-quantization-high | JAZwLCYuVDc_000022_000032 | baby waking up | 0.829208 | 10 | 2 |
| rgb-quantization-high | 1rItq8U-FZg_000011_000021 | bench pressing | 0.814827 | 19 | 1 |
| solarization-high | DRqG1pbQF80_000002_000012 | baby waking up | 0.812278 | 10 | 35 |

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
- This report covers one frozen model × one large-scale web-video Kinetics dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
