# SSV2 × frozen dinov2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| freeze-tail-low | motion | curve | low | 0.002897 | 0.002715 | 0.003086 | -0.002000 | -0.008000 | 0.004000 | 0.006667 | 0.000667 |
| freeze-tail-mid | motion | curve | mid | 0.020854 | 0.019792 | 0.022039 | 0.005333 | -0.005333 | 0.018000 | 0.028667 | -0.001333 |
| freeze-tail-high | motion | curve | high | 0.055140 | 0.052384 | 0.058008 | 0.020667 | 0.007983 | 0.034667 | 0.050000 | 0.007333 |
| color-low | appearance | curve | low | 0.000062 | 0.000059 | 0.000065 | -0.000667 | -0.002000 | 0.000000 | 0.000000 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000836 | 0.000806 | 0.000867 | -0.001333 | -0.004667 | 0.002000 | 0.002000 | -0.000667 |
| color-high | appearance | curve | high | 0.007709 | 0.007456 | 0.007961 | -0.004667 | -0.012667 | 0.003333 | 0.011333 | 0.000000 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.048301 | 0.046690 | 0.049982 | 0.006667 | -0.006000 | 0.019350 | 0.037333 | 0.012000 |
| rgb-quantization-low | appearance | curve | low | 0.058086 | 0.055495 | 0.060796 | 0.004000 | -0.009333 | 0.018000 | 0.040667 | -0.004667 |
| rgb-quantization-mid | appearance | curve | mid | 0.170238 | 0.163850 | 0.176062 | 0.037333 | 0.019333 | 0.055333 | 0.084000 | 0.022000 |
| rgb-quantization-high | appearance | curve | high | 0.397436 | 0.388424 | 0.406367 | 0.114667 | 0.092000 | 0.138683 | 0.163333 | 0.068000 |
| solarization-low | appearance | curve | low | 0.146863 | 0.140998 | 0.152551 | 0.030667 | 0.010667 | 0.047333 | 0.074000 | 0.016000 |
| solarization-mid | appearance | curve | mid | 0.124750 | 0.120616 | 0.128961 | 0.013333 | -0.001333 | 0.030667 | 0.060000 | 0.013333 |
| solarization-high | appearance | curve | high | 0.354163 | 0.347302 | 0.361288 | 0.085333 | 0.062000 | 0.108667 | 0.144000 | 0.046000 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | 177943 | Pouring something out of something | 0.901705 | 20 | 47 | False |
| rgb-quantization-high | 115405 | Throwing something in the air and letting it fall | 0.885216 | 8 | 47 | False |
| rgb-quantization-high | 45500 | Pouring something into something | 0.883288 | 23 | 37 | False |
| rgb-quantization-high | 83567 | Trying to pour something into something, but missing so it spills next to it | 0.880359 | 20 | 47 | False |
| rgb-quantization-high | 211506 | Moving something and something away from each other | 0.879645 | 35 | 37 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | 58339 | Pouring something out of something | 0.860610 | 23 | 37 |
| rgb-quantization-high | 201535 | Opening something | 0.844737 | 17 | 47 |
| rgb-quantization-high | 10154 | Turning the camera right while filming something | 0.833264 | 48 | 47 |
| rgb-quantization-high | 126184 | Putting something that cannot actually stand upright upright on the table, so it falls on its side | 0.826998 | 35 | 37 |
| rgb-quantization-high | 81747 | Digging something out of something | 0.826724 | 3 | 37 |

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
