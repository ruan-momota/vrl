# Kinetics × frozen SlowFast linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.186045 | 0.180176 | 0.192506 | 0.076000 | 0.061333 | 0.093333 | 0.089333 | 0.124667 |
| freeze-tail-low | motion | curve | low | 0.030776 | 0.029614 | 0.031900 | 0.016000 | 0.006667 | 0.025350 | 0.026000 | 0.014667 |
| freeze-tail-mid | motion | curve | mid | 0.088915 | 0.086304 | 0.091497 | 0.032667 | 0.021333 | 0.044017 | 0.045333 | 0.044667 |
| freeze-tail-high | motion | curve | high | 0.180018 | 0.174849 | 0.184863 | 0.070667 | 0.056000 | 0.086683 | 0.086667 | 0.164667 |
| color-low | appearance | curve | low | 0.000060 | 0.000057 | 0.000064 | 0.000667 | 0.000000 | 0.002000 | 0.000667 | 0.001333 |
| color-mid | appearance | curve | mid | 0.001357 | 0.001305 | 0.001414 | 0.001333 | -0.002000 | 0.004667 | 0.002667 | 0.002667 |
| color-high | appearance | curve | high | 0.013334 | 0.012802 | 0.013895 | 0.012667 | 0.006000 | 0.019333 | 0.015333 | 0.010000 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.024946 | 0.023139 | 0.026814 | 0.018000 | 0.009333 | 0.027333 | 0.024000 | 0.024000 |
| rgb-quantization-low | appearance | curve | low | 0.036992 | 0.034491 | 0.039641 | 0.019333 | 0.010667 | 0.028000 | 0.024667 | 0.016667 |
| rgb-quantization-mid | appearance | curve | mid | 0.127057 | 0.120678 | 0.133011 | 0.100000 | 0.083317 | 0.117333 | 0.110667 | 0.110000 |
| rgb-quantization-high | appearance | curve | high | 0.308103 | 0.299289 | 0.317292 | 0.325333 | 0.301333 | 0.349333 | 0.336000 | 0.339333 |
| solarization-low | appearance | curve | low | 0.169477 | 0.162686 | 0.176576 | 0.168667 | 0.149333 | 0.189333 | 0.179333 | 0.168667 |
| solarization-mid | appearance | curve | mid | 0.188262 | 0.181727 | 0.195575 | 0.212000 | 0.191317 | 0.234683 | 0.222667 | 0.197333 |
| solarization-high | appearance | curve | high | 0.346678 | 0.339380 | 0.354315 | 0.416000 | 0.391333 | 0.444000 | 0.427333 | 0.420667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | DImuhCtkoUU_000019_000029 | baby waking up | 0.819251 | 10 | 43 | True |
| rgb-quantization-high | 8Y_6ExwVlxA_000030_000040 | brushing teeth | 0.817743 | 37 | 20 | True |
| rgb-quantization-high | BDLsv-zlIno_000030_000040 | blasting sand | 0.796516 | 23 | 1 | True |
| rgb-quantization-high | -Ps3PSV5Cps_000012_000022 | blasting sand | 0.792148 | 23 | 16 | True |
| solarization-low | 8Y_6ExwVlxA_000030_000040 | brushing teeth | 0.787020 | 37 | 36 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | DImuhCtkoUU_000019_000029 | baby waking up | 0.819251 | 10 | 43 |
| rgb-quantization-high | 8Y_6ExwVlxA_000030_000040 | brushing teeth | 0.817743 | 37 | 20 |
| rgb-quantization-high | BDLsv-zlIno_000030_000040 | blasting sand | 0.796516 | 23 | 1 |
| rgb-quantization-high | -Ps3PSV5Cps_000012_000022 | blasting sand | 0.792148 | 23 | 16 |
| solarization-low | 8Y_6ExwVlxA_000030_000040 | brushing teeth | 0.787020 | 37 | 36 |

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
