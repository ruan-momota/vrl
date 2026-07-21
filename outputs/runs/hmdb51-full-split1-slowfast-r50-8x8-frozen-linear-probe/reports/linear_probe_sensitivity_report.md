# HMDB51 × frozen SlowFast linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.039396 | 0.038294 | 0.040552 | 0.028871 | 0.014419 | 0.043307 | 0.061024 | 0.008530 |
| freeze-tail-mid | motion | curve | mid | 0.115225 | 0.112829 | 0.117496 | 0.093832 | 0.073474 | 0.113517 | 0.128609 | 0.064304 |
| freeze-tail-high | motion | curve | high | 0.230303 | 0.226349 | 0.234843 | 0.222441 | 0.198802 | 0.246719 | 0.253281 | 0.238189 |
| color-low | appearance | curve | low | 0.000092 | 0.000088 | 0.000096 | 0.000000 | -0.002625 | 0.002625 | 0.001312 | -0.004593 |
| color-mid | appearance | curve | mid | 0.001658 | 0.001601 | 0.001715 | 0.000000 | -0.005249 | 0.005249 | 0.005249 | -0.001969 |
| color-high | appearance | curve | high | 0.014255 | 0.013774 | 0.014726 | 0.001969 | -0.008530 | 0.011171 | 0.022966 | 0.003281 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.066687 | 0.064791 | 0.068718 | 0.040682 | 0.024278 | 0.057087 | 0.081365 | 0.022310 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.240534 | 0.235073 | 0.246013 | 0.172572 | 0.148950 | 0.196211 | 0.211286 | 0.094488 |
| rgb-quantization-low | appearance | curve | low | 0.040760 | 0.038285 | 0.043574 | 0.031496 | 0.017044 | 0.045932 | 0.056430 | 0.017060 |
| rgb-quantization-mid | appearance | curve | mid | 0.137132 | 0.131680 | 0.142707 | 0.090551 | 0.068898 | 0.111549 | 0.135171 | 0.063648 |
| rgb-quantization-high | appearance | curve | high | 0.303059 | 0.295662 | 0.310158 | 0.273622 | 0.246063 | 0.301854 | 0.314304 | 0.179790 |
| solarization-low | appearance | curve | low | 0.119740 | 0.114362 | 0.125800 | 0.110236 | 0.089879 | 0.131234 | 0.143701 | 0.057743 |
| solarization-mid | appearance | curve | mid | 0.125090 | 0.120123 | 0.130398 | 0.127953 | 0.106955 | 0.148950 | 0.164042 | 0.077428 |
| solarization-high | appearance | curve | high | 0.310127 | 0.303795 | 0.316197 | 0.322178 | 0.291995 | 0.351050 | 0.362861 | 0.213911 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | 5GreatHollywoodKisses_kiss_h_cm_np2_ri_goo_1 | kiss | 0.877001 | 22 | 46 | True |
| solarization-high | 5GreatHollywoodKisses_kiss_h_cm_np2_ri_goo_1 | kiss | 0.849264 | 22 | 46 | True |
| rgb-quantization-high | 5GreatHollywoodKisses_kiss_u_nm_np2_ri_goo_2 | kiss | 0.787719 | 22 | 46 | True |
| rgb-quantization-high | TVs_Best_Kisses_Top_50_(52_to_41)_kiss_h_nm_np2_le_goo_3 | kiss | 0.784828 | 22 | 38 | True |
| rgb-quantization-mid | 5GreatHollywoodKisses_kiss_h_cm_np2_ri_goo_1 | kiss | 0.781080 | 22 | 24 | True |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| rgb-quantization-high | 5GreatHollywoodKisses_kiss_h_cm_np2_ri_goo_1 | kiss | 0.877001 | 22 | 46 |
| solarization-high | 5GreatHollywoodKisses_kiss_h_cm_np2_ri_goo_1 | kiss | 0.849264 | 22 | 46 |
| rgb-quantization-high | 5GreatHollywoodKisses_kiss_u_nm_np2_ri_goo_2 | kiss | 0.787719 | 22 | 46 |
| rgb-quantization-high | TVs_Best_Kisses_Top_50_(52_to_41)_kiss_h_nm_np2_le_goo_3 | kiss | 0.784828 | 22 | 38 |
| rgb-quantization-mid | 5GreatHollywoodKisses_kiss_h_cm_np2_ri_goo_1 | kiss | 0.781080 | 22 | 24 |

## Data quality and failures

Fail-fast extraction was used. All extraction artifacts succeeded: True. 
All sampled frame-index and sampling-strategy checks passed: True.

| artifact_label | dataset_size | successful_samples | failed_samples |
| --- | --- | --- | --- |
| train_original | 3551 | 3551 | 0 |
| heldout_original | 1524 | 1524 | 0 |
| freeze-tail-low | 1524 | 1524 | 0 |
| freeze-tail-mid | 1524 | 1524 | 0 |
| freeze-tail-high | 1524 | 1524 | 0 |
| color-low | 1524 | 1524 | 0 |
| color-mid | 1524 | 1524 | 0 |
| color-high | 1524 | 1524 | 0 |
| spatial-blur-mid | 1524 | 1524 | 0 |
| temporal-shuffle-mid | 1524 | 1524 | 0 |
| rgb-quantization-low | 1524 | 1524 | 0 |
| rgb-quantization-mid | 1524 | 1524 | 0 |
| rgb-quantization-high | 1524 | 1524 | 0 |
| solarization-low | 1524 | 1524 | 0 |
| solarization-mid | 1524 | 1524 | 0 |
| solarization-high | 1524 | 1524 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one action-recognition HMDB51 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
