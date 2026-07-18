# UCF101 × frozen VideoMAE linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.012222 | 0.011236 | 0.013251 | 0.249333 | 0.224667 | 0.274017 | 0.280000 | 0.105333 |
| freeze-tail-low | motion | curve | low | 0.000428 | 0.000388 | 0.000476 | 0.007333 | -0.002000 | 0.016667 | 0.021333 | 0.009333 |
| freeze-tail-mid | motion | curve | mid | 0.002368 | 0.002140 | 0.002623 | 0.044000 | 0.029333 | 0.058667 | 0.068000 | 0.040667 |
| freeze-tail-high | motion | curve | high | 0.006342 | 0.005807 | 0.006945 | 0.134000 | 0.112667 | 0.153333 | 0.163333 | 0.153333 |
| color-low | appearance | curve | low | 0.000002 | 0.000002 | 0.000002 | 0.002000 | -0.001333 | 0.005333 | 0.003333 | 0.002000 |
| color-mid | appearance | curve | mid | 0.000048 | 0.000046 | 0.000050 | 0.010000 | 0.003333 | 0.018000 | 0.016667 | 0.000000 |
| color-high | appearance | curve | high | 0.000478 | 0.000462 | 0.000493 | 0.056667 | 0.042667 | 0.070667 | 0.069333 | 0.008667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.009448 | 0.009282 | 0.009602 | 0.292000 | 0.267333 | 0.318017 | 0.313333 | 0.252000 |
| rgb-quantization-low | appearance | curve | low | 0.003098 | 0.002974 | 0.003215 | 0.129333 | 0.110667 | 0.148667 | 0.150000 | 0.054000 |
| rgb-quantization-mid | appearance | curve | mid | 0.008380 | 0.008139 | 0.008621 | 0.341333 | 0.314000 | 0.367350 | 0.357333 | 0.266667 |
| rgb-quantization-high | appearance | curve | high | 0.017310 | 0.016928 | 0.017725 | 0.565333 | 0.540650 | 0.591333 | 0.574667 | 0.497333 |
| solarization-low | appearance | curve | low | 0.005709 | 0.005377 | 0.006048 | 0.194000 | 0.170667 | 0.214667 | 0.212667 | 0.170000 |
| solarization-mid | appearance | curve | mid | 0.006742 | 0.006425 | 0.007076 | 0.254000 | 0.231333 | 0.278683 | 0.269333 | 0.179333 |
| solarization-high | appearance | curve | high | 0.013412 | 0.013002 | 0.013804 | 0.466000 | 0.440650 | 0.491350 | 0.478000 | 0.426667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | v_CliffDiving_g23_c01 | CliffDiving | 0.139000 | 14 | 12 | True |
| temporal-shuffle-mid | v_CliffDiving_g19_c07 | CliffDiving | 0.133561 | 14 | 26 | True |
| freeze-tail-high | v_CliffDiving_g17_c05 | CliffDiving | 0.129649 | 14 | 14 | False |
| temporal-shuffle-mid | v_CliffDiving_g01_c01 | CliffDiving | 0.128499 | 14 | 12 | True |
| temporal-shuffle-mid | v_LongJump_g10_c03 | LongJump | 0.127956 | 35 | 26 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | v_CliffDiving_g23_c01 | CliffDiving | 0.139000 | 14 | 12 |
| temporal-shuffle-mid | v_CliffDiving_g19_c07 | CliffDiving | 0.133561 | 14 | 26 |
| temporal-shuffle-mid | v_CliffDiving_g01_c01 | CliffDiving | 0.128499 | 14 | 12 |
| temporal-shuffle-mid | v_CliffDiving_g05_c06 | CliffDiving | 0.122225 | 14 | 12 |
| temporal-shuffle-mid | v_CliffDiving_g12_c05 | CliffDiving | 0.118891 | 14 | 26 |

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
- This report covers one frozen model × one appearance-rich/context-correlated UCF101 dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
