# diving48 × frozen SlowFast linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.205339 | 0.197959 | 0.212195 | 0.022917 | -0.006250 | 0.054219 | 0.072917 | 0.010417 |
| freeze-tail-low | motion | curve | low | 0.045089 | 0.042611 | 0.047634 | 0.010417 | -0.012500 | 0.035417 | 0.041667 | 0.008333 |
| freeze-tail-mid | motion | curve | mid | 0.136849 | 0.130632 | 0.143042 | 0.016667 | -0.012500 | 0.050000 | 0.075000 | 0.000000 |
| freeze-tail-high | motion | curve | high | 0.264780 | 0.255043 | 0.274053 | 0.033333 | 0.004167 | 0.064583 | 0.077083 | 0.016667 |
| color-low | appearance | curve | low | 0.000033 | 0.000031 | 0.000035 | 0.000000 | -0.006250 | 0.006250 | 0.002083 | -0.002083 |
| color-mid | appearance | curve | mid | 0.000902 | 0.000854 | 0.000951 | 0.000000 | -0.010417 | 0.010417 | 0.006250 | -0.002083 |
| color-high | appearance | curve | high | 0.009252 | 0.008777 | 0.009735 | 0.004167 | -0.014583 | 0.025000 | 0.025000 | 0.004167 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.005341 | 0.005036 | 0.005669 | -0.004167 | -0.018750 | 0.010417 | 0.012500 | -0.010417 |
| rgb-quantization-low | appearance | curve | low | 0.016253 | 0.015076 | 0.017422 | 0.004167 | -0.016719 | 0.025000 | 0.029167 | 0.004167 |
| rgb-quantization-mid | appearance | curve | mid | 0.070204 | 0.066337 | 0.073900 | 0.022917 | -0.002135 | 0.050000 | 0.058333 | 0.004167 |
| rgb-quantization-high | appearance | curve | high | 0.239190 | 0.230546 | 0.248015 | 0.031250 | -0.002083 | 0.062500 | 0.077083 | 0.016667 |
| solarization-low | appearance | curve | low | 0.069927 | 0.062047 | 0.077810 | 0.014583 | -0.008333 | 0.037500 | 0.037500 | 0.008333 |
| solarization-mid | appearance | curve | mid | 0.102497 | 0.094527 | 0.110318 | 0.002083 | -0.020833 | 0.025000 | 0.033333 | -0.006250 |
| solarization-high | appearance | curve | high | 0.296716 | 0.288479 | 0.304884 | 0.045833 | 0.014583 | 0.077083 | 0.083333 | 0.012500 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | heldout_25_jMEYIEYkpY0_00056 | Inward_25som_NoTwis_TUCK | 0.683268 | 31 | 26 | False |
| solarization-high | heldout_25_jMEYIEYkpY0_00056 | Inward_25som_NoTwis_TUCK | 0.645152 | 31 | 20 | False |
| solarization-mid | heldout_02_Le6xdQ2OO8w_00258 | Back_15som_NoTwis_PIKE | 0.620358 | 21 | 20 | False |
| rgb-quantization-high | heldout_25_jMEYIEYkpY0_00056 | Inward_25som_NoTwis_TUCK | 0.609356 | 31 | 20 | False |
| solarization-high | heldout_19_ovWCmIMMkRI_00046 | Forward_35som_NoTwis_TUCK | 0.607872 | 6 | 8 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | heldout_11_MbzIpx8kAD0_00017 | Back_Dive_NoTwis_TUCK | 0.536316 | 11 | 14 |
| freeze-tail-high | heldout_23_bSsVWVfYU4w_00158 | Inward_15som_NoTwis_TUCK | 0.535117 | 23 | 14 |
| solarization-high | heldout_05_sFO6XlfgxNQ_00088 | Back_25som_NoTwis_PIKE | 0.502131 | 5 | 20 |
| freeze-tail-high | heldout_11_CVAfPfVFulQ_00015 | Back_Dive_NoTwis_TUCK | 0.477274 | 11 | 14 |
| freeze-tail-high | heldout_23_bSsVWVfYU4w_00144 | Inward_15som_NoTwis_TUCK | 0.470971 | 23 | 14 |

## Data quality and failures

Fail-fast extraction was used. All extraction artifacts succeeded: True. 
All sampled frame-index and sampling-strategy checks passed: True.

| artifact_label | dataset_size | successful_samples | failed_samples |
| --- | --- | --- | --- |
| train_original | 1600 | 1600 | 0 |
| heldout_original | 480 | 480 | 0 |
| temporal-shuffle-mid | 480 | 480 | 0 |
| freeze-tail-low | 480 | 480 | 0 |
| freeze-tail-mid | 480 | 480 | 0 |
| freeze-tail-high | 480 | 480 | 0 |
| color-low | 480 | 480 | 0 |
| color-mid | 480 | 480 | 0 |
| color-high | 480 | 480 | 0 |
| spatial-blur-mid | 480 | 480 | 0 |
| rgb-quantization-low | 480 | 480 | 0 |
| rgb-quantization-mid | 480 | 480 | 0 |
| rgb-quantization-high | 480 | 480 | 0 |
| solarization-low | 480 | 480 | 0 |
| solarization-mid | 480 | 480 | 0 |
| solarization-high | 480 | 480 | 0 |

## Interpretation boundaries

- Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.
- Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.
- The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.
- This report covers one frozen model × one dataset cell. It must not be generalized to another model or dataset before the remaining matrix cells are evaluated.
