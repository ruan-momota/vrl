# diving48 × frozen V-JEPA2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.120855 | 0.118615 | 0.123329 | 0.022917 | -0.010417 | 0.054167 | 0.070833 | 0.033333 |
| freeze-tail-low | motion | curve | low | 0.042486 | 0.040368 | 0.044766 | -0.018750 | -0.045833 | 0.008333 | 0.039583 | 0.008333 |
| freeze-tail-mid | motion | curve | mid | 0.103013 | 0.100204 | 0.106104 | 0.012500 | -0.020833 | 0.045833 | 0.072917 | 0.041667 |
| freeze-tail-high | motion | curve | high | 0.188189 | 0.183875 | 0.192350 | 0.027083 | -0.002083 | 0.058333 | 0.070833 | 0.045833 |
| color-low | appearance | curve | low | 0.000024 | 0.000023 | 0.000025 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.002083 |
| color-mid | appearance | curve | mid | 0.000510 | 0.000492 | 0.000529 | -0.006250 | -0.016667 | 0.004167 | 0.004167 | 0.008333 |
| color-high | appearance | curve | high | 0.005127 | 0.004953 | 0.005313 | -0.002083 | -0.022917 | 0.018750 | 0.027083 | 0.008333 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.058510 | 0.056329 | 0.060707 | 0.010417 | -0.016667 | 0.035417 | 0.045833 | 0.012500 |
| rgb-quantization-low | appearance | curve | low | 0.010299 | 0.009710 | 0.010851 | 0.004167 | -0.020833 | 0.027083 | 0.039583 | 0.014583 |
| rgb-quantization-mid | appearance | curve | mid | 0.025904 | 0.024897 | 0.026843 | -0.008333 | -0.037500 | 0.018750 | 0.045833 | -0.006250 |
| rgb-quantization-high | appearance | curve | high | 0.063926 | 0.061684 | 0.065974 | 0.008333 | -0.020833 | 0.037552 | 0.060417 | 0.018750 |
| solarization-low | appearance | curve | low | 0.023556 | 0.021785 | 0.025446 | -0.008333 | -0.033333 | 0.018750 | 0.035417 | 0.022917 |
| solarization-mid | appearance | curve | mid | 0.027285 | 0.025634 | 0.028794 | -0.002083 | -0.027083 | 0.020885 | 0.037500 | 0.025000 |
| solarization-high | appearance | curve | high | 0.066452 | 0.064562 | 0.068255 | 0.033333 | 0.004115 | 0.062500 | 0.072917 | 0.037500 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | heldout_14_CVAfPfVFulQ_00033 | Forward_1som_NoTwis_PIKE | 0.352945 | 11 | 1 | False |
| freeze-tail-high | heldout_23_MbzIpx8kAD0_00063 | Inward_15som_NoTwis_TUCK | 0.330259 | 11 | 16 | False |
| freeze-tail-high | heldout_31_MbzIpx8kAD0_00003 | Reverse_Dive_NoTwis_TUCK | 0.328370 | 31 | 2 | True |
| freeze-tail-high | heldout_23_Le6xdQ2OO8w_00046 | Inward_15som_NoTwis_TUCK | 0.322953 | 20 | 2 | False |
| freeze-tail-high | heldout_00_Le6xdQ2OO8w_00130 | Back_15som_05Twis_FREE | 0.316285 | 23 | 23 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | heldout_31_MbzIpx8kAD0_00003 | Reverse_Dive_NoTwis_TUCK | 0.328370 | 31 | 2 |
| freeze-tail-high | heldout_29_zBXtMcI41z8_00055 | Reverse_35som_NoTwis_TUCK | 0.273781 | 29 | 2 |
| freeze-tail-high | heldout_11_MbzIpx8kAD0_00017 | Back_Dive_NoTwis_TUCK | 0.244063 | 11 | 2 |
| freeze-tail-high | heldout_11_Le6xdQ2OO8w_00029 | Back_Dive_NoTwis_TUCK | 0.238128 | 11 | 2 |
| freeze-tail-high | heldout_11_ovWCmIMMkRI_00022 | Back_Dive_NoTwis_TUCK | 0.236511 | 11 | 6 |

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
