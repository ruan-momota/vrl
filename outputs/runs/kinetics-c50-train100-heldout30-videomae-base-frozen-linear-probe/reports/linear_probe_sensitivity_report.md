# Kinetics × frozen VideoMAE linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.000667 | 0.000603 | 0.000730 | -0.010667 | -0.020000 | -0.000667 | 0.013333 | 0.001333 |
| freeze-tail-mid | motion | curve | mid | 0.003331 | 0.003064 | 0.003608 | 0.014667 | 0.000667 | 0.030667 | 0.050667 | 0.002000 |
| freeze-tail-high | motion | curve | high | 0.008744 | 0.008137 | 0.009365 | 0.042667 | 0.024000 | 0.062667 | 0.094667 | 0.026000 |
| color-low | appearance | curve | low | 0.000002 | 0.000002 | 0.000002 | 0.001333 | 0.000000 | 0.003333 | 0.001333 | -0.000667 |
| color-mid | appearance | curve | mid | 0.000040 | 0.000038 | 0.000042 | 0.003333 | -0.004667 | 0.012000 | 0.014667 | 0.001333 |
| color-high | appearance | curve | high | 0.000356 | 0.000343 | 0.000369 | 0.020000 | 0.006667 | 0.034000 | 0.046667 | 0.010667 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.002634 | 0.002477 | 0.002793 | 0.040667 | 0.026667 | 0.057333 | 0.066000 | 0.015333 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.018172 | 0.016988 | 0.019329 | 0.150667 | 0.129317 | 0.174667 | 0.196667 | 0.018000 |
| rgb-quantization-low | appearance | curve | low | 0.003389 | 0.003199 | 0.003579 | 0.086667 | 0.066000 | 0.107333 | 0.130000 | 0.025333 |
| rgb-quantization-mid | appearance | curve | mid | 0.009375 | 0.008975 | 0.009793 | 0.188667 | 0.165333 | 0.214667 | 0.232667 | 0.051333 |
| rgb-quantization-high | appearance | curve | high | 0.019081 | 0.018451 | 0.019766 | 0.272667 | 0.245983 | 0.299333 | 0.306667 | 0.090667 |
| solarization-low | appearance | curve | low | 0.005862 | 0.005544 | 0.006230 | 0.118667 | 0.096000 | 0.142000 | 0.159333 | 0.044000 |
| solarization-mid | appearance | curve | mid | 0.006950 | 0.006585 | 0.007355 | 0.146667 | 0.123317 | 0.170667 | 0.191333 | 0.044000 |
| solarization-high | appearance | curve | high | 0.014112 | 0.013623 | 0.014670 | 0.244000 | 0.218667 | 0.271333 | 0.286000 | 0.076000 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | -QJz_YH0TMA_000335_000345 | barbequing | 0.200359 | 31 | 48 | False |
| rgb-quantization-high | 2emrOciST7I_000014_000024 | applauding | 0.167397 | 49 | 44 | False |
| temporal-shuffle-mid | 1UqN3hbK7x4_000003_000013 | applauding | 0.167237 | 3 | 3 | False |
| temporal-shuffle-mid | 1CeODw5s6Ho_000018_000028 | auctioning | 0.140704 | 3 | 48 | False |
| temporal-shuffle-mid | AUpo8x-nXBs_000017_000027 | applauding | 0.139941 | 40 | 16 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| temporal-shuffle-mid | UyWbZRdnXKc_000086_000096 | bee keeping | 0.132827 | 17 | 0 |
| temporal-shuffle-mid | 5H1VnOA4wfE_000029_000039 | bungee jumping | 0.124467 | 40 | 29 |
| solarization-high | 4Le0QDJSwEI_000142_000152 | bungee jumping | 0.123653 | 40 | 47 |
| temporal-shuffle-mid | 29ZTZQtOTsg_000003_000013 | bungee jumping | 0.115730 | 40 | 16 |
| freeze-tail-high | 6wmzxqaEBd4_000018_000028 | capoeira | 0.111227 | 43 | 3 |

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
