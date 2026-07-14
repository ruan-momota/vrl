# Kinetics × frozen V-JEPA2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.019219 | 0.018178 | 0.020342 | 0.016667 | 0.003333 | 0.029333 | 0.042667 | 0.008667 |
| freeze-tail-mid | motion | curve | mid | 0.044797 | 0.043075 | 0.046560 | 0.026000 | 0.010667 | 0.042667 | 0.064000 | 0.017333 |
| freeze-tail-high | motion | curve | high | 0.095246 | 0.092630 | 0.098176 | 0.094000 | 0.074667 | 0.114000 | 0.136667 | 0.038667 |
| color-low | appearance | curve | low | 0.000038 | 0.000036 | 0.000040 | 0.000000 | -0.002000 | 0.002000 | 0.000667 | -0.001333 |
| color-mid | appearance | curve | mid | 0.000835 | 0.000814 | 0.000859 | -0.004000 | -0.010000 | 0.002000 | 0.005333 | 0.002667 |
| color-high | appearance | curve | high | 0.009485 | 0.009233 | 0.009755 | 0.004667 | -0.006667 | 0.015333 | 0.027333 | -0.003333 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.098850 | 0.096361 | 0.101411 | 0.047333 | 0.030650 | 0.064000 | 0.086000 | 0.115333 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.085428 | 0.083526 | 0.087411 | 0.202000 | 0.179333 | 0.225333 | 0.233333 | 0.092667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 6wmzxqaEBd4_000018_000028 | capoeira | 0.398779 | 43 | 3 | True |
| freeze-tail-high | 8hYoYEHAk-c_000003_000013 | answering questions | 0.374616 | 2 | 49 | True |
| freeze-tail-high | 52wT9kPLElI_000001_000011 | beatboxing | 0.352202 | 4 | 2 | False |
| freeze-tail-low | 52wT9kPLElI_000001_000011 | beatboxing | 0.349655 | 4 | 2 | False |
| freeze-tail-high | 7oecv-OO4QU_000021_000031 | bobsledding | 0.349630 | 28 | 28 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 6wmzxqaEBd4_000018_000028 | capoeira | 0.398779 | 43 | 3 |
| freeze-tail-high | 8hYoYEHAk-c_000003_000013 | answering questions | 0.374616 | 2 | 49 |
| freeze-tail-high | 82rXGnBFGac_000025_000035 | bee keeping | 0.309910 | 17 | 2 |
| spatial-blur-mid | AUpo8x-nXBs_000017_000027 | applauding | 0.300445 | 3 | 16 |
| freeze-tail-high | -7sTNNI1Bcg_000075_000085 | bowling | 0.285467 | 31 | 24 |

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
