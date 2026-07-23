# Kinetics × frozen V-JEPA2 linear-probe sensitivity experiment

This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.

## Perturbation summary

| artifact_label | group | role | strength | mean_cosine_distance | cosine_distance_ci_lower | cosine_distance_ci_upper | linear_probe_accuracy_drop | linear_probe_accuracy_drop_ci_lower | linear_probe_accuracy_drop_ci_upper | correct_to_incorrect_rate | knn_accuracy_drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-low | motion | curve | low | 0.019411 | 0.018338 | 0.020533 | 0.022000 | 0.009317 | 0.036000 | 0.046000 | 0.012000 |
| freeze-tail-mid | motion | curve | mid | 0.045006 | 0.043312 | 0.046771 | 0.031333 | 0.016650 | 0.046667 | 0.066667 | 0.016000 |
| freeze-tail-high | motion | curve | high | 0.095408 | 0.092913 | 0.098267 | 0.096000 | 0.075983 | 0.115333 | 0.135333 | 0.037333 |
| color-low | appearance | curve | low | 0.000049 | 0.000047 | 0.000051 | 0.002000 | -0.001333 | 0.006000 | 0.004000 | 0.000000 |
| color-mid | appearance | curve | mid | 0.000843 | 0.000822 | 0.000866 | 0.004000 | -0.002000 | 0.010667 | 0.010000 | 0.000000 |
| color-high | appearance | curve | high | 0.009453 | 0.009203 | 0.009725 | 0.009333 | -0.002000 | 0.020667 | 0.031333 | -0.003333 |
| spatial-blur-mid | appearance | fixed_mid |  | 0.097690 | 0.095158 | 0.100218 | 0.051333 | 0.033983 | 0.067350 | 0.089333 | 0.106667 |
| temporal-shuffle-mid | motion | fixed_mid |  | 0.085608 | 0.083692 | 0.087612 | 0.210000 | 0.187333 | 0.233350 | 0.239333 | 0.093333 |
| rgb-quantization-low | appearance | curve | low | 0.018836 | 0.018007 | 0.019608 | 0.008000 | -0.004017 | 0.020000 | 0.035333 | 0.000000 |
| rgb-quantization-mid | appearance | curve | mid | 0.053126 | 0.051405 | 0.054715 | 0.061333 | 0.043983 | 0.078683 | 0.097333 | 0.037333 |
| rgb-quantization-high | appearance | curve | high | 0.115410 | 0.112921 | 0.117755 | 0.169333 | 0.144650 | 0.192000 | 0.207333 | 0.115333 |
| solarization-low | appearance | curve | low | 0.047898 | 0.046289 | 0.049638 | 0.065333 | 0.048650 | 0.082000 | 0.095333 | 0.055333 |
| solarization-mid | appearance | curve | mid | 0.047007 | 0.045570 | 0.048523 | 0.075333 | 0.058667 | 0.092667 | 0.102667 | 0.062667 |
| solarization-high | appearance | curve | high | 0.096612 | 0.094574 | 0.098706 | 0.182000 | 0.158650 | 0.206683 | 0.212667 | 0.138667 |

KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.

## Representative qualitative samples

Largest representation shifts:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction | correct_to_incorrect |
| --- | --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 6wmzxqaEBd4_000018_000028 | capoeira | 0.397502 | 43 | 3 | True |
| freeze-tail-high | 8hYoYEHAk-c_000003_000013 | answering questions | 0.372894 | 2 | 49 | True |
| freeze-tail-high | 7oecv-OO4QU_000021_000031 | bobsledding | 0.350372 | 28 | 28 | False |
| rgb-quantization-high | Ai1Y9G3N9pA_000006_000016 | baby waking up | 0.349903 | 10 | 16 | True |
| freeze-tail-high | 52wT9kPLElI_000001_000011 | beatboxing | 0.346613 | 4 | 2 | False |

Correct-to-incorrect examples:

| artifact_label | video_id | label_name | cosine_distance | original_prediction | perturbed_prediction |
| --- | --- | --- | --- | --- | --- |
| freeze-tail-high | 6wmzxqaEBd4_000018_000028 | capoeira | 0.397502 | 43 | 3 |
| freeze-tail-high | 8hYoYEHAk-c_000003_000013 | answering questions | 0.372894 | 2 | 49 |
| rgb-quantization-high | Ai1Y9G3N9pA_000006_000016 | baby waking up | 0.349903 | 10 | 16 |
| rgb-quantization-high | -Ps3PSV5Cps_000012_000022 | blasting sand | 0.333862 | 23 | 16 |
| freeze-tail-high | 82rXGnBFGac_000025_000035 | bee keeping | 0.311913 | 17 | 2 |

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
