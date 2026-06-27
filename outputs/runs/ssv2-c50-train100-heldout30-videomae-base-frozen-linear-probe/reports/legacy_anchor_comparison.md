# Legacy-anchor comparison

Legacy KNN k=1 and current KNN k=5 are not directly comparable. Only the explicitly declared common perturbations compare representation shift directly.

| legacy_perturbation | current_artifact_label | legacy_mean_cosine_distance | current_mean_cosine_distance | mean_cosine_distance_delta_current_minus_legacy | legacy_knn_k | current_knn_k | comparability_note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| temporal_shuffle | temporal-shuffle-mid | 0.022874 | 0.023021 | 0.000148 | 1 | 5 | The sampled-frame set and temporal-shuffle family match, but the deterministic shuffle seed changed from legacy seed 0 to current seed 42; compare the representation-shift magnitude as a stability check, not an exact repeat. |
| freeze_tail | freeze-tail-mid | 0.004984 | 0.004984 | 0.000000 | 1 | 5 | Both runs use freeze_start_fraction=0.5 and the same frozen VideoMAE/SSV2 setup; KNN drops differ in k (legacy k=1, current k=5) and are not directly comparable. |
