# Experimental Report: Motion and Appearance Sensitivity of Frozen Video Representations

Report date: 2026-07-18

## 1. Experimental Topic

This project studies how frozen pretrained video and image models depend on motion / temporal information and appearance / spatial information when videos are perturbed.

The core question is:

> When the model encoder is kept frozen, how do its representations and representation-based classification results change after video frame order, action continuity, color, texture, and spatial details are modified?

This experiment does not simply label a model as "understanding motion" or "depending on appearance." The more precise unit of analysis is:

```text
model x dataset x intervention
```

In other words, the same model can behave differently on different datasets, and the same dataset can produce different patterns under different models and perturbations. The project therefore focuses on the interaction among model architecture, dataset cue structure, and intervention type.

## 2. Experimental Setup

### 2.1 Models

The experiment uses three frozen models:

| Model | Role | Output dim |
| --- | --- | ---: |
| VideoMAE `MCG-NJU/videomae-base` | Video Transformer representation model | 768 |
| SlowFast R50 8x8 `facebookresearch/pytorchvideo:slowfast_r50` | Video CNN-style spatiotemporal representation model | 9216 |
| DINOv2 `facebook/dinov2-base` frame-mean | Frame-wise image encoder with simple temporal mean pooling, used as a static frame-level baseline | 768 |

DINOv2 is not a temporal video model. It encodes each frame independently and averages the CLS embeddings over 16 frames. Therefore, its `temporal_shuffle` result is mainly a sanity check: if the same set of frames is preserved and only the order is changed, the frame-mean embedding should remain almost unchanged.

### 2.2 Datasets

The experiment uses three controlled balanced subsets:

| Dataset | Role | Train | Held-out | Classes |
| --- | --- | ---: | ---: | ---: |
| SSV2 C50 | motion-oriented contrast | 5,000 | 1,500 | 50 |
| UCF101 C50 | appearance-rich / context-correlated contrast | 5,000 | 1,500 | 50 |
| Diving48 C32 | fine-grained motion / pose contrast | 1,600 | 480 | 32 |

SSV2 usually emphasizes object interaction processes and temporal order, so it is used as a motion-oriented contrast. UCF101 contains strong scene, person, object, equipment, and context cues, so it is used as an appearance-rich / context-correlated contrast. Diving48 is a more fine-grained action and pose classification task. This experiment only uses the C32 train50 / heldout15 subset and does not claim to be a full Diving48 benchmark.

### 2.3 Perturbations

Each `model x dataset` cell contains 2 original artifacts and 14 held-out perturbation artifacts. Perturbations are applied only to held-out queries; the original train embeddings are always kept unchanged.

| Information type | Perturbation | Strength | Meaning |
| --- | --- | --- | --- |
| motion / temporal | `temporal_shuffle` | mid | Preserves the same frame set but shuffles frame order |
| motion / temporal | `freeze_tail` | low / mid / high | Freezes frames after a temporal point, reducing action progression |
| appearance / spatial | `color_transform` | low / mid / high | Applies one fixed color transform to the whole clip, avoiding frame-to-frame flicker |
| appearance / spatial | `spatial_blur` | mid | Applies spatial blur to each frame, reducing texture and local details |
| appearance / spatial | `rgb_quantization` | low / mid / high | Quantizes each RGB channel to 16 / 8 / 4 fixed levels |
| appearance / spatial | `solarization` | low / mid / high | Inverts values at or above thresholds 192 / 128 / 64 |

RGB quantization and solarization strengths were frozen using only the train-only pixel audit, not embedding shift or held-out accuracy. Each mapping remains fixed across the whole clip to avoid frame-wise random flicker.

## 3. Experimental Data Pipeline

The current main experiment uses a run-scoped pipeline:

```text
normalized video index
  -> deterministic clip sampling
  -> optional perturbation on held-out clips
  -> model-specific preprocessing
  -> frozen encoder embedding extraction
  -> artifact alignment check
  -> train-only linear probe
  -> paired perturbation evaluation
  -> bootstrap / KNN / reports / plots
```

The main code boundaries are:

| Module | Responsibility |
| --- | --- |
| `src/data/` | Normalized indexes and dataset adapters for SSV2, UCF101, and Diving48 |
| `src/video/` | Video decoding, sampling, and clip-level perturbations |
| `src/models/` | Encoder adapters for VideoMAE, SlowFast, and DINOv2 |
| `src/pipeline/extract.py` | Reads run configs and extracts embedding artifacts |
| `src/pipeline/evaluate.py` | Trains the frozen linear probe and evaluates perturbation sensitivity |
| `src/evaluation/` | Linear probe, KNN, bootstrap, reporting, and alignment |

Example experiment entry points:

```bash
uv run python -m src.pipeline.extract --run-config <extraction-config.json>
uv run python -m src.pipeline.evaluate --config <evaluation-config.json>
```

All main matrix configs are stored under:

```text
configs/runs/
```

The final 3x3 summary is stored under:

```text
outputs/reports/diving48_3x3/
outputs/plots/diving48_3x3/
```

## 4. Experimental Metrics

### 4.1 Original Linear-Probe Accuracy

This is the main baseline classification metric. The protocol is:

```text
train a single-layer linear classifier on original train embeddings
evaluate accuracy on original held-out embeddings
```

The encoder is always frozen, and the linear probe uses only original train embeddings. This metric measures how much label-relevant information is available in the frozen representation.

### 4.2 Original KNN k=5 Accuracy

KNN is an auxiliary diagnostic metric, not the primary classification metric. The protocol is:

```text
for each held-out embedding, find its 5 nearest neighbors in train embedding space
vote using the neighbors' labels
```

It reflects whether the local neighborhood structure of the embedding space is organized by class.

### 4.3 Mean / Median Cosine Distance

This is the representation sensitivity metric:

```text
cosine_distance(z_original, z_perturbed)
```

Here `z_original` and `z_perturbed` are the embeddings of the same held-out video before and after perturbation.

A larger value means the perturbation causes a larger representation shift. This metric does not depend on a classifier and does not directly indicate whether classification becomes incorrect.

### 4.4 Linear-Probe Accuracy Drop

This is the main label-related sensitivity metric:

```text
LP drop = original held-out accuracy - perturbed held-out accuracy
```

A positive value means the accuracy decreases after perturbation, indicating that the perturbation damages classification-relevant information in the representation. A value close to 0 means the classification result is largely stable. A negative value should usually not be interpreted as the perturbation being beneficial; it is more often caused by net cancellation under low baseline accuracy or sample-level variation.

### 4.5 Correct-to-Incorrect Rate

This metric counts the fraction of samples that were originally predicted correctly but became incorrect after perturbation:

```text
original correct -> perturbed incorrect
```

It more directly describes how many originally correct samples are broken by the perturbation. However, accuracy drop can also be offset by incorrect-to-correct cases, so these metrics should be interpreted together.

### 4.6 KNN Accuracy Drop and Prediction-Change Rate

KNN drop measures the change in KNN accuracy after perturbation. Prediction-change rate measures whether the nearest-neighbor vote changes after perturbation. They are used to observe whether the perturbation changes the local embedding neighborhood, but they remain auxiliary diagnostics.

### 4.7 Bootstrap Confidence Interval

The experiment applies video-level paired bootstrap to metrics such as cosine distance, linear-probe drop, and correct-to-incorrect rate. Paired bootstrap preserves the original/perturbed pairing for the same video, which is more appropriate for this experiment than independent resampling.

### 4.8 Quality / Alignment Audit

The quality audit verifies that:

- artifact sample counts are correct;
- failed samples are 0;
- original and perturbed artifacts have aligned video IDs, labels, and sample order;
- frame indices and sampling strategies are aligned;
- DINOv2 `temporal_shuffle` is approximately 0, matching the frame-mean sanity expectation.
## 5. Final Experimental Results

This section uses the unified 14-perturbation evaluation completed on 2026-07-18. The complete 9 × 14 = 126 rows and confidence intervals are in `outputs/reports/diving48_3x3/matrix_perturbation_summary.csv`. The report focuses on baselines, fixed-mid comparisons, and dose-response curves instead of averaging all appearance rows and obscuring model–dataset interactions.

### 5.1 Baselines

| Dataset | Model | Train | Held-out | Emb dim | Original LP acc. | KNN k=5 acc. |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SSV2 | VideoMAE | 5000 | 1500 | 768 | 24.6% | 9.9% |
| UCF101 | VideoMAE | 5000 | 1500 | 768 | 85.3% | 83.6% |
| Diving48 | VideoMAE | 1600 | 480 | 768 | 7.3% | 3.8% |
| SSV2 | SlowFast R50 8x8 | 5000 | 1500 | 9216 | 33.9% | 20.3% |
| UCF101 | SlowFast R50 8x8 | 5000 | 1500 | 9216 | 99.4% | 99.3% |
| Diving48 | SlowFast R50 8x8 | 1600 | 480 | 9216 | 9.4% | 7.3% |
| SSV2 | DINOv2 frame-mean | 5000 | 1500 | 768 | 29.7% | 19.4% |
| UCF101 | DINOv2 frame-mean | 5000 | 1500 | 768 | 99.0% | 97.7% |
| Diving48 | DINOv2 frame-mean | 1600 | 480 | 768 | 9.6% | 9.0% |

Chance is approximately 2.0% for SSV2/UCF101 and 3.125% for Diving48. All Diving48 baselines are above chance but remain low, so its accuracy drops must be interpreted together with original accuracy and representation shift.

### 5.2 Train-only Pixel Audit

Parameters were frozen before model inference:

- RGB quantization low/mid/high = 16/8/4 levels;
- solarization low/mid/high = thresholds 192/128/64.

The audit covered all three datasets and both 16/32-frame profiles, selecting the first five decodable train clips per class under stable video-ID ordering. All 36 aggregate conditions passed shape, dtype, frame-count, and temporal-alignment checks, with zero decode failures.

Normalized RGB MAD for quantization increased from 0.0145–0.0162 at low, through 0.0314–0.0346 at mid, to 0.0729–0.0851 at high; per-channel output levels were exactly 16/8/4. Solarization MAD increased from 0.0894–0.0994 at low, through 0.1366–0.1687 at mid, to 0.2136–0.2485 at high, while the inversion ratio increased as the threshold decreased. Contact sheets show a clearly visible mid intervention and a severe but still recognizable high intervention.

Consequently, a small downstream LP drop cannot be dismissed as an ineffective input manipulation. Pixel intervention validity and decision-boundary sensitivity are different questions.

### 5.3 Fixed-mid Comparison

| Model | Dataset | Shuffle LP drop | Blur LP drop | Quant. LP drop | Solar. LP drop | Quant. mean cos. | Solar. mean cos. |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VideoMAE | SSV2 | 0.1800 | 0.0333 | 0.1007 | 0.0507 | 0.0161 | 0.0095 |
| VideoMAE | UCF101 | 0.2493 | 0.2920 | 0.3413 | 0.2540 | 0.0084 | 0.0067 |
| VideoMAE | Diving48 | 0.0354 | 0.0146 | 0.0063 | 0.0083 | 0.0072 | 0.0028 |
| SlowFast R50 8x8 | SSV2 | 0.2053 | 0.0007 | 0.0920 | 0.0627 | 0.2243 | 0.1845 |
| SlowFast R50 8x8 | UCF101 | 0.0460 | 0.0140 | 0.0387 | 0.0753 | 0.0956 | 0.1556 |
| SlowFast R50 8x8 | Diving48 | 0.0229 | -0.0042 | 0.0229 | 0.0021 | 0.0702 | 0.1025 |
| DINOv2 frame-mean | SSV2 | 0.0000 | 0.0067 | 0.0373 | 0.0133 | 0.1702 | 0.1247 |
| DINOv2 frame-mean | UCF101 | 0.0000 | 0.0020 | 0.0113 | 0.0173 | 0.0813 | 0.1234 |
| DINOv2 frame-mean | Diving48 | 0.0000 | 0.0042 | 0.0188 | 0.0083 | 0.1593 | 0.1186 |

The new appearance interventions resolve the weak-control concern. Except in the low-baseline Diving48 setting, quantization-mid and solarization-mid LP drops are generally well above color-mid, and they produce clear representation shifts in SlowFast and DINOv2. VideoMAE × UCF101 is most sensitive: quantization-mid drops accuracy by 34.1 percentage points and solarization-mid by 25.4 points.

### 5.4 Low → Mid → High Dose Curves

| Model | Dataset | RGB quantization LP drop | Solarization LP drop |
| --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.0687 → 0.1007 → 0.1213 | 0.0607 → 0.0507 → 0.0753 |
| VideoMAE | UCF101 | 0.1293 → 0.3413 → 0.5653 | 0.1940 → 0.2540 → 0.4660 |
| VideoMAE | Diving48 | 0.0000 → 0.0063 → -0.0146 | -0.0063 → 0.0083 → 0.0146 |
| SlowFast R50 8x8 | SSV2 | 0.0340 → 0.0920 → 0.1693 | 0.0820 → 0.0627 → 0.1580 |
| SlowFast R50 8x8 | UCF101 | 0.0080 → 0.0387 → 0.1780 | 0.0627 → 0.0753 → 0.2647 |
| SlowFast R50 8x8 | Diving48 | 0.0042 → 0.0229 → 0.0312 | 0.0146 → 0.0021 → 0.0458 |
| DINOv2 frame-mean | SSV2 | 0.0040 → 0.0373 → 0.1147 | 0.0307 → 0.0133 → 0.0853 |
| DINOv2 frame-mean | UCF101 | 0.0007 → 0.0113 → 0.0647 | 0.0127 → 0.0173 → 0.1160 |
| DINOv2 frame-mean | Diving48 | 0.0000 → 0.0188 → 0.0458 | 0.0083 → 0.0083 → 0.0521 |

Strictly monotonic pixel MAD does not require strictly monotonic LP drop. Pixel MAD measures input magnitude; LP drop measures whether the induced representation change follows a label-relevant direction that crosses the current decision boundary. Most quantization curves and high-strength results show the expected increase. Local low/mid non-monotonicity under solarization indicates that the semantic location of nonlinear pixel inversion matters beyond a scalar severity measure. Negative Diving48 drops should not be described as beneficial; they reflect net cancellation between incorrect-to-correct and correct-to-incorrect cases under a low baseline.

### 5.5 Quality Audit and Evaluation Regression Check

All nine quality-audit overall statuses are `True`. Each cell contains 2 original and 14 perturbation artifacts, for 16 total; all failed-sample counts are zero. Every cell produced 14 linear-probe reports, 14 KNN reports, 14 sensitivity reports, and 15 prediction artifacts. All 70 sampled pairing records per cell match in frame indices and sampling strategy.

One reproducibility boundary emerged when refitting the linear probe: evaluation uses GPU `device=auto` LBFGS. Relative to the earlier 8-perturbation snapshot, five cells reproduced baseline and legacy LP rows exactly, while four changed by −0.47 to +0.83 percentage points in baseline accuracy. Near-tied validation scores also changed the L2 tie outcome for Diving48-DINOv2. Legacy cosine and KNN metrics were unchanged, ruling out an embedding or perturbation-pipeline regression. This report does not mix probes; it consistently uses the complete 14-perturbation rerun within each cell. Future extensions should retain the fitted probe artifact or use deterministic CPU fitting.

## 6. Analysis

### 6.1 Appearance Interventions Are Now Effective

The original `color_transform` produced small representation shifts and label drops in most cells, leaving “model stability” confounded with “weak intervention.” RGB quantization and solarization provide three aligned layers of evidence:

1. the train-only pixel audit confirms real input changes and increasing low-to-high severity;
2. mean cosine distance rises clearly across all model families;
3. multiple SSV2/UCF101 cells exhibit material LP drops.

The supervisor's request for stronger appearance perturbations was therefore well founded, and the extension materially improves the experiment's discriminative value.

### 6.2 Model–Dataset Interaction Dominates a Single Motion/Appearance Ranking

VideoMAE × UCF101 is highly sensitive to quantization and solarization, with quantization-mid exceeding spatial blur. SlowFast × SSV2 shows large quantization/solarization representation shifts and moderate label drops. DINOv2 has strong appearance representation shifts on all datasets but smaller UCF101 LP drops, partly because the baseline is nearly saturated and the movement does not cross many class boundaries.

These patterns do not support describing a model as having one fixed dependence on motion or appearance. Conclusions should remain at the `model × dataset × intervention` level.

### 6.3 Representation and Label Sensitivity Are Distinct

SlowFast × Diving48 solarization-mid has mean cosine distance 0.1025 but LP drop 0.0021. DINOv2 × SSV2 quantization-mid has cosine distance 0.1702 and drop 0.0373. Conversely, VideoMAE × UCF101 quantization-mid has cosine distance only 0.0084 but LP drop 0.3413.

Cosine shift therefore answers whether the representation moved; LP drop answers whether it crossed the current label boundary. Neither can substitute for the other.

### 6.4 Temporal Conclusions Retain the DINOv2 Boundary

DINOv2 temporal-shuffle mean cosine distance is approximately 3×10⁻⁸ on all datasets, with zero LP drop, matching frame-mean order invariance. VideoMAE and SlowFast on SSV2 show shuffle LP drops of 0.1800 and 0.2053. This is a useful temporal contrast, but it does not establish human-like motion understanding.

### 6.5 Diving48 Has Limited Drop Dynamic Range

Diving48 baseline accuracy is only 7.3%–9.6%. Under this condition, a substantial appearance or temporal embedding shift can yield a small or negative net LP drop. Diving48 should therefore be treated as a fine-grained, small-sample interaction rather than a strong model-ranking benchmark.

## 7. Contribution to the Research Question

1. The project completes a unified 3 models × 3 datasets × 14 perturbations matrix under the same train-only probe, paired evaluation, and quality-audit protocol.
2. Train-only pixel evidence establishes effective appearance interventions without post-hoc accuracy-based parameter selection.
3. Strong appearance interventions reveal sensitivities hidden by the original color control, while also showing that an effective input intervention need not cause a large label drop.
4. The DINOv2 frame-mean sanity check separates static readability from frame-order sensitivity.
5. Results are interpreted as model, dataset, and intervention interactions, not a single averaged motion/appearance score.

## 8. Limitations

1. Perturbations are not clean causal isolations. Temporal interventions also alter naturalness and redundancy; quantization/solarization can alter texture, edges, and local shape cues.
2. DINOv2 is not a temporal video model and is used only as a static frame-level baseline.
3. Results cover controlled subsets and cannot be directly generalized to full benchmarks.
4. Checkpoint pretraining data may overlap with evaluation datasets, especially UCF101, so these are not strict OOD claims.
5. Low Diving48 baselines limit the dynamic range of accuracy drops.
6. GPU LBFGS linear-probe refitting has small numerical instability. The final results consistently use one complete rerun, but future reproduction should freeze the probe or use deterministic CPU fitting.
7. Pixel MAD, cosine shift, and LP drop measure different levels of evidence; no single metric establishes causal dependence.

## 9. Conclusion

The project completes a unified frozen-representation perturbation-sensitivity experiment. RGB quantization and solarization are stronger than the original color control at the pixel, representation, and—across multiple cells—classification levels, resolving the weak-appearance-perturbation concern. The evidence is clearest on UCF101 and SSV2; Diving48 primarily exposes the interpretive limitation of low baseline accuracy.

The supported conclusion is not that a model depends exclusively on motion or appearance. Sensitivity is jointly determined by model architecture, dataset cue structure, and the specific intervention. An effective appearance intervention can cause a large representation shift without breaking the label boundary, while a small cosine shift can still correspond to a large accuracy drop. Both levels must be reported separately.

## 10. Result Files

Main results:

- `outputs/reports/diving48_3x3/diving48_3x3_summary.md`
- `outputs/reports/diving48_3x3/matrix_baselines.csv`
- `outputs/reports/diving48_3x3/matrix_perturbation_summary.csv`
- `outputs/reports/diving48_3x3/matrix_quality_summary.csv`
- `outputs/reports/diving48_3x3/matrix_interaction_notes.md`
- `outputs/reports/diving48_3x3/quan_solar_pixel_audit.csv`
- `outputs/reports/diving48_3x3/quan_solar_pixel_audit.json`

Main figures:

- `outputs/plots/diving48_3x3/matrix_fixed_mid_accuracy_drop.svg`
- `outputs/plots/diving48_3x3/matrix_fixed_mid_representation_shift.svg`
- `outputs/plots/diving48_3x3/matrix_strength_curves_accuracy_drop.svg`
- `outputs/plots/diving48_3x3/matrix_strength_curves_representation_shift.svg`

