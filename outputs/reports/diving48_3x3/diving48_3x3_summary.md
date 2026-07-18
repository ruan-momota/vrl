# Diving48 3 x 3 Summary

Generated: 2026-07-18

This summary reads the nine completed run reports and does not re-extract embeddings. The matrix covers three models, VideoMAE, SlowFast R50 8x8, and DINOv2 frame-mean, across three datasets: SSV2 C50, UCF101 C50, and Diving48 C32.

Diving48 uses the balanced `c32_train50_heldout15` subset: 1,600 train videos, 480 held-out videos, 32 classes, with 50 train and 15 held-out videos per class. Its role in this project is a fine-grained motion / pose contrast, not a full Diving48 benchmark.

## Input Runs

- `ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe`
- `diving48-c32-train50-heldout15-videomae-base-frozen-linear-probe`
- `ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe`
- `diving48-c32-train50-heldout15-slowfast-r50-8x8-frozen-linear-probe`
- `ssv2-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe`
- `diving48-c32-train50-heldout15-dinov2-base-frame-mean-frozen-linear-probe`

Quality audit overall status: `True`. Extraction succeeded with 0 failed samples in all nine cells. The train-only pixel audit fixed RGB levels at 16/8/4 and solarization thresholds at 192/128/64 before model inference; normalized pixel MAD increased monotonically from low to high for both interventions on every dataset and frame profile.

## Baseline

| Model | Dataset | Embedding dim | LP original acc. | KNN k=5 acc. |
| --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 768 | 24.6% | 9.9% |
| VideoMAE | UCF101 | 768 | 85.3% | 83.6% |
| VideoMAE | Diving48 | 768 | 7.3% | 3.8% |
| SlowFast R50 8x8 | SSV2 | 9216 | 33.9% | 20.3% |
| SlowFast R50 8x8 | UCF101 | 9216 | 99.4% | 99.3% |
| SlowFast R50 8x8 | Diving48 | 9216 | 9.4% | 7.3% |
| DINOv2 frame-mean | SSV2 | 768 | 29.7% | 19.4% |
| DINOv2 frame-mean | UCF101 | 768 | 99.0% | 97.7% |
| DINOv2 frame-mean | Diving48 | 768 | 9.6% | 9.0% |

DINOv2 frame-mean reaches 99.0% LP accuracy on UCF101, close to the saturated SlowFast result. This indicates that the current UCF101 C50 subset has strong static appearance/context readability. On SSV2, DINOv2 reaches 29.7%, showing that this SSV2 subset also contains static cues. Model differences should therefore not be written as a one-dimensional motion ranking.

Diving48 baselines are low overall: VideoMAE 7.3%, SlowFast 9.4%, and DINOv2 9.6%. These results are above random 1/32 but far below UCF101, consistent with the fine-grained action/pose difficulty and limited train50 setting. In the report, this should be treated as dataset difficulty and model-dataset interaction, not as a reason for post-hoc subset changes.

## Fixed-mid Interventions

| Model | Dataset | Temporal shuffle LP drop | Spatial blur LP drop | RGB quant. LP drop | Solarization LP drop | Temporal shuffle mean cos. | Spatial blur mean cos. | RGB quant. mean cos. | Solarization mean cos. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.1800 | 0.0333 | 0.1007 | 0.0507 | 0.0230 | 0.0065 | 0.0161 | 0.0095 |
| VideoMAE | UCF101 | 0.2493 | 0.2920 | 0.3413 | 0.2540 | 0.0122 | 0.0094 | 0.0084 | 0.0067 |
| VideoMAE | Diving48 | 0.0354 | 0.0146 | 0.0063 | 0.0083 | 0.0802 | 0.0011 | 0.0072 | 0.0028 |
| SlowFast R50 8x8 | SSV2 | 0.2053 | 0.0007 | 0.0920 | 0.0627 | 0.2218 | 0.0761 | 0.2243 | 0.1845 |
| SlowFast R50 8x8 | UCF101 | 0.0460 | 0.0140 | 0.0387 | 0.0753 | 0.1614 | 0.0608 | 0.0956 | 0.1556 |
| SlowFast R50 8x8 | Diving48 | 0.0229 | -0.0042 | 0.0229 | 0.0021 | 0.2053 | 0.0053 | 0.0702 | 0.1025 |
| DINOv2 frame-mean | SSV2 | 0.0000 | 0.0067 | 0.0373 | 0.0133 | 3.79e-08 | 0.0483 | 0.1702 | 0.1247 |
| DINOv2 frame-mean | UCF101 | 0.0000 | 0.0020 | 0.0113 | 0.0173 | 3.20e-08 | 0.0394 | 0.0813 | 0.1234 |
| DINOv2 frame-mean | Diving48 | 0.0000 | 0.0042 | 0.0188 | 0.0083 | 3.71e-08 | 0.0033 | 0.1593 | 0.1186 |

DINOv2 `temporal-shuffle-mid` is a sanity check: mean cosine distance is 3.79e-08 on SSV2, 3.20e-08 on UCF101, and 3.71e-08 on Diving48. This matches the frame-mean design, which is insensitive to frame order.

On Diving48, SlowFast temporal-shuffle representation shift is 0.2053, but LP drop is only 0.0229. This shows that a large embedding shift does not necessarily translate into an equally large label-related drop, especially in a low-baseline, small-sample, fine-grained dataset.

The new appearance interventions are materially stronger than the original color control. At fixed mid strength, VideoMAE LP drops are 0.1007 / 0.0507 for RGB quantization / solarization on SSV2 and 0.3413 / 0.2540 on UCF101. SlowFast and DINOv2 also show clear representation shifts even where label drop is smaller. This supports the intended interpretation: an effective intervention can move embeddings without necessarily crossing the current label boundary.

## Strength Curves

| Model | Dataset | Freeze-tail LP drop low->mid->high | Color LP drop low->mid->high | RGB quant. LP drop low->mid->high | Solarization LP drop low->mid->high |
| --- | --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.0213 -> 0.0567 -> 0.1260 | 0.0020 -> 0.0007 -> 0.0100 | 0.0687 -> 0.1007 -> 0.1213 | 0.0607 -> 0.0507 -> 0.0753 |
| VideoMAE | UCF101 | 0.0073 -> 0.0440 -> 0.1340 | 0.0020 -> 0.0100 -> 0.0567 | 0.1293 -> 0.3413 -> 0.5653 | 0.1940 -> 0.2540 -> 0.4660 |
| VideoMAE | Diving48 | 0.0000 -> 0.0063 -> 0.0354 | 0.0000 -> 0.0042 -> 0.0063 | 0.0000 -> 0.0063 -> -0.0146 | -0.0063 -> 0.0083 -> 0.0146 |
| SlowFast R50 8x8 | SSV2 | 0.0340 -> 0.0960 -> 0.1940 | -0.0020 -> 0.0007 -> 0.0053 | 0.0340 -> 0.0920 -> 0.1693 | 0.0820 -> 0.0627 -> 0.1580 |
| SlowFast R50 8x8 | UCF101 | 0.0033 -> 0.0093 -> 0.0387 | 0.0000 -> 0.0000 -> 0.0000 | 0.0080 -> 0.0387 -> 0.1780 | 0.0627 -> 0.0753 -> 0.2647 |
| SlowFast R50 8x8 | Diving48 | 0.0104 -> 0.0167 -> 0.0333 | 0.0000 -> 0.0000 -> 0.0042 | 0.0042 -> 0.0229 -> 0.0312 | 0.0146 -> 0.0021 -> 0.0458 |
| DINOv2 frame-mean | SSV2 | -0.0020 -> 0.0053 -> 0.0207 | -0.0007 -> -0.0013 -> -0.0047 | 0.0040 -> 0.0373 -> 0.1147 | 0.0307 -> 0.0133 -> 0.0853 |
| DINOv2 frame-mean | UCF101 | 0.0013 -> 0.0000 -> 0.0027 | 0.0000 -> 0.0000 -> 0.0020 | 0.0007 -> 0.0113 -> 0.0647 | 0.0127 -> 0.0173 -> 0.1160 |
| DINOv2 frame-mean | Diving48 | -0.0021 -> 0.0063 -> 0.0146 | 0.0000 -> 0.0000 -> -0.0021 | 0.0000 -> 0.0188 -> 0.0458 | 0.0083 -> 0.0083 -> 0.0521 |

DINOv2 `freeze-tail` changes the frame-content distribution, but that does not imply temporal modeling. RGB quantization and solarization are stronger photometric interventions whose parameters were frozen using the train-only pixel audit, not model accuracy. Diving48 strength curves should be read together with the low baseline rather than interpreted from individual drop values alone.

## Figures

- `outputs/plots/diving48_3x3/matrix_fixed_mid_accuracy_drop.svg`
- `outputs/plots/diving48_3x3/matrix_fixed_mid_representation_shift.svg`
- `outputs/plots/diving48_3x3/matrix_strength_curves_accuracy_drop.svg`
- `outputs/plots/diving48_3x3/matrix_strength_curves_representation_shift.svg`
- `outputs/reports/diving48_3x3/quan_solar_pixel_audit.csv`

## Conclusions

1. DINOv2 temporal-shuffle results validate the order-insensitivity of the frame-mean baseline. This is not a bug; it is the key interpretation boundary of this baseline.
2. UCF101 C50 can be distinguished well by strong static image representations, supporting its role as an appearance-rich / context-correlated contrast.
3. SSV2 C50 also contains usable static cues. DINOv2 has a higher SSV2 baseline than VideoMAE, but it does not use frame order, so the SSV2 result should not be equated directly with motion understanding.
4. Diving48 adds a more fine-grained, lower-sample action/pose contrast. All three models have low baselines, indicating that this subset is harder than the current UCF101 and SSV2 subsets.
5. Video-model label drops under temporal perturbation provide a useful contrast against DINOv2's near-zero temporal-shuffle drop. However, DINOv2 freeze-tail effects should be interpreted as frame-content distribution changes.
6. RGB quantization and solarization address the weak-appearance-control concern: they produce much larger pixel and embedding changes than color transform, and often larger label drops, especially on SSV2 and UCF101.
7. Representation shift and label-related drop are not always synchronized and should be reported separately. Monotonic pixel strength does not imply a strictly monotonic LP-drop curve.

## Limitations

- DINOv2 is not a temporal video model; it is a frame-wise static representation baseline.
- Current results cover SSV2/UCF101 C50 train100/heldout30 and Diving48 C32 train50/heldout15, and should not be directly generalized to full datasets.
- Perturbation sensitivity is not a clean causal isolation of motion versus appearance.
- Checkpoint pretraining-overlap risk should remain in the report, especially for UCF101-related claims.
- The low Diving48 baseline limits the dynamic range of drop metrics, so original accuracy, representation shift, and paired accuracy drop must be reported together.
- The evaluation config uses GPU `device=auto` for LBFGS probe fitting. A complete rerun changed four baseline accuracies by -0.47 to +0.83 percentage points relative to the earlier 8-perturbation snapshot, although embedding-derived cosine and KNN results were unchanged. The final report consistently uses the single complete 14-perturbation rerun within each cell; future extensions should retain the fitted probe or use a deterministic CPU fit.
