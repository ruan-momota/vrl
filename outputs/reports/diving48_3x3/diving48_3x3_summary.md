# Diving48 3 x 3 Summary

Generated: 2026-07-07

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

Quality audit overall status: `True`. Extraction succeeded for all nine cells, and all three Diving48 runs have 0 failed samples.

## Baseline

| Model | Dataset | Embedding dim | LP original acc. | KNN k=5 acc. |
| --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 768 | 25.1% | 9.9% |
| VideoMAE | UCF101 | 768 | 85.3% | 83.6% |
| VideoMAE | Diving48 | 768 | 7.5% | 3.8% |
| SlowFast R50 8x8 | SSV2 | 9216 | 33.9% | 20.3% |
| SlowFast R50 8x8 | UCF101 | 9216 | 99.4% | 99.3% |
| SlowFast R50 8x8 | Diving48 | 9216 | 9.0% | 7.3% |
| DINOv2 frame-mean | SSV2 | 768 | 29.7% | 19.4% |
| DINOv2 frame-mean | UCF101 | 768 | 99.0% | 97.7% |
| DINOv2 frame-mean | Diving48 | 768 | 8.7% | 9.0% |

DINOv2 frame-mean reaches 99.0% LP accuracy on UCF101, close to the saturated SlowFast result. This indicates that the current UCF101 C50 subset has strong static appearance/context readability. On SSV2, DINOv2 reaches 29.7%, showing that this SSV2 subset also contains static cues. Model differences should therefore not be written as a one-dimensional motion ranking.

Diving48 baselines are low overall: VideoMAE 7.5%, SlowFast 9.0%, and DINOv2 8.7%. These results are above random 1/32 but far below UCF101, consistent with the fine-grained action/pose difficulty and limited train50 setting. In the report, this should be treated as dataset difficulty and model-dataset interaction, not as a reason for post-hoc subset changes.

## Fixed-mid Interventions

| Model | Dataset | Temporal shuffle LP drop | Spatial blur LP drop | Temporal shuffle mean cos. | Spatial blur mean cos. |
| --- | --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.1767 | 0.0273 | 0.0230 | 0.0065 |
| VideoMAE | UCF101 | 0.2493 | 0.2920 | 0.0122 | 0.0094 |
| VideoMAE | Diving48 | 0.0375 | 0.0188 | 0.0802 | 0.0011 |
| SlowFast R50 8x8 | SSV2 | 0.2053 | 0.0007 | 0.2218 | 0.0761 |
| SlowFast R50 8x8 | UCF101 | 0.0460 | 0.0140 | 0.1614 | 0.0608 |
| SlowFast R50 8x8 | Diving48 | 0.0188 | -0.0083 | 0.2053 | 0.0053 |
| DINOv2 frame-mean | SSV2 | 0.0000 | 0.0067 | 3.79e-08 | 0.0483 |
| DINOv2 frame-mean | UCF101 | 0.0000 | 0.0020 | 3.20e-08 | 0.0394 |
| DINOv2 frame-mean | Diving48 | 0.0000 | 0.0021 | 3.71e-08 | 0.0033 |

DINOv2 `temporal-shuffle-mid` is a sanity check: mean cosine distance is 3.79e-08 on SSV2, 3.20e-08 on UCF101, and 3.71e-08 on Diving48. This matches the frame-mean design, which is insensitive to frame order.

On Diving48, SlowFast temporal-shuffle representation shift is 0.2053, but LP drop is only 0.0188. This shows that a large embedding shift does not necessarily translate into an equally large label-related drop, especially in a low-baseline, small-sample, fine-grained dataset.

## Strength Curves

| Model | Dataset | Freeze-tail LP drop low->mid->high | Color LP drop low->mid->high |
| --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.0207 -> 0.0500 -> 0.1173 | 0.0007 -> 0.0013 -> 0.0073 |
| VideoMAE | UCF101 | 0.0073 -> 0.0440 -> 0.1340 | 0.0020 -> 0.0100 -> 0.0567 |
| VideoMAE | Diving48 | 0.0021 -> 0.0083 -> 0.0396 | -0.0021 -> 0.0063 -> 0.0000 |
| SlowFast R50 8x8 | SSV2 | 0.0340 -> 0.0960 -> 0.1940 | -0.0020 -> 0.0007 -> 0.0053 |
| SlowFast R50 8x8 | UCF101 | 0.0033 -> 0.0093 -> 0.0387 | 0.0000 -> 0.0000 -> 0.0000 |
| SlowFast R50 8x8 | Diving48 | 0.0042 -> 0.0083 -> 0.0250 | 0.0000 -> -0.0042 -> -0.0042 |
| DINOv2 frame-mean | SSV2 | -0.0020 -> 0.0053 -> 0.0207 | -0.0007 -> -0.0013 -> -0.0047 |
| DINOv2 frame-mean | UCF101 | 0.0013 -> 0.0000 -> 0.0027 | 0.0000 -> 0.0000 -> 0.0020 |
| DINOv2 frame-mean | Diving48 | 0.0000 -> -0.0146 -> -0.0125 | -0.0042 -> -0.0063 -> -0.0063 |

DINOv2 `freeze-tail` changes the frame-content distribution, but that does not imply temporal modeling. `color_transform` mainly increases representation shift, while label drops are usually small. Diving48 strength curves should be read together with the low baseline rather than interpreted from individual drop values alone.

## Figures

- `outputs/plots/diving48_3x3/matrix_fixed_mid_accuracy_drop.svg`
- `outputs/plots/diving48_3x3/matrix_fixed_mid_representation_shift.svg`
- `outputs/plots/diving48_3x3/matrix_strength_curves_accuracy_drop.svg`
- `outputs/plots/diving48_3x3/matrix_strength_curves_representation_shift.svg`

## Conclusions

1. DINOv2 temporal-shuffle results validate the order-insensitivity of the frame-mean baseline. This is not a bug; it is the key interpretation boundary of this baseline.
2. UCF101 C50 can be distinguished well by strong static image representations, supporting its role as an appearance-rich / context-correlated contrast.
3. SSV2 C50 also contains usable static cues. DINOv2 has a higher SSV2 baseline than VideoMAE, but it does not use frame order, so the SSV2 result should not be equated directly with motion understanding.
4. Diving48 adds a more fine-grained, lower-sample action/pose contrast. All three models have low baselines, indicating that this subset is harder than the current UCF101 and SSV2 subsets.
5. Video-model label drops under temporal perturbation provide a useful contrast against DINOv2's near-zero temporal-shuffle drop. However, DINOv2 freeze-tail effects should be interpreted as frame-content distribution changes.
6. Representation shift and label-related drop are not always synchronized and should be reported separately.

## Limitations

- DINOv2 is not a temporal video model; it is a frame-wise static representation baseline.
- Current results cover SSV2/UCF101 C50 train100/heldout30 and Diving48 C32 train50/heldout15, and should not be directly generalized to full datasets.
- Perturbation sensitivity is not a clean causal isolation of motion versus appearance.
- Checkpoint pretraining-overlap risk should remain in the report, especially for UCF101-related claims.
- The low Diving48 baseline limits the dynamic range of drop metrics, so original accuracy, representation shift, and paired accuracy drop must be reported together.
