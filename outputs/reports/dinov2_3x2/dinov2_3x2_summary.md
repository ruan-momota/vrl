# DINOv2 3 x 2 Summary

Generated: 2026-07-06

This summary reads the six completed run reports and does not re-extract embeddings. The DINOv2 cells use a frame-wise image encoder: 16 frames are encoded independently, CLS tokens are averaged, and the result is treated as a static image-representation baseline.

## Input Runs

- `ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe`
- `ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe`
- `ssv2-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe`

Quality audit overall status: `True`. Extraction succeeded for all six cells, and both DINOv2 runs have 0 failed samples.

## Baseline

| Model | Dataset | Embedding dim | LP original acc. | KNN k=5 acc. |
| --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 768 | 25.1% | 9.9% |
| VideoMAE | UCF101 | 768 | 85.3% | 83.6% |
| SlowFast R50 8x8 | SSV2 | 9216 | 33.9% | 20.3% |
| SlowFast R50 8x8 | UCF101 | 9216 | 99.4% | 99.3% |
| DINOv2 frame-mean | SSV2 | 768 | 29.7% | 19.4% |
| DINOv2 frame-mean | UCF101 | 768 | 99.0% | 97.7% |

DINOv2 frame-mean reaches 99.0% LP accuracy on UCF101, close to the saturated SlowFast result. This indicates that the current UCF101 C50 subset has strong static appearance/context readability. On SSV2, DINOv2 reaches 29.7%, showing that this SSV2 subset also contains static cues. Model differences should therefore not be written as a one-dimensional motion ranking.

## Fixed-mid Interventions

| Model | Dataset | Temporal shuffle LP drop | Spatial blur LP drop | Temporal shuffle mean cos. | Spatial blur mean cos. |
| --- | --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.1767 | 0.0273 | 0.0230 | 0.0065 |
| VideoMAE | UCF101 | 0.2493 | 0.2920 | 0.0122 | 0.0094 |
| SlowFast R50 8x8 | SSV2 | 0.2053 | 0.0007 | 0.2218 | 0.0761 |
| SlowFast R50 8x8 | UCF101 | 0.0460 | 0.0140 | 0.1614 | 0.0608 |
| DINOv2 frame-mean | SSV2 | 0.0000 | 0.0067 | 3.79e-08 | 0.0483 |
| DINOv2 frame-mean | UCF101 | 0.0000 | 0.0020 | 3.20e-08 | 0.0394 |

DINOv2 `temporal-shuffle-mid` is a sanity check: mean cosine distance is 3.79e-08 on SSV2 and 3.20e-08 on UCF101, and both LP drops are 0.0000. This matches the frame-mean design, which is insensitive to frame order.

## Strength Curves

| Model | Dataset | Freeze-tail LP drop low->mid->high | Color LP drop low->mid->high |
| --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.0207 -> 0.0500 -> 0.1173 | 0.0007 -> 0.0013 -> 0.0073 |
| VideoMAE | UCF101 | 0.0073 -> 0.0440 -> 0.1340 | 0.0020 -> 0.0100 -> 0.0567 |
| SlowFast R50 8x8 | SSV2 | 0.0340 -> 0.0960 -> 0.1940 | -0.0020 -> 0.0007 -> 0.0053 |
| SlowFast R50 8x8 | UCF101 | 0.0033 -> 0.0093 -> 0.0387 | 0.0000 -> 0.0000 -> 0.0000 |
| DINOv2 frame-mean | SSV2 | -0.0020 -> 0.0053 -> 0.0207 | -0.0007 -> -0.0013 -> -0.0047 |
| DINOv2 frame-mean | UCF101 | 0.0013 -> 0.0000 -> 0.0027 | 0.0000 -> 0.0000 -> 0.0020 |

DINOv2 `freeze-tail` changes the frame-content distribution, but that does not imply temporal modeling. Compared with VideoMAE and SlowFast, DINOv2 has much smaller freeze-tail label drops overall. `color_transform` mainly increases representation shift, while label drops are usually small.

## Figures

- `outputs/plots/dinov2_3x2/matrix_fixed_mid_accuracy_drop.svg`
- `outputs/plots/dinov2_3x2/matrix_fixed_mid_representation_shift.svg`
- `outputs/plots/dinov2_3x2/matrix_strength_curves_accuracy_drop.svg`
- `outputs/plots/dinov2_3x2/matrix_strength_curves_representation_shift.svg`

## Conclusions

1. DINOv2 temporal-shuffle results validate the order-insensitivity of the frame-mean baseline. This is not a bug; it is the key interpretation boundary of this baseline.
2. UCF101 C50 can be distinguished well by strong static image representations, supporting its role as an appearance-rich / context-correlated contrast.
3. SSV2 C50 also contains usable static cues. DINOv2 has a higher SSV2 baseline than VideoMAE, but it does not use frame order, so the SSV2 result should not be equated directly with motion understanding.
4. Video-model label drops under temporal perturbation provide a useful contrast against DINOv2's near-zero temporal-shuffle drop. However, DINOv2 freeze-tail effects should be interpreted as frame-content distribution changes.
5. Representation shift and label-related drop are not always synchronized and should be reported separately.

## Limitations

- DINOv2 is not a temporal video model; it is a frame-wise static representation baseline.
- Current results cover only the controlled C50 train100/heldout30 subsets and should not be directly generalized to full datasets.
- Perturbation sensitivity is not a clean causal isolation of motion versus appearance.
- Checkpoint pretraining-overlap risk should remain in the report, especially for UCF101-related claims.
