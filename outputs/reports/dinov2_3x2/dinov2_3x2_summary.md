# DINOv2 3 x 2 Summary

生成日期：2026-07-06

本汇总读取六个已经完成的 run report，没有重新提取 embedding。新增 DINOv2 cell 使用 frame-wise image encoder：16 帧独立编码，取 CLS token 后做 simple mean，作为静态图像表征 baseline。

## 输入 Run

- `ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe`
- `ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe`
- `ssv2-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe`
- `ucf101-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe`

Quality audit overall status: `True`。六个 cell 的 extraction 全部成功，DINOv2 两个 run 的 failed samples 为 0。

## Baseline

| Model | Dataset | Embedding dim | LP original acc. | KNN k=5 acc. |
| --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 768 | 25.1% | 9.9% |
| VideoMAE | UCF101 | 768 | 85.3% | 83.6% |
| SlowFast R50 8x8 | SSV2 | 9216 | 33.9% | 20.3% |
| SlowFast R50 8x8 | UCF101 | 9216 | 99.4% | 99.3% |
| DINOv2 frame-mean | SSV2 | 768 | 29.7% | 19.4% |
| DINOv2 frame-mean | UCF101 | 768 | 99.0% | 97.7% |

UCF101 上 DINOv2 frame-mean 达到 99.0% LP accuracy，接近 SlowFast 的饱和结果。这说明当前 UCF101 C50 子集具有很强的静态 appearance/context 可读性。SSV2 上 DINOv2 为 29.7%，说明该 SSV2 子集也存在静态线索；因此不能把模型差异写成单一 motion ranking。

## Fixed-mid Interventions

| Model | Dataset | Temporal shuffle LP drop | Spatial blur LP drop | Temporal shuffle mean cos. | Spatial blur mean cos. |
| --- | --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.1767 | 0.0273 | 0.0230 | 0.0065 |
| VideoMAE | UCF101 | 0.2493 | 0.2920 | 0.0122 | 0.0094 |
| SlowFast R50 8x8 | SSV2 | 0.2053 | 0.0007 | 0.2218 | 0.0761 |
| SlowFast R50 8x8 | UCF101 | 0.0460 | 0.0140 | 0.1614 | 0.0608 |
| DINOv2 frame-mean | SSV2 | 0.0000 | 0.0067 | 3.79e-08 | 0.0483 |
| DINOv2 frame-mean | UCF101 | 0.0000 | 0.0020 | 3.20e-08 | 0.0394 |

DINOv2 的 `temporal-shuffle-mid` 是 sanity check：SSV2 mean cosine distance 为 3.79e-08，UCF101 为 3.20e-08，两个 LP drop 都是 0.0000。这符合 frame mean 对帧顺序不敏感的设计。

## Strength Curves

| Model | Dataset | Freeze-tail LP drop low->mid->high | Color LP drop low->mid->high |
| --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.0207 -> 0.0500 -> 0.1173 | 0.0007 -> 0.0013 -> 0.0073 |
| VideoMAE | UCF101 | 0.0073 -> 0.0440 -> 0.1340 | 0.0020 -> 0.0100 -> 0.0567 |
| SlowFast R50 8x8 | SSV2 | 0.0340 -> 0.0960 -> 0.1940 | -0.0020 -> 0.0007 -> 0.0053 |
| SlowFast R50 8x8 | UCF101 | 0.0033 -> 0.0093 -> 0.0387 | 0.0000 -> 0.0000 -> 0.0000 |
| DINOv2 frame-mean | SSV2 | -0.0020 -> 0.0053 -> 0.0207 | -0.0007 -> -0.0013 -> -0.0047 |
| DINOv2 frame-mean | UCF101 | 0.0013 -> 0.0000 -> 0.0027 | 0.0000 -> 0.0000 -> 0.0020 |

DINOv2 的 `freeze-tail` 会改变帧内容分布，但不表示它具有 temporal modeling。相比 VideoMAE 和 SlowFast，DINOv2 的 freeze-tail label drop 整体小得多。`color_transform` 对 DINOv2 和其他模型一样主要表现为 representation shift 增加，label drop 通常较小。

## 图

- `outputs/plots/dinov2_3x2/matrix_fixed_mid_accuracy_drop.svg`
- `outputs/plots/dinov2_3x2/matrix_fixed_mid_representation_shift.svg`
- `outputs/plots/dinov2_3x2/matrix_strength_curves_accuracy_drop.svg`
- `outputs/plots/dinov2_3x2/matrix_strength_curves_representation_shift.svg`

## 结论

1. DINOv2 的 temporal-shuffle 结果验证了 frame-mean baseline 的顺序不敏感性；这不是 bug，而是该 baseline 的核心解释边界。
2. UCF101 C50 可由强静态图像表征很好地区分，支持其作为 appearance-rich / context-correlated contrast 的角色。
3. SSV2 C50 也含可用静态线索；DINOv2 的 SSV2 baseline 高于 VideoMAE，但它不使用帧序，因此不能把 SSV2 结果简单等同于 motion understanding。
4. 视频模型在 temporal perturbation 上的 label drop 与 DINOv2 的近零 temporal-shuffle drop 形成有用对照；但 freeze-tail 对 DINOv2 的影响应解释为帧内容分布变化。
5. Representation shift 和 label-related drop 仍然不总是同步，汇总时应分别报告。

## 限制

- DINOv2 不是视频时序模型，只是 frame-wise static representation baseline。
- 当前结果只覆盖 C50 train100/heldout30 的受控子集，不直接外推到完整数据集。
- Perturbation sensitivity 不是 motion / appearance 的干净因果隔离。
- Checkpoint 预训练数据重叠风险仍需在报告中保留，尤其是 UCF101 相关结论。
