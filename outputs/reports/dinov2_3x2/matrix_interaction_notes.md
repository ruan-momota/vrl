# DINOv2 3 x 2 Interaction Notes

本文件只总结已经完成的 `3 模型 x 2 数据集` C50 主矩阵。新增 DINOv2 结果作为 frame-wise image baseline，而不是 temporal video encoder。

## 状态

- 六个 cell 均有 2 个 original artifact 和 8 个 held-out perturbation artifact。
- Quality audit overall status: `True`；DINOv2 两个 run 的 failed samples 均为 0。
- DINOv2 SSV2 original LP accuracy 为 29.7%，UCF101 为 99.0%。

## DINOv2 sanity check

- SSV2 `temporal-shuffle-mid`: mean cosine distance 3.79e-08，LP drop 0.0000。
- UCF101 `temporal-shuffle-mid`: mean cosine distance 3.20e-08，LP drop 0.0000。

这符合 frame-mean DINOv2 的预期：帧集合相同而顺序改变时，简单平均后的 embedding 基本不变。

## 主要 interaction

- DINOv2 在 UCF101 上接近饱和，LP accuracy 为 99.0%，说明该 C50 子集可由强静态图像表征很好地区分。
- DINOv2 在 SSV2 上 LP accuracy 为 29.7%，高于 VideoMAE 但低于 SlowFast；这说明 SSV2 C50 仍含可用静态线索，不能把 VideoMAE/SlowFast 与 DINOv2 的差异简单归因给 motion understanding。
- DINOv2 的 `freeze-tail-high` 在 SSV2 上 LP drop 为 0.0207，远小于视频模型的 high-strength freeze-tail drop；对 DINOv2 来说这主要是帧内容分布变化，不是 temporal modeling 证据。
- DINOv2 的 UCF101 `spatial-blur-mid` LP drop 为 0.0020，很小；相比之下 VideoMAE x UCF101 的 blur drop 仍是本矩阵中更强的 appearance-related label effect。

## 写作边界

- UCF101 仍应写作 `appearance-rich / context-correlated contrast`，不要写成纯 appearance-biased 数据集。
- DINOv2 结果不能证明或否定 motion understanding；它主要说明静态 frame-level representation 在当前 C50 子集中的可读性。
- Representation shift 和 label-related drop 仍需分开报告。
