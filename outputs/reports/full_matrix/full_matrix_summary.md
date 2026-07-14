# Full Model x Dataset Matrix Summary

This summary reads the 13 completed run reports from both branches
and does not re-extract embeddings or fabricate cross-cell interpretation --
that is left for a follow-up write-up. Perturbation strengths and protocol
match across all cells (frozen linear probe + KNN baseline, plus 2-8
perturbation artifacts per cell depending on which matrix the cell belongs
to; see each cell's own `linear_probe_sensitivity_report.md` for its full
perturbation set).

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
- `hmdb51-full-split1-vjepa2-vitl-fpc64-256-frozen-linear-probe`
- `kinetics-c50-train100-heldout30-videomae-base-frozen-linear-probe`
- `kinetics-c50-train100-heldout30-dismo-motion-extractor-large-frozen-linear-probe`
- `kinetics-c50-train100-heldout30-vjepa2-vitl-fpc64-256-frozen-linear-probe`

Quality audit overall status across all cells: `True`.

## Baseline

| Model | Dataset | Train n | Heldout n | Embedding dim | LP original acc. | KNN k=5 acc. | Quality OK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 5000 | 1500 | 768 | 25.1% | 9.9% | True |
| VideoMAE | UCF101 | 5000 | 1500 | 768 | 85.3% | 83.6% | True |
| VideoMAE | Diving48 | 1600 | 480 | 768 | 7.5% | 3.8% | True |
| SlowFast R50 8x8 | SSV2 | 5000 | 1500 | 9216 | 33.9% | 20.3% | True |
| SlowFast R50 8x8 | UCF101 | 5000 | 1500 | 9216 | 99.4% | 99.3% | True |
| SlowFast R50 8x8 | Diving48 | 1600 | 480 | 9216 | 9.0% | 7.3% | True |
| DINOv2 frame-mean | SSV2 | 5000 | 1500 | 768 | 29.7% | 19.4% | True |
| DINOv2 frame-mean | UCF101 | 5000 | 1500 | 768 | 99.0% | 97.7% | True |
| DINOv2 frame-mean | Diving48 | 1600 | 480 | 768 | 8.7% | 9.0% | True |
| V-JEPA2 | HMDB51 | 3551 | 1524 | 1024 | 62.8% | 46.9% | True |
| VideoMAE | Kinetics | 5000 | 1500 | 768 | 45.7% | 19.5% | True |
| DisMo | Kinetics | 5000 | 1500 | 128 | 34.6% | 20.3% | True |
| V-JEPA2 | Kinetics | 5000 | 1500 | 1024 | 71.8% | 52.1% | True |

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
| V-JEPA2 | HMDB51 | 0.1844 | 0.1404 | 0.0907 | 0.1470 |
| VideoMAE | Kinetics | 0.1480 | 0.0253 | 0.0181 | 0.0026 |
| DisMo | Kinetics | 0.1073 | 0.0167 | 0.3773 | 0.0090 |
| V-JEPA2 | Kinetics | 0.2020 | 0.0473 | 0.0854 | 0.0989 |

(Not every cell has both fixed-mid perturbations -- HMDB51/Kinetics cells
have the full 8-perturbation matrix, matching this table; entries show
`n/a` only if a cell genuinely lacks that specific artifact.)

## Figures

- `outputs/plots/full_matrix/matrix_fixed_mid_accuracy_drop.svg`
- `outputs/plots/full_matrix/matrix_fixed_mid_representation_shift.svg`
- `outputs/plots/full_matrix/matrix_strength_curves_accuracy_drop.svg`
- `outputs/plots/full_matrix/matrix_strength_curves_representation_shift.svg`

## Full data

- `outputs/reports/full_matrix/matrix_baselines.csv`
- `outputs/reports/full_matrix/matrix_perturbation_summary.csv` (includes freeze_tail/color_transform strength curves for every cell that has them)
- `outputs/reports/full_matrix/matrix_quality_summary.csv`
