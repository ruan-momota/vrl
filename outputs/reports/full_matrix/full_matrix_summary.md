# Full Model x Dataset Matrix Summary

This summary reads the 21 completed run reports from both branches
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
- `hmdb51-full-split1-videomae-base-frozen-linear-probe`
- `hmdb51-full-split1-slowfast-r50-8x8-frozen-linear-probe`
- `kinetics-c50-train100-heldout30-videomae-base-frozen-linear-probe`
- `kinetics-c50-train100-heldout30-dismo-motion-extractor-large-frozen-linear-probe`
- `kinetics-c50-train100-heldout30-vjepa2-vitl-fpc64-256-frozen-linear-probe`
- `hmdb51-full-split1-dinov2-base-frame-mean-frozen-linear-probe`
- `kinetics-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe`
- `kinetics-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe`
- `diving48-c32-train50-heldout15-dismo-motion-extractor-large-frozen-linear-probe`
- `diving48-c32-train50-heldout15-vjepa2-vitl-fpc64-256-frozen-linear-probe`
- `hmdb51-full-split1-dismo-motion-extractor-large-frozen-linear-probe`

Quality audit overall status across all cells: `True`.

## Baseline

| Model | Dataset | Train n | Heldout n | Embedding dim | LP original acc. | KNN k=5 acc. | Quality OK |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 5000 | 1500 | 768 | 24.6% | 9.9% | True |
| VideoMAE | UCF101 | 5000 | 1500 | 768 | 85.3% | 83.6% | True |
| VideoMAE | Diving48 | 1600 | 480 | 768 | 7.3% | 3.8% | True |
| SlowFast R50 8x8 | SSV2 | 5000 | 1500 | 9216 | 33.9% | 20.3% | True |
| SlowFast R50 8x8 | UCF101 | 5000 | 1500 | 9216 | 99.4% | 99.3% | True |
| SlowFast R50 8x8 | Diving48 | 1600 | 480 | 9216 | 9.4% | 7.3% | True |
| DINOv2 frame-mean | SSV2 | 5000 | 1500 | 768 | 29.7% | 19.4% | True |
| DINOv2 frame-mean | UCF101 | 5000 | 1500 | 768 | 99.0% | 97.7% | True |
| DINOv2 frame-mean | Diving48 | 1600 | 480 | 768 | 9.6% | 9.0% | True |
| V-JEPA2 | HMDB51 | 3551 | 1524 | 1024 | 62.8% | 46.9% | True |
| VideoMAE | HMDB51 | 3551 | 1524 | 768 | 36.0% | 16.8% | True |
| SlowFast R50 8x8 | HMDB51 | 3551 | 1524 | 9216 | 71.9% | 61.4% | True |
| VideoMAE | Kinetics | 5000 | 1500 | 768 | 45.7% | 19.5% | True |
| DisMo | Kinetics | 5000 | 1500 | 128 | 34.6% | 20.3% | True |
| V-JEPA2 | Kinetics | 5000 | 1500 | 1024 | 71.8% | 52.1% | True |
| DINOv2 frame-mean | HMDB51 | 3551 | 1524 | 768 | 63.3% | 53.0% | True |
| DINOv2 frame-mean | Kinetics | 5000 | 1500 | 768 | 83.1% | 78.4% | True |
| SlowFast R50 8x8 | Kinetics | 5000 | 1500 | 9216 | 92.1% | 89.2% | True |
| DisMo | Diving48 | 1600 | 480 | 128 | 9.8% | 7.3% | True |
| V-JEPA2 | Diving48 | 1600 | 480 | 1024 | 8.5% | 8.3% | True |
| DisMo | HMDB51 | 3551 | 1524 | 128 | 47.3% | 37.2% | True |

## Fixed-mid Interventions

| Model | Dataset | Temporal shuffle LP drop | Spatial blur LP drop | Temporal shuffle mean cos. | Spatial blur mean cos. |
| --- | --- | --- | --- | --- | --- |
| VideoMAE | SSV2 | 0.1800 | 0.0333 | 0.0230 | 0.0065 |
| VideoMAE | UCF101 | 0.2493 | 0.2920 | 0.0122 | 0.0094 |
| VideoMAE | Diving48 | 0.0354 | 0.0146 | 0.0802 | 0.0011 |
| SlowFast R50 8x8 | SSV2 | 0.2053 | 0.0007 | 0.2218 | 0.0761 |
| SlowFast R50 8x8 | UCF101 | 0.0460 | 0.0140 | 0.1614 | 0.0608 |
| SlowFast R50 8x8 | Diving48 | 0.0229 | -0.0042 | 0.2053 | 0.0053 |
| DINOv2 frame-mean | SSV2 | 0.0000 | 0.0067 | 3.79e-08 | 0.0483 |
| DINOv2 frame-mean | UCF101 | 0.0000 | 0.0020 | 3.20e-08 | 0.0394 |
| DINOv2 frame-mean | Diving48 | 0.0000 | 0.0042 | 3.71e-08 | 0.0033 |
| V-JEPA2 | HMDB51 | 0.1844 | 0.1404 | 0.0907 | 0.1470 |
| VideoMAE | HMDB51 | 0.1496 | 0.1437 | 0.0143 | 0.0091 |
| SlowFast R50 8x8 | HMDB51 | 0.1732 | 0.0433 | 0.2407 | 0.0666 |
| VideoMAE | Kinetics | 0.1480 | 0.0253 | 0.0181 | 0.0026 |
| DisMo | Kinetics | 0.1073 | 0.0167 | 0.3773 | 0.0090 |
| V-JEPA2 | Kinetics | 0.2020 | 0.0473 | 0.0854 | 0.0989 |
| DINOv2 frame-mean | HMDB51 | 0.0000 | 0.0190 | 3.31e-08 | 0.0379 |
| DINOv2 frame-mean | Kinetics | 0.0000 | 0.0053 | 3.53e-08 | 0.0149 |
| SlowFast R50 8x8 | Kinetics | 0.0760 | 0.0180 | 0.1860 | 0.0249 |
| DisMo | Diving48 | 0.0458 | 0.0083 | 0.7094 | 0.0026 |
| V-JEPA2 | Diving48 | 0.0229 | 0.0104 | 0.1209 | 0.0585 |
| DisMo | HMDB51 | 0.1883 | 0.0295 | 0.3622 | 0.0177 |

(Not every cell has both fixed-mid perturbations -- HMDB51/Kinetics cells
have the full 8-perturbation matrix, matching this table; entries show
`n/a` only if a cell genuinely lacks that specific artifact.)

## Motion vs Appearance Bias

Raw accuracy drop and raw cosine distance are each confounded by something
that has nothing to do with motion/appearance bias: accuracy drop is capped
by how much original accuracy there was to lose (e.g. VideoMAE x Diving48
starts at 7.5%, so it cannot show a large drop regardless of sensitivity),
and cosine distance magnitude is set by each model's own embedding
geometry (SlowFast's 9216-d space produces larger raw distances than
DINOv2's 768-d space at the same nominal perturbation strength). The two
columns below correct for that, using only the matched motion/appearance
pair at the same nominal strength (temporal-shuffle-mid vs
spatial-blur-mid) within each cell:

- **Behavioral bias** = `correct_to_incorrect_rate(shuffle) -
  correct_to_incorrect_rate(blur)`. This rate already conditions on
  originally-correct predictions, so it stays meaningful even for
  low-accuracy cells. Positive => losing temporal order flips more
  originally-correct predictions than blurring appearance does.
- **Representational bias** = `log2(mean_cosine_distance(shuffle) /
  mean_cosine_distance(blur))`, computed within one cell's own embedding
  space so the arbitrary per-model distance scale cancels out. Positive =>
  shuffling frame order moves the representation further than blurring
  does. Clipped to +-10 (1024x) since a couple of cells hit a true zero.

| Model | Dataset | Motion flip rate (shuffle) | Appearance flip rate (blur) | Behavioral bias (motion - appearance) | Repr. log2(motion/appearance), mid |
| --- | --- | --- | --- | --- | --- |
| DisMo | HMDB51 | 0.2618 | 0.0689 | 0.1929 | 4.3570 |
| SlowFast R50 8x8 | SSV2 | 0.2333 | 0.0647 | 0.1687 | 1.5428 |
| V-JEPA2 | Kinetics | 0.2333 | 0.0860 | 0.1473 | -0.2105 |
| VideoMAE | Kinetics | 0.1913 | 0.0573 | 0.1340 | 2.7789 |
| DisMo | Kinetics | 0.1680 | 0.0393 | 0.1287 | 5.3939 |
| SlowFast R50 8x8 | HMDB51 | 0.2113 | 0.0833 | 0.1280 | 1.8530 |
| VideoMAE | SSV2 | 0.2027 | 0.0840 | 0.1187 | 1.8304 |
| DisMo | Diving48 | 0.0833 | 0.0125 | 0.0708 | 8.0978 |
| SlowFast R50 8x8 | Kinetics | 0.0893 | 0.0240 | 0.0653 | 2.8988 |
| SlowFast R50 8x8 | Diving48 | 0.0729 | 0.0125 | 0.0604 | 5.2648 |
| V-JEPA2 | HMDB51 | 0.2323 | 0.1877 | 0.0446 | -0.6977 |
| VideoMAE | Diving48 | 0.0562 | 0.0229 | 0.0333 | 6.2328 |
| SlowFast R50 8x8 | UCF101 | 0.0487 | 0.0160 | 0.0327 | 1.4077 |
| V-JEPA2 | Diving48 | 0.0708 | 0.0458 | 0.0250 | 1.0465 |
| VideoMAE | HMDB51 | 0.1916 | 0.1870 | 0.0046 | 0.6494 |
| DINOv2 frame-mean | UCF101 | 0.0000 | 0.0040 | -0.0040 | -10.0000 |
| DINOv2 frame-mean | Diving48 | 0.0000 | 0.0104 | -0.0104 | -10.0000 |
| DINOv2 frame-mean | Kinetics | 0.0000 | 0.0120 | -0.0120 | -10.0000 |
| VideoMAE | UCF101 | 0.2800 | 0.3133 | -0.0333 | 0.3714 |
| DINOv2 frame-mean | SSV2 | 0.0000 | 0.0373 | -0.0373 | -10.0000 |
| DINOv2 frame-mean | HMDB51 | 0.0000 | 0.0459 | -0.0459 | -10.0000 |

DINOv2 frame-mean's `0.0000` motion columns are not a rounding artifact:
frame-mean pooling averages per-frame features, and a mean is exactly
invariant to the order the frames are averaged in, so temporal-shuffle
cannot move that representation at all by construction -- this is a
property of the pooling operation, not evidence that DINOv2's *frame-level*
features are appearance-only.

Model averages (unweighted mean across the datasets each model was run on
-- coverage differs by model, e.g. DisMo only has a Kinetics cell, so these
are not fully apples-to-apples across rows):

| Model | Datasets averaged | Mean behavioral bias | Mean repr. log2 ratio, mid |
| --- | --- | --- | --- |
| DisMo | Diving48, HMDB51, Kinetics | 0.1308 | 5.9496 |
| SlowFast R50 8x8 | Diving48, HMDB51, Kinetics, SSV2, UCF101 | 0.0910 | 2.5934 |
| V-JEPA2 | Diving48, HMDB51, Kinetics | 0.0723 | 0.0461 |
| VideoMAE | Diving48, HMDB51, Kinetics, SSV2, UCF101 | 0.0515 | 2.3726 |
| DINOv2 frame-mean | Diving48, HMDB51, Kinetics, SSV2, UCF101 | -0.0219 | -10.0000 |

Cells where representational and behavioral bias disagree in sign -- a
larger embedding shift from one perturbation type does not always mean the
frozen linear probe's decision is more sensitive to it:

- VideoMAE x UCF101: representation shifts more from motion (log2 ratio 0.37), but behaviorally the linear probe flips more predictions from appearance (bias -0.0333).
- V-JEPA2 x HMDB51: representation shifts more from appearance (log2 ratio -0.70), but behaviorally the linear probe flips more predictions from motion (bias 0.0446).
- V-JEPA2 x Kinetics: representation shifts more from appearance (log2 ratio -0.21), but behaviorally the linear probe flips more predictions from motion (bias 0.1473).

## Figures

- `outputs/plots/full_matrix/matrix_fixed_mid_accuracy_drop.svg`
- `outputs/plots/full_matrix/matrix_fixed_mid_representation_shift.svg`
- `outputs/plots/full_matrix/matrix_strength_curves_accuracy_drop.svg`
- `outputs/plots/full_matrix/matrix_strength_curves_representation_shift.svg`
- `outputs/plots/full_matrix/matrix_motion_appearance_scatter.svg`
- `outputs/plots/full_matrix/matrix_motion_appearance_bias_ratio.svg`
- `outputs/plots/full_matrix/matrix_per_cell_grid.svg` -- one small chart per
  model x dataset cell (dataset rows x model columns, blank = not run),
  each showing all 8 perturbation artifacts on a shared y-axis so bar
  heights are comparable across the whole grid, not just within a cell.

## Full data

- `outputs/reports/full_matrix/matrix_baselines.csv`
- `outputs/reports/full_matrix/matrix_perturbation_summary.csv` (includes freeze_tail/color_transform strength curves for every cell that has them)
- `outputs/reports/full_matrix/matrix_quality_summary.csv`
- `outputs/reports/full_matrix/matrix_motion_appearance_bias.csv`
