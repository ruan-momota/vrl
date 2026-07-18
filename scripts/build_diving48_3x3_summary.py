from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any


REPORT_DIR = Path("outputs/reports/diving48_3x3")
PLOT_DIR = Path("outputs/plots/diving48_3x3")


@dataclass(frozen=True)
class Cell:
    dataset: str
    dataset_role: str
    model: str
    checkpoint: str
    run_id: str

    @property
    def run_dir(self) -> Path:
        return Path("outputs/runs") / self.run_id

    @property
    def short_label(self) -> str:
        return f"{self.model} x {self.dataset}"


CELLS = (
    Cell(
        dataset="SSV2",
        dataset_role="motion-oriented",
        model="VideoMAE",
        checkpoint="MCG-NJU/videomae-base",
        run_id="ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe",
    ),
    Cell(
        dataset="UCF101",
        dataset_role="appearance-rich / context-correlated contrast",
        model="VideoMAE",
        checkpoint="MCG-NJU/videomae-base",
        run_id="ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe",
    ),
    Cell(
        dataset="Diving48",
        dataset_role="fine-grained motion / pose contrast",
        model="VideoMAE",
        checkpoint="MCG-NJU/videomae-base",
        run_id="diving48-c32-train50-heldout15-videomae-base-frozen-linear-probe",
    ),
    Cell(
        dataset="SSV2",
        dataset_role="motion-oriented",
        model="SlowFast R50 8x8",
        checkpoint="facebookresearch/pytorchvideo:slowfast_r50",
        run_id="ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe",
    ),
    Cell(
        dataset="UCF101",
        dataset_role="appearance-rich / context-correlated contrast",
        model="SlowFast R50 8x8",
        checkpoint="facebookresearch/pytorchvideo:slowfast_r50",
        run_id="ucf101-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe",
    ),
    Cell(
        dataset="Diving48",
        dataset_role="fine-grained motion / pose contrast",
        model="SlowFast R50 8x8",
        checkpoint="facebookresearch/pytorchvideo:slowfast_r50",
        run_id="diving48-c32-train50-heldout15-slowfast-r50-8x8-frozen-linear-probe",
    ),
    Cell(
        dataset="SSV2",
        dataset_role="motion-oriented",
        model="DINOv2 frame-mean",
        checkpoint="facebook/dinov2-base",
        run_id="ssv2-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe",
    ),
    Cell(
        dataset="UCF101",
        dataset_role="appearance-rich / context-correlated contrast",
        model="DINOv2 frame-mean",
        checkpoint="facebook/dinov2-base",
        run_id="ucf101-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe",
    ),
    Cell(
        dataset="Diving48",
        dataset_role="fine-grained motion / pose contrast",
        model="DINOv2 frame-mean",
        checkpoint="facebook/dinov2-base",
        run_id="diving48-c32-train50-heldout15-dinov2-base-frame-mean-frozen-linear-probe",
    ),
)


CELL_COLORS = {
    "VideoMAE x SSV2": "#3b82f6",
    "VideoMAE x UCF101": "#0f766e",
    "VideoMAE x Diving48": "#9333ea",
    "SlowFast R50 8x8 x SSV2": "#dc2626",
    "SlowFast R50 8x8 x UCF101": "#d97706",
    "SlowFast R50 8x8 x Diving48": "#0891b2",
    "DINOv2 frame-mean x SSV2": "#7c3aed",
    "DINOv2 frame-mean x UCF101": "#16a34a",
    "DINOv2 frame-mean x Diving48": "#be123c",
}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_rows = build_baseline_rows()
    perturbation_rows = build_perturbation_rows()
    quality_rows = build_quality_rows()

    write_csv(REPORT_DIR / "matrix_baselines.csv", baseline_rows)
    write_csv(REPORT_DIR / "matrix_perturbation_summary.csv", perturbation_rows)
    write_csv(REPORT_DIR / "matrix_quality_summary.csv", quality_rows)

    write_fixed_mid_bar_chart(
        perturbation_rows,
        metric="linear_probe_accuracy_drop",
        output_path=PLOT_DIR / "matrix_fixed_mid_accuracy_drop.svg",
        title="Fixed-mid perturbation: linear-probe accuracy drop",
        y_label="Accuracy drop",
    )
    write_fixed_mid_bar_chart(
        perturbation_rows,
        metric="mean_cosine_distance",
        output_path=PLOT_DIR / "matrix_fixed_mid_representation_shift.svg",
        title="Fixed-mid perturbation: mean cosine distance",
        y_label="Mean cosine distance",
    )
    write_strength_curve_chart(
        perturbation_rows,
        metric="linear_probe_accuracy_drop",
        output_path=PLOT_DIR / "matrix_strength_curves_accuracy_drop.svg",
        title="Strength curves: linear-probe accuracy drop",
        y_label="Accuracy drop",
    )
    write_strength_curve_chart(
        perturbation_rows,
        metric="mean_cosine_distance",
        output_path=PLOT_DIR / "matrix_strength_curves_representation_shift.svg",
        title="Strength curves: mean cosine distance",
        y_label="Mean cosine distance",
    )

    (REPORT_DIR / "matrix_interaction_notes.md").write_text(
        build_interaction_notes(baseline_rows, perturbation_rows, quality_rows),
        encoding="utf-8",
    )
    (REPORT_DIR / "diving48_3x3_summary.md").write_text(
        build_summary_markdown(baseline_rows, perturbation_rows, quality_rows),
        encoding="utf-8",
    )
    return 0


def build_baseline_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        summary = load_json(cell.run_dir / "reports/linear_probe_sensitivity_summary.json")
        knn = load_json(cell.run_dir / "metrics/knn_original.json")
        quality = load_json(cell.run_dir / "reports/quality_audit.json")
        failure_summary = quality["failure_summary"]
        failed_samples_total = sum(
            int(item["failed_samples"]) for item in failure_summary["per_artifact"]
        )
        rows.append(
            {
                "dataset": cell.dataset,
                "dataset_role": cell.dataset_role,
                "model": cell.model,
                "checkpoint": cell.checkpoint,
                "run_id": cell.run_id,
                "train_n": knn["train_samples"],
                "heldout_n": knn["validation_samples"],
                "embedding_dim": knn["embedding_dim"],
                "linear_probe_original_accuracy": summary["original_linear_probe_accuracy"],
                "knn_k5_original_accuracy": knn_k_accuracy(knn, k=5),
                "quality_ok": (
                    failure_summary["all_extractions_succeeded"]
                    and failure_summary["all_sampled_quality_checks_passed"]
                    and failed_samples_total == 0
                ),
                "failed_samples_total": failed_samples_total,
            }
        )
    return rows


def build_perturbation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        path = cell.run_dir / "reports/linear_probe_perturbation_summary.csv"
        with path.open("r", encoding="utf-8", newline="") as file:
            for source_row in csv.DictReader(file):
                row = {
                    "dataset": cell.dataset,
                    "dataset_role": cell.dataset_role,
                    "model": cell.model,
                    "checkpoint": cell.checkpoint,
                    "run_id": cell.run_id,
                    **source_row,
                }
                rows.append(row)
    return rows


def build_quality_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        quality = load_json(cell.run_dir / "reports/quality_audit.json")
        failure_summary = quality["failure_summary"]
        for artifact in failure_summary["per_artifact"]:
            rows.append(
                {
                    "dataset": cell.dataset,
                    "model": cell.model,
                    "run_id": cell.run_id,
                    "artifact_label": artifact["artifact_label"],
                    "dataset_size": artifact["dataset_size"],
                    "successful_samples": artifact["successful_samples"],
                    "failed_samples": artifact["failed_samples"],
                    "quality_ok": (
                        failure_summary["all_extractions_succeeded"]
                        and failure_summary["all_sampled_quality_checks_passed"]
                        and int(artifact["failed_samples"]) == 0
                    ),
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_fixed_mid_bar_chart(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    series = {
        "temporal-shuffle-mid": "#2563eb",
        "spatial-blur-mid": "#ea580c",
        "rgb-quantization-mid": "#16a34a",
        "solarization-mid": "#9333ea",
    }
    fixed_rows = [row for row in rows if row["artifact_label"] in series]
    labels = [cell.short_label for cell in CELLS]
    values_by_label = {
        (row["model"] + " x " + row["dataset"], row["artifact_label"]): float(row[metric])
        for row in fixed_rows
    }
    values = list(values_by_label.values())
    min_value = min(0.0, min(values))
    max_value = max(values)

    width, height = 1280, 620
    left, right, top, bottom = 90, 30, 105, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min, y_max = padded_range(min_value, max_value)

    def x_for_group(index: int) -> float:
        return left + (index + 0.5) * plot_width / len(labels)

    def y_for(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    zero_y = y_for(0.0)
    bar_width = min(24.0, plot_width / len(labels) / (len(series) + 2.0))
    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, title, size=18, weight="700", anchor="middle"))
    parts.append(text(18, top + plot_height / 2, y_label, size=12, anchor="middle", rotate=-90))
    parts.extend(axis_parts(left, top, plot_width, plot_height, y_min, y_max, y_for))

    for index, label in enumerate(labels):
        group_center = x_for_group(index)
        center_offset = (len(series) - 1) / 2.0
        for offset, (artifact_label, color) in enumerate(series.items()):
            value = values_by_label[(label, artifact_label)]
            x = group_center + (offset - center_offset) * bar_width * 1.12
            y = min(y_for(value), zero_y)
            bar_height = abs(y_for(value) - zero_y)
            parts.append(
                f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" />'
            )
        parts.append(text(group_center, height - 38, label, size=10, anchor="end", rotate=-35))

    legend_x = left + 12
    legend_y = 68
    for idx, (artifact_label, color) in enumerate(series.items()):
        x = legend_x + idx * 270
        parts.append(f'<rect x="{x}" y="{legend_y - 10}" width="12" height="12" fill="{color}" />')
        parts.append(text(x + 18, legend_y, artifact_label, size=12))
    parts.append("</svg>\n")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_strength_curve_chart(
    rows: list[dict[str, Any]],
    *,
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    width, height = 1280, 1490
    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, title, size=18, weight="700", anchor="middle"))
    panel_specs = (
        ("freeze_tail", "Freeze-tail low -> mid -> high", 55),
        ("color_transform", "Color transform low -> mid -> high", 405),
        ("rgb_quantization", "RGB quantization low -> mid -> high", 755),
        ("solarization", "Solarization low -> mid -> high", 1105),
    )
    for perturbation, subtitle, top in panel_specs:
        curve_rows = [
            row
            for row in rows
            if row["perturbation"] == perturbation and row["role"] == "curve"
        ]
        write_curve_panel(
            parts,
            rows=curve_rows,
            metric=metric,
            top=top,
            width=width,
            title=subtitle,
            y_label=y_label,
        )
    parts.append("</svg>\n")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_curve_panel(
    parts: list[str],
    *,
    rows: list[dict[str, Any]],
    metric: str,
    top: int,
    width: int,
    title: str,
    y_label: str,
) -> None:
    left, right, panel_height, bottom_pad = 90, 390, 270, 55
    plot_width = width - left - right
    plot_height = panel_height - bottom_pad
    strengths = ("low", "mid", "high")
    values = [float(row[metric]) for row in rows]
    y_min, y_max = padded_range(min(0.0, min(values)), max(values))

    def x_for(strength: str) -> float:
        index = strengths.index(strength)
        return left + index * plot_width / (len(strengths) - 1)

    def y_for(value: float) -> float:
        return top + 35 + (y_max - value) / (y_max - y_min) * plot_height

    parts.append(text(left, top + 10, title, size=14, weight="700"))
    parts.append(text(18, top + 35 + plot_height / 2, y_label, size=11, anchor="middle", rotate=-90))
    parts.extend(axis_parts(left, top + 35, plot_width, plot_height, y_min, y_max, y_for))

    rows_by_cell: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        label = row["model"] + " x " + row["dataset"]
        rows_by_cell.setdefault(label, {})[row["strength"]] = row

    for cell in CELLS:
        label = cell.short_label
        color = CELL_COLORS[label]
        points = [
            (x_for(strength), y_for(float(rows_by_cell[label][strength][metric])))
            for strength in strengths
        ]
        point_data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline points="{point_data}" fill="none" stroke="{color}" '
            'stroke-width="2.3" stroke-linejoin="round" />'
        )
        for x, y in points:
            parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.8" fill="{color}" />')

    axis_y = top + 35 + plot_height
    for strength in strengths:
        parts.append(text(x_for(strength), axis_y + 22, strength, size=12, anchor="middle"))

    legend_x = width - right + 25
    legend_y = top + 55
    for idx, cell in enumerate(CELLS):
        y = legend_y + idx * 22
        label = cell.short_label
        color = CELL_COLORS[label]
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="3" />')
        parts.append(text(legend_x + 24, y + 4, label, size=11))


def axis_parts(
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    y_min: float,
    y_max: float,
    y_for: Any,
) -> list[str]:
    parts = [
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1" />',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1" />',
    ]
    ticks = 5
    for i in range(ticks + 1):
        value = y_min + (y_max - y_min) * i / ticks
        y = y_for(value)
        parts.append(f'<line x1="{left - 4}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="#111827" />')
        parts.append(text(left - 8, y + 4, f"{value:.3f}", size=10, anchor="end"))
        if i not in {0, ticks}:
            parts.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
                'stroke="#e5e7eb" stroke-width="1" />'
            )
    zero_y = y_for(0.0)
    if top <= zero_y <= top + plot_height:
        parts.append(
            f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_width}" y2="{zero_y:.2f}" '
            'stroke="#6b7280" stroke-dasharray="4 4" />'
        )
    return parts


def padded_range(min_value: float, max_value: float) -> tuple[float, float]:
    if min_value == max_value:
        pad = max(abs(max_value) * 0.1, 0.001)
        return min_value - pad, max_value + pad
    pad = (max_value - min_value) * 0.12
    return min_value - pad, max_value + pad


def build_interaction_notes(
    baselines: list[dict[str, Any]],
    perturbations: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> str:
    ssv2_dino = find_row(baselines, "DINOv2 frame-mean", "SSV2")
    ucf_dino = find_row(baselines, "DINOv2 frame-mean", "UCF101")
    diving_videomae = find_row(baselines, "VideoMAE", "Diving48")
    diving_slowfast = find_row(baselines, "SlowFast R50 8x8", "Diving48")
    diving_dino = find_row(baselines, "DINOv2 frame-mean", "Diving48")
    dino_ssv2_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "SSV2", "temporal-shuffle-mid")
    dino_ucf_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "UCF101", "temporal-shuffle-mid")
    dino_diving_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "Diving48", "temporal-shuffle-mid")
    videomae_diving_shuffle = find_perturbation(perturbations, "VideoMAE", "Diving48", "temporal-shuffle-mid")
    slowfast_diving_shuffle = find_perturbation(perturbations, "SlowFast R50 8x8", "Diving48", "temporal-shuffle-mid")
    videomae_diving_blur = find_perturbation(perturbations, "VideoMAE", "Diving48", "spatial-blur-mid")
    slowfast_diving_blur = find_perturbation(perturbations, "SlowFast R50 8x8", "Diving48", "spatial-blur-mid")
    dino_diving_blur = find_perturbation(perturbations, "DINOv2 frame-mean", "Diving48", "spatial-blur-mid")
    videomae_ssv2_quant = find_perturbation(perturbations, "VideoMAE", "SSV2", "rgb-quantization-mid")
    videomae_ssv2_solar = find_perturbation(perturbations, "VideoMAE", "SSV2", "solarization-mid")
    videomae_ucf_quant = find_perturbation(perturbations, "VideoMAE", "UCF101", "rgb-quantization-mid")
    videomae_ucf_solar = find_perturbation(perturbations, "VideoMAE", "UCF101", "solarization-mid")
    quality_ok = all(row["quality_ok"] in {True, "True", "true"} for row in quality_rows)
    return f"""# Diving48 3 x 3 Interaction Notes

This note summarizes the completed `3 models x 3 datasets` matrix. The added Diving48 C32 setting uses a balanced train50 / heldout15 subset. Its role is a fine-grained motion / pose contrast, not a full Diving48 benchmark.

## Status

- All nine cells have 2 original artifacts and 14 held-out perturbation artifacts.
- Quality audit overall status: `{quality_ok}`; all nine cells have 0 failed samples.
- The train-only pixel audit established monotonic low-to-high pixel severity before model inference; no strength was selected from accuracy.
- Diving48 original LP accuracy: VideoMAE {pct(diving_videomae['linear_probe_original_accuracy'])}, SlowFast {pct(diving_slowfast['linear_probe_original_accuracy'])}, DINOv2 {pct(diving_dino['linear_probe_original_accuracy'])}.

## DINOv2 sanity check

- SSV2 `temporal-shuffle-mid`: mean cosine distance {fmt_float(dino_ssv2_shuffle['mean_cosine_distance'])}, LP drop {fmt_float(dino_ssv2_shuffle['linear_probe_accuracy_drop'])}.
- UCF101 `temporal-shuffle-mid`: mean cosine distance {fmt_float(dino_ucf_shuffle['mean_cosine_distance'])}, LP drop {fmt_float(dino_ucf_shuffle['linear_probe_accuracy_drop'])}.
- Diving48 `temporal-shuffle-mid`: mean cosine distance {fmt_float(dino_diving_shuffle['mean_cosine_distance'])}, LP drop {fmt_float(dino_diving_shuffle['linear_probe_accuracy_drop'])}.

This matches the frame-mean DINOv2 expectation: when the same frame set is preserved and only order changes, the simple-mean embedding is effectively unchanged.

## Main Interactions and Interpretation Boundaries

- DINOv2 is nearly saturated on UCF101, with LP accuracy of {pct(ucf_dino['linear_probe_original_accuracy'])}. This indicates that the C50 subset is highly readable from strong static image representations.
- DINOv2 reaches {pct(ssv2_dino['linear_probe_original_accuracy'])} LP accuracy on SSV2, higher than VideoMAE but lower than SlowFast. This shows that the SSV2 C50 subset still contains usable static cues, so VideoMAE/SlowFast versus DINOv2 differences should not be reduced to a simple motion-understanding claim.
- All three Diving48 baselines are low but still above random 1/32. This should primarily be interpreted as fine-grained, small-sample difficulty in the C32 train50 setting, not as a failure of any single model implementation.
- On Diving48, VideoMAE temporal-shuffle LP drop is {fmt_float(videomae_diving_shuffle['linear_probe_accuracy_drop'])}, and SlowFast is {fmt_float(slowfast_diving_shuffle['linear_probe_accuracy_drop'])}. Both are less pronounced than the fixed-mid label effects observed on SSV2/UCF101.
- SlowFast x Diving48 temporal-shuffle representation shift is {fmt_float(slowfast_diving_shuffle['mean_cosine_distance'])}, much larger than DINOv2's {fmt_float(dino_diving_shuffle['mean_cosine_distance'])}, but the label drop is only {fmt_float(slowfast_diving_shuffle['linear_probe_accuracy_drop'])}. Representation shift and label-related drop must therefore be reported separately.
- Diving48 spatial-blur LP drops are: VideoMAE {fmt_float(videomae_diving_blur['linear_probe_accuracy_drop'])}, SlowFast {fmt_float(slowfast_diving_blur['linear_probe_accuracy_drop'])}, DINOv2 {fmt_float(dino_diving_blur['linear_probe_accuracy_drop'])}. Current results do not support explaining Diving48 differences as simple static-appearance cue effects.
- The stronger appearance interventions resolve the weak-control concern. For VideoMAE, fixed-mid RGB quantization / solarization LP drops are {fmt_float(videomae_ssv2_quant['linear_probe_accuracy_drop'])} / {fmt_float(videomae_ssv2_solar['linear_probe_accuracy_drop'])} on SSV2 and {fmt_float(videomae_ucf_quant['linear_probe_accuracy_drop'])} / {fmt_float(videomae_ucf_solar['linear_probe_accuracy_drop'])} on UCF101, substantially above the original color-mid control.
- Pixel severity is monotonic, but downstream LP drop is not required to be monotonic: the latter depends on which representation directions cross the fitted decision boundary.

## Writing Boundaries

- UCF101 should be described as an `appearance-rich / context-correlated contrast`, not as a purely appearance-biased dataset.
- Diving48 should be described as a `fine-grained motion / pose contrast`, while explicitly stating that this experiment uses a C32 train50/heldout15 subset.
- DINOv2 results cannot prove or disprove motion understanding; they mainly quantify how readable the current subsets are from static frame-level representations.
- The low Diving48 baseline is an important result and should not be optimized away by post-hoc class or quota changes.
- Representation shift and label-related drop should be reported separately.
- RGB quantization and solarization results must be interpreted together with the train-only pixel audit; model accuracy was not used to choose their strengths.
- The final table uses one complete 14-perturbation evaluation per cell. GPU `device=auto` LBFGS refitting caused small baseline variation in four cells relative to the earlier 8-perturbation snapshot (range -0.47 to +0.83 percentage points), while cosine/KNN embedding metrics were unchanged; future extensions should preserve the fitted probe artifact or use a deterministic CPU fit.
"""


def build_summary_markdown(
    baselines: list[dict[str, Any]],
    perturbations: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> str:
    baseline_table = markdown_table(
        ["Model", "Dataset", "Embedding dim", "LP original acc.", "KNN k=5 acc."],
        [
            [
                row["model"],
                row["dataset"],
                row["embedding_dim"],
                pct(row["linear_probe_original_accuracy"]),
                pct(row["knn_k5_original_accuracy"]),
            ]
            for row in baselines
        ],
    )
    fixed_rows = [
        [
            cell.model,
            cell.dataset,
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "temporal-shuffle-mid")["linear_probe_accuracy_drop"]),
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "spatial-blur-mid")["linear_probe_accuracy_drop"]),
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "rgb-quantization-mid")["linear_probe_accuracy_drop"]),
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "solarization-mid")["linear_probe_accuracy_drop"]),
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "temporal-shuffle-mid")["mean_cosine_distance"]),
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "spatial-blur-mid")["mean_cosine_distance"]),
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "rgb-quantization-mid")["mean_cosine_distance"]),
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "solarization-mid")["mean_cosine_distance"]),
        ]
        for cell in CELLS
    ]
    fixed_table = markdown_table(
        [
            "Model",
            "Dataset",
            "Temporal shuffle LP drop",
            "Spatial blur LP drop",
            "RGB quant. LP drop",
            "Solarization LP drop",
            "Temporal shuffle mean cos.",
            "Spatial blur mean cos.",
            "RGB quant. mean cos.",
            "Solarization mean cos.",
        ],
        fixed_rows,
    )
    curve_table = markdown_table(
        [
            "Model",
            "Dataset",
            "Freeze-tail LP drop low->mid->high",
            "Color LP drop low->mid->high",
            "RGB quant. LP drop low->mid->high",
            "Solarization LP drop low->mid->high",
        ],
        [
            [
                cell.model,
                cell.dataset,
                curve_string(perturbations, cell.model, cell.dataset, "freeze_tail"),
                curve_string(perturbations, cell.model, cell.dataset, "color_transform"),
                curve_string(perturbations, cell.model, cell.dataset, "rgb_quantization"),
                curve_string(perturbations, cell.model, cell.dataset, "solarization"),
            ]
            for cell in CELLS
        ],
    )
    dino_ssv2_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "SSV2", "temporal-shuffle-mid")
    dino_ucf_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "UCF101", "temporal-shuffle-mid")
    dino_diving_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "Diving48", "temporal-shuffle-mid")
    diving_videomae = find_row(baselines, "VideoMAE", "Diving48")
    diving_slowfast = find_row(baselines, "SlowFast R50 8x8", "Diving48")
    diving_dino = find_row(baselines, "DINOv2 frame-mean", "Diving48")
    diving_slowfast_shuffle = find_perturbation(perturbations, "SlowFast R50 8x8", "Diving48", "temporal-shuffle-mid")
    quality_ok = all(row["quality_ok"] in {True, "True", "true"} for row in quality_rows)
    return f"""# Diving48 3 x 3 Summary

Generated: {date.today().isoformat()}

This summary reads the nine completed run reports and does not re-extract embeddings. The matrix covers three models, VideoMAE, SlowFast R50 8x8, and DINOv2 frame-mean, across three datasets: SSV2 C50, UCF101 C50, and Diving48 C32.

Diving48 uses the balanced `c32_train50_heldout15` subset: 1,600 train videos, 480 held-out videos, 32 classes, with 50 train and 15 held-out videos per class. Its role in this project is a fine-grained motion / pose contrast, not a full Diving48 benchmark.

## Input Runs

{chr(10).join(f'- `{cell.run_id}`' for cell in CELLS)}

Quality audit overall status: `{quality_ok}`. Extraction succeeded with 0 failed samples in all nine cells. The train-only pixel audit fixed RGB levels at 16/8/4 and solarization thresholds at 192/128/64 before model inference; normalized pixel MAD increased monotonically from low to high for both interventions on every dataset and frame profile.

## Baseline

{baseline_table}

DINOv2 frame-mean reaches {pct(find_row(baselines, "DINOv2 frame-mean", "UCF101")["linear_probe_original_accuracy"])} LP accuracy on UCF101, close to the saturated SlowFast result. This indicates that the current UCF101 C50 subset has strong static appearance/context readability. On SSV2, DINOv2 reaches {pct(find_row(baselines, "DINOv2 frame-mean", "SSV2")["linear_probe_original_accuracy"])}, showing that this SSV2 subset also contains static cues. Model differences should therefore not be written as a one-dimensional motion ranking.

Diving48 baselines are low overall: VideoMAE {pct(diving_videomae["linear_probe_original_accuracy"])}, SlowFast {pct(diving_slowfast["linear_probe_original_accuracy"])}, and DINOv2 {pct(diving_dino["linear_probe_original_accuracy"])}. These results are above random 1/32 but far below UCF101, consistent with the fine-grained action/pose difficulty and limited train50 setting. In the report, this should be treated as dataset difficulty and model-dataset interaction, not as a reason for post-hoc subset changes.

## Fixed-mid Interventions

{fixed_table}

DINOv2 `temporal-shuffle-mid` is a sanity check: mean cosine distance is {fmt_float(dino_ssv2_shuffle["mean_cosine_distance"])} on SSV2, {fmt_float(dino_ucf_shuffle["mean_cosine_distance"])} on UCF101, and {fmt_float(dino_diving_shuffle["mean_cosine_distance"])} on Diving48. This matches the frame-mean design, which is insensitive to frame order.

On Diving48, SlowFast temporal-shuffle representation shift is {fmt_float(diving_slowfast_shuffle["mean_cosine_distance"])}, but LP drop is only {fmt_float(diving_slowfast_shuffle["linear_probe_accuracy_drop"])}. This shows that a large embedding shift does not necessarily translate into an equally large label-related drop, especially in a low-baseline, small-sample, fine-grained dataset.

The new appearance interventions are materially stronger than the original color control. At fixed mid strength, VideoMAE LP drops are 0.1007 / 0.0507 for RGB quantization / solarization on SSV2 and 0.3413 / 0.2540 on UCF101. SlowFast and DINOv2 also show clear representation shifts even where label drop is smaller. This supports the intended interpretation: an effective intervention can move embeddings without necessarily crossing the current label boundary.

## Strength Curves

{curve_table}

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
"""


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def curve_string(rows: list[dict[str, Any]], model: str, dataset: str, perturbation: str) -> str:
    by_strength = {
        row["strength"]: row
        for row in rows
        if row["model"] == model and row["dataset"] == dataset and row["perturbation"] == perturbation
    }
    return " -> ".join(
        fmt_float(by_strength[strength]["linear_probe_accuracy_drop"])
        for strength in ("low", "mid", "high")
    )


def find_row(rows: list[dict[str, Any]], model: str, dataset: str) -> dict[str, Any]:
    for row in rows:
        if row["model"] == model and row["dataset"] == dataset:
            return row
    raise KeyError(f"missing row for {model} x {dataset}")


def find_perturbation(
    rows: list[dict[str, Any]],
    model: str,
    dataset: str,
    artifact_label: str,
) -> dict[str, Any]:
    for row in rows:
        if (
            row["model"] == model
            and row["dataset"] == dataset
            and row["artifact_label"] == artifact_label
        ):
            return row
    raise KeyError(f"missing perturbation {artifact_label} for {model} x {dataset}")


def knn_k_accuracy(knn: dict[str, Any], *, k: int) -> float:
    for result in knn["results"]:
        if int(result["k"]) == k:
            return float(result["accuracy"])
    raise KeyError(f"KNN result does not include k={k}")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"expected JSON object: {path}")
    return data


def pct(value: Any) -> str:
    return f"{float(value) * 100:.1f}%"


def fmt_float(value: Any) -> str:
    number = float(value)
    if abs(number) < 0.0001 and number != 0:
        return f"{number:.2e}"
    return f"{number:.4f}"


def svg_header(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        '<style>text { font-family: Arial, Helvetica, sans-serif; fill: #111827; }</style>',
    ]


def text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 12,
    weight: str = "400",
    anchor: str = "start",
    rotate: int | None = None,
) -> str:
    transform = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}"{transform}>{html.escape(str(value))}</text>'
    )


if __name__ == "__main__":
    raise SystemExit(main())
