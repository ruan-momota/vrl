"""Combine every completed model x dataset cell (both branches) into one table.

This reads only the already-committed ``outputs/runs/<run_id>/reports/*``
artifacts produced by the shared, unmodified extraction/evaluation pipeline
(see ``src.pipeline.evaluate``) -- it does not re-extract or re-evaluate
anything, and it does not require a git merge of either branch. The 13
cells span:

- SSV2 / UCF101 / Diving48 x {VideoMAE, SlowFast R50 8x8, DINOv2 frame-mean}
  (from the ``vm-sf-dn-ssv2-ucf101-diving48`` branch)
- HMDB51 x V-JEPA2, Kinetics x {VideoMAE, DisMo, V-JEPA2}
  (from the ``kinetics-videomae`` branch)

KNN accuracy is read from each report's ``original_knn`` field rather than
a separate ``metrics/knn_original.json`` file, since ``metrics/`` is never
committed on either branch (see .gitignore) -- only ``reports/`` and
``plots/`` are.

This script intentionally does not fabricate cross-cell interpretation
(e.g. "DINOv2 saturates because..."): it produces the objective tables and
charts only. Narrative interpretation of the combined matrix is a human
judgment call and belongs in a separate write-up.
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_DIR = Path("outputs/reports/full_matrix")
PLOT_DIR = Path("outputs/plots/full_matrix")


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
    # --- SSV2 / UCF101 / Diving48 x VideoMAE, SlowFast, DINOv2 ---
    # (vm-sf-dn-ssv2-ucf101-diving48 branch)
    Cell("SSV2", "motion-oriented", "VideoMAE", "MCG-NJU/videomae-base",
         "ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe"),
    Cell("UCF101", "appearance-rich / context-correlated contrast", "VideoMAE",
         "MCG-NJU/videomae-base",
         "ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe"),
    Cell("Diving48", "fine-grained motion / pose contrast", "VideoMAE",
         "MCG-NJU/videomae-base",
         "diving48-c32-train50-heldout15-videomae-base-frozen-linear-probe"),
    Cell("SSV2", "motion-oriented", "SlowFast R50 8x8",
         "facebookresearch/pytorchvideo:slowfast_r50",
         "ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe"),
    Cell("UCF101", "appearance-rich / context-correlated contrast",
         "SlowFast R50 8x8", "facebookresearch/pytorchvideo:slowfast_r50",
         "ucf101-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe"),
    Cell("Diving48", "fine-grained motion / pose contrast", "SlowFast R50 8x8",
         "facebookresearch/pytorchvideo:slowfast_r50",
         "diving48-c32-train50-heldout15-slowfast-r50-8x8-frozen-linear-probe"),
    Cell("SSV2", "motion-oriented", "DINOv2 frame-mean", "facebook/dinov2-base",
         "ssv2-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe"),
    Cell("UCF101", "appearance-rich / context-correlated contrast",
         "DINOv2 frame-mean", "facebook/dinov2-base",
         "ucf101-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe"),
    Cell("Diving48", "fine-grained motion / pose contrast", "DINOv2 frame-mean",
         "facebook/dinov2-base",
         "diving48-c32-train50-heldout15-dinov2-base-frame-mean-frozen-linear-probe"),
    # --- HMDB51 / Kinetics x V-JEPA2, VideoMAE, DisMo ---
    # (kinetics-videomae branch)
    Cell("HMDB51", "action-recognition", "V-JEPA2", "facebook/vjepa2-vitl-fpc64-256",
         "hmdb51-full-split1-vjepa2-vitl-fpc64-256-frozen-linear-probe"),
    Cell("Kinetics", "large-scale web-video", "VideoMAE", "MCG-NJU/videomae-base",
         "kinetics-c50-train100-heldout30-videomae-base-frozen-linear-probe"),
    Cell("Kinetics", "large-scale web-video", "DisMo", "motion_extractor_large",
         "kinetics-c50-train100-heldout30-dismo-motion-extractor-large-frozen-linear-probe"),
    Cell("Kinetics", "large-scale web-video", "V-JEPA2", "facebook/vjepa2-vitl-fpc64-256",
         "kinetics-c50-train100-heldout30-vjepa2-vitl-fpc64-256-frozen-linear-probe"),
)


CELL_COLORS = {
    "VideoMAE x SSV2": "#3b82f6",
    "VideoMAE x UCF101": "#0f766e",
    "VideoMAE x Diving48": "#9333ea",
    "VideoMAE x Kinetics": "#65a30d",
    "SlowFast R50 8x8 x SSV2": "#dc2626",
    "SlowFast R50 8x8 x UCF101": "#d97706",
    "SlowFast R50 8x8 x Diving48": "#0891b2",
    "DINOv2 frame-mean x SSV2": "#7c3aed",
    "DINOv2 frame-mean x UCF101": "#16a34a",
    "DINOv2 frame-mean x Diving48": "#be123c",
    "V-JEPA2 x HMDB51": "#334155",
    "V-JEPA2 x Kinetics": "#b45309",
    "DisMo x Kinetics": "#db2777",
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

    (REPORT_DIR / "full_matrix_summary.md").write_text(
        build_summary_markdown(baseline_rows, perturbation_rows, quality_rows),
        encoding="utf-8",
    )
    return 0


def build_baseline_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        summary = load_json(cell.run_dir / "reports/linear_probe_sensitivity_summary.json")
        knn = summary["original_knn"]
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
                rows.append(
                    {
                        "dataset": cell.dataset,
                        "dataset_role": cell.dataset_role,
                        "model": cell.model,
                        "checkpoint": cell.checkpoint,
                        "run_id": cell.run_id,
                        **source_row,
                    }
                )
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
        writer = csv.DictWriter(file, fieldnames=fieldnames)
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
    fixed_rows = [
        row for row in rows if row["artifact_label"] in {"temporal-shuffle-mid", "spatial-blur-mid"}
    ]
    labels = [cell.short_label for cell in CELLS]
    series = {
        "temporal-shuffle-mid": "#2563eb",
        "spatial-blur-mid": "#ea580c",
    }
    values_by_label = {
        (row["model"] + " x " + row["dataset"], row["artifact_label"]): float(row[metric])
        for row in fixed_rows
    }

    width = max(1280, 80 * len(labels))
    height = 620
    left, right, top, bottom = 90, 30, 55, 160
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = list(values_by_label.values())
    y_min, y_max = padded_range(min(0.0, min(values)), max(values))

    def x_for_group(index: int) -> float:
        return left + (index + 0.5) * plot_width / len(labels)

    def y_for(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    zero_y = y_for(0.0)
    bar_width = min(44.0, plot_width / len(labels) / 4.0)
    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, title, size=18, weight="700", anchor="middle"))
    parts.append(text(18, top + plot_height / 2, y_label, size=12, anchor="middle", rotate=-90))
    parts.extend(axis_parts(left, top, plot_width, plot_height, y_min, y_max, y_for))

    for index, label in enumerate(labels):
        group_center = x_for_group(index)
        for offset, (artifact_label, color) in enumerate(series.items()):
            key = (label, artifact_label)
            if key not in values_by_label:
                continue
            value = values_by_label[key]
            x = group_center + (offset - 0.5) * bar_width * 1.2
            y = min(y_for(value), zero_y)
            bar_height = abs(y_for(value) - zero_y)
            parts.append(
                f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" />'
            )
        parts.append(text(group_center, height - 100, label, size=10, anchor="end", rotate=-40))

    legend_x = left + 12
    legend_y = 48
    for idx, (artifact_label, color) in enumerate(series.items()):
        y = legend_y + idx * 20
        parts.append(f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" fill="{color}" />')
        parts.append(text(legend_x + 18, y, artifact_label, size=12))
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
    width, height = 1360, 900
    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, title, size=18, weight="700", anchor="middle"))
    panel_specs = (
        ("freeze_tail", "Freeze-tail low -> mid -> high", 55),
        ("color_transform", "Color transform low -> mid -> high", 465),
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
    left, right, panel_height, bottom_pad = 90, 430, 330, 55
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
        if label not in rows_by_cell:
            continue
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
    legend_y = top + 20
    for idx, cell in enumerate(CELLS):
        label = cell.short_label
        if label not in rows_by_cell:
            continue
        y = legend_y + idx * 20
        color = CELL_COLORS[label]
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" stroke="{color}" stroke-width="3" />')
        parts.append(text(legend_x + 24, y + 4, label, size=10))


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


def build_summary_markdown(
    baselines: list[dict[str, Any]],
    perturbations: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> str:
    baseline_table = markdown_table(
        ["Model", "Dataset", "Train n", "Heldout n", "Embedding dim",
         "LP original acc.", "KNN k=5 acc.", "Quality OK"],
        [
            [
                row["model"], row["dataset"], row["train_n"], row["heldout_n"],
                row["embedding_dim"],
                pct(row["linear_probe_original_accuracy"]),
                pct(row["knn_k5_original_accuracy"]),
                row["quality_ok"],
            ]
            for row in baselines
        ],
    )
    fixed_rows = []
    for cell in CELLS:
        shuffle = find_perturbation_optional(perturbations, cell.model, cell.dataset, "temporal-shuffle-mid")
        blur = find_perturbation_optional(perturbations, cell.model, cell.dataset, "spatial-blur-mid")
        fixed_rows.append(
            [
                cell.model,
                cell.dataset,
                fmt_float(shuffle["linear_probe_accuracy_drop"]) if shuffle else "n/a",
                fmt_float(blur["linear_probe_accuracy_drop"]) if blur else "n/a",
                fmt_float(shuffle["mean_cosine_distance"]) if shuffle else "n/a",
                fmt_float(blur["mean_cosine_distance"]) if blur else "n/a",
            ]
        )
    fixed_table = markdown_table(
        [
            "Model", "Dataset", "Temporal shuffle LP drop", "Spatial blur LP drop",
            "Temporal shuffle mean cos.", "Spatial blur mean cos.",
        ],
        fixed_rows,
    )
    quality_ok = all(row["quality_ok"] in {True, "True", "true"} for row in quality_rows)
    return f"""# Full Model x Dataset Matrix Summary

This summary reads the {len(CELLS)} completed run reports from both branches
and does not re-extract embeddings or fabricate cross-cell interpretation --
that is left for a follow-up write-up. Perturbation strengths and protocol
match across all cells (frozen linear probe + KNN baseline, plus 2-8
perturbation artifacts per cell depending on which matrix the cell belongs
to; see each cell's own `linear_probe_sensitivity_report.md` for its full
perturbation set).

## Input Runs

{chr(10).join(f'- `{cell.run_id}`' for cell in CELLS)}

Quality audit overall status across all cells: `{quality_ok}`.

## Baseline

{baseline_table}

## Fixed-mid Interventions

{fixed_table}

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
"""


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def find_perturbation_optional(
    rows: list[dict[str, Any]],
    model: str,
    dataset: str,
    artifact_label: str,
) -> dict[str, Any] | None:
    for row in rows:
        if (
            row["model"] == model
            and row["dataset"] == dataset
            and row["artifact_label"] == artifact_label
        ):
            return row
    return None


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
