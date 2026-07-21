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
import math
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
    Cell("HMDB51", "action-recognition", "VideoMAE", "MCG-NJU/videomae-base",
         "hmdb51-full-split1-videomae-base-frozen-linear-probe"),
    Cell("HMDB51", "action-recognition", "SlowFast R50 8x8", "facebookresearch/pytorchvideo:slowfast_r50",
         "hmdb51-full-split1-slowfast-r50-8x8-frozen-linear-probe"),
    Cell("Kinetics", "large-scale web-video", "VideoMAE", "MCG-NJU/videomae-base",
         "kinetics-c50-train100-heldout30-videomae-base-frozen-linear-probe"),
    Cell("Kinetics", "large-scale web-video", "DisMo", "motion_extractor_large",
         "kinetics-c50-train100-heldout30-dismo-motion-extractor-large-frozen-linear-probe"),
    Cell("Kinetics", "large-scale web-video", "V-JEPA2", "facebook/vjepa2-vitl-fpc64-256",
         "kinetics-c50-train100-heldout30-vjepa2-vitl-fpc64-256-frozen-linear-probe"),
    # --- filled in 2026-07-19: SLURM (LMU IfI CIP pool) + RunPod ---
    Cell("HMDB51", "action-recognition", "DINOv2 frame-mean", "facebook/dinov2-base",
         "hmdb51-full-split1-dinov2-base-frame-mean-frozen-linear-probe"),
    Cell("Kinetics", "large-scale web-video", "DINOv2 frame-mean", "facebook/dinov2-base",
         "kinetics-c50-train100-heldout30-dinov2-base-frame-mean-frozen-linear-probe"),
    Cell("Kinetics", "large-scale web-video", "SlowFast R50 8x8", "facebookresearch/pytorchvideo:slowfast_r50",
         "kinetics-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe"),
    Cell("Diving48", "fine-grained motion / pose contrast", "DisMo", "motion_extractor_large",
         "diving48-c32-train50-heldout15-dismo-motion-extractor-large-frozen-linear-probe"),
    Cell("Diving48", "fine-grained motion / pose contrast", "V-JEPA2", "facebook/vjepa2-vitl-fpc64-256",
         "diving48-c32-train50-heldout15-vjepa2-vitl-fpc64-256-frozen-linear-probe"),
    Cell("HMDB51", "action-recognition", "DisMo", "motion_extractor_large",
         "hmdb51-full-split1-dismo-motion-extractor-large-frozen-linear-probe"),
    Cell("SSV2", "motion-oriented", "DisMo", "motion_extractor_large",
         "ssv2-c50-train100-heldout30-dismo-motion-extractor-large-frozen-linear-probe"),
    Cell("UCF101", "appearance-rich / context-correlated contrast", "V-JEPA2",
         "facebook/vjepa2-vitl-fpc64-256",
         "ucf101-c50-train100-heldout30-vjepa2-vitl-fpc64-256-frozen-linear-probe"),
)


DATASET_ORDER = ("SSV2", "UCF101", "Diving48", "HMDB51", "Kinetics")
MODEL_ORDER = ("VideoMAE", "SlowFast R50 8x8", "DINOv2 frame-mean", "V-JEPA2", "DisMo")
ARTIFACT_ORDER = (
    "temporal-shuffle-mid",
    "freeze-tail-low", "freeze-tail-mid", "freeze-tail-high",
    "color-low", "color-mid", "color-high",
    "spatial-blur-mid",
    "rgb-quantization-low", "rgb-quantization-mid", "rgb-quantization-high",
    "solarization-low", "solarization-mid", "solarization-high",
)
GROUP_COLORS = {"motion": "#2563eb", "appearance": "#ea580c"}


CELL_COLORS = {
    "VideoMAE x SSV2": "#3b82f6",
    "VideoMAE x UCF101": "#0f766e",
    "VideoMAE x Diving48": "#9333ea",
    "VideoMAE x Kinetics": "#65a30d",
    "VideoMAE x HMDB51": "#1d4ed8",
    "SlowFast R50 8x8 x SSV2": "#dc2626",
    "SlowFast R50 8x8 x UCF101": "#d97706",
    "SlowFast R50 8x8 x Diving48": "#0891b2",
    "SlowFast R50 8x8 x HMDB51": "#991b1b",
    "DINOv2 frame-mean x SSV2": "#7c3aed",
    "DINOv2 frame-mean x UCF101": "#16a34a",
    "DINOv2 frame-mean x Diving48": "#be123c",
    "V-JEPA2 x HMDB51": "#334155",
    "V-JEPA2 x Kinetics": "#b45309",
    "V-JEPA2 x Diving48": "#475569",
    "V-JEPA2 x UCF101": "#64748b",
    "DisMo x Kinetics": "#db2777",
    "DisMo x Diving48": "#c026d3",
    "DisMo x HMDB51": "#9f1239",
    "DisMo x SSV2": "#f472b6",
    "DINOv2 frame-mean x HMDB51": "#a855f7",
    "DINOv2 frame-mean x Kinetics": "#059669",
    "SlowFast R50 8x8 x Kinetics": "#ea580c",
}

# For the strength-curve chart: one color per model, one dash pattern per
# dataset, so lines for the same model (different datasets) read as "the
# same series, different condition" instead of needing 13 unrelated colors.
MODEL_COLORS = {
    "VideoMAE": "#2563eb",
    "SlowFast R50 8x8": "#dc2626",
    "DINOv2 frame-mean": "#7c3aed",
    "V-JEPA2": "#334155",
    "DisMo": "#db2777",
}

# Comparability reminder: deterministic_center_clip samples num_frames
# *consecutive* native frames, so a model's num_frames controls how much
# real-world video duration it sees, not just temporal resolution -- e.g. at
# 30fps, 16 frames = ~0.53s vs V-JEPA2's 64 frames = ~2.13s of the same
# source clip. Motion perturbations (freeze_tail/temporal_shuffle) are
# defined as a fraction of that sampled window, so this confounds any
# cross-model claim about motion-perturbation magnitude unless window_frames
# is set (not yet applied to any cell -- see project memory). Displayed on
# the grid so it stays visible rather than a footnote nobody rereads.
NUM_FRAMES_BY_MODEL = {
    "VideoMAE": 16,
    "SlowFast R50 8x8": 32,
    "DINOv2 frame-mean": 16,
    "V-JEPA2": 64,
    "DisMo": 16,
}
DATASET_DASH = {
    "SSV2": None,
    "UCF101": "12 5",
    "Diving48": "1 4",
    "HMDB51": "9 3 2 3",
    "Kinetics": "5 3",
}


def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    baseline_rows = build_baseline_rows()
    perturbation_rows = build_perturbation_rows()
    quality_rows = build_quality_rows()
    bias_rows = build_bias_rows(perturbation_rows)

    write_csv(REPORT_DIR / "matrix_baselines.csv", baseline_rows)
    write_csv(REPORT_DIR / "matrix_perturbation_summary.csv", perturbation_rows)
    write_csv(REPORT_DIR / "matrix_quality_summary.csv", quality_rows)
    write_csv(REPORT_DIR / "matrix_motion_appearance_bias.csv", bias_rows)

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
    write_bias_scatter_chart(
        bias_rows,
        output_path=PLOT_DIR / "matrix_motion_appearance_scatter.svg",
    )
    write_bias_ratio_bar_chart(
        bias_rows,
        output_path=PLOT_DIR / "matrix_motion_appearance_bias_ratio.svg",
    )
    write_matrix_grid_chart(
        perturbation_rows,
        baseline_rows,
        output_path=PLOT_DIR / "matrix_per_cell_grid.svg",
    )

    (REPORT_DIR / "full_matrix_summary.md").write_text(
        build_summary_markdown(baseline_rows, perturbation_rows, quality_rows, bias_rows),
        encoding="utf-8",
    )
    return 0


def build_bias_rows(perturbation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-cell motion-vs-appearance bias, contrasting matched perturbation pairs.

    Two complementary metrics, since neither is comparable across cells alone:

    - Behavioral: correct_to_incorrect_rate for temporal-shuffle-mid vs
      spatial-blur-mid. This conditions on originally-correct predictions,
      so it stays meaningful even for cells with a low original accuracy
      (e.g. VideoMAE x Diving48 at 7.5%), unlike raw accuracy drop which is
      capped by how much accuracy there was to lose.
    - Representational: mean_cosine_distance for the same pair, expressed as
      log2(motion / appearance). Raw cosine distance is not comparable
      across embedding spaces (SlowFast's 9216-d space produces much larger
      distances than DINOv2's 768-d space for the same nominal strength),
      but the *ratio* within one cell's own embedding space is.
    """
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        shuffle = find_perturbation_optional(perturbation_rows, cell.model, cell.dataset, "temporal-shuffle-mid")
        blur = find_perturbation_optional(perturbation_rows, cell.model, cell.dataset, "spatial-blur-mid")
        freeze_high = find_perturbation_optional(perturbation_rows, cell.model, cell.dataset, "freeze-tail-high")
        color_high = find_perturbation_optional(perturbation_rows, cell.model, cell.dataset, "color-high")
        if shuffle is None or blur is None:
            continue
        motion_flip = float(shuffle["correct_to_incorrect_rate"])
        appearance_flip = float(blur["correct_to_incorrect_rate"])
        row = {
            "model": cell.model,
            "dataset": cell.dataset,
            "dataset_role": cell.dataset_role,
            "motion_flip_rate": motion_flip,
            "appearance_flip_rate": appearance_flip,
            "behavioral_bias_diff": motion_flip - appearance_flip,
            "motion_mean_cosine_distance": float(shuffle["mean_cosine_distance"]),
            "appearance_mean_cosine_distance": float(blur["mean_cosine_distance"]),
            "repr_bias_log2_ratio": log2_ratio(
                float(shuffle["mean_cosine_distance"]), float(blur["mean_cosine_distance"])
            ),
        }
        if freeze_high is not None and color_high is not None:
            row["repr_bias_log2_ratio_strong"] = log2_ratio(
                float(freeze_high["mean_cosine_distance"]), float(color_high["mean_cosine_distance"])
            )
        else:
            row["repr_bias_log2_ratio_strong"] = ""
        rows.append(row)
    return rows


def log2_ratio(motion_value: float, appearance_value: float, *, eps: float = 1e-9) -> float:
    """log2(motion/appearance), clipped to +-10 (1024x) so a near-zero value

    from genuine order-invariance (e.g. DINOv2 frame-mean pooling makes
    temporal-shuffle cosine distance exactly 0.0) doesn't blow up the scale
    -- the clipped value still reads as "as motion-insensitive as this chart
    can show," which is the correct qualitative signal.
    """
    ratio = (motion_value + eps) / (appearance_value + eps)
    return max(-10.0, min(10.0, math.log2(ratio)))


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
    width, height = 1360, 1720
    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, title, size=18, weight="700", anchor="middle"))
    parts.append(
        text(
            width / 2, 46,
            "Cells built before the matrix expansion have no rgb-quantization/solarization data "
            "and are omitted from those two panels",
            size=10, anchor="middle",
        )
    )
    panel_specs = (
        ("freeze_tail", "Freeze-tail low -> mid -> high", 65),
        ("color_transform", "Color transform low -> mid -> high", 475),
        ("rgb_quantization", "RGB quantization low -> mid -> high", 885),
        ("solarization", "Solarization low -> mid -> high", 1295),
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
        color = MODEL_COLORS[cell.model]
        dash = DATASET_DASH[cell.dataset]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        points = [
            (x_for(strength), y_for(float(rows_by_cell[label][strength][metric])))
            for strength in strengths
        ]
        point_data = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        parts.append(
            f'<polyline points="{point_data}" fill="none" stroke="{color}" '
            f'stroke-width="2.3" stroke-linejoin="round"{dash_attr} />'
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
        color = MODEL_COLORS[cell.model]
        dash = DATASET_DASH[cell.dataset]
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 18}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"{dash_attr} />'
        )
        parts.append(text(legend_x + 24, y + 4, label, size=10))


def write_bias_scatter_chart(bias_rows: list[dict[str, Any]], *, output_path: Path) -> None:
    width, height = 900, 820
    left, right, top, bottom = 90, 260, 55, 90
    plot_width = width - left - right
    plot_height = height - top - bottom

    parts = svg_header(width, height)
    parts.append(
        text(
            width / 2, 28,
            "Motion vs appearance flip rate (correct-to-incorrect)",
            size=18, weight="700", anchor="middle",
        )
    )
    axis_max = max(
        0.05,
        max(row["motion_flip_rate"] for row in bias_rows),
        max(row["appearance_flip_rate"] for row in bias_rows),
    ) * 1.1

    def x_for(value: float) -> float:
        return left + value / axis_max * plot_width

    def y_for(value: float) -> float:
        return top + (axis_max - value) / axis_max * plot_height

    parts.append(
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" '
        'stroke="#111827" stroke-width="1" />'
    )
    parts.append(
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1" />'
    )
    parts.append(
        f'<line x1="{x_for(0):.2f}" y1="{y_for(0):.2f}" x2="{x_for(axis_max):.2f}" y2="{y_for(axis_max):.2f}" '
        'stroke="#9ca3af" stroke-dasharray="5 5" />'
    )
    parts.append(text(left + plot_width / 2, height - 45, "Appearance flip rate (spatial-blur-mid)", size=12, anchor="middle"))
    parts.append(text(28, top + plot_height / 2, "Motion flip rate (temporal-shuffle-mid)", size=12, anchor="middle", rotate=-90))
    parts.append(text(left + plot_width - 10, y_for(axis_max) + 14, "motion-biased side", size=11, anchor="end", weight="600"))
    parts.append(text(left + 10, y_for(0) - 8, "appearance-biased side", size=11, anchor="start", weight="600"))

    ticks = 5
    for i in range(ticks + 1):
        value = axis_max * i / ticks
        parts.append(text(x_for(value), top + plot_height + 18, f"{value:.2f}", size=10, anchor="middle"))
        parts.append(text(left - 8, y_for(value) + 4, f"{value:.2f}", size=10, anchor="end"))

    for row in bias_rows:
        label = row["model"] + " x " + row["dataset"]
        color = CELL_COLORS[label]
        cx = x_for(row["appearance_flip_rate"])
        cy = y_for(row["motion_flip_rate"])
        parts.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="7" fill="{color}" stroke="#111827" stroke-width="0.8" />')

    legend_x = width - right + 25
    legend_y = top + 10
    for idx, cell in enumerate(CELLS):
        label = cell.short_label
        y = legend_y + idx * 20
        color = CELL_COLORS[label]
        parts.append(f'<circle cx="{legend_x}" cy="{y}" r="5.5" fill="{color}" stroke="#111827" stroke-width="0.8" />')
        parts.append(text(legend_x + 12, y + 4, label, size=10))

    parts.append("</svg>\n")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_bias_ratio_bar_chart(bias_rows: list[dict[str, Any]], *, output_path: Path) -> None:
    ordered = sorted(bias_rows, key=lambda row: row["repr_bias_log2_ratio"], reverse=True)
    labels = [f'{row["model"]} x {row["dataset"]}' for row in ordered]

    width = max(1000, 70 * len(labels))
    height = 620
    left, right, top, bottom = 60, 30, 70, 190
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min, y_max = -10.5, 10.5

    def x_for_group(index: int) -> float:
        return left + (index + 0.5) * plot_width / len(labels)

    def y_for(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    zero_y = y_for(0.0)
    bar_width = min(46.0, plot_width / len(labels) * 0.6)

    parts = svg_header(width, height)
    parts.append(
        text(
            width / 2, 26,
            "Representation-shift bias: log2(motion cos. dist. / appearance cos. dist.), mid strength",
            size=15, weight="700", anchor="middle",
        )
    )
    parts.append(text(16, top + plot_height / 2, "log2 ratio (motion / appearance)", size=11, anchor="middle", rotate=-90))
    parts.extend(axis_parts(left, top, plot_width, plot_height, y_min, y_max, y_for))
    parts.append(text(left + plot_width - 5, top + 14, "motion-dominant", size=10, anchor="end", weight="600"))
    parts.append(text(left + plot_width - 5, top + plot_height - 6, "appearance-dominant", size=10, anchor="end", weight="600"))

    for index, row in enumerate(ordered):
        value = row["repr_bias_log2_ratio"]
        center = x_for_group(index)
        color = "#2563eb" if value >= 0 else "#ea580c"
        y = min(y_for(value), zero_y)
        bar_height = abs(y_for(value) - zero_y)
        parts.append(
            f'<rect x="{center - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
            f'height="{max(bar_height, 1.0):.2f}" fill="{color}" />'
        )
        parts.append(text(center, height - 130, labels[index], size=10, anchor="end", rotate=-40))

    parts.append("</svg>\n")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_matrix_grid_chart(
    perturbation_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    """One small multiple per model x dataset cell, laid out as a literal grid.

    Rows are datasets, columns are models -- the same axes as the CELLS
    matrix -- so a blank tile means that combination was never run, and a
    reader can visually compare either a whole row (one dataset, every
    model) or a whole column (one model, every dataset). Every tile shares
    the same y-axis so bar heights are comparable across the entire grid,
    not just within a tile.
    """
    by_cell: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in perturbation_rows:
        by_cell.setdefault((row["model"], row["dataset"]), {})[row["artifact_label"]] = row
    baseline_by_cell = {(row["model"], row["dataset"]): row for row in baseline_rows}

    metric = "correct_to_incorrect_rate"
    all_values = [
        float(row[metric]) for row in perturbation_rows if row["artifact_label"] in ARTIFACT_ORDER
    ]
    y_max = max(all_values) * 1.12

    label_col_w, header_h = 130, 66
    tile_w, tile_h = 190, 150
    width = label_col_w + tile_w * len(MODEL_ORDER)
    height = header_h + tile_h * len(DATASET_ORDER) + 70

    parts = svg_header(width, height)
    parts.append(
        text(
            width / 2, 22,
            "Per-cell perturbation profile (correct-to-incorrect rate, up to 14 artifacts "
            "-- cells built before the matrix expansion show only the original 8)",
            size=15, weight="700", anchor="middle",
        )
    )
    parts.append(
        text(
            width / 2, 40,
            "Frame counts below each model differ (16-64) and are NOT yet normalized to a "
            "shared window -- motion-perturbation magnitude is not directly comparable across models",
            size=10, weight="400", anchor="middle",
        )
    )

    for col, model in enumerate(MODEL_ORDER):
        cx = label_col_w + (col + 0.5) * tile_w
        parts.append(text(cx, header_h + 20, model, size=12, weight="700", anchor="middle"))
        frames = NUM_FRAMES_BY_MODEL.get(model)
        if frames is not None:
            parts.append(
                text(
                    cx, header_h + 36, f"{frames} frames sampled",
                    size=10, weight="400", anchor="middle",
                )
            )

    for row_idx, dataset in enumerate(DATASET_ORDER):
        row_y = header_h + 40 + row_idx * tile_h
        parts.append(
            text(label_col_w - 10, row_y + tile_h / 2, dataset, size=12, weight="700", anchor="end")
        )
        for col, model in enumerate(MODEL_ORDER):
            tile_x = label_col_w + col * tile_w
            parts.append(
                f'<rect x="{tile_x + 4}" y="{row_y + 4}" width="{tile_w - 8}" height="{tile_h - 8}" '
                'fill="none" stroke="#e5e7eb" stroke-width="1" />'
            )
            cell_key = (model, dataset)
            if cell_key not in by_cell:
                parts.append(
                    text(
                        tile_x + tile_w / 2, row_y + tile_h / 2, "no run",
                        size=10, anchor="middle",
                    )
                )
                continue
            write_grid_tile(
                parts,
                artifacts=by_cell[cell_key],
                baseline=baseline_by_cell.get(cell_key),
                metric=metric,
                y_max=y_max,
                x=tile_x + 4,
                y=row_y + 4,
                w=tile_w - 8,
                h=tile_h - 8,
            )

    legend_y = height - 22
    parts.append(f'<rect x="{label_col_w}" y="{legend_y - 10}" width="12" height="12" fill="{GROUP_COLORS["motion"]}" />')
    parts.append(text(label_col_w + 18, legend_y, "motion perturbation", size=11))
    parts.append(f'<rect x="{label_col_w + 170}" y="{legend_y - 10}" width="12" height="12" fill="{GROUP_COLORS["appearance"]}" />')
    parts.append(text(label_col_w + 188, legend_y, "appearance perturbation", size=11))
    parts.append(
        text(
            label_col_w + 400, legend_y,
            "bar order (left to right): shuffle, freeze-low/mid/high, color-low/mid/high, blur, "
            "rgb-quant-low/mid/high, solarization-low/mid/high",
            size=9,
        )
    )

    parts.append("</svg>\n")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_grid_tile(
    parts: list[str],
    *,
    artifacts: dict[str, dict[str, Any]],
    baseline: dict[str, Any] | None,
    metric: str,
    y_max: float,
    x: float,
    y: float,
    w: float,
    h: float,
) -> None:
    top_pad, bottom_pad = 18, 4
    plot_h = h - top_pad - bottom_pad
    baseline_acc = float(baseline["linear_probe_original_accuracy"]) if baseline else None
    title = f"LP orig. {baseline_acc * 100:.0f}%" if baseline_acc is not None else ""
    parts.append(text(x + w / 2, y + 12, title, size=9, anchor="middle"))

    n = len(ARTIFACT_ORDER)
    gap = 2.0
    bar_w = (w - gap * (n - 1)) / n
    for i, artifact_label in enumerate(ARTIFACT_ORDER):
        row = artifacts.get(artifact_label)
        bx = x + i * (bar_w + gap)
        baseline_y = y + top_pad + plot_h
        if row is None:
            continue
        value = max(0.0, float(row[metric]))
        color = GROUP_COLORS.get(row["group"], "#6b7280")
        bar_h = 0.0 if y_max <= 0 else min(1.0, value / y_max) * plot_h
        parts.append(
            f'<rect x="{bx:.2f}" y="{baseline_y - bar_h:.2f}" width="{bar_w:.2f}" '
            f'height="{max(bar_h, 0.6):.2f}" fill="{color}" />'
        )
    parts.append(
        f'<line x1="{x}" y1="{y + top_pad + plot_h:.2f}" x2="{x + w}" y2="{y + top_pad + plot_h:.2f}" '
        'stroke="#9ca3af" stroke-width="1" />'
    )


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
    bias_rows: list[dict[str, Any]],
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

    bias_table = markdown_table(
        [
            "Model", "Dataset", "Motion flip rate (shuffle)", "Appearance flip rate (blur)",
            "Behavioral bias (motion - appearance)", "Repr. log2(motion/appearance), mid",
        ],
        [
            [
                row["model"], row["dataset"],
                fmt_float(row["motion_flip_rate"]),
                fmt_float(row["appearance_flip_rate"]),
                fmt_float(row["behavioral_bias_diff"]),
                fmt_float(row["repr_bias_log2_ratio"]),
            ]
            for row in sorted(bias_rows, key=lambda r: r["behavioral_bias_diff"], reverse=True)
        ],
    )

    model_bias_table = markdown_table(
        ["Model", "Datasets averaged", "Mean behavioral bias", "Mean repr. log2 ratio, mid"],
        model_bias_rows(bias_rows),
    )

    divergent = [
        row for row in bias_rows
        if row["behavioral_bias_diff"] * row["repr_bias_log2_ratio"] < 0
        and abs(row["behavioral_bias_diff"]) > 0.01
    ]
    if divergent:
        divergent_lines = "\n".join(
            f'- {row["model"]} x {row["dataset"]}: representation shifts more from '
            f'{"appearance" if row["repr_bias_log2_ratio"] < 0 else "motion"} '
            f'(log2 ratio {row["repr_bias_log2_ratio"]:.2f}), but behaviorally the linear '
            f'probe flips more predictions from '
            f'{"motion" if row["behavioral_bias_diff"] > 0 else "appearance"} '
            f'(bias {row["behavioral_bias_diff"]:.4f}).'
            for row in divergent
        )
        divergent_block = f"""
Cells where representational and behavioral bias disagree in sign -- a
larger embedding shift from one perturbation type does not always mean the
frozen linear probe's decision is more sensitive to it:

{divergent_lines}
"""
    else:
        divergent_block = ""

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

{bias_table}

DINOv2 frame-mean's `0.0000` motion columns are not a rounding artifact:
frame-mean pooling averages per-frame features, and a mean is exactly
invariant to the order the frames are averaged in, so temporal-shuffle
cannot move that representation at all by construction -- this is a
property of the pooling operation, not evidence that DINOv2's *frame-level*
features are appearance-only.

Model averages (unweighted mean across the datasets each model was run on
-- coverage differs by model, e.g. DisMo only has a Kinetics cell, so these
are not fully apples-to-apples across rows):

{model_bias_table}
{divergent_block}
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


def model_bias_rows(bias_rows: list[dict[str, Any]]) -> list[list[Any]]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in bias_rows:
        by_model.setdefault(row["model"], []).append(row)
    rows = []
    for model, cells in by_model.items():
        datasets = ", ".join(sorted(row["dataset"] for row in cells))
        mean_behavioral = sum(row["behavioral_bias_diff"] for row in cells) / len(cells)
        mean_repr = sum(row["repr_bias_log2_ratio"] for row in cells) / len(cells)
        rows.append([model, datasets, fmt_float(mean_behavioral), fmt_float(mean_repr)])
    rows.sort(key=lambda row: float(row[2]), reverse=True)
    return rows


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
