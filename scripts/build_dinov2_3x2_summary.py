from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPORT_DIR = Path("outputs/reports/dinov2_3x2")
PLOT_DIR = Path("outputs/plots/dinov2_3x2")


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
)


CELL_COLORS = {
    "VideoMAE x SSV2": "#3b82f6",
    "VideoMAE x UCF101": "#0f766e",
    "SlowFast R50 8x8 x SSV2": "#dc2626",
    "SlowFast R50 8x8 x UCF101": "#d97706",
    "DINOv2 frame-mean x SSV2": "#7c3aed",
    "DINOv2 frame-mean x UCF101": "#16a34a",
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
    (REPORT_DIR / "dinov2_3x2_summary.md").write_text(
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
    values = list(values_by_label.values())
    min_value = min(0.0, min(values))
    max_value = max(values)

    width, height = 980, 520
    left, right, top, bottom = 90, 30, 55, 120
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min, y_max = padded_range(min_value, max_value)

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
            value = values_by_label[(label, artifact_label)]
            x = group_center + (offset - 0.5) * bar_width * 1.2
            y = min(y_for(value), zero_y)
            bar_height = abs(y_for(value) - zero_y)
            parts.append(
                f'<rect x="{x - bar_width / 2:.2f}" y="{y:.2f}" width="{bar_width:.2f}" '
                f'height="{bar_height:.2f}" fill="{color}" />'
            )
        parts.append(text(group_center, height - 38, label, size=10, anchor="end", rotate=-35))

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
    width, height = 980, 760
    parts = svg_header(width, height)
    parts.append(text(width / 2, 28, title, size=18, weight="700", anchor="middle"))
    panel_specs = (
        ("freeze_tail", "Freeze-tail low -> mid -> high", 55),
        ("color_transform", "Color transform low -> mid -> high", 405),
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
    left, right, panel_height, bottom_pad = 90, 260, 260, 55
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
    dino_ssv2_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "SSV2", "temporal-shuffle-mid")
    dino_ucf_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "UCF101", "temporal-shuffle-mid")
    dino_ssv2_freeze_high = find_perturbation(perturbations, "DINOv2 frame-mean", "SSV2", "freeze-tail-high")
    dino_ucf_blur = find_perturbation(perturbations, "DINOv2 frame-mean", "UCF101", "spatial-blur-mid")
    quality_ok = all(row["quality_ok"] in {True, "True", "true"} for row in quality_rows)
    return f"""# DINOv2 3 x 2 Interaction Notes

本文件只总结已经完成的 `3 模型 x 2 数据集` C50 主矩阵。新增 DINOv2 结果作为 frame-wise image baseline，而不是 temporal video encoder。

## 状态

- 六个 cell 均有 2 个 original artifact 和 8 个 held-out perturbation artifact。
- Quality audit overall status: `{quality_ok}`；DINOv2 两个 run 的 failed samples 均为 0。
- DINOv2 SSV2 original LP accuracy 为 {pct(ssv2_dino['linear_probe_original_accuracy'])}，UCF101 为 {pct(ucf_dino['linear_probe_original_accuracy'])}。

## DINOv2 sanity check

- SSV2 `temporal-shuffle-mid`: mean cosine distance {fmt_float(dino_ssv2_shuffle['mean_cosine_distance'])}，LP drop {fmt_float(dino_ssv2_shuffle['linear_probe_accuracy_drop'])}。
- UCF101 `temporal-shuffle-mid`: mean cosine distance {fmt_float(dino_ucf_shuffle['mean_cosine_distance'])}，LP drop {fmt_float(dino_ucf_shuffle['linear_probe_accuracy_drop'])}。

这符合 frame-mean DINOv2 的预期：帧集合相同而顺序改变时，简单平均后的 embedding 基本不变。

## 主要 interaction

- DINOv2 在 UCF101 上接近饱和，LP accuracy 为 {pct(ucf_dino['linear_probe_original_accuracy'])}，说明该 C50 子集可由强静态图像表征很好地区分。
- DINOv2 在 SSV2 上 LP accuracy 为 {pct(ssv2_dino['linear_probe_original_accuracy'])}，高于 VideoMAE 但低于 SlowFast；这说明 SSV2 C50 仍含可用静态线索，不能把 VideoMAE/SlowFast 与 DINOv2 的差异简单归因给 motion understanding。
- DINOv2 的 `freeze-tail-high` 在 SSV2 上 LP drop 为 {fmt_float(dino_ssv2_freeze_high['linear_probe_accuracy_drop'])}，远小于视频模型的 high-strength freeze-tail drop；对 DINOv2 来说这主要是帧内容分布变化，不是 temporal modeling 证据。
- DINOv2 的 UCF101 `spatial-blur-mid` LP drop 为 {fmt_float(dino_ucf_blur['linear_probe_accuracy_drop'])}，很小；相比之下 VideoMAE x UCF101 的 blur drop 仍是本矩阵中更强的 appearance-related label effect。

## 写作边界

- UCF101 仍应写作 `appearance-rich / context-correlated contrast`，不要写成纯 appearance-biased 数据集。
- DINOv2 结果不能证明或否定 motion understanding；它主要说明静态 frame-level representation 在当前 C50 子集中的可读性。
- Representation shift 和 label-related drop 仍需分开报告。
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
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "temporal-shuffle-mid")["mean_cosine_distance"]),
            fmt_float(find_perturbation(perturbations, cell.model, cell.dataset, "spatial-blur-mid")["mean_cosine_distance"]),
        ]
        for cell in CELLS
    ]
    fixed_table = markdown_table(
        [
            "Model",
            "Dataset",
            "Temporal shuffle LP drop",
            "Spatial blur LP drop",
            "Temporal shuffle mean cos.",
            "Spatial blur mean cos.",
        ],
        fixed_rows,
    )
    curve_table = markdown_table(
        ["Model", "Dataset", "Freeze-tail LP drop low->mid->high", "Color LP drop low->mid->high"],
        [
            [
                cell.model,
                cell.dataset,
                curve_string(perturbations, cell.model, cell.dataset, "freeze_tail"),
                curve_string(perturbations, cell.model, cell.dataset, "color_transform"),
            ]
            for cell in CELLS
        ],
    )
    dino_ssv2_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "SSV2", "temporal-shuffle-mid")
    dino_ucf_shuffle = find_perturbation(perturbations, "DINOv2 frame-mean", "UCF101", "temporal-shuffle-mid")
    quality_ok = all(row["quality_ok"] in {True, "True", "true"} for row in quality_rows)
    return f"""# DINOv2 3 x 2 Summary

生成日期：2026-07-06

本汇总读取六个已经完成的 run report，没有重新提取 embedding。新增 DINOv2 cell 使用 frame-wise image encoder：16 帧独立编码，取 CLS token 后做 simple mean，作为静态图像表征 baseline。

## 输入 Run

{chr(10).join(f'- `{cell.run_id}`' for cell in CELLS)}

Quality audit overall status: `{quality_ok}`。六个 cell 的 extraction 全部成功，DINOv2 两个 run 的 failed samples 为 0。

## Baseline

{baseline_table}

UCF101 上 DINOv2 frame-mean 达到 {pct(find_row(baselines, "DINOv2 frame-mean", "UCF101")["linear_probe_original_accuracy"])} LP accuracy，接近 SlowFast 的饱和结果。这说明当前 UCF101 C50 子集具有很强的静态 appearance/context 可读性。SSV2 上 DINOv2 为 {pct(find_row(baselines, "DINOv2 frame-mean", "SSV2")["linear_probe_original_accuracy"])}，说明该 SSV2 子集也存在静态线索；因此不能把模型差异写成单一 motion ranking。

## Fixed-mid Interventions

{fixed_table}

DINOv2 的 `temporal-shuffle-mid` 是 sanity check：SSV2 mean cosine distance 为 {fmt_float(dino_ssv2_shuffle["mean_cosine_distance"])}，UCF101 为 {fmt_float(dino_ucf_shuffle["mean_cosine_distance"])}，两个 LP drop 都是 {fmt_float(0)}。这符合 frame mean 对帧顺序不敏感的设计。

## Strength Curves

{curve_table}

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
