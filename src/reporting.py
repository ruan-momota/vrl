from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

from src.embedding_extraction import load_embedding_artifact


DEFAULT_LOG_DIR = Path("outputs/logs")
DEFAULT_REPORT_DIR = Path("outputs/reports")
DEFAULT_PLOT_DIR = Path("outputs/plots")
BASE_NAME = "ssv2_validation100_videomae_base_16f_mean"
DEFAULT_ORIGINAL_ARTIFACT = Path(
    "outputs/embeddings/ssv2_validation100_videomae_base_16f_mean_original.pt"
)
FIRST_ROUND_PERTURBATIONS = {
    "temporal_reverse",
    "temporal_shuffle",
    "freeze_tail",
    "single_frame",
    "grayscale",
    "center_occlusion",
}


def build_baseline_rows(baseline_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric, metric_rows in baseline_report["metrics"].items():
        for row in metric_rows:
            rows.append(
                {
                    "split": "validation100",
                    "train_samples": baseline_report["train_samples"],
                    "validation_samples": baseline_report["validation_samples"],
                    "train_class_count": baseline_report["train_class_count"],
                    "validation_class_count": baseline_report["validation_class_count"],
                    "common_class_count": baseline_report["common_class_count"],
                    "metric": metric,
                    "k": row["k"],
                    "all_correct": row["all_correct"],
                    "all_total": row["all_total"],
                    "all_accuracy": row["all_accuracy"],
                    "train_seen_correct": row["train_seen_correct"],
                    "train_seen_total": row["train_seen_total"],
                    "train_seen_accuracy": row["train_seen_accuracy"],
                }
            )
    return rows


def build_perturbation_rows(
    sensitivity_summary: dict[str, Any],
    knn_drop_summary: dict[str, Any],
    *,
    k: int = 1,
) -> list[dict[str, Any]]:
    k_results = {
        item["perturbation"]: item
        for item in knn_drop_summary["ranked_by_train_seen_accuracy_drop"][str(k)]
    }
    rows: list[dict[str, Any]] = []
    for sensitivity in sensitivity_summary["ranked_by_mean_cosine_distance"]:
        perturbation = sensitivity["perturbation"]
        knn = k_results[perturbation]
        rows.append(
            {
                "perturbation": perturbation,
                "group": sensitivity["perturbation_group"],
                "mean_cosine_distance": sensitivity["mean_cosine_distance"],
                "median_cosine_distance": sensitivity["median_cosine_distance"],
                "mean_l2_distance": sensitivity["mean_l2_distance"],
                "mean_relative_l2_distance": sensitivity["mean_relative_l2_distance"],
                "knn_k": k,
                "all_perturbed_accuracy": knn["all_perturbed_accuracy"],
                "all_accuracy_drop": knn["all_absolute_accuracy_drop"],
                "train_seen_perturbed_accuracy": knn["train_seen_perturbed_accuracy"],
                "train_seen_accuracy_drop": knn["train_seen_absolute_accuracy_drop"],
                "prediction_change_rate": knn["all_prediction_change_rate"],
            }
        )
    return rows


def build_class_rows(class_report: dict[str, Any], *, top_n: int = 20) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in class_report["classes"][:top_n]:
        rows.append(
            {
                "label_id": item["label_id"],
                "label_name": item["label_name"],
                "sample_count": item["sample_count"],
                "mean_cosine_distance_across_perturbations": item[
                    "mean_cosine_distance_across_perturbations"
                ],
                "strongest_perturbation": item["strongest_perturbation"]["name"],
                "strongest_mean_cosine_distance": item["strongest_perturbation"][
                    "mean_cosine_distance"
                ],
                "motion_group_mean_cosine_distance": item["group_mean_cosine_distance"].get(
                    "motion"
                ),
                "appearance_group_mean_cosine_distance": item[
                    "group_mean_cosine_distance"
                ].get("appearance"),
            }
        )
    return rows


def build_sweep_rows(sweep_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary in sweep_summaries:
        trend = summary.get("trend_analysis") or {}
        for case in summary["cases"]:
            k1 = case["knn_by_k"]["1"]
            rows.append(
                {
                    "sweep_name": summary["sweep_name"],
                    "base_perturbation": summary["base_perturbation"],
                    "group": summary["group"],
                    "parameter": summary["parameter"],
                    "case_label": case["label"],
                    "case_value": case["value"],
                    "mean_cosine_distance": case["mean_cosine_distance"],
                    "median_cosine_distance": case["median_cosine_distance"],
                    "mean_l2_distance": case["mean_l2_distance"],
                    "k1_all_perturbed_accuracy": k1["all_perturbed_accuracy"],
                    "k1_all_accuracy_drop": k1["all_accuracy_drop"],
                    "k1_train_seen_perturbed_accuracy": k1[
                        "train_seen_perturbed_accuracy"
                    ],
                    "k1_train_seen_accuracy_drop": k1["train_seen_accuracy_drop"],
                    "k1_prediction_change_rate": k1["prediction_change_rate"],
                    "mean_cosine_distance_direction": trend.get(
                        "mean_cosine_distance_direction"
                    ),
                    "k1_all_accuracy_drop_direction": trend.get(
                        "k1_all_accuracy_drop_direction"
                    ),
                }
            )
    return rows


def build_qualitative_sample_records(
    sensitivity_reports: list[dict[str, Any]],
    knn_reports: list[dict[str, Any]],
    *,
    k: int = 1,
    sample_metadata_by_index: dict[int, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    knn_by_perturbation = {report["perturbation"]: report for report in knn_reports}
    records: list[dict[str, Any]] = []
    for sensitivity in sensitivity_reports:
        perturbation = sensitivity["perturbation"]
        knn_report = knn_by_perturbation[perturbation]
        knn_by_index = {
            int(record["sample_index"]): record
            for record in knn_report["sample_prediction_changes"]
        }
        for sample in sensitivity["sample_metrics"]:
            sample_index = int(sample["sample_index"])
            prediction = knn_by_index[sample_index]["by_k"][str(k)]
            metadata = (
                {}
                if sample_metadata_by_index is None
                else sample_metadata_by_index.get(sample_index, {})
            )
            records.append(
                {
                    "perturbation": perturbation,
                    "group": sensitivity["perturbation_group"],
                    "perturbation_config": sensitivity.get("perturbation_config", {}),
                    "sample_index": sample_index,
                    "video_id": sample["video_id"],
                    "video_path": metadata.get("video_path"),
                    "label_id": sample["label_id"],
                    "label_name": sample["label_name"],
                    "frame_indices": metadata.get("frame_indices"),
                    "sampling_strategy": metadata.get("sampling_strategy"),
                    "cosine_distance": sample["cosine_distance"],
                    "l2_distance": sample["l2_distance"],
                    "relative_l2_distance": sample["relative_l2_distance"],
                    "train_label_seen": knn_by_index[sample_index]["train_label_seen"],
                    "original_prediction": prediction["original_prediction"],
                    "perturbed_prediction": prediction["perturbed_prediction"],
                    "original_correct": prediction["original_correct"],
                    "perturbed_correct": prediction["perturbed_correct"],
                    "prediction_changed": prediction["prediction_changed"],
                    "correct_to_incorrect": prediction["correct_to_incorrect"],
                    "incorrect_to_correct": prediction["incorrect_to_correct"],
                }
            )
    return records


def build_qualitative_sample_summary(
    records: list[dict[str, Any]],
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    return {
        "largest_embedding_shift": sorted(
            records,
            key=lambda item: item["cosine_distance"],
            reverse=True,
        )[:top_n],
        "smallest_embedding_shift": sorted(
            records,
            key=lambda item: item["cosine_distance"],
        )[:top_n],
        "correct_to_incorrect": sorted(
            [record for record in records if record["correct_to_incorrect"]],
            key=lambda item: item["cosine_distance"],
            reverse=True,
        )[:top_n],
        "incorrect_to_correct": sorted(
            [record for record in records if record["incorrect_to_correct"]],
            key=lambda item: item["cosine_distance"],
            reverse=True,
        )[:top_n],
        "high_shift_prediction_unchanged": sorted(
            [record for record in records if not record["prediction_changed"]],
            key=lambda item: item["cosine_distance"],
            reverse=True,
        )[:top_n],
        "low_shift_prediction_changed": sorted(
            [record for record in records if record["prediction_changed"]],
            key=lambda item: item["cosine_distance"],
        )[:top_n],
    }


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def render_markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_value(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


def render_experiment_report(
    *,
    baseline_rows: list[dict[str, Any]],
    perturbation_rows: list[dict[str, Any]],
    class_rows: list[dict[str, Any]],
    sweep_rows: list[dict[str, Any]],
    qualitative_summary: dict[str, Any],
) -> str:
    top_shift = qualitative_summary["largest_embedding_shift"][0]
    sections = [
        "# SSV2 VideoMAE Sensitivity Report",
        "",
        "## Original KNN Baseline",
        "",
        render_markdown_table(
            baseline_rows,
            [
                "metric",
                "k",
                "all_accuracy",
                "train_seen_accuracy",
                "all_correct",
                "train_seen_correct",
            ],
        ),
        "",
        "## Perturbation Summary",
        "",
        render_markdown_table(
            perturbation_rows,
            [
                "perturbation",
                "group",
                "mean_cosine_distance",
                "all_accuracy_drop",
                "train_seen_accuracy_drop",
                "prediction_change_rate",
            ],
        ),
        "",
        "## Sweep Summary",
        "",
        render_markdown_table(
            sweep_rows,
            [
                "sweep_name",
                "case_label",
                "mean_cosine_distance",
                "k1_all_accuracy_drop",
                "k1_prediction_change_rate",
            ],
        ),
        "",
        "## Class-Level Candidates",
        "",
        render_markdown_table(
            class_rows[:10],
            [
                "label_id",
                "label_name",
                "sample_count",
                "strongest_perturbation",
                "mean_cosine_distance_across_perturbations",
            ],
        ),
        "",
        "## Qualitative Sample Candidates",
        "",
        "Largest embedding shift:",
        "",
        render_markdown_table(
            [top_shift],
            [
                "perturbation",
                "video_id",
                "label_name",
                "cosine_distance",
                "prediction_changed",
            ],
        ),
    ]
    return "\n".join(sections) + "\n"


def write_bar_chart_svg(
    rows: list[dict[str, Any]],
    *,
    label_key: str,
    value_key: str,
    title: str,
    path: str | Path,
    width: int = 960,
    height: int = 520,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    margin_left = 190
    margin_right = 40
    margin_top = 56
    margin_bottom = 32
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(float(row[value_key]) for row in rows) if rows else 1.0
    max_value = max(max_value, 1e-12)
    bar_gap = 10
    bar_height = max(14, (plot_height - bar_gap * (len(rows) - 1)) / max(len(rows), 1))

    body = [
        _svg_text(width / 2, 28, title, "middle", size=20, weight="700"),
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="1"/>',
    ]
    for index, row in enumerate(rows):
        y = margin_top + index * (bar_height + bar_gap)
        value = float(row[value_key])
        bar_width = value / max_value * plot_width
        label = str(row[label_key])
        body.append(_svg_text(margin_left - 10, y + bar_height * 0.68, label, "end", size=13))
        body.append(
            f'<rect x="{margin_left}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="#4c78a8"/>'
        )
        body.append(
            _svg_text(
                margin_left + bar_width + 6,
                y + bar_height * 0.68,
                _format_float(value, digits=4),
                "start",
                size=12,
            )
        )
    output_path.write_text(_svg_document(width, height, body), encoding="utf-8")


def write_line_chart_svg(
    series: list[dict[str, Any]],
    *,
    title: str,
    path: str | Path,
    width: int = 960,
    height: int = 560,
) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    margin_left = 80
    margin_right = 220
    margin_top = 58
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_points = [point for item in series for point in item["points"]]
    x_values = [float(point["x"]) for point in all_points]
    y_values = [float(point["y"]) for point in all_points]
    min_x, max_x = min(x_values), max(x_values)
    min_y, max_y = min(y_values), max(y_values)
    if min_x == max_x:
        max_x += 1.0
    if min_y == max_y:
        max_y += 1.0

    colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2", "#e45756"]
    body = [
        _svg_text(width / 2, 30, title, "middle", size=20, weight="700"),
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#444" stroke-width="1"/>',
    ]
    for index, item in enumerate(series):
        color = colors[index % len(colors)]
        points = [
            (
                margin_left + (float(point["x"]) - min_x) / (max_x - min_x) * plot_width,
                margin_top + plot_height - (float(point["y"]) - min_y) / (max_y - min_y) * plot_height,
                point,
            )
            for point in item["points"]
        ]
        path_points = " ".join(f"{x:.2f},{y:.2f}" for x, y, _ in points)
        body.append(
            f'<polyline points="{path_points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for x, y, point in points:
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>')
            body.append(_svg_text(x, y - 8, str(point["label"]), "middle", size=11))
        legend_y = margin_top + index * 24
        body.append(f'<rect x="{width - margin_right + 24}" y="{legend_y - 10}" width="14" height="14" fill="{color}"/>')
        body.append(_svg_text(width - margin_right + 44, legend_y + 2, item["name"], "start", size=13))

    body.extend(
        [
            _svg_text(margin_left + plot_width / 2, height - 22, "sweep value", "middle", size=13),
            _svg_text(18, margin_top + plot_height / 2, "mean cosine distance", "middle", size=13, rotate=-90),
        ]
    )
    output_path.write_text(_svg_document(width, height, body), encoding="utf-8")


def run_reporting(
    *,
    log_dir: str | Path = DEFAULT_LOG_DIR,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    plot_dir: str | Path = DEFAULT_PLOT_DIR,
) -> dict[str, Any]:
    logs = Path(log_dir)
    reports = Path(report_dir)
    plots = Path(plot_dir)
    reports.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)

    baseline = _load_json(logs / f"{BASE_NAME}_original_baseline_interpretability.json")
    sensitivity_summary = _load_json(
        logs / f"{BASE_NAME}_all_perturbations_sensitivity_summary.json"
    )
    knn_drop_summary = _load_json(logs / f"{BASE_NAME}_all_perturbations_knn_drop_cosine.json")
    class_report = _load_json(logs / f"{BASE_NAME}_class_sensitivity.json")
    original_artifact = load_embedding_artifact(DEFAULT_ORIGINAL_ARTIFACT)
    sample_metadata_by_index = {
        index: metadata
        for index, metadata in enumerate(original_artifact.get("sample_metadata", []))
    }
    sweep_summaries = [
        _load_json(path)
        for path in sorted(logs.glob(f"{BASE_NAME}_*_sweep_summary.json"))
        if "all_sweeps" not in path.name
    ]
    sensitivity_reports = [
        _load_json(logs / f"{BASE_NAME}_{perturbation}_sensitivity.json")
        for perturbation in sorted(FIRST_ROUND_PERTURBATIONS)
    ]
    knn_reports = [
        _load_json(logs / f"{BASE_NAME}_{perturbation}_knn_cosine.json")
        for perturbation in sorted(FIRST_ROUND_PERTURBATIONS)
    ]

    baseline_rows = build_baseline_rows(baseline)
    perturbation_rows = build_perturbation_rows(sensitivity_summary, knn_drop_summary, k=1)
    class_rows = build_class_rows(class_report, top_n=20)
    sweep_rows = build_sweep_rows(sweep_summaries)
    qualitative_records = build_qualitative_sample_records(
        sensitivity_reports,
        knn_reports,
        k=1,
        sample_metadata_by_index=sample_metadata_by_index,
    )
    qualitative_summary = build_qualitative_sample_summary(qualitative_records, top_n=10)

    output_files = {
        "baseline_table": reports / "baseline_table.csv",
        "perturbation_table": reports / "perturbation_summary_table.csv",
        "class_table": reports / "class_sensitivity_table.csv",
        "sweep_table": reports / "sweep_summary_table.csv",
        "qualitative_samples": reports / "qualitative_samples.json",
        "markdown_report": reports / "ssv2_videomae_sensitivity_report.md",
        "sensitivity_plot": plots / "perturbation_mean_cosine_distance.svg",
        "knn_drop_plot": plots / "perturbation_k1_accuracy_drop.svg",
        "sweep_plot": plots / "sweep_mean_cosine_distance.svg",
    }

    write_csv(baseline_rows, output_files["baseline_table"])
    write_csv(perturbation_rows, output_files["perturbation_table"])
    write_csv(class_rows, output_files["class_table"])
    write_csv(sweep_rows, output_files["sweep_table"])
    write_json(qualitative_summary, output_files["qualitative_samples"])
    output_files["markdown_report"].write_text(
        render_experiment_report(
            baseline_rows=baseline_rows,
            perturbation_rows=perturbation_rows,
            class_rows=class_rows,
            sweep_rows=sweep_rows,
            qualitative_summary=qualitative_summary,
        ),
        encoding="utf-8",
    )
    write_bar_chart_svg(
        perturbation_rows,
        label_key="perturbation",
        value_key="mean_cosine_distance",
        title="Perturbation Mean Cosine Distance",
        path=output_files["sensitivity_plot"],
    )
    write_bar_chart_svg(
        sorted(perturbation_rows, key=lambda item: item["all_accuracy_drop"], reverse=True),
        label_key="perturbation",
        value_key="all_accuracy_drop",
        title="Perturbation KNN k=1 Accuracy Drop",
        path=output_files["knn_drop_plot"],
    )
    write_line_chart_svg(
        _sweep_line_series(sweep_rows),
        title="Sweep Mean Cosine Distance",
        path=output_files["sweep_plot"],
    )
    return {key: str(value) for key, value in output_files.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tables, qualitative samples, and SVG plots from experiment JSON reports."
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--plot-dir", type=Path, default=DEFAULT_PLOT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_reporting(
        log_dir=args.log_dir,
        report_dir=args.report_dir,
        plot_dir=args.plot_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _sweep_line_series(sweep_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sweep: dict[str, list[dict[str, Any]]] = {}
    for row in sweep_rows:
        if not isinstance(row["case_value"], (int, float)) or isinstance(
            row["case_value"],
            bool,
        ):
            continue
        by_sweep.setdefault(row["sweep_name"], []).append(row)
    return [
        {
            "name": sweep_name,
            "points": [
                {
                    "x": row["case_value"],
                    "y": row["mean_cosine_distance"],
                    "label": row["case_label"],
                }
                for row in sorted(rows, key=lambda item: float(item["case_value"]))
            ],
        }
        for sweep_name, rows in by_sweep.items()
    ]


def _format_markdown_value(value: Any) -> str:
    if isinstance(value, float):
        return _format_float(value, digits=6)
    if value is None:
        return ""
    return str(value)


def _format_float(value: float, *, digits: int) -> str:
    return f"{value:.{digits}f}"


def _svg_document(width: int, height: int, body: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _svg_text(
    x: float,
    y: float,
    text: str,
    anchor: str,
    *,
    size: int,
    weight: str = "400",
    rotate: int | None = None,
) -> str:
    escaped = html.escape(text)
    transform = "" if rotate is None else f' transform="rotate({rotate} {x:.2f} {y:.2f})"'
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Arial, sans-serif" font-size="{size}" font-weight="{weight}"'
        f'{transform}>{escaped}</text>'
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
