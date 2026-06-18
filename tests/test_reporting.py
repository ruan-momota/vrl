from __future__ import annotations

from pathlib import Path

from src.reporting import (
    build_perturbation_rows,
    build_qualitative_sample_records,
    build_qualitative_sample_summary,
    render_markdown_table,
    write_bar_chart_svg,
    write_line_chart_svg,
)


def test_build_perturbation_rows_joins_sensitivity_and_knn() -> None:
    rows = build_perturbation_rows(
        sensitivity_summary={
            "ranked_by_mean_cosine_distance": [
                {
                    "perturbation": "single_frame",
                    "perturbation_group": "motion",
                    "mean_cosine_distance": 0.2,
                    "median_cosine_distance": 0.1,
                    "mean_l2_distance": 2.0,
                    "mean_relative_l2_distance": 0.3,
                }
            ]
        },
        knn_drop_summary={
            "ranked_by_train_seen_accuracy_drop": {
                "1": [
                    {
                        "perturbation": "single_frame",
                        "all_perturbed_accuracy": 0.02,
                        "all_absolute_accuracy_drop": 0.0,
                        "train_seen_perturbed_accuracy": 0.04,
                        "train_seen_absolute_accuracy_drop": 0.0,
                        "all_prediction_change_rate": 0.77,
                    }
                ]
            }
        },
        k=1,
    )

    assert rows == [
        {
            "perturbation": "single_frame",
            "group": "motion",
            "mean_cosine_distance": 0.2,
            "median_cosine_distance": 0.1,
            "mean_l2_distance": 2.0,
            "mean_relative_l2_distance": 0.3,
            "knn_k": 1,
            "all_perturbed_accuracy": 0.02,
            "all_accuracy_drop": 0.0,
            "train_seen_perturbed_accuracy": 0.04,
            "train_seen_accuracy_drop": 0.0,
            "prediction_change_rate": 0.77,
        }
    ]


def test_build_qualitative_sample_summary_categories() -> None:
    records = build_qualitative_sample_records(
        sensitivity_reports=[
            {
                "perturbation": "temporal_reverse",
                "perturbation_group": "motion",
                "sample_metrics": [
                    _sample_metric(0, 0.9),
                    _sample_metric(1, 0.1),
                ],
            }
        ],
        knn_reports=[
            {
                "perturbation": "temporal_reverse",
                "sample_prediction_changes": [
                    _prediction_record(0, changed=True, correct_to_incorrect=True),
                    _prediction_record(1, changed=False, correct_to_incorrect=False),
                ],
            }
        ],
        k=1,
    )
    summary = build_qualitative_sample_summary(records, top_n=1)

    assert summary["largest_embedding_shift"][0]["sample_index"] == 0
    assert summary["smallest_embedding_shift"][0]["sample_index"] == 1
    assert summary["correct_to_incorrect"][0]["sample_index"] == 0
    assert summary["high_shift_prediction_unchanged"][0]["sample_index"] == 1


def test_render_markdown_table_formats_floats() -> None:
    markdown = render_markdown_table(
        [{"name": "a", "value": 0.123456789}],
        ["name", "value"],
    )

    assert "| name | value |" in markdown
    assert "0.123457" in markdown


def test_svg_writers_create_files(tmp_path: Path) -> None:
    bar_path = tmp_path / "bar.svg"
    line_path = tmp_path / "line.svg"

    write_bar_chart_svg(
        [{"label": "a", "value": 0.1}, {"label": "b", "value": 0.2}],
        label_key="label",
        value_key="value",
        title="Bar",
        path=bar_path,
    )
    write_line_chart_svg(
        [
            {
                "name": "series",
                "points": [
                    {"x": 0.1, "y": 0.2, "label": "low"},
                    {"x": 0.2, "y": 0.4, "label": "high"},
                ],
            }
        ],
        title="Line",
        path=line_path,
    )

    assert bar_path.read_text(encoding="utf-8").startswith("<svg")
    assert "<rect" in bar_path.read_text(encoding="utf-8")
    assert "<polyline" in line_path.read_text(encoding="utf-8")


def _sample_metric(index: int, cosine_distance: float) -> dict:
    return {
        "sample_index": index,
        "video_id": f"video-{index}",
        "label_id": index,
        "label_name": f"Label {index}",
        "cosine_distance": cosine_distance,
        "l2_distance": cosine_distance * 10,
        "relative_l2_distance": cosine_distance,
    }


def _prediction_record(
    index: int,
    *,
    changed: bool,
    correct_to_incorrect: bool,
) -> dict:
    return {
        "sample_index": index,
        "train_label_seen": True,
        "by_k": {
            "1": {
                "original_prediction": 1,
                "perturbed_prediction": 2 if changed else 1,
                "original_correct": correct_to_incorrect,
                "perturbed_correct": False,
                "prediction_changed": changed,
                "correct_to_incorrect": correct_to_incorrect,
                "incorrect_to_correct": False,
            }
        },
    }
