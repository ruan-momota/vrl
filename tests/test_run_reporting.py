from __future__ import annotations

from pathlib import Path

import pytest
from src.evaluation.run_reporting import (
    build_failure_summary,
    build_legacy_anchor_comparison,
    build_qualitative_records,
    build_qualitative_summary,
    build_run_provenance,
)


def test_qualitative_records_join_continuous_shift_and_probe_prediction() -> None:
    records = build_qualitative_records(
        {
            "perturbation": "temporal_shuffle",
            "perturbation_group": "motion",
            "sample_metrics": [
                {
                    "sample_index": 0,
                    "video_id": "a",
                    "label_id": 1,
                    "label_name": "Example",
                    "cosine_distance": 0.8,
                    "l2_distance": 2.0,
                }
            ],
        },
        {
            "sample_predictions": [
                {
                    "sample_index": 0,
                    "video_id": "a",
                    "original_prediction": 1,
                    "perturbed_prediction": 2,
                    "original_correct": True,
                    "perturbed_correct": False,
                    "correct_to_incorrect": True,
                    "prediction_changed": True,
                }
            ]
        },
        artifact_label="temporal-shuffle-mid",
    )
    summary = build_qualitative_summary(records, top_n=1)

    assert summary["record_count"] == 1
    assert summary["correct_to_incorrect"][0]["video_id"] == "a"


def test_failure_summary_reports_fail_fast_and_quality_mismatches() -> None:
    artifact = {
        "summary": {
            "dataset_size": 2,
            "successful_samples": 2,
            "failed_samples": 0,
        }
    }
    summary = build_failure_summary(
        {"original": artifact},
        [
            {
                "frame_indices_match": True,
                "sampling_strategy_match": True,
            }
        ],
    )

    assert summary["all_extractions_succeeded"] is True
    assert summary["all_sampled_quality_checks_passed"] is True


def test_legacy_anchor_comparison_only_uses_declared_matches(tmp_path: Path) -> None:
    legacy_summary = tmp_path / "legacy_summary.csv"
    legacy_summary.write_text(
        "perturbation,mean_cosine_distance,knn_k,all_accuracy_drop\n"
        "freeze_tail,0.01,1,0.02\n",
        encoding="utf-8",
    )
    legacy_baseline = tmp_path / "legacy_baseline.csv"
    legacy_baseline.write_text(
        "metric,k,all_accuracy\ncosine,5,0.1\n",
        encoding="utf-8",
    )

    report = build_legacy_anchor_comparison(
        legacy_perturbation_summary=legacy_summary,
        legacy_baseline_table=legacy_baseline,
        current_rows=[
            {
                "artifact_label": "freeze-tail-mid",
                "mean_cosine_distance": 0.015,
                "knn_k": 5,
                "knn_accuracy_drop": 0.03,
            }
        ],
        current_original_knn={"results": [{"k": 5, "accuracy": 0.11}]},
        matches=[
            {
                "legacy_perturbation": "freeze_tail",
                "current_artifact_label": "freeze-tail-mid",
                "comparability_note": "same temporal probe",
            }
        ],
    )

    comparison = report["shared_perturbation_comparisons"][0]
    assert comparison["mean_cosine_distance_delta_current_minus_legacy"] == pytest.approx(0.005)
    assert report["cosine_k5_original_baseline"] == {
        "legacy_accuracy": 0.1,
        "current_accuracy": 0.11,
    }


def test_provenance_keeps_observed_device_without_requiring_host_details(tmp_path: Path) -> None:
    artifact_path = tmp_path / "artifact.pt"
    artifact_path.write_bytes(b"embedding")

    provenance = build_run_provenance(
        run_dir=tmp_path,
        artifact_paths={"original": artifact_path},
        artifacts={"original": {"model_metadata": {"device": "cuda"}}},
        hardware={"environment_recording_policy": "portable"},
    )

    assert provenance["hardware"] == {
        "environment_recording_policy": "portable",
        "observed_encoder_devices": ["cuda"],
    }
