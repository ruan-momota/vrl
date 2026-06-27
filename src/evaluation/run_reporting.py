"""Final report helpers for one run-scoped frozen-embedding experiment."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def build_qualitative_records(
    sensitivity_report: dict[str, Any],
    linear_probe_report: dict[str, Any],
    *,
    artifact_label: str,
) -> list[dict[str, Any]]:
    """Join continuous representation shifts with paired probe outcomes."""
    predictions = {
        int(record["sample_index"]): record
        for record in linear_probe_report["sample_predictions"]
    }
    records: list[dict[str, Any]] = []
    for sample in sensitivity_report["sample_metrics"]:
        sample_index = int(sample["sample_index"])
        prediction = predictions.get(sample_index)
        if prediction is None:
            raise ValueError(f"Missing linear-probe prediction for sample {sample_index}")
        if str(prediction["video_id"]) != str(sample["video_id"]):
            raise ValueError(f"Mismatched video ID for sample {sample_index}")
        records.append(
            {
                "artifact_label": artifact_label,
                "perturbation": sensitivity_report["perturbation"],
                "group": sensitivity_report.get("perturbation_group"),
                "sample_index": sample_index,
                "video_id": sample["video_id"],
                "label_id": sample["label_id"],
                "label_name": sample["label_name"],
                "cosine_distance": sample["cosine_distance"],
                "l2_distance": sample["l2_distance"],
                "original_prediction": prediction["original_prediction"],
                "perturbed_prediction": prediction["perturbed_prediction"],
                "original_correct": prediction["original_correct"],
                "perturbed_correct": prediction["perturbed_correct"],
                "correct_to_incorrect": prediction["correct_to_incorrect"],
                "prediction_changed": prediction["prediction_changed"],
            }
        )
    return records


def build_qualitative_summary(records: list[dict[str, Any]], *, top_n: int = 10) -> dict[str, Any]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    by_descending_shift = sorted(records, key=lambda item: float(item["cosine_distance"]), reverse=True)
    by_ascending_shift = list(reversed(by_descending_shift))
    return {
        "record_count": len(records),
        "largest_embedding_shift": by_descending_shift[:top_n],
        "correct_to_incorrect": [
            item for item in by_descending_shift if item["correct_to_incorrect"]
        ][:top_n],
        "high_shift_prediction_unchanged": [
            item for item in by_descending_shift if not item["prediction_changed"]
        ][:top_n],
        "low_shift_prediction_changed": [
            item for item in by_ascending_shift if item["prediction_changed"]
        ][:top_n],
    }


def build_failure_summary(
    artifacts: dict[str, dict[str, Any]],
    quality_records: list[dict[str, Any]],
) -> dict[str, Any]:
    extraction = []
    for label, artifact in artifacts.items():
        summary = artifact.get("summary") if isinstance(artifact.get("summary"), dict) else {}
        extraction.append(
            {
                "artifact_label": label,
                "dataset_size": summary.get("dataset_size"),
                "successful_samples": summary.get("successful_samples"),
                "failed_samples": summary.get("failed_samples"),
            }
        )
    failed_extractions = [
        item for item in extraction if int(item["failed_samples"] or 0) > 0
    ]
    quality_mismatches = [
        item
        for item in quality_records
        if not item.get("frame_indices_match") or not item.get("sampling_strategy_match")
    ]
    return {
        "extraction_policy": "fail_fast_no_silent_sample_replacement",
        "per_artifact": extraction,
        "failed_extractions": failed_extractions,
        "quality_sample_count": len(quality_records),
        "quality_mismatches": quality_mismatches,
        "all_extractions_succeeded": not failed_extractions,
        "all_sampled_quality_checks_passed": not quality_mismatches,
    }


def build_run_provenance(
    *,
    run_dir: str | Path,
    artifact_paths: dict[str, str | Path],
    artifacts: dict[str, dict[str, Any]],
    hardware: dict[str, Any],
) -> dict[str, Any]:
    """Record reproducibility facts available after extraction and evaluation."""
    resolved_run_dir = Path(run_dir)
    manifest_path = resolved_run_dir / "manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else None
    observed_devices = sorted(
        {
            str(artifact.get("model_metadata", {}).get("device"))
            for artifact in artifacts.values()
            if isinstance(artifact.get("model_metadata"), dict)
            and artifact["model_metadata"].get("device") is not None
        }
    )
    artifact_details = {
        label: {
            "path": str(path),
            "sha256": _sha256(Path(path)),
            "size_bytes": Path(path).stat().st_size,
            "summary": artifact.get("summary"),
            "model_metadata": artifact.get("model_metadata"),
        }
        for label, path in artifact_paths.items()
        for artifact in [artifacts[label]]
    }
    return {
        "format_version": 1,
        "run_id": None if manifest is None else manifest.get("run_id"),
        "manifest_path": str(manifest_path),
        "code_commit": None if manifest is None else manifest.get("code_commit"),
        "created_at_utc": None if manifest is None else manifest.get("created_at_utc"),
        "artifact_commands": None if manifest is None else manifest.get("artifact_commands"),
        "evaluation_commands": None if manifest is None else manifest.get("evaluation_commands"),
        "hardware": {
            **hardware,
            "observed_encoder_devices": observed_devices,
        },
        "artifacts": artifact_details,
    }


def build_legacy_anchor_comparison(
    *,
    legacy_perturbation_summary: str | Path,
    legacy_baseline_table: str | Path,
    current_rows: list[dict[str, Any]],
    current_original_knn: dict[str, Any],
    matches: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare only explicitly declared, method-compatible legacy probes."""
    legacy_by_name = {
        str(row["perturbation"]): row
        for row in _read_csv(Path(legacy_perturbation_summary))
    }
    current_by_label = {str(row["artifact_label"]): row for row in current_rows}
    comparisons: list[dict[str, Any]] = []
    for match in matches:
        legacy_name = str(match["legacy_perturbation"])
        current_label = str(match["current_artifact_label"])
        legacy = legacy_by_name.get(legacy_name)
        current = current_by_label.get(current_label)
        if legacy is None or current is None:
            raise ValueError(
                f"Legacy comparison match is unavailable: {legacy_name!r} / {current_label!r}"
            )
        comparisons.append(
            {
                "legacy_perturbation": legacy_name,
                "current_artifact_label": current_label,
                "legacy_mean_cosine_distance": _float(legacy, "mean_cosine_distance"),
                "current_mean_cosine_distance": float(current["mean_cosine_distance"]),
                "mean_cosine_distance_delta_current_minus_legacy": float(
                    current["mean_cosine_distance"]
                )
                - _float(legacy, "mean_cosine_distance"),
                "legacy_knn_k": _int(legacy, "knn_k"),
                "legacy_knn_accuracy_drop": _float(legacy, "all_accuracy_drop"),
                "current_knn_k": int(current["knn_k"]),
                "current_knn_accuracy_drop": float(current["knn_accuracy_drop"]),
                "comparability_note": str(match.get("comparability_note", "")),
            }
        )
    legacy_baseline = _read_csv(Path(legacy_baseline_table))
    legacy_k5 = next(
        (
            row
            for row in legacy_baseline
            if row.get("metric") == "cosine" and _int(row, "k") == 5
        ),
        None,
    )
    current_k5 = next(
        (
            row
            for row in current_original_knn.get("results", [])
            if int(row["k"]) == 5
        ),
        None,
    )
    return {
        "format_version": 1,
        "method_boundary": (
            "Legacy KNN k=1 and current KNN k=5 are not directly comparable. "
            "Only the explicitly declared common perturbations compare representation shift directly."
        ),
        "shared_perturbation_comparisons": comparisons,
        "cosine_k5_original_baseline": {
            "legacy_accuracy": None if legacy_k5 is None else _float(legacy_k5, "all_accuracy"),
            "current_accuracy": None if current_k5 is None else float(current_k5["accuracy"]),
        },
    }


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return data


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing {key} in comparison row")
    return float(value)


def _int(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    if value is None:
        raise ValueError(f"Missing {key} in comparison row")
    return int(value)
