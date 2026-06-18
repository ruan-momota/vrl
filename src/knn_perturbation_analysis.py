from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.artifact_alignment import check_paired_embedding_alignment
from src.embedding_extraction import load_embedding_artifact
from src.knn_baseline import (
    DistanceMetric,
    knn_predict,
    normalize_k_values,
    validate_embedding_pair,
)
from src.embedding_sensitivity import compute_embedding_distances


DEFAULT_MATRIX_PATH = Path("configs/ssv2_videomae_perturbation_matrix.json")
DEFAULT_OUTPUT_DIR = Path("outputs/logs")


def evaluate_knn_perturbation_drop(
    *,
    train_artifact: dict[str, Any],
    original_validation_artifact: dict[str, Any],
    perturbed_validation_artifact: dict[str, Any],
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
    metric: DistanceMetric = "cosine",
    query_batch_size: int = 1024,
    train_artifact_path: str | Path | None = None,
    original_validation_artifact_path: str | Path | None = None,
    perturbed_validation_artifact_path: str | Path | None = None,
    perturbation_name: str | None = None,
    perturbation_group: str | None = None,
) -> dict[str, Any]:
    """Compare original and perturbed validation KNN predictions."""
    validate_embedding_pair(train_artifact, original_validation_artifact)
    validate_embedding_pair(train_artifact, perturbed_validation_artifact)
    alignment = check_paired_embedding_alignment(
        original_validation_artifact,
        perturbed_validation_artifact,
    )
    if metric not in {"cosine", "l2"}:
        raise ValueError(f"Unsupported KNN metric: {metric}")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")

    train_embeddings, train_labels = _labeled_tensors(train_artifact, name="train")
    original_embeddings, validation_labels = _labeled_tensors(
        original_validation_artifact,
        name="original validation",
    )
    perturbed_embeddings, perturbed_labels = _labeled_tensors(
        perturbed_validation_artifact,
        name="perturbed validation",
    )
    if not torch.equal(validation_labels, perturbed_labels):
        raise ValueError("original and perturbed validation label_ids must match")

    normalized_k_values = normalize_k_values(k_values, train_count=train_embeddings.shape[0])
    original_predictions = knn_predict(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=original_embeddings,
        k_values=normalized_k_values,
        metric=metric,
        query_batch_size=query_batch_size,
    )
    perturbed_predictions = knn_predict(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=perturbed_embeddings,
        k_values=normalized_k_values,
        metric=metric,
        query_batch_size=query_batch_size,
    )
    embedding_distances = compute_embedding_distances(
        original_embeddings,
        perturbed_embeddings,
    )

    train_seen_mask = _train_seen_mask(train_labels, validation_labels)
    results = [
        _build_k_result(
            k=k,
            labels=validation_labels,
            train_seen_mask=train_seen_mask,
            original_predictions=original_predictions[k],
            perturbed_predictions=perturbed_predictions[k],
            cosine_distance=embedding_distances["cosine_distance"],
        )
        for k in normalized_k_values
    ]
    resolved_name = perturbation_name or _infer_perturbation_name(perturbed_validation_artifact)
    sample_prediction_changes = _build_sample_prediction_changes(
        original_validation_artifact=original_validation_artifact,
        labels=validation_labels,
        train_seen_mask=train_seen_mask,
        k_values=normalized_k_values,
        original_predictions=original_predictions,
        perturbed_predictions=perturbed_predictions,
        embedding_distances=embedding_distances,
    )

    return {
        "format_version": 1,
        "report_type": "knn_perturbation_drop",
        "dataset_name": _nested_value(train_artifact, "config", "dataset_name")
        or _nested_value(original_validation_artifact, "config", "dataset_name"),
        "model_checkpoint": _nested_value(train_artifact, "model_metadata", "checkpoint"),
        "embedding_type": _nested_value(train_artifact, "model_metadata", "embedding_type"),
        "metric": metric,
        "normalized_embeddings": metric == "cosine",
        "k_values": list(normalized_k_values),
        "query_batch_size": query_batch_size,
        "train_artifact_path": None if train_artifact_path is None else str(train_artifact_path),
        "original_validation_artifact_path": None
        if original_validation_artifact_path is None
        else str(original_validation_artifact_path),
        "perturbed_validation_artifact_path": None
        if perturbed_validation_artifact_path is None
        else str(perturbed_validation_artifact_path),
        "perturbation": resolved_name,
        "perturbation_group": perturbation_group,
        "perturbation_config": _perturbation_config(perturbed_validation_artifact),
        "train_samples": int(train_embeddings.shape[0]),
        "validation_samples": int(validation_labels.numel()),
        "train_seen_validation_samples": int(train_seen_mask.sum().item()),
        "train_unseen_validation_samples": int((~train_seen_mask).sum().item()),
        "alignment": alignment.to_dict(),
        "results": results,
        "sample_prediction_changes": sample_prediction_changes,
    }


def run_knn_perturbation_drop(
    *,
    train_artifact_path: str | Path,
    original_validation_artifact_path: str | Path,
    perturbed_validation_artifact_path: str | Path,
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
    metric: DistanceMetric = "cosine",
    query_batch_size: int = 1024,
    perturbation_name: str | None = None,
    perturbation_group: str | None = None,
) -> dict[str, Any]:
    train_path = Path(train_artifact_path)
    original_path = Path(original_validation_artifact_path)
    perturbed_path = Path(perturbed_validation_artifact_path)
    return evaluate_knn_perturbation_drop(
        train_artifact=load_embedding_artifact(train_path),
        original_validation_artifact=load_embedding_artifact(original_path),
        perturbed_validation_artifact=load_embedding_artifact(perturbed_path),
        k_values=k_values,
        metric=metric,
        query_batch_size=query_batch_size,
        train_artifact_path=train_path,
        original_validation_artifact_path=original_path,
        perturbed_validation_artifact_path=perturbed_path,
        perturbation_name=perturbation_name,
        perturbation_group=perturbation_group,
    )


def build_all_perturbations_knn_drop_summary(
    reports: list[dict[str, Any]],
    *,
    matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one KNN perturbation report is required")

    compact = [_compact_report(report) for report in reports]
    k_values = [int(k) for k in reports[0]["k_values"]]
    return {
        "format_version": 1,
        "report_type": "knn_perturbation_drop_summary",
        "matrix_name": None if matrix is None else matrix.get("matrix_name"),
        "dataset_name": reports[0].get("dataset_name"),
        "model_checkpoint": reports[0].get("model_checkpoint"),
        "embedding_type": reports[0].get("embedding_type"),
        "metric": reports[0].get("metric"),
        "k_values": k_values,
        "train_artifact_path": reports[0].get("train_artifact_path"),
        "original_validation_artifact_path": reports[0].get(
            "original_validation_artifact_path"
        ),
        "perturbation_count": len(reports),
        "validation_samples": reports[0].get("validation_samples"),
        "train_seen_validation_samples": reports[0].get("train_seen_validation_samples"),
        "perturbations": compact,
        "ranked_by_all_accuracy_drop": {
            str(k): sorted(
                (_compact_k_result(item, k) for item in compact),
                key=lambda item: item["all_absolute_accuracy_drop"],
                reverse=True,
            )
            for k in k_values
        },
        "ranked_by_train_seen_accuracy_drop": {
            str(k): sorted(
                (_compact_k_result(item, k) for item in compact),
                key=lambda item: item["train_seen_absolute_accuracy_drop"],
                reverse=True,
            )
            for k in k_values
        },
        "group_summaries": _build_group_summaries(reports),
    }


def run_matrix_knn_perturbation_analysis(
    *,
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
    train_artifact_path: str | Path | None = None,
    original_validation_artifact_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    metric: DistanceMetric | None = None,
    k_values: list[int] | tuple[int, ...] | None = None,
    query_batch_size: int = 1024,
    overwrite: bool = False,
) -> dict[str, Any]:
    matrix = _load_json(Path(matrix_path))
    scope = matrix.get("scope", {})
    resolved_train_path = Path(train_artifact_path or scope["train_reference_artifact"])
    resolved_original_path = Path(
        original_validation_artifact_path or scope["validation_original_artifact"]
    )
    resolved_metric = metric or scope.get("primary_knn_metric", "cosine")
    if resolved_metric not in {"cosine", "l2"}:
        raise ValueError(f"Unsupported KNN metric: {resolved_metric}")
    resolved_k_values = k_values or tuple(int(k) for k in scope.get("knn_k_values", [1, 5, 10]))
    output_path = Path(output_dir)

    reports: list[dict[str, Any]] = []
    per_perturbation_outputs: list[dict[str, str]] = []
    for entry in matrix.get("first_round", []):
        perturbation_name = str(entry["name"])
        perturbed_path = Path(entry["output_path"])
        report = run_knn_perturbation_drop(
            train_artifact_path=resolved_train_path,
            original_validation_artifact_path=resolved_original_path,
            perturbed_validation_artifact_path=perturbed_path,
            k_values=resolved_k_values,
            metric=resolved_metric,  # type: ignore[arg-type]
            query_batch_size=query_batch_size,
            perturbation_name=perturbation_name,
            perturbation_group=entry.get("group"),
        )
        report_path = _knn_report_path(perturbed_path, output_path, metric=resolved_metric)
        save_json_report(report, report_path, overwrite=overwrite)
        reports.append(report)
        per_perturbation_outputs.append(
            {
                "perturbation": perturbation_name,
                "report_path": str(report_path),
            }
        )

    summary = build_all_perturbations_knn_drop_summary(reports, matrix=matrix)
    summary_path = output_path / (
        f"ssv2_validation100_videomae_base_16f_mean_"
        f"all_perturbations_knn_drop_{resolved_metric}.json"
    )
    save_json_report(summary, summary_path, overwrite=overwrite)
    return {
        "matrix_path": str(matrix_path),
        "train_artifact_path": str(resolved_train_path),
        "original_validation_artifact_path": str(resolved_original_path),
        "metric": resolved_metric,
        "k_values": list(resolved_k_values),
        "per_perturbation_reports": per_perturbation_outputs,
        "summary_report_path": str(summary_path),
    }


def save_json_report(
    report: dict[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute KNN accuracy drop for first-round perturbation artifacts."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--train-artifact", type=Path, default=None)
    parser.add_argument("--original-validation-artifact", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", choices=["cosine", "l2"], default=None)
    parser.add_argument("--k", type=int, nargs="+", default=None)
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_matrix_knn_perturbation_analysis(
        matrix_path=args.matrix,
        train_artifact_path=args.train_artifact,
        original_validation_artifact_path=args.original_validation_artifact,
        output_dir=args.output_dir,
        metric=args.metric,
        k_values=args.k,
        query_batch_size=args.query_batch_size,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _build_k_result(
    *,
    k: int,
    labels: torch.Tensor,
    train_seen_mask: torch.Tensor,
    original_predictions: torch.Tensor,
    perturbed_predictions: torch.Tensor,
    cosine_distance: torch.Tensor,
) -> dict[str, Any]:
    original_correct = original_predictions == labels
    perturbed_correct = perturbed_predictions == labels
    all_original = _accuracy_counts(original_correct)
    all_perturbed = _accuracy_counts(perturbed_correct)
    train_seen_original = _accuracy_counts(original_correct, mask=train_seen_mask)
    train_seen_perturbed = _accuracy_counts(perturbed_correct, mask=train_seen_mask)
    all_prediction_changed = original_predictions != perturbed_predictions
    train_seen_prediction_changed = all_prediction_changed & train_seen_mask

    return {
        "k": k,
        "all": _drop_metrics(all_original, all_perturbed),
        "train_seen": _drop_metrics(train_seen_original, train_seen_perturbed),
        "prediction_changes": {
            "all": _prediction_change_counts(
                original_correct=original_correct,
                perturbed_correct=perturbed_correct,
                prediction_changed=all_prediction_changed,
            ),
            "train_seen": _prediction_change_counts(
                original_correct=original_correct,
                perturbed_correct=perturbed_correct,
                prediction_changed=train_seen_prediction_changed,
                mask=train_seen_mask,
            ),
        },
        "embedding_shift_by_prediction_change": {
            "all": _embedding_shift_by_prediction_change(
                prediction_changed=all_prediction_changed,
                cosine_distance=cosine_distance,
            ),
            "train_seen": _embedding_shift_by_prediction_change(
                prediction_changed=all_prediction_changed,
                cosine_distance=cosine_distance,
                mask=train_seen_mask,
            ),
        },
    }


def _drop_metrics(
    original: dict[str, int | float],
    perturbed: dict[str, int | float],
) -> dict[str, Any]:
    original_accuracy = float(original["accuracy"])
    perturbed_accuracy = float(perturbed["accuracy"])
    absolute_drop = original_accuracy - perturbed_accuracy
    relative_drop = absolute_drop / original_accuracy if original_accuracy > 0 else None
    return {
        "original_correct": original["correct"],
        "original_total": original["total"],
        "original_accuracy": original_accuracy,
        "perturbed_correct": perturbed["correct"],
        "perturbed_total": perturbed["total"],
        "perturbed_accuracy": perturbed_accuracy,
        "absolute_accuracy_drop": absolute_drop,
        "relative_accuracy_drop": relative_drop,
    }


def _accuracy_counts(
    correct: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
) -> dict[str, int | float]:
    values = correct if mask is None else correct[mask]
    total = int(values.numel())
    count = int(values.sum().item()) if total else 0
    return {
        "correct": count,
        "total": total,
        "accuracy": count / total if total else 0.0,
    }


def _prediction_change_counts(
    *,
    original_correct: torch.Tensor,
    perturbed_correct: torch.Tensor,
    prediction_changed: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, int | float]:
    if mask is not None:
        original_correct = original_correct[mask]
        perturbed_correct = perturbed_correct[mask]
        prediction_changed = prediction_changed[mask]
    total = int(original_correct.numel())
    changed = int(prediction_changed.sum().item()) if total else 0
    correct_to_incorrect = int((original_correct & ~perturbed_correct).sum().item())
    incorrect_to_correct = int((~original_correct & perturbed_correct).sum().item())
    return {
        "total": total,
        "prediction_changed": changed,
        "prediction_change_rate": changed / total if total else 0.0,
        "correct_to_incorrect": correct_to_incorrect,
        "incorrect_to_correct": incorrect_to_correct,
        "correct_stayed_correct": int((original_correct & perturbed_correct).sum().item()),
        "incorrect_stayed_incorrect": int(
            (~original_correct & ~perturbed_correct).sum().item()
        ),
    }


def _embedding_shift_by_prediction_change(
    *,
    prediction_changed: torch.Tensor,
    cosine_distance: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float | int | None]:
    if mask is not None:
        prediction_changed = prediction_changed[mask]
        cosine_distance = cosine_distance[mask]
    changed_values = cosine_distance[prediction_changed]
    unchanged_values = cosine_distance[~prediction_changed]
    return {
        "changed_count": int(changed_values.numel()),
        "unchanged_count": int(unchanged_values.numel()),
        "changed_mean_cosine_distance": _optional_mean(changed_values),
        "unchanged_mean_cosine_distance": _optional_mean(unchanged_values),
        "changed_median_cosine_distance": _optional_median(changed_values),
        "unchanged_median_cosine_distance": _optional_median(unchanged_values),
    }


def _build_sample_prediction_changes(
    *,
    original_validation_artifact: dict[str, Any],
    labels: torch.Tensor,
    train_seen_mask: torch.Tensor,
    k_values: tuple[int, ...],
    original_predictions: dict[int, torch.Tensor],
    perturbed_predictions: dict[int, torch.Tensor],
    embedding_distances: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    video_ids = original_validation_artifact["video_ids"]
    metadata = original_validation_artifact.get("sample_metadata", [])
    records: list[dict[str, Any]] = []
    for index, video_id in enumerate(video_ids):
        sample_metadata = metadata[index] if index < len(metadata) else {}
        label_id = int(labels[index].item())
        by_k: dict[str, Any] = {}
        for k in k_values:
            original_prediction = int(original_predictions[k][index].item())
            perturbed_prediction = int(perturbed_predictions[k][index].item())
            original_correct = original_prediction == label_id
            perturbed_correct = perturbed_prediction == label_id
            by_k[str(k)] = {
                "original_prediction": original_prediction,
                "perturbed_prediction": perturbed_prediction,
                "original_correct": original_correct,
                "perturbed_correct": perturbed_correct,
                "prediction_changed": original_prediction != perturbed_prediction,
                "correct_to_incorrect": original_correct and not perturbed_correct,
                "incorrect_to_correct": (not original_correct) and perturbed_correct,
            }
        records.append(
            {
                "sample_index": index,
                "video_id": str(video_id),
                "label_id": label_id,
                "label_name": _metadata_str(sample_metadata, "label_name"),
                "train_label_seen": bool(train_seen_mask[index].item()),
                "embedding_shift": {
                    "cosine_similarity": float(
                        embedding_distances["cosine_similarity"][index].item()
                    ),
                    "cosine_distance": float(
                        embedding_distances["cosine_distance"][index].item()
                    ),
                    "l2_distance": float(embedding_distances["l2_distance"][index].item()),
                    "relative_l2_distance": float(
                        embedding_distances["relative_l2_distance"][index].item()
                    ),
                },
                "by_k": by_k,
            }
        )
    return records


def _compact_report(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "perturbation": report["perturbation"],
        "perturbation_group": report.get("perturbation_group"),
        "results": [
            {
                "k": result["k"],
                "all_original_accuracy": result["all"]["original_accuracy"],
                "all_perturbed_accuracy": result["all"]["perturbed_accuracy"],
                "all_absolute_accuracy_drop": result["all"]["absolute_accuracy_drop"],
                "all_relative_accuracy_drop": result["all"]["relative_accuracy_drop"],
                "train_seen_original_accuracy": result["train_seen"]["original_accuracy"],
                "train_seen_perturbed_accuracy": result["train_seen"][
                    "perturbed_accuracy"
                ],
                "train_seen_absolute_accuracy_drop": result["train_seen"][
                    "absolute_accuracy_drop"
                ],
                "train_seen_relative_accuracy_drop": result["train_seen"][
                    "relative_accuracy_drop"
                ],
                "all_prediction_change_rate": result["prediction_changes"]["all"][
                    "prediction_change_rate"
                ],
                "train_seen_prediction_change_rate": result["prediction_changes"][
                    "train_seen"
                ]["prediction_change_rate"],
                "all_correct_to_incorrect": result["prediction_changes"]["all"][
                    "correct_to_incorrect"
                ],
                "all_incorrect_to_correct": result["prediction_changes"]["all"][
                    "incorrect_to_correct"
                ],
                "all_changed_mean_cosine_distance": result[
                    "embedding_shift_by_prediction_change"
                ]["all"]["changed_mean_cosine_distance"],
                "all_unchanged_mean_cosine_distance": result[
                    "embedding_shift_by_prediction_change"
                ]["all"]["unchanged_mean_cosine_distance"],
            }
            for result in report["results"]
        ],
    }


def _compact_k_result(compact_report: dict[str, Any], k: int) -> dict[str, Any]:
    for result in compact_report["results"]:
        if int(result["k"]) == k:
            return {
                "perturbation": compact_report["perturbation"],
                "perturbation_group": compact_report.get("perturbation_group"),
                **result,
            }
    raise ValueError(f"Report for {compact_report['perturbation']} does not include k={k}")


def _build_group_summaries(reports: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(report.get("perturbation_group")) for report in reports})
    group_summaries: dict[str, Any] = {}
    for group in groups:
        group_reports = [
            report for report in reports if str(report.get("perturbation_group")) == group
        ]
        group_summaries[group] = {
            "perturbation_count": len(group_reports),
            "perturbations": [str(report["perturbation"]) for report in group_reports],
            "by_k": {
                str(k): _summarize_group_k(group_reports, k)
                for k in reports[0]["k_values"]
            },
        }
    return group_summaries


def _summarize_group_k(reports: list[dict[str, Any]], k: int) -> dict[str, Any]:
    k_results = [_result_for_k(report, k) for report in reports]
    return {
        "mean_all_absolute_accuracy_drop": _mean(
            result["all"]["absolute_accuracy_drop"] for result in k_results
        ),
        "mean_train_seen_absolute_accuracy_drop": _mean(
            result["train_seen"]["absolute_accuracy_drop"] for result in k_results
        ),
        "mean_all_prediction_change_rate": _mean(
            result["prediction_changes"]["all"]["prediction_change_rate"]
            for result in k_results
        ),
        "mean_train_seen_prediction_change_rate": _mean(
            result["prediction_changes"]["train_seen"]["prediction_change_rate"]
            for result in k_results
        ),
    }


def _result_for_k(report: dict[str, Any], k: int) -> dict[str, Any]:
    for result in report["results"]:
        if int(result["k"]) == int(k):
            return result
    raise ValueError(f"Report for {report['perturbation']} does not include k={k}")


def _mean(values: Any) -> float:
    sequence = [float(value) for value in values]
    if not sequence:
        raise ValueError("Cannot average an empty sequence")
    return sum(sequence) / len(sequence)


def _optional_mean(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    return float(values.detach().cpu().float().mean().item())


def _optional_median(values: torch.Tensor) -> float | None:
    if values.numel() == 0:
        return None
    return float(torch.quantile(values.detach().cpu().float(), 0.5).item())


def _labeled_tensors(
    artifact: dict[str, Any],
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = artifact["embeddings"]
    label_ids = artifact.get("label_ids")
    if label_ids is None:
        raise ValueError(f"{name} artifact does not contain label_ids")
    return embeddings.detach().cpu().float(), label_ids.detach().cpu().long()


def _train_seen_mask(train_labels: torch.Tensor, validation_labels: torch.Tensor) -> torch.Tensor:
    train_label_set = {int(label_id) for label_id in train_labels.detach().cpu().long().tolist()}
    return torch.tensor(
        [int(label_id) in train_label_set for label_id in validation_labels.tolist()],
        dtype=torch.bool,
    )


def _knn_report_path(
    perturbed_artifact_path: Path,
    output_dir: Path,
    *,
    metric: str,
) -> Path:
    return output_dir / f"{perturbed_artifact_path.stem}_knn_{metric}.json"


def _infer_perturbation_name(artifact: dict[str, Any]) -> str:
    config = _perturbation_config(artifact)
    value = config.get("name")
    if value is not None:
        return str(value)
    return "unknown"


def _perturbation_config(artifact: dict[str, Any]) -> dict[str, Any]:
    config = artifact.get("config")
    if isinstance(config, dict) and isinstance(config.get("perturbation"), dict):
        return dict(config["perturbation"])
    options = artifact.get("extraction_options")
    if isinstance(options, dict) and isinstance(options.get("perturbation"), dict):
        return dict(options["perturbation"])
    return {}


def _nested_value(
    artifact: dict[str, Any],
    section: str,
    key: str,
) -> str | None:
    section_value = artifact.get(section)
    if not isinstance(section_value, dict):
        return None
    value = section_value.get(key)
    return None if value is None else str(value)


def _metadata_str(sample_metadata: Any, key: str) -> str | None:
    if not isinstance(sample_metadata, dict):
        return None
    value = sample_metadata.get(key)
    return None if value is None else str(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
