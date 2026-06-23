from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from src.embedding_extraction import load_embedding_artifact
from src.knn_baseline import (
    DistanceMetric,
    knn_predict,
    normalize_k_values,
    validate_embedding_pair,
)


DEFAULT_TRAIN_ARTIFACT = Path("outputs/embeddings/train_original.pt")
DEFAULT_VALIDATION_ARTIFACT = Path("outputs/embeddings/validation_original.pt")
DEFAULT_OUTPUT_PATH = Path(
    "outputs/logs/"
    "ssv2_50c_train100_val30_videomae_base_16f_"
    "original_baseline_interpretability.json"
)


def build_baseline_interpretability_report(
    *,
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    metrics: Sequence[DistanceMetric] = ("cosine", "l2"),
    k_values: Sequence[int] = (1, 5, 10),
    query_batch_size: int = 1024,
    train_artifact_path: str | Path | None = None,
    validation_artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the baseline schema consumed by ``src.reporting``.

    KNN predictions are evaluated over all validation rows and separately over
    rows whose ground-truth label appears in the train reference set.
    """
    validate_embedding_pair(train_artifact, validation_artifact)
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")

    train_embeddings, train_labels = _labeled_tensors(train_artifact, name="train")
    validation_embeddings, validation_labels = _labeled_tensors(
        validation_artifact,
        name="validation",
    )
    resolved_metrics = _normalize_metrics(metrics)
    resolved_k_values = normalize_k_values(k_values, train_count=train_embeddings.shape[0])

    train_label_ids = sorted(set(train_labels.tolist()))
    validation_label_ids = sorted(set(validation_labels.tolist()))
    train_label_set = set(train_label_ids)
    train_seen_mask = torch.tensor(
        [label_id in train_label_set for label_id in validation_labels.tolist()],
        dtype=torch.bool,
    )

    metric_reports: dict[str, list[dict[str, int | float]]] = {}
    for metric in resolved_metrics:
        predictions_by_k = knn_predict(
            train_embeddings=train_embeddings,
            train_labels=train_labels,
            query_embeddings=validation_embeddings,
            k_values=resolved_k_values,
            metric=metric,
            query_batch_size=query_batch_size,
        )
        metric_reports[metric] = [
            _k_result(
                predictions=predictions_by_k[k],
                labels=validation_labels,
                train_seen_mask=train_seen_mask,
                k=k,
            )
            for k in resolved_k_values
        ]

    train_label_ids_set = set(train_label_ids)
    validation_label_ids_set = set(validation_label_ids)
    return {
        "format_version": 1,
        "report_type": "baseline_interpretability",
        "dataset_name": _nested_value(train_artifact, "config", "dataset_name"),
        "train_split": _nested_value(train_artifact, "config", "split"),
        "validation_split": _nested_value(validation_artifact, "config", "split"),
        "train_artifact_path": None
        if train_artifact_path is None
        else str(train_artifact_path),
        "validation_artifact_path": None
        if validation_artifact_path is None
        else str(validation_artifact_path),
        "model_checkpoint": _nested_value(train_artifact, "model_metadata", "checkpoint"),
        "embedding_type": _nested_value(
            train_artifact,
            "model_metadata",
            "embedding_type",
        ),
        "embedding_dim": int(train_embeddings.shape[1]),
        "query_batch_size": query_batch_size,
        "train_samples": int(train_embeddings.shape[0]),
        "validation_samples": int(validation_embeddings.shape[0]),
        "train_class_count": len(train_label_ids),
        "validation_class_count": len(validation_label_ids),
        "common_class_count": len(train_label_ids_set & validation_label_ids_set),
        "train_only_class_count": len(train_label_ids_set - validation_label_ids_set),
        "validation_only_class_count": len(validation_label_ids_set - train_label_ids_set),
        "train_label_ids": train_label_ids,
        "validation_label_ids": validation_label_ids,
        "validation_samples_with_train_seen_label": int(train_seen_mask.sum().item()),
        "validation_samples_with_train_unseen_label": int((~train_seen_mask).sum().item()),
        "metrics": metric_reports,
    }


def run_baseline_interpretability(
    *,
    train_artifact_path: str | Path,
    validation_artifact_path: str | Path,
    metrics: Sequence[DistanceMetric] = ("cosine", "l2"),
    k_values: Sequence[int] = (1, 5, 10),
    query_batch_size: int = 1024,
) -> dict[str, Any]:
    train_path = Path(train_artifact_path)
    validation_path = Path(validation_artifact_path)
    return build_baseline_interpretability_report(
        train_artifact=load_embedding_artifact(train_path),
        validation_artifact=load_embedding_artifact(validation_path),
        metrics=metrics,
        k_values=k_values,
        query_batch_size=query_batch_size,
        train_artifact_path=train_path,
        validation_artifact_path=validation_path,
    )


def save_baseline_interpretability_report(
    report: dict[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Baseline report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _labeled_tensors(
    artifact: dict[str, Any],
    *,
    name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings = artifact["embeddings"].detach().cpu().float()
    labels = artifact.get("label_ids")
    if labels is None:
        raise ValueError(f"{name} artifact does not contain label_ids")
    return embeddings, labels.detach().cpu().long()


def _normalize_metrics(metrics: Sequence[DistanceMetric]) -> tuple[DistanceMetric, ...]:
    if not metrics:
        raise ValueError("At least one metric is required")
    normalized: list[DistanceMetric] = []
    for metric in metrics:
        if metric not in {"cosine", "l2"}:
            raise ValueError(f"Unsupported KNN metric: {metric}")
        if metric not in normalized:
            normalized.append(metric)
    return tuple(normalized)


def _k_result(
    *,
    predictions: torch.Tensor,
    labels: torch.Tensor,
    train_seen_mask: torch.Tensor,
    k: int,
) -> dict[str, int | float]:
    correct = predictions == labels
    all_correct = int(correct.sum().item())
    all_total = int(correct.numel())
    train_seen_correct = int((correct & train_seen_mask).sum().item())
    train_seen_total = int(train_seen_mask.sum().item())
    return {
        "k": k,
        "all_correct": all_correct,
        "all_total": all_total,
        "all_accuracy": all_correct / all_total if all_total else 0.0,
        "train_seen_correct": train_seen_correct,
        "train_seen_total": train_seen_total,
        "train_seen_accuracy": (
            train_seen_correct / train_seen_total if train_seen_total else 0.0
        ),
    }


def _nested_value(artifact: dict[str, Any], section: str, key: str) -> str | None:
    value = artifact.get(section)
    if not isinstance(value, dict) or value.get(key) is None:
        return None
    return str(value[key])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate class-coverage and KNN baseline report for reporting.py."
    )
    parser.add_argument("--train-artifact", type=Path, default=DEFAULT_TRAIN_ARTIFACT)
    parser.add_argument(
        "--validation-artifact",
        type=Path,
        default=DEFAULT_VALIDATION_ARTIFACT,
    )
    parser.add_argument(
        "--metric",
        choices=["cosine", "l2"],
        nargs="+",
        default=["cosine", "l2"],
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run_baseline_interpretability(
        train_artifact_path=args.train_artifact,
        validation_artifact_path=args.validation_artifact,
        metrics=tuple(args.metric),
        k_values=tuple(args.k),
        query_batch_size=args.query_batch_size,
    )
    save_baseline_interpretability_report(
        report,
        args.output_path,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output_path": str(args.output_path), **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
