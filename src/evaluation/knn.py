"""K-nearest-neighbour diagnostics over embedding artifacts."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from src.artifacts import load_embedding_artifact, validate_embedding_artifact


DistanceMetric = Literal["cosine", "l2"]


@dataclass(frozen=True)
class KNNMetricResult:
    k: int
    correct: int
    total: int
    accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KNNBaselineReport:
    dataset_name: str | None
    train_split: str | None
    validation_split: str | None
    train_artifact_path: str | None
    validation_artifact_path: str | None
    model_checkpoint: str | None
    embedding_type: str | None
    metric: DistanceMetric
    normalized_embeddings: bool
    train_samples: int
    validation_samples: int
    embedding_dim: int
    query_batch_size: int
    total_seconds: float
    results: tuple[KNNMetricResult, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["results"] = [result.to_dict() for result in self.results]
        return data


def run_knn_baseline(
    *,
    train_artifact_path: str | Path,
    validation_artifact_path: str | Path,
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
    metric: DistanceMetric = "cosine",
    query_batch_size: int = 1024,
) -> KNNBaselineReport:
    train_path = Path(train_artifact_path)
    validation_path = Path(validation_artifact_path)
    train_artifact = load_embedding_artifact(train_path)
    validation_artifact = load_embedding_artifact(validation_path)
    return evaluate_knn_baseline(
        train_artifact=train_artifact,
        validation_artifact=validation_artifact,
        k_values=k_values,
        metric=metric,
        query_batch_size=query_batch_size,
        train_artifact_path=str(train_path),
        validation_artifact_path=str(validation_path),
    )


def evaluate_knn_baseline(
    *,
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    k_values: list[int] | tuple[int, ...] = (1, 5, 10),
    metric: DistanceMetric = "cosine",
    query_batch_size: int = 1024,
    train_artifact_path: str | None = None,
    validation_artifact_path: str | None = None,
) -> KNNBaselineReport:
    validate_embedding_pair(train_artifact, validation_artifact)
    if metric not in {"cosine", "l2"}:
        raise ValueError(f"Unsupported KNN metric: {metric}")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")

    train_embeddings, train_labels = _labeled_tensors(train_artifact, name="train")
    validation_embeddings, validation_labels = _labeled_tensors(
        validation_artifact,
        name="validation",
    )
    normalized_k_values = normalize_k_values(k_values, train_count=train_embeddings.shape[0])

    start = time.perf_counter()
    predictions_by_k = knn_predict(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=validation_embeddings,
        k_values=normalized_k_values,
        metric=metric,
        query_batch_size=query_batch_size,
    )
    total_seconds = time.perf_counter() - start

    results: list[KNNMetricResult] = []
    for k in normalized_k_values:
        predictions = predictions_by_k[k]
        correct = int((predictions == validation_labels).sum().item())
        total = int(validation_labels.numel())
        results.append(
            KNNMetricResult(
                k=k,
                correct=correct,
                total=total,
                accuracy=correct / total if total else 0.0,
            )
        )

    return KNNBaselineReport(
        dataset_name=_shared_metadata_value(
            train_artifact,
            validation_artifact,
            "config",
            "dataset_name",
        ),
        train_split=_nested_value(train_artifact, "config", "split"),
        validation_split=_nested_value(validation_artifact, "config", "split"),
        train_artifact_path=train_artifact_path,
        validation_artifact_path=validation_artifact_path,
        model_checkpoint=_shared_metadata_value(
            train_artifact,
            validation_artifact,
            "model_metadata",
            "checkpoint",
        ),
        embedding_type=_shared_metadata_value(
            train_artifact,
            validation_artifact,
            "model_metadata",
            "embedding_type",
        ),
        metric=metric,
        normalized_embeddings=metric == "cosine",
        train_samples=int(train_embeddings.shape[0]),
        validation_samples=int(validation_embeddings.shape[0]),
        embedding_dim=int(train_embeddings.shape[1]),
        query_batch_size=query_batch_size,
        total_seconds=total_seconds,
        results=tuple(results),
    )


def knn_predict(
    *,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    query_embeddings: torch.Tensor,
    k_values: list[int] | tuple[int, ...],
    metric: DistanceMetric = "cosine",
    query_batch_size: int = 1024,
) -> dict[int, torch.Tensor]:
    if train_embeddings.ndim != 2:
        raise ValueError("train_embeddings must have shape [N, D]")
    if query_embeddings.ndim != 2:
        raise ValueError("query_embeddings must have shape [Q, D]")
    if train_embeddings.shape[1] != query_embeddings.shape[1]:
        raise ValueError(
            "train and query embedding dimensions must match, "
            f"got {train_embeddings.shape[1]} and {query_embeddings.shape[1]}"
        )
    if train_labels.shape != (train_embeddings.shape[0],):
        raise ValueError("train_labels must have one label per train embedding")
    if metric not in {"cosine", "l2"}:
        raise ValueError(f"Unsupported KNN metric: {metric}")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size must be positive")

    normalized_k_values = normalize_k_values(k_values, train_count=train_embeddings.shape[0])
    max_k = max(normalized_k_values)
    train_vectors = train_embeddings.detach().cpu().float()
    query_vectors = query_embeddings.detach().cpu().float()
    labels = train_labels.detach().cpu().long()
    if metric == "cosine":
        train_vectors = F.normalize(train_vectors, p=2, dim=1)

    prediction_chunks: dict[int, list[torch.Tensor]] = {k: [] for k in normalized_k_values}
    for start in range(0, query_vectors.shape[0], query_batch_size):
        query_chunk = query_vectors[start : start + query_batch_size]
        if metric == "cosine":
            query_chunk = F.normalize(query_chunk, p=2, dim=1)
            scores = query_chunk @ train_vectors.T
            top_indices = scores.topk(max_k, dim=1, largest=True, sorted=True).indices
        else:
            distances = torch.cdist(query_chunk, train_vectors, p=2)
            top_indices = distances.topk(max_k, dim=1, largest=False, sorted=True).indices

        top_labels = labels[top_indices]
        for k in normalized_k_values:
            prediction_chunks[k].append(majority_vote_by_rank(top_labels[:, :k]))

    return {
        k: torch.cat(chunks, dim=0) if chunks else torch.empty((0,), dtype=torch.long)
        for k, chunks in prediction_chunks.items()
    }


def majority_vote_by_rank(topk_label_ids: torch.Tensor) -> torch.Tensor:
    """Vote over nearest labels; ties go to the label with the nearest neighbor."""
    if topk_label_ids.ndim != 2:
        raise ValueError("topk_label_ids must have shape [Q, K]")
    predictions: list[int] = []
    for row in topk_label_ids.detach().cpu().long().tolist():
        counts: dict[int, int] = {}
        first_rank: dict[int, int] = {}
        for rank, label_id in enumerate(row):
            counts[label_id] = counts.get(label_id, 0) + 1
            first_rank.setdefault(label_id, rank)
        prediction = min(
            counts,
            key=lambda label_id: (-counts[label_id], first_rank[label_id], label_id),
        )
        predictions.append(prediction)
    return torch.tensor(predictions, dtype=torch.long)


def validate_embedding_pair(
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
) -> None:
    validate_embedding_artifact(train_artifact)
    validate_embedding_artifact(validation_artifact)
    train_embeddings, train_labels = _labeled_tensors(train_artifact, name="train")
    validation_embeddings, validation_labels = _labeled_tensors(
        validation_artifact,
        name="validation",
    )

    if train_embeddings.shape[0] == 0:
        raise ValueError("train artifact contains no embeddings")
    if validation_embeddings.shape[0] == 0:
        raise ValueError("validation artifact contains no embeddings")
    if train_embeddings.shape[1] != validation_embeddings.shape[1]:
        raise ValueError(
            "train and validation embedding dimensions must match, "
            f"got {train_embeddings.shape[1]} and {validation_embeddings.shape[1]}"
        )
    if train_labels.numel() != train_embeddings.shape[0]:
        raise ValueError("train label count does not match train embeddings")
    if validation_labels.numel() != validation_embeddings.shape[0]:
        raise ValueError("validation label count does not match validation embeddings")

    _require_shared_metadata(train_artifact, validation_artifact, "model_metadata", "checkpoint")
    _require_shared_metadata(
        train_artifact,
        validation_artifact,
        "model_metadata",
        "embedding_type",
    )
    _require_same_run_id(train_artifact, validation_artifact)


def normalize_k_values(
    k_values: list[int] | tuple[int, ...],
    *,
    train_count: int,
) -> tuple[int, ...]:
    if not k_values:
        raise ValueError("At least one K value is required")
    if train_count <= 0:
        raise ValueError("train_count must be positive")

    normalized: list[int] = []
    seen: set[int] = set()
    for k in k_values:
        if k <= 0:
            raise ValueError(f"K values must be positive, got {k}")
        if k > train_count:
            raise ValueError(f"K={k} is larger than train sample count {train_count}")
        if k not in seen:
            normalized.append(int(k))
            seen.add(int(k))
    return tuple(normalized)


def save_knn_report(
    report: KNNBaselineReport,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"KNN report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a KNN baseline over saved embedding artifacts."
    )
    parser.add_argument(
        "--train-artifact",
        type=Path,
        default=Path("outputs/embeddings/ssv2_debug_train_videomae_base_16f_mean.pt"),
    )
    parser.add_argument(
        "--validation-artifact",
        type=Path,
        default=Path("outputs/embeddings/ssv2_debug_validation_videomae_base_16f_mean.pt"),
    )
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--query-batch-size", type=int, default=1024)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_knn_from_args(args: argparse.Namespace) -> KNNBaselineReport:
    report = run_knn_baseline(
        train_artifact_path=args.train_artifact,
        validation_artifact_path=args.validation_artifact,
        k_values=args.k,
        metric=args.metric,
        query_batch_size=args.query_batch_size,
    )
    if args.output_path is not None:
        save_knn_report(report, args.output_path, overwrite=args.overwrite)
    return report


def main() -> int:
    report = run_knn_from_args(parse_args())
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0


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


def _shared_metadata_value(
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    section: str,
    key: str,
) -> str | None:
    train_value = _nested_value(train_artifact, section, key)
    validation_value = _nested_value(validation_artifact, section, key)
    if train_value is not None:
        return train_value
    return validation_value


def _require_shared_metadata(
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
    section: str,
    key: str,
) -> None:
    train_value = _nested_value(train_artifact, section, key)
    validation_value = _nested_value(validation_artifact, section, key)
    if train_value is not None and validation_value is not None and train_value != validation_value:
        raise ValueError(
            f"train and validation artifacts use different {section}.{key}: "
            f"{train_value!r} vs {validation_value!r}"
        )


def _require_same_run_id(
    train_artifact: dict[str, Any],
    validation_artifact: dict[str, Any],
) -> None:
    train_run_id = train_artifact.get("run_id")
    validation_run_id = validation_artifact.get("run_id")
    if train_run_id is not None and validation_run_id is not None and train_run_id != validation_run_id:
        raise ValueError(
            "train and validation artifacts belong to different runs: "
            f"{train_run_id!r} vs {validation_run_id!r}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
