"""Versioned embedding artifact contract shared by pipeline and evaluation.

This module deliberately has no dataset, model, or evaluation dependency.  It
is the only persistence boundary between embedding extraction and downstream
analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class EmbeddingExtractionSummary:
    dataset_size: int
    successful_samples: int
    failed_samples: int
    batch_count: int
    total_seconds: float
    average_batch_seconds: float
    embeddings_shape: tuple[int, ...]
    embeddings_dtype: str
    labels_shape: tuple[int, ...] | None
    output_path: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["embeddings_shape"] = list(self.embeddings_shape)
        data["labels_shape"] = None if self.labels_shape is None else list(self.labels_shape)
        return data


@dataclass(frozen=True)
class EmbeddingExtractionResult:
    embeddings: torch.Tensor
    label_ids: torch.Tensor | None
    video_ids: list[str]
    sample_metadata: list[dict[str, Any]]
    frame_indices: torch.Tensor
    summary: EmbeddingExtractionSummary


def save_embedding_artifact(
    result: EmbeddingExtractionResult,
    output_path: str | Path,
    *,
    config_snapshot: dict[str, Any],
    model_metadata: dict[str, Any] | Any,
    extraction_options: dict[str, Any] | None = None,
    run_id: str | None = None,
    artifact_metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and persist a schema-versioned embedding artifact."""
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Embedding output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata_to_dict = getattr(model_metadata, "to_dict", None)
    model_metadata_dict = metadata_to_dict() if callable(metadata_to_dict) else dict(model_metadata)
    summary = EmbeddingExtractionSummary(
        **{
            **result.summary.to_dict(),
            "embeddings_shape": tuple(result.summary.embeddings_shape),
            "labels_shape": None
            if result.summary.labels_shape is None
            else tuple(result.summary.labels_shape),
            "output_path": str(path),
        }
    )
    artifact = {
        "format_version": 2,
        "embeddings": result.embeddings,
        "label_ids": result.label_ids,
        "video_ids": result.video_ids,
        "sample_metadata": result.sample_metadata,
        "frame_indices": result.frame_indices,
        "config": config_snapshot,
        "model_metadata": model_metadata_dict,
        "extraction_options": extraction_options or {},
        "run_id": run_id,
        "artifact_metadata": artifact_metadata or {},
        "summary": summary.to_dict(),
    }
    validate_embedding_artifact(artifact)
    torch.save(artifact, path)
    return artifact


def load_embedding_artifact(path: str | Path) -> dict[str, Any]:
    """Load and validate an artifact from any supported schema version."""
    artifact = torch.load(Path(path), map_location="cpu")
    validate_embedding_artifact(artifact)
    return artifact


def validate_embedding_result(result: EmbeddingExtractionResult) -> None:
    """Validate in-memory extraction output before it is persisted."""
    artifact = {
        "embeddings": result.embeddings,
        "label_ids": result.label_ids,
        "video_ids": result.video_ids,
        "sample_metadata": result.sample_metadata,
        "frame_indices": result.frame_indices,
    }
    validate_embedding_artifact(artifact)


def validate_embedding_artifact(artifact: dict[str, Any]) -> None:
    """Validate fields required by all current and legacy embedding artifacts."""
    format_version = artifact.get("format_version")
    if format_version is not None and (
        not isinstance(format_version, int) or format_version < 1
    ):
        raise TypeError("format_version must be a positive integer when present")

    run_id = artifact.get("run_id")
    if run_id is not None and not isinstance(run_id, str):
        raise TypeError("run_id must be a string or None")

    artifact_metadata = artifact.get("artifact_metadata")
    if artifact_metadata is not None and not isinstance(artifact_metadata, dict):
        raise TypeError("artifact_metadata must be a dictionary when present")

    embeddings = artifact.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("Embedding artifact must contain a tensor at key 'embeddings'")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [N, D], got {tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or Inf values")

    sample_count = embeddings.shape[0]
    video_ids = artifact.get("video_ids")
    if not isinstance(video_ids, list) or len(video_ids) != sample_count:
        raise ValueError("video_ids must be a list with one item per embedding")

    label_ids = artifact.get("label_ids")
    if label_ids is not None:
        if not isinstance(label_ids, torch.Tensor):
            raise TypeError("label_ids must be a tensor or None")
        if label_ids.shape != (sample_count,):
            raise ValueError(
                f"label_ids must have shape [{sample_count}], got {tuple(label_ids.shape)}"
            )

    sample_metadata = artifact.get("sample_metadata")
    if not isinstance(sample_metadata, list) or len(sample_metadata) != sample_count:
        raise ValueError("sample_metadata must be a list with one item per embedding")

    frame_indices = artifact.get("frame_indices")
    if not isinstance(frame_indices, torch.Tensor) or frame_indices.ndim != 2:
        raise ValueError("frame_indices must be a rank-2 tensor")
    if frame_indices.shape[0] != sample_count:
        raise ValueError("frame_indices must have one row per embedding")
