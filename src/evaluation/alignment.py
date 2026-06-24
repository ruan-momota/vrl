"""Paired embedding artifact compatibility checks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch


class ArtifactAlignmentError(ValueError):
    """Raised when two embedding artifacts cannot be compared row-by-row."""


@dataclass(frozen=True)
class ArtifactAlignmentReport:
    original_samples: int
    perturbed_samples: int
    original_embedding_dim: int
    perturbed_embedding_dim: int
    video_ids_match: bool
    label_ids_match: bool
    sample_count_match: bool
    embedding_dim_match: bool
    frame_indices_match: bool
    run_id_match: bool
    model_metadata_match: bool
    sampling_match: bool

    @property
    def aligned(self) -> bool:
        return all(
            (
                self.video_ids_match,
                self.label_ids_match,
                self.sample_count_match,
                self.embedding_dim_match,
                self.frame_indices_match,
                self.run_id_match,
                self.model_metadata_match,
                self.sampling_match,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "aligned": self.aligned}


def check_paired_embedding_alignment(
    original_artifact: dict[str, Any],
    perturbed_artifact: dict[str, Any],
    *,
    raise_on_mismatch: bool = True,
) -> ArtifactAlignmentReport:
    """Check whether two embedding artifacts are safe for row-wise comparison."""
    original_embeddings = _embedding_tensor(original_artifact, artifact_name="original")
    perturbed_embeddings = _embedding_tensor(perturbed_artifact, artifact_name="perturbed")

    report = ArtifactAlignmentReport(
        original_samples=int(original_embeddings.shape[0]),
        perturbed_samples=int(perturbed_embeddings.shape[0]),
        original_embedding_dim=int(original_embeddings.shape[1]),
        perturbed_embedding_dim=int(perturbed_embeddings.shape[1]),
        video_ids_match=_video_ids(original_artifact, artifact_name="original")
        == _video_ids(perturbed_artifact, artifact_name="perturbed"),
        label_ids_match=_tensor_equal_or_both_none(
            original_artifact.get("label_ids"),
            perturbed_artifact.get("label_ids"),
        ),
        sample_count_match=original_embeddings.shape[0] == perturbed_embeddings.shape[0],
        embedding_dim_match=original_embeddings.shape[1] == perturbed_embeddings.shape[1],
        frame_indices_match=_tensor_equal_or_both_none(
            original_artifact.get("frame_indices"),
            perturbed_artifact.get("frame_indices"),
        ),
        run_id_match=original_artifact.get("run_id") == perturbed_artifact.get("run_id"),
        model_metadata_match=_model_signature(original_artifact)
        == _model_signature(perturbed_artifact),
        sampling_match=_sampling_signature(original_artifact)
        == _sampling_signature(perturbed_artifact),
    )
    if raise_on_mismatch and not report.aligned:
        mismatches = [
            key
            for key, value in report.to_dict().items()
            if key.endswith("_match") and value is False
        ]
        raise ArtifactAlignmentError(
            "Embedding artifacts are not row-aligned: " + ", ".join(mismatches)
        )
    return report


def _embedding_tensor(artifact: dict[str, Any], *, artifact_name: str) -> torch.Tensor:
    embeddings = artifact.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2:
        raise TypeError(f"{artifact_name} artifact must contain rank-2 tensor embeddings")
    return embeddings


def _video_ids(artifact: dict[str, Any], *, artifact_name: str) -> list[str]:
    video_ids = artifact.get("video_ids")
    if not isinstance(video_ids, list):
        raise TypeError(f"{artifact_name} artifact must contain list video_ids")
    return [str(video_id) for video_id in video_ids]


def _tensor_equal_or_both_none(left: Any, right: Any) -> bool:
    if left is None and right is None:
        return True
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    return False


def _model_signature(artifact: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    metadata = artifact.get("model_metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, dict):
        return (("invalid", repr(metadata)),)
    relevant_keys = ("encoder_name", "checkpoint", "embedding_type", "model_type")
    return tuple(
        (key, repr(metadata.get(key)))
        for key in relevant_keys
        if key in metadata
    )


def _sampling_signature(artifact: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    config = artifact.get("config")
    if config is None:
        return None
    if not isinstance(config, dict):
        return (("invalid", repr(config)),)
    relevant_keys = ("num_frames", "sampling_strategy", "image_size", "video_decoder")
    return tuple(
        (key, repr(config.get(key)))
        for key in relevant_keys
        if key in config
    )
