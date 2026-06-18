from __future__ import annotations

import pytest
import torch

from src.artifact_alignment import (
    ArtifactAlignmentError,
    check_paired_embedding_alignment,
)


def test_check_paired_embedding_alignment_accepts_matching_artifacts() -> None:
    original = _artifact()
    perturbed = _artifact()

    report = check_paired_embedding_alignment(original, perturbed)

    assert report.aligned
    assert report.to_dict()["aligned"] is True
    assert report.original_samples == 2
    assert report.perturbed_samples == 2
    assert report.original_embedding_dim == 3
    assert report.perturbed_embedding_dim == 3


@pytest.mark.parametrize(
    ("override", "mismatch"),
    [
        ({"video_ids": ["b", "a"]}, "video_ids_match"),
        ({"label_ids": torch.tensor([1, 9])}, "label_ids_match"),
        ({"embeddings": torch.zeros((1, 3))}, "sample_count_match"),
        ({"embeddings": torch.zeros((2, 4))}, "embedding_dim_match"),
        ({"frame_indices": torch.tensor([[0, 2], [1, 2]])}, "frame_indices_match"),
    ],
)
def test_check_paired_embedding_alignment_reports_mismatch(
    override: dict,
    mismatch: str,
) -> None:
    original = _artifact()
    perturbed = _artifact(**override)

    report = check_paired_embedding_alignment(
        original,
        perturbed,
        raise_on_mismatch=False,
    )

    assert not report.aligned
    assert report.to_dict()[mismatch] is False


def test_check_paired_embedding_alignment_raises_on_mismatch() -> None:
    original = _artifact()
    perturbed = _artifact(video_ids=["b", "a"])

    with pytest.raises(ArtifactAlignmentError, match="video_ids_match"):
        check_paired_embedding_alignment(original, perturbed)


def _artifact(**override):
    artifact = {
        "embeddings": torch.zeros((2, 3)),
        "label_ids": torch.tensor([1, 2]),
        "video_ids": ["a", "b"],
        "frame_indices": torch.tensor([[0, 1], [0, 1]]),
    }
    artifact.update(override)
    return artifact
