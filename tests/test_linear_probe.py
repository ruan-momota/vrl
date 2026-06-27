from __future__ import annotations

import pytest
import torch

from src.evaluation.bootstrap import BootstrapConfig
from src.evaluation.linear_probe import (
    LinearProbeConfig,
    build_paired_linear_probe_report,
    evaluate_frozen_linear_probe,
    fit_frozen_linear_probe,
    stratified_train_validation_split,
)


def test_stratified_train_validation_split_is_reproducible_and_keeps_each_class() -> None:
    labels = torch.tensor([0] * 5 + [1] * 5 + [2] * 5)

    first = stratified_train_validation_split(labels, validation_fraction=0.4, seed=9)
    second = stratified_train_validation_split(labels, validation_fraction=0.4, seed=9)

    assert torch.equal(first["train_indices"], second["train_indices"])
    assert torch.equal(first["validation_indices"], second["validation_indices"])
    for label in (0, 1, 2):
        assert labels[first["train_indices"]].eq(label).any()
        assert labels[first["validation_indices"]].eq(label).any()


def test_frozen_linear_probe_selects_on_train_only_and_reports_paired_drop() -> None:
    train = _artifact(
        embeddings=torch.tensor(
            [[1.0, 0.0]] * 6 + [[0.0, 1.0]] * 6,
        ),
        labels=torch.tensor([0] * 6 + [1] * 6),
        video_prefix="train",
    )
    config = LinearProbeConfig(
        validation_fraction=0.33,
        split_seed=5,
        training_seed=7,
        l2_values=(0.0, 1e-3),
        max_iterations=40,
        feature_normalization="l2",
    )

    probe = fit_frozen_linear_probe(train, config=config)
    original = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([0, 1]),
        video_prefix="heldout",
    )
    perturbed = _artifact(
        embeddings=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        labels=torch.tensor([0, 1]),
        video_prefix="heldout",
    )

    original_evaluation = evaluate_frozen_linear_probe(probe, original)
    perturbed_evaluation = evaluate_frozen_linear_probe(probe, perturbed)
    report = build_paired_linear_probe_report(
        original_evaluation,
        perturbed_evaluation,
        bootstrap_config=BootstrapConfig(resamples=50, seed=3),
        perturbation_name="toy_swap",
        perturbation_group="motion",
    )

    assert probe.metadata["training_protocol"].startswith("stratified_train_validation")
    assert len(probe.metadata["selection_candidates"]) == 2
    assert original_evaluation.accuracy == pytest.approx(1.0)
    assert perturbed_evaluation.accuracy == pytest.approx(0.0)
    assert report["accuracy_drop"] == pytest.approx(1.0)
    assert report["correct_to_incorrect_count"] == 2
    assert report["accuracy_drop_bootstrap"]["statistics"]["mean"]["point_estimate"] == pytest.approx(
        1.0
    )


def test_linear_probe_rejects_incompatible_evaluation_embedding_dimension() -> None:
    train = _artifact(
        embeddings=torch.tensor([[1.0, 0.0]] * 3 + [[0.0, 1.0]] * 3),
        labels=torch.tensor([0] * 3 + [1] * 3),
        video_prefix="train",
    )
    probe = fit_frozen_linear_probe(
        train,
        config=LinearProbeConfig(l2_values=(0.0,), max_iterations=20),
    )
    incompatible = _artifact(
        embeddings=torch.zeros((2, 3)),
        labels=torch.tensor([0, 1]),
        video_prefix="heldout",
    )

    with pytest.raises(ValueError, match="Embedding dimension"):
        evaluate_frozen_linear_probe(probe, incompatible)


def _artifact(
    *,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    video_prefix: str,
) -> dict:
    sample_count = embeddings.shape[0]
    return {
        "format_version": 2,
        "embeddings": embeddings,
        "label_ids": labels,
        "video_ids": [f"{video_prefix}-{index}" for index in range(sample_count)],
        "sample_metadata": [{} for _ in range(sample_count)],
        "frame_indices": torch.zeros((sample_count, 4), dtype=torch.long),
        "run_id": "toy-run",
        "artifact_metadata": {},
    }
