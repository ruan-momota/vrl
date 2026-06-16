from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.knn_baseline import (
    evaluate_knn_baseline,
    knn_predict,
    majority_vote_by_rank,
    normalize_k_values,
    run_knn_baseline,
    save_knn_report,
)


def test_majority_vote_by_rank_breaks_ties_by_nearest_label() -> None:
    topk_label_ids = torch.tensor(
        [
            [7, 3, 3, 7],
            [4, 5, 6, 7],
        ]
    )

    predictions = majority_vote_by_rank(topk_label_ids)

    assert predictions.tolist() == [7, 4]


def test_knn_predict_cosine_majority_vote() -> None:
    train_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    train_labels = torch.tensor([1, 1, 2, 2])
    query_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    predictions = knn_predict(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=query_embeddings,
        k_values=(1, 3),
        metric="cosine",
        query_batch_size=1,
    )

    assert predictions[1].tolist() == [1, 2]
    assert predictions[3].tolist() == [1, 2]


def test_knn_predict_l2() -> None:
    train_embeddings = torch.tensor(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    train_labels = torch.tensor([0, 1, 2])
    query_embeddings = torch.tensor([[1.8, 0.1]])

    predictions = knn_predict(
        train_embeddings=train_embeddings,
        train_labels=train_labels,
        query_embeddings=query_embeddings,
        k_values=(1,),
        metric="l2",
    )

    assert predictions[1].tolist() == [1]


def test_evaluate_knn_baseline_from_artifacts() -> None:
    train_artifact = _artifact(
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.0, 1.0],
                [0.2, 0.8],
            ]
        ),
        labels=torch.tensor([5, 5, 6, 6]),
        split="debug_train",
    )
    validation_artifact = _artifact(
        embeddings=torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        labels=torch.tensor([5, 6]),
        split="debug_validation",
    )

    report = evaluate_knn_baseline(
        train_artifact=train_artifact,
        validation_artifact=validation_artifact,
        k_values=(1, 3),
        metric="cosine",
        query_batch_size=1,
        train_artifact_path="train.pt",
        validation_artifact_path="validation.pt",
    )

    assert report.dataset_name == "ssv2"
    assert report.train_split == "debug_train"
    assert report.validation_split == "debug_validation"
    assert report.train_samples == 4
    assert report.validation_samples == 2
    assert report.embedding_dim == 2
    assert report.normalized_embeddings is True
    assert [result.to_dict() for result in report.results] == [
        {"k": 1, "correct": 2, "total": 2, "accuracy": 1.0},
        {"k": 3, "correct": 2, "total": 2, "accuracy": 1.0},
    ]


def test_run_knn_baseline_loads_artifacts_and_saves_report(tmp_path: Path) -> None:
    train_path = tmp_path / "train.pt"
    validation_path = tmp_path / "validation.pt"
    torch.save(
        _artifact(
            embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            labels=torch.tensor([1, 2]),
            split="debug_train",
        ),
        train_path,
    )
    torch.save(
        _artifact(
            embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            labels=torch.tensor([1, 2]),
            split="debug_validation",
        ),
        validation_path,
    )

    report = run_knn_baseline(
        train_artifact_path=train_path,
        validation_artifact_path=validation_path,
        k_values=(1,),
    )
    output_path = tmp_path / "report.json"
    save_knn_report(report, output_path)

    assert report.results[0].accuracy == 1.0
    assert '"accuracy": 1.0' in output_path.read_text(encoding="utf-8")


def test_evaluate_knn_baseline_rejects_mismatched_embedding_dims() -> None:
    train_artifact = _artifact(
        embeddings=torch.zeros((2, 3)),
        labels=torch.tensor([1, 2]),
        split="debug_train",
    )
    validation_artifact = _artifact(
        embeddings=torch.zeros((2, 4)),
        labels=torch.tensor([1, 2]),
        split="debug_validation",
    )

    with pytest.raises(ValueError, match="embedding dimensions"):
        evaluate_knn_baseline(
            train_artifact=train_artifact,
            validation_artifact=validation_artifact,
        )


def test_normalize_k_values_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="larger than train sample count"):
        normalize_k_values((1, 4), train_count=3)


def _artifact(
    *,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    split: str,
) -> dict:
    sample_count = embeddings.shape[0]
    return {
        "format_version": 1,
        "embeddings": embeddings.float(),
        "label_ids": labels.long(),
        "video_ids": [f"video-{index}" for index in range(sample_count)],
        "sample_metadata": [{"split": split} for _ in range(sample_count)],
        "frame_indices": torch.zeros((sample_count, 1), dtype=torch.long),
        "config": {
            "dataset_name": "ssv2",
            "split": split,
        },
        "model_metadata": {
            "checkpoint": "tiny-videomae",
            "embedding_type": "last_hidden_state_mean_pool",
        },
        "summary": {
            "embeddings_shape": list(embeddings.shape),
        },
    }
