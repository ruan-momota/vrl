from __future__ import annotations

import torch

from src.evaluation.baseline_interpretability import build_baseline_interpretability_report


def test_baseline_report_tracks_class_coverage_and_train_seen_metrics() -> None:
    train_artifact = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([0, 1]),
        split="train",
    )
    validation_artifact = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        labels=torch.tensor([0, 1, 2]),
        split="validation",
    )

    report = build_baseline_interpretability_report(
        train_artifact=train_artifact,
        validation_artifact=validation_artifact,
        metrics=("cosine", "l2"),
        k_values=(1,),
    )

    assert report["train_class_count"] == 2
    assert report["validation_class_count"] == 3
    assert report["common_class_count"] == 2
    assert report["validation_only_class_count"] == 1
    assert report["validation_samples_with_train_seen_label"] == 2
    assert report["validation_samples_with_train_unseen_label"] == 1
    assert report["metrics"]["cosine"] == [
        {
            "k": 1,
            "all_correct": 2,
            "all_total": 3,
            "all_accuracy": 2 / 3,
            "train_seen_correct": 2,
            "train_seen_total": 2,
            "train_seen_accuracy": 1.0,
        }
    ]
    assert report["metrics"]["l2"][0]["all_accuracy"] == 2 / 3


def _artifact(*, embeddings: torch.Tensor, labels: torch.Tensor, split: str) -> dict:
    sample_count = embeddings.shape[0]
    return {
        "format_version": 1,
        "embeddings": embeddings.float(),
        "label_ids": labels.long(),
        "video_ids": [f"video-{index}" for index in range(sample_count)],
        "sample_metadata": [{} for _ in range(sample_count)],
        "frame_indices": torch.zeros((sample_count, 1), dtype=torch.long),
        "config": {"dataset_name": "ssv2", "split": split},
        "model_metadata": {
            "checkpoint": "tiny-videomae",
            "embedding_type": "last_hidden_state_mean_pool",
        },
        "summary": {"embeddings_shape": list(embeddings.shape)},
    }
