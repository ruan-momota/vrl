from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.artifact_alignment import ArtifactAlignmentError
from src.knn_perturbation_analysis import (
    build_all_perturbations_knn_drop_summary,
    evaluate_knn_perturbation_drop,
    run_matrix_knn_perturbation_analysis,
)


def test_evaluate_knn_perturbation_drop_reports_all_and_train_seen_metrics() -> None:
    train = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([1, 2]),
        split="train",
    )
    original = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]]),
        labels=torch.tensor([1, 2, 3]),
        split="validation",
        video_ids=["a", "b", "c"],
    )
    perturbed = _artifact(
        embeddings=torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.9, 0.1]]),
        labels=torch.tensor([1, 2, 3]),
        split="validation",
        video_ids=["a", "b", "c"],
        perturbation="temporal_reverse",
    )

    report = evaluate_knn_perturbation_drop(
        train_artifact=train,
        original_validation_artifact=original,
        perturbed_validation_artifact=perturbed,
        k_values=(1,),
        metric="cosine",
        perturbation_name="temporal_reverse",
        perturbation_group="motion",
    )

    result = report["results"][0]
    assert report["train_seen_validation_samples"] == 2
    assert report["train_unseen_validation_samples"] == 1
    assert result["all"]["original_accuracy"] == pytest.approx(2 / 3)
    assert result["all"]["perturbed_accuracy"] == pytest.approx(1 / 3)
    assert result["all"]["absolute_accuracy_drop"] == pytest.approx(1 / 3)
    assert result["all"]["relative_accuracy_drop"] == pytest.approx(0.5)
    assert result["train_seen"]["original_accuracy"] == pytest.approx(1.0)
    assert result["train_seen"]["perturbed_accuracy"] == pytest.approx(0.5)
    assert result["prediction_changes"]["all"]["prediction_changed"] == 1
    assert result["prediction_changes"]["all"]["correct_to_incorrect"] == 1
    assert report["sample_prediction_changes"][0]["by_k"]["1"] == {
        "original_prediction": 1,
        "perturbed_prediction": 2,
        "original_correct": True,
        "perturbed_correct": False,
        "prediction_changed": True,
        "correct_to_incorrect": True,
        "incorrect_to_correct": False,
    }
    assert report["sample_prediction_changes"][2]["train_label_seen"] is False


def test_evaluate_knn_perturbation_drop_reports_incorrect_to_correct() -> None:
    train = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([1, 2]),
        split="train",
    )
    original = _artifact(
        embeddings=torch.tensor([[0.0, 1.0]]),
        labels=torch.tensor([1]),
        split="validation",
    )
    perturbed = _artifact(
        embeddings=torch.tensor([[1.0, 0.0]]),
        labels=torch.tensor([1]),
        split="validation",
        perturbation="single_frame",
    )

    report = evaluate_knn_perturbation_drop(
        train_artifact=train,
        original_validation_artifact=original,
        perturbed_validation_artifact=perturbed,
        k_values=(1,),
    )

    result = report["results"][0]
    assert result["all"]["absolute_accuracy_drop"] == pytest.approx(-1.0)
    assert result["prediction_changes"]["all"]["incorrect_to_correct"] == 1
    assert report["sample_prediction_changes"][0]["by_k"]["1"]["incorrect_to_correct"] is True


def test_evaluate_knn_perturbation_drop_rejects_unaligned_artifacts() -> None:
    train = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([1, 2]),
        split="train",
    )
    original = _artifact(video_ids=["a", "b"])
    perturbed = _artifact(video_ids=["b", "a"])

    with pytest.raises(ArtifactAlignmentError):
        evaluate_knn_perturbation_drop(
            train_artifact=train,
            original_validation_artifact=original,
            perturbed_validation_artifact=perturbed,
            k_values=(1,),
        )


def test_build_all_perturbations_knn_drop_summary_ranks_drops() -> None:
    first = _report("single_frame", "motion", all_drop=0.2, train_seen_drop=0.5)
    second = _report("grayscale", "appearance", all_drop=-0.1, train_seen_drop=0.0)

    summary = build_all_perturbations_knn_drop_summary([first, second])

    assert summary["perturbation_count"] == 2
    assert summary["ranked_by_all_accuracy_drop"]["1"][0]["perturbation"] == "single_frame"
    assert summary["ranked_by_train_seen_accuracy_drop"]["1"][0][
        "train_seen_absolute_accuracy_drop"
    ] == pytest.approx(0.5)
    assert summary["group_summaries"]["motion"]["by_k"]["1"][
        "mean_all_absolute_accuracy_drop"
    ] == pytest.approx(0.2)


def test_run_matrix_knn_perturbation_analysis_writes_reports(tmp_path: Path) -> None:
    train_path = tmp_path / "train.pt"
    original_path = tmp_path / "original.pt"
    perturbed_path = tmp_path / "single_frame.pt"
    torch.save(
        _artifact(
            embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            labels=torch.tensor([1, 2]),
            split="train",
        ),
        train_path,
    )
    torch.save(
        _artifact(
            embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            labels=torch.tensor([1, 2]),
            split="validation",
        ),
        original_path,
    )
    torch.save(
        _artifact(
            embeddings=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
            labels=torch.tensor([1, 2]),
            split="validation",
            perturbation="single_frame",
        ),
        perturbed_path,
    )
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "matrix_name": "tiny",
                "scope": {
                    "train_reference_artifact": str(train_path),
                    "validation_original_artifact": str(original_path),
                    "primary_knn_metric": "cosine",
                    "knn_k_values": [1],
                },
                "first_round": [
                    {
                        "name": "single_frame",
                        "group": "motion",
                        "output_path": str(perturbed_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    result = run_matrix_knn_perturbation_analysis(
        matrix_path=matrix_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert result["summary_report_path"].endswith("_knn_drop_cosine.json")
    assert Path(result["summary_report_path"]).exists()
    assert (output_dir / "single_frame_knn_cosine.json").exists()


def _artifact(
    *,
    embeddings: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    split: str = "validation",
    video_ids: list[str] | None = None,
    perturbation: str = "none",
) -> dict:
    embeddings = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]) if embeddings is None else embeddings
    )
    labels = torch.tensor([1, 2]) if labels is None else labels
    video_ids = [f"video-{index}" for index in range(embeddings.shape[0])] if video_ids is None else video_ids
    return {
        "format_version": 1,
        "embeddings": embeddings.float(),
        "label_ids": labels.long(),
        "video_ids": video_ids,
        "sample_metadata": [
            {
                "label_id": int(labels[index].item()),
                "label_name": f"Class {int(labels[index].item())}",
                "video_id": video_ids[index],
            }
            for index in range(embeddings.shape[0])
        ],
        "frame_indices": torch.arange(2).repeat(embeddings.shape[0], 1),
        "config": {
            "dataset_name": "ssv2",
            "split": split,
            "perturbation": {"name": perturbation},
        },
        "model_metadata": {
            "checkpoint": "tiny-videomae",
            "embedding_type": "last_hidden_state_mean_pool",
        },
        "summary": {"embeddings_shape": list(embeddings.shape)},
    }


def _report(
    perturbation: str,
    group: str,
    *,
    all_drop: float,
    train_seen_drop: float,
) -> dict:
    return {
        "dataset_name": "ssv2",
        "model_checkpoint": "tiny-videomae",
        "embedding_type": "last_hidden_state_mean_pool",
        "metric": "cosine",
        "k_values": [1],
        "train_artifact_path": "train.pt",
        "original_validation_artifact_path": "original.pt",
        "validation_samples": 10,
        "train_seen_validation_samples": 5,
        "perturbation": perturbation,
        "perturbation_group": group,
        "results": [
            {
                "k": 1,
                "all": {
                    "original_accuracy": 0.4,
                    "perturbed_accuracy": 0.4 - all_drop,
                    "absolute_accuracy_drop": all_drop,
                    "relative_accuracy_drop": all_drop / 0.4,
                },
                "train_seen": {
                    "original_accuracy": 0.8,
                    "perturbed_accuracy": 0.8 - train_seen_drop,
                    "absolute_accuracy_drop": train_seen_drop,
                    "relative_accuracy_drop": train_seen_drop / 0.8,
                },
                "prediction_changes": {
                    "all": {
                        "prediction_change_rate": 0.2,
                        "correct_to_incorrect": 1,
                        "incorrect_to_correct": 0,
                    },
                    "train_seen": {
                        "prediction_change_rate": 0.4,
                    },
                },
                "embedding_shift_by_prediction_change": {
                    "all": {
                        "changed_mean_cosine_distance": 0.3,
                        "unchanged_mean_cosine_distance": 0.1,
                    },
                    "train_seen": {
                        "changed_mean_cosine_distance": 0.4,
                        "unchanged_mean_cosine_distance": 0.2,
                    },
                },
            }
        ],
    }
