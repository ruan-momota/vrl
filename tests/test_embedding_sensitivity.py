from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.evaluation.alignment import ArtifactAlignmentError
from src.evaluation.bootstrap import BootstrapConfig
from src.evaluation.sensitivity import (
    build_all_perturbations_summary,
    build_class_sensitivity_report,
    build_embedding_sensitivity_report,
    compute_embedding_distances,
    run_matrix_sensitivity,
    summarize_values,
)


def test_compute_embedding_distances() -> None:
    original = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    perturbed = torch.tensor([[0.0, 1.0], [0.0, 1.0]])

    distances = compute_embedding_distances(original, perturbed)

    assert distances["cosine_similarity"].tolist() == pytest.approx([0.0, 1.0])
    assert distances["cosine_distance"].tolist() == pytest.approx([1.0, 0.0])
    assert distances["l2_distance"].tolist() == pytest.approx([2**0.5, 1.0])
    assert distances["relative_l2_distance"].tolist() == pytest.approx([2**0.5, 0.5])


def test_summarize_values_uses_population_std() -> None:
    summary = summarize_values(torch.tensor([1.0, 2.0, 3.0, 4.0]))

    assert summary["count"] == 4
    assert summary["mean"] == pytest.approx(2.5)
    assert summary["median"] == pytest.approx(2.5)
    assert summary["std"] == pytest.approx(1.25**0.5)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(4.0)


def test_build_embedding_sensitivity_report_from_aligned_artifacts() -> None:
    original = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([7, 8]),
        label_names=["Left", "Up"],
    )
    perturbed = _artifact(
        embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        labels=torch.tensor([7, 8]),
        label_names=["Left", "Up"],
        perturbation="temporal_reverse",
    )

    report = build_embedding_sensitivity_report(
        original_artifact=original,
        perturbed_artifact=perturbed,
        original_artifact_path="original.pt",
        perturbed_artifact_path="reverse.pt",
        perturbation_name="temporal_reverse",
        perturbation_group="motion",
        bootstrap_config=BootstrapConfig(resamples=40, seed=11),
    )

    assert report["alignment"]["aligned"] is True
    assert report["perturbation"] == "temporal_reverse"
    assert report["summary"]["sample_count"] == 2
    assert report["summary"]["metrics"]["cosine_distance"]["mean"] == pytest.approx(0.5)
    assert report["sample_metrics"][1]["label_id"] == 8
    assert report["sample_metrics"][1]["label_name"] == "Up"
    assert report["sample_metrics"][1]["cosine_distance"] == pytest.approx(1.0)
    assert report["sample_metrics"][1]["perturbation_config"]["name"] == "temporal_reverse"
    assert report["bootstrap"]["cosine_distance"]["statistics"]["mean"][
        "point_estimate"
    ] == pytest.approx(0.5)


def test_build_embedding_sensitivity_report_rejects_misalignment() -> None:
    original = _artifact(video_ids=["a", "b"])
    perturbed = _artifact(video_ids=["b", "a"])

    with pytest.raises(ArtifactAlignmentError):
        build_embedding_sensitivity_report(
            original_artifact=original,
            perturbed_artifact=perturbed,
        )


def test_build_all_perturbations_summary_compares_groups() -> None:
    motion = _report_with_sample_distances(
        "temporal_reverse",
        "motion",
        [0.3, 0.5],
        [3.0, 5.0],
    )
    appearance = _report_with_sample_distances(
        "grayscale",
        "appearance",
        [0.1, 0.2],
        [1.0, 2.0],
    )

    summary = build_all_perturbations_summary([motion, appearance])

    assert summary["perturbation_count"] == 2
    assert summary["ranked_by_mean_cosine_distance"][0]["perturbation"] == "temporal_reverse"
    assert summary["group_comparison"]["larger_shift_group"] == "motion"
    assert summary["group_summaries"]["motion"]["metrics"]["cosine_distance"][
        "mean_of_perturbation_means"
    ]["mean"] == pytest.approx(0.4)


def test_build_class_sensitivity_report_ranks_per_perturbation() -> None:
    motion = _report_with_class_distances(
        "temporal_reverse",
        "motion",
        {
            (1, "Class A"): [0.9],
            (2, "Class B"): [0.1],
        },
    )
    appearance = _report_with_class_distances(
        "grayscale",
        "appearance",
        {
            (1, "Class A"): [0.2],
            (2, "Class B"): [0.8],
        },
    )

    report = build_class_sensitivity_report([motion, appearance], top_k=1)

    assert report["class_count"] == 2
    by_label = {item["label_id"]: item for item in report["classes"]}
    assert by_label[1]["perturbations"]["temporal_reverse"][
        "rank_by_mean_cosine_distance"
    ] == 1
    assert by_label[2]["perturbations"]["grayscale"]["rank_by_mean_cosine_distance"] == 1
    assert report["most_sensitive_classes_by_mean_cosine_distance"][0][
        "mean_cosine_distance_across_perturbations"
    ] == pytest.approx(0.55)


def test_run_matrix_sensitivity_writes_reports(tmp_path: Path) -> None:
    original_path = tmp_path / "original.pt"
    reverse_path = tmp_path / "reverse.pt"
    grayscale_path = tmp_path / "grayscale.pt"
    torch.save(_artifact(), original_path)
    torch.save(_artifact(embeddings=torch.tensor([[0.0, 1.0], [1.0, 0.0]])), reverse_path)
    torch.save(_artifact(embeddings=torch.tensor([[1.0, 0.0], [1.0, 0.0]])), grayscale_path)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(
        json.dumps(
            {
                "matrix_name": "tiny",
                "scope": {"validation_original_artifact": str(original_path)},
                "first_round": [
                    {
                        "name": "temporal_reverse",
                        "group": "motion",
                        "output_path": str(reverse_path),
                    },
                    {
                        "name": "grayscale",
                        "group": "appearance",
                        "output_path": str(grayscale_path),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "reports"

    result = run_matrix_sensitivity(
        matrix_path=matrix_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert Path(result["summary_report_path"]).exists()
    assert Path(result["class_report_path"]).exists()
    assert (output_dir / "reverse_sensitivity.json").exists()
    assert (output_dir / "grayscale_sensitivity.json").exists()


def _artifact(
    *,
    embeddings: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    label_names: list[str] | None = None,
    video_ids: list[str] | None = None,
    perturbation: str = "none",
) -> dict:
    embeddings = (
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]) if embeddings is None else embeddings
    )
    labels = torch.tensor([1, 2]) if labels is None else labels
    video_ids = ["a", "b"] if video_ids is None else video_ids
    label_names = ["Class A", "Class B"] if label_names is None else label_names
    return {
        "format_version": 1,
        "embeddings": embeddings.float(),
        "label_ids": labels.long(),
        "video_ids": video_ids,
        "sample_metadata": [
            {
                "label_id": int(labels[index].item()),
                "label_name": label_names[index],
                "video_id": video_ids[index],
            }
            for index in range(embeddings.shape[0])
        ],
        "frame_indices": torch.tensor([[0, 1], [0, 1]]),
        "config": {
            "dataset_name": "ssv2",
            "split": "validation",
            "perturbation": {
                "name": perturbation,
                "seed": 0,
                "frame_index": None,
                "freeze_start_fraction": 0.5,
                "occlusion_size_fraction": 0.25,
                "occlusion_fill_value": 0,
            },
        },
        "model_metadata": {
            "checkpoint": "tiny-videomae",
            "embedding_type": "last_hidden_state_mean_pool",
        },
        "summary": {
            "embeddings_shape": list(embeddings.shape),
        },
    }


def _report_with_sample_distances(
    perturbation: str,
    group: str,
    cosine_distances: list[float],
    l2_distances: list[float],
) -> dict:
    sample_metrics = [
        {
            "sample_index": index,
            "video_id": f"video-{index}",
            "label_id": index,
            "label_name": f"Class {index}",
            "perturbation": perturbation,
            "perturbation_group": group,
            "cosine_distance": cosine_distances[index],
            "cosine_similarity": 1.0 - cosine_distances[index],
            "l2_distance": l2_distances[index],
            "relative_l2_distance": l2_distances[index] / 10.0,
        }
        for index in range(len(cosine_distances))
    ]
    return {
        "dataset_name": "ssv2",
        "split": "validation",
        "model_checkpoint": "tiny-videomae",
        "embedding_type": "last_hidden_state_mean_pool",
        "original_artifact_path": "original.pt",
        "perturbation": perturbation,
        "perturbation_group": group,
        "summary": {
            "sample_count": len(sample_metrics),
            "metrics": {
                "cosine_distance": {
                    "mean": sum(cosine_distances) / len(cosine_distances),
                    "median": cosine_distances[0],
                    "std": 0.0,
                    "min": min(cosine_distances),
                    "max": max(cosine_distances),
                },
                "l2_distance": {
                    "mean": sum(l2_distances) / len(l2_distances),
                    "median": l2_distances[0],
                },
                "relative_l2_distance": {
                    "mean": sum(l2_distances) / len(l2_distances) / 10.0,
                    "median": l2_distances[0] / 10.0,
                },
            },
        },
        "sample_metrics": sample_metrics,
    }


def _report_with_class_distances(
    perturbation: str,
    group: str,
    distances_by_class: dict[tuple[int, str], list[float]],
) -> dict:
    sample_metrics = []
    for label_id, label_name in distances_by_class:
        for value in distances_by_class[(label_id, label_name)]:
            sample_metrics.append(
                {
                    "sample_index": len(sample_metrics),
                    "video_id": f"video-{len(sample_metrics)}",
                    "label_id": label_id,
                    "label_name": label_name,
                    "perturbation": perturbation,
                    "perturbation_group": group,
                    "cosine_distance": value,
                    "cosine_similarity": 1.0 - value,
                    "l2_distance": value * 10.0,
                    "relative_l2_distance": value,
                }
            )
    return {
        "dataset_name": "ssv2",
        "split": "validation",
        "model_checkpoint": "tiny-videomae",
        "embedding_type": "last_hidden_state_mean_pool",
        "perturbation": perturbation,
        "perturbation_group": group,
        "sample_metrics": sample_metrics,
    }
