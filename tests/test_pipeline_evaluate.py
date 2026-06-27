from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

from src.pipeline.evaluate import RunEvaluationConfig, run_linear_probe_evaluation


def test_evaluation_module_exposes_a_python_m_cli() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "src.pipeline.evaluate", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--config" in result.stdout


def test_run_linear_probe_evaluation_writes_run_scoped_metrics_reports_and_quality_audit(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "runs" / "toy-run"
    _save_artifact(
        run_dir / "embeddings/train/original.pt",
        embeddings=torch.tensor([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4),
        labels=torch.tensor([0] * 4 + [1] * 4),
        video_ids=[f"train-{index}" for index in range(8)],
        split="train",
    )
    heldout_ids = ["heldout-0", "heldout-1"]
    _save_artifact(
        run_dir / "embeddings/validation/original.pt",
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([0, 1]),
        video_ids=heldout_ids,
        split="validation",
    )
    _save_artifact(
        run_dir / "embeddings/validation/perturbations/toy-swap.pt",
        embeddings=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        labels=torch.tensor([0, 1]),
        video_ids=heldout_ids,
        split="validation",
        perturbation={"name": "toy_swap", "constant_across_frames": True},
    )
    config_path = tmp_path / "evaluation.json"
    config_path.write_text(
        json.dumps(
            {
                "run_id": "toy-run",
                "output_root": str(tmp_path / "runs"),
                "train_original": "embeddings/train/original.pt",
                "heldout_original": "embeddings/validation/original.pt",
                "perturbations": [
                    {
                        "name": "toy_swap",
                        "artifact": "embeddings/validation/perturbations/toy-swap.pt",
                        "group": "motion",
                        "role": "fixed_mid",
                    }
                ],
                "linear_probe": {
                    "validation_fraction": 0.25,
                    "split_seed": 1,
                    "training_seed": 2,
                    "l2_values": [0.0],
                    "max_iterations": 30,
                    "feature_normalization": "l2",
                    "device": "cpu",
                },
                "bootstrap": {"resamples": 20, "seed": 4, "confidence_level": 0.95},
                "knn": {"metric": "cosine", "k_values": [1]},
            }
        ),
        encoding="utf-8",
    )

    result = run_linear_probe_evaluation(RunEvaluationConfig.from_file(config_path))

    summary = json.loads(Path(result["summary_json"]).read_text(encoding="utf-8"))
    assert summary["run_id"] == "toy-run"
    assert summary["original_linear_probe_accuracy"] == 1.0
    assert summary["rows"][0]["linear_probe_accuracy_drop"] == 1.0
    assert summary["qualitative_samples"]["record_count"] == 2
    assert summary["failure_summary"]["all_extractions_succeeded"] is True
    assert Path(result["linear_probe"]).exists()
    assert Path(result["summary_csv"]).exists()
    assert Path(result["provenance"]).exists()
    quality = json.loads(Path(result["quality_audit"]).read_text(encoding="utf-8"))
    assert quality["records"][0]["frame_indices_match"] is True


def _save_artifact(
    path: Path,
    *,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    video_ids: list[str],
    split: str,
    perturbation: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = embeddings.shape[0]
    perturbation_data = perturbation or {"name": "none"}
    torch.save(
        {
            "format_version": 2,
            "embeddings": embeddings,
            "label_ids": labels,
            "video_ids": video_ids,
            "sample_metadata": [
                {
                    "video_id": video_id,
                    "video_path": f"/videos/{video_id}.webm",
                    "frame_indices": [0, 1, 2, 3],
                    "sampling_strategy": "deterministic_center_clip",
                    "perturbation": perturbation_data,
                }
                for video_id in video_ids
            ],
            "frame_indices": torch.tensor([[0, 1, 2, 3]] * sample_count),
            "config": {
                "dataset_name": "toy",
                "split": split,
                "num_frames": 4,
                "sampling_strategy": "deterministic_center_clip",
                "image_size": 16,
            },
            "model_metadata": {
                "encoder_name": "tiny",
                "checkpoint": "tiny",
                "embedding_type": "mean",
            },
            "run_id": "toy-run",
            "artifact_metadata": {},
        },
        path,
    )
