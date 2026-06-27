from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.data.ssv2 import SSV2DatasetAdapter
from src.artifacts import load_embedding_artifact
from src.pipeline.extract import run_extraction
from src.pipeline.run import RunConfig, RunPaths, write_manifest
from tests.test_video_io import _write_tiny_video


def _config(tmp_path: Path, *, perturbation: dict | None = None) -> RunConfig:
    return RunConfig(
        dataset_name="ssv2",
        subset_id="c50",
        split="validation",
        index_path="data/ssv2/index/validation.jsonl",
        model_name="videomae",
        model_checkpoint="tiny",
        num_frames=4,
        sampling_strategy="deterministic_center_clip",
        image_size=16,
        batch_size=1,
        num_workers=0,
        device="cpu",
        seed=7,
        deterministic=True,
        perturbation=perturbation or {"name": "none"},
        output_root=str(tmp_path / "runs"),
    )


def test_run_paths_are_deterministic_and_isolated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = RunPaths(config)

    assert paths.run_dir.name.startswith("ssv2-c50-videomae-4f-")
    assert paths.artifact_path() == paths.run_dir / "embeddings/validation/original.pt"

    perturbed = RunPaths(_config(tmp_path, perturbation={"name": "temporal_shuffle"}))
    assert perturbed.run_dir == paths.run_dir
    assert perturbed.artifact_path().name == "temporal-shuffle.pt"


def test_manifest_reuses_only_matching_config(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths = RunPaths(config)
    manifest = write_manifest(paths, command=["python", "-m", "example"])

    assert manifest["run_id"] == config.resolved_run_id
    assert paths.manifest_path.exists()
    assert json.loads(paths.manifest_path.read_text(encoding="utf-8"))["experiment_config"] == (
        config.experiment_definition()
    )
    assert json.loads(paths.manifest_path.read_text(encoding="utf-8"))["artifact_configs"] == {
        "embeddings/validation/original.pt": config.to_dict()
    }
    assert json.loads(paths.manifest_path.read_text(encoding="utf-8"))["artifact_commands"] == {
        "embeddings/validation/original.pt": ["python", "-m", "example"]
    }
    assert write_manifest(paths)["run_id"] == config.resolved_run_id

    perturbed_config = _config(tmp_path, perturbation={"name": "temporal_shuffle"})
    write_manifest(RunPaths(perturbed_config))
    artifact_configs = json.loads(paths.manifest_path.read_text(encoding="utf-8"))["artifact_configs"]
    assert set(artifact_configs) == {
        "embeddings/validation/original.pt",
        "embeddings/validation/perturbations/temporal-shuffle.pt",
    }

    changed = RunConfig(
        **{
            **_config(tmp_path).to_dict(),
            "model_checkpoint": "other-checkpoint",
        }
    )
    changed_paths = RunPaths(
        RunConfig(
            **{**changed.to_dict(), "run_id": config.resolved_run_id}
        )
    )
    with pytest.raises(FileExistsError, match="different config"):
        write_manifest(changed_paths)


def test_manifest_allows_split_specific_index_paths_with_one_run_id(tmp_path: Path) -> None:
    train_config = RunConfig(
        **{
            **_config(tmp_path).to_dict(),
            "split": "train",
            "index_path": "data/ssv2/index/train.jsonl",
            "run_id": "shared-run",
        }
    )
    heldout_config = RunConfig(
        **{
            **_config(tmp_path).to_dict(),
            "split": "validation",
            "index_path": "data/ssv2/index/validation.jsonl",
            "run_id": "shared-run",
        }
    )

    write_manifest(RunPaths(train_config))
    manifest = write_manifest(RunPaths(heldout_config))

    assert manifest["run_id"] == "shared-run"
    assert set(manifest["artifact_configs"]) == {
        "embeddings/train/original.pt",
        "embeddings/validation/original.pt",
    }


def test_generic_pipeline_writes_run_scoped_artifact(tmp_path: Path) -> None:
    video_path = _write_tiny_video(tmp_path / "sample.mp4", frame_count=5)
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "video_id": "sample",
                "video_path": str(video_path),
                "label_id": 3,
                "label_name": "Example",
                "split": "train",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = RunConfig(
        **{
            **_config(tmp_path).to_dict(),
            "index_path": str(index_path),
            "split": "train",
            "num_frames": 4,
        }
    )

    result = run_extraction(
        config,
        encoder=TinyEncoder(),
        dataset_adapter=SSV2DatasetAdapter(),
        show_progress=False,
    )

    artifact = load_embedding_artifact(result["artifact_path"])
    assert result["run_id"] == config.resolved_run_id
    assert Path(result["manifest_path"]).exists()
    assert artifact["run_id"] == config.resolved_run_id
    assert artifact["sample_metadata"][0]["source_dataset"] == "ssv2"
    assert artifact["sample_metadata"][0]["subset_id"] == "c50"
    assert artifact["embeddings"].shape == (1, 2)


class TinyEncoder:
    name = "tiny"

    def preprocess(self, frames) -> torch.Tensor:
        return torch.from_numpy(frames).permute(0, 3, 1, 2).float()

    def encode(self, pixel_values: torch.Tensor, *, device=None) -> torch.Tensor:
        mean = pixel_values.to(device).mean(dim=(1, 2, 3, 4))
        return torch.stack([mean, mean + 1], dim=1)

    def metadata(self) -> dict:
        return {"encoder_name": self.name, "checkpoint": "tiny"}
