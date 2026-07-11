from __future__ import annotations

from pathlib import Path

import torch

from src.artifacts import load_embedding_artifact, save_embedding_artifact
from src.data.indexed_dataset import IndexedVideoDataset
from src.pipeline.extraction import extract_embeddings

from tests.test_video_io import _write_tiny_video


def test_extract_embeddings_and_save_reload_artifact(tmp_path: Path) -> None:
    index_path = _write_index_with_two_videos(tmp_path)
    dataset = IndexedVideoDataset(
        index_path,
        num_frames=4,
        transform=_to_model_tensor,
    )
    result = extract_embeddings(
        encoder=TinyEncoder(),
        dataset=dataset,
        batch_size=2,
        num_workers=0,
        device="cpu",
        show_progress=False,
    )

    assert result.embeddings.shape == (2, 4)
    assert result.label_ids is not None
    assert result.label_ids.tolist() == [3, 4]
    assert result.video_ids == ["sample-1", "sample-2"]
    assert result.frame_indices.shape == (2, 4)
    assert result.summary.successful_samples == 2
    assert result.summary.failed_samples == 0

    output_path = tmp_path / "embeddings.pt"
    save_embedding_artifact(
        result,
        output_path,
        config_snapshot={
            "dataset_name": "ssv2",
            "split": "debug_train",
            "num_frames": 4,
        },
        model_metadata={"checkpoint": "tiny", "embedding_type": "last_hidden_state_mean_pool"},
    )
    artifact = load_embedding_artifact(output_path)

    assert torch.equal(artifact["embeddings"], result.embeddings)
    assert artifact["label_ids"].tolist() == [3, 4]
    assert artifact["video_ids"] == ["sample-1", "sample-2"]
    assert artifact["summary"]["embeddings_shape"] == [2, 4]
    assert artifact["summary"]["output_path"] == str(output_path)
    assert artifact["config"]["split"] == "debug_train"


def test_load_embedding_artifact_accepts_legacy_v1_schema(tmp_path: Path) -> None:
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "format_version": 1,
            "embeddings": torch.tensor([[1.0, 0.0]]),
            "label_ids": torch.tensor([3]),
            "video_ids": ["legacy-sample"],
            "sample_metadata": [{"video_id": "legacy-sample"}],
            "frame_indices": torch.tensor([[0, 1, 2, 3]]),
            "config": {"num_frames": 4, "sampling_strategy": "deterministic_center_clip"},
            "model_metadata": {"checkpoint": "legacy-videomae"},
            "summary": {"embeddings_shape": [1, 2]},
        },
        path,
    )

    artifact = load_embedding_artifact(path)

    assert artifact["format_version"] == 1
    assert artifact["video_ids"] == ["legacy-sample"]


def _write_index_with_two_videos(tmp_path: Path) -> Path:
    first = _write_tiny_video(tmp_path / "sample-1.mp4", frame_count=5)
    second = _write_tiny_video(tmp_path / "sample-2.mp4", frame_count=6)
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        "\n".join(
            [
                '{"video_id": "sample-1", "video_path": "'
                + str(first)
                + '", "label_id": 3, "label_name": "First", "split": "train"}',
                '{"video_id": "sample-2", "video_path": "'
                + str(second)
                + '", "label_id": 4, "label_name": "Second", "split": "train"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return index_path


def _to_model_tensor(frames) -> torch.Tensor:
    return torch.from_numpy(frames).permute(0, 3, 1, 2).float().contiguous()


class TinyEncoder:
    name = "tiny"

    def encode(self, pixel_values: torch.Tensor, *, device: str | torch.device | None = None) -> torch.Tensor:
        base = pixel_values.to(device).float().mean(dim=(1, 2, 3, 4))
        return torch.stack([base, base + 1, base + 2, base + 3], dim=1)
