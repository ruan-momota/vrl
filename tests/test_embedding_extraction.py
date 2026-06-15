from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from src.embedding_extraction import (
    extract_embeddings,
    load_embedding_artifact,
    save_embedding_artifact,
)
from src.ssv2_dataset import SSV2ClipDataset

from tests.test_video_io import _write_tiny_video


def test_extract_embeddings_and_save_reload_artifact(tmp_path: Path) -> None:
    index_path = _write_index_with_two_videos(tmp_path)
    dataset = SSV2ClipDataset(
        index_path,
        num_frames=4,
        transform=_to_model_tensor,
    )
    model = TinyEmbeddingModel()

    result = extract_embeddings(
        model=model,
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


class TinyEmbeddingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        base = pixel_values.float().mean(dim=(1, 2, 3, 4)) + self.anchor
        hidden = torch.stack([base, base + 1, base + 2, base + 3], dim=1)
        return SimpleNamespace(last_hidden_state=hidden.unsqueeze(1).repeat(1, 2, 1))
