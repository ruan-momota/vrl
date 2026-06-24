from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.indexed_dataset import IndexedVideoDataset, collate_video_batch
from src.models.videomae_preprocessing import VideoMAEClipTransform

from tests.test_video_io import _write_tiny_video


def test_videomae_clip_transform_outputs_model_layout() -> None:
    pytest.importorskip("PIL")
    frames = np.zeros((4, 18, 20, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 255

    transform = VideoMAEClipTransform(image_size=16)
    pixel_values = transform(frames)

    assert pixel_values.shape == (4, 3, 16, 16)
    assert pixel_values.dtype == torch.float32
    assert torch.isfinite(pixel_values).all()


def test_indexed_video_dataset_collates_videomae_batch(tmp_path: Path) -> None:
    pytest.importorskip("PIL")
    video_path = _write_tiny_video(tmp_path / "sample.mp4", frame_count=6)
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        '{"video_id": "sample", "video_path": "'
        + str(video_path)
        + '", "label_id": 7, "label_name": "Example", "split": "train"}\n',
        encoding="utf-8",
    )

    dataset = IndexedVideoDataset(
        index_path,
        num_frames=4,
        transform=VideoMAEClipTransform(image_size=16),
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=collate_video_batch,
    )
    batch = next(iter(loader))

    assert batch["pixel_values"].shape == (1, 4, 3, 16, 16)
    assert batch["pixel_values"].dtype == torch.float32
    assert batch["label_ids"].tolist() == [7]
    assert batch["video_ids"] == ["sample"]
    assert batch["frame_indices"].shape == (1, 4)
    assert batch["metadata"][0]["source_dataset"] is None
