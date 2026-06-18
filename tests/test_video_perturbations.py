from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.ssv2_dataset import SSV2ClipDataset, collate_video_batch
from src.video_perturbations import (
    VideoPerturbation,
    VideoPerturbationConfig,
    apply_video_perturbation,
)

from tests.test_video_io import _write_tiny_video


def test_temporal_reverse_records_frame_order() -> None:
    frames = _toy_frames(frame_count=4)

    result = apply_video_perturbation(
        frames,
        VideoPerturbationConfig(name="temporal_reverse"),
    )

    assert result.metadata["frame_order"] == [3, 2, 1, 0]
    assert np.array_equal(result.frames, frames[[3, 2, 1, 0]])
    assert result.frames.flags.c_contiguous


def test_temporal_shuffle_is_deterministic_per_video_id() -> None:
    frames = _toy_frames(frame_count=8)
    config = VideoPerturbationConfig(name="temporal_shuffle", seed=17)

    first = apply_video_perturbation(frames, config, video_id="sample-1")
    second = apply_video_perturbation(frames, config, video_id="sample-1")

    assert first.metadata["frame_order"] == second.metadata["frame_order"]
    assert np.array_equal(first.frames, second.frames)
    assert sorted(first.metadata["frame_order"]) == list(range(8))


def test_freeze_tail_repeats_tail_from_boundary() -> None:
    frames = _toy_frames(frame_count=5)

    result = apply_video_perturbation(
        frames,
        VideoPerturbationConfig(name="freeze_tail", freeze_start_fraction=0.4),
    )

    assert result.metadata["freeze_start_index"] == 2
    assert np.array_equal(result.frames[:2], frames[:2])
    assert np.array_equal(result.frames[2], frames[2])
    assert np.array_equal(result.frames[3], frames[2])
    assert np.array_equal(result.frames[4], frames[2])


def test_single_frame_repeats_selected_frame() -> None:
    frames = _toy_frames(frame_count=4)

    result = apply_video_perturbation(
        frames,
        VideoPerturbationConfig(name="single_frame", frame_index=1),
    )

    assert result.metadata["source_frame_index"] == 1
    assert np.array_equal(result.frames, np.repeat(frames[1:2], 4, axis=0))


def test_grayscale_copies_luminance_to_all_channels() -> None:
    frames = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    frames[..., 0] = 100
    frames[..., 1] = 50
    frames[..., 2] = 10

    result = apply_video_perturbation(
        frames,
        VideoPerturbationConfig(name="grayscale"),
    )

    assert np.array_equal(result.frames[..., 0], result.frames[..., 1])
    assert np.array_equal(result.frames[..., 1], result.frames[..., 2])
    assert result.frames.dtype == np.uint8


def test_center_occlusion_fills_center_box() -> None:
    frames = np.full((2, 6, 8, 3), 100, dtype=np.uint8)

    result = apply_video_perturbation(
        frames,
        VideoPerturbationConfig(
            name="center_occlusion",
            occlusion_size_fraction=0.5,
            occlusion_fill_value=7,
        ),
    )

    assert result.metadata["box"] == {"x0": 2, "y0": 1, "width": 4, "height": 3}
    assert np.all(result.frames[:, 1:4, 2:6, :] == 7)
    assert np.all(result.frames[:, 0, :, :] == 100)


def test_dataset_can_return_perturbed_and_original_clips(tmp_path: Path) -> None:
    video_path = _write_tiny_video(tmp_path / "sample.mp4", frame_count=5)
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        '{"video_id": "sample", "video_path": "'
        + str(video_path)
        + '", "label_id": 7, "label_name": "Example", "split": "train"}\n',
        encoding="utf-8",
    )
    dataset = SSV2ClipDataset(
        index_path,
        num_frames=4,
        transform=_to_channel_mean_tensor,
        perturbation=VideoPerturbation(
            VideoPerturbationConfig(name="single_frame", frame_index=1)
        ),
        include_original_clip=True,
    )

    item = dataset[0]
    batch = collate_video_batch([item])

    assert item["metadata"]["perturbation"]["name"] == "single_frame"
    assert item["metadata"]["perturbation"]["source_frame_index"] == 1
    assert item["pixel_values"].shape == item["original_pixel_values"].shape
    assert torch.equal(item["pixel_values"][0], item["pixel_values"][1])
    assert batch["original_pixel_values"].shape == batch["pixel_values"].shape
    assert batch["label_ids"].tolist() == [7]


def test_dataset_failure_includes_sample_and_perturbation_context(tmp_path: Path) -> None:
    missing_video_path = tmp_path / "missing.mp4"
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        '{"video_id": "missing-sample", "video_path": "'
        + str(missing_video_path)
        + '", "label_id": 7, "label_name": "Example", "split": "train"}\n',
        encoding="utf-8",
    )
    dataset = SSV2ClipDataset(
        index_path,
        num_frames=4,
        perturbation=VideoPerturbation(
            VideoPerturbationConfig(name="center_occlusion")
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        dataset[0]

    message = str(exc_info.value)
    assert "video_id=missing-sample" in message
    assert str(missing_video_path) in message
    assert "'name': 'center_occlusion'" in message


def _toy_frames(*, frame_count: int) -> np.ndarray:
    frames = np.zeros((frame_count, 3, 4, 3), dtype=np.uint8)
    for frame_index in range(frame_count):
        frames[frame_index, :, :, :] = frame_index * 10
    return frames


def _to_channel_mean_tensor(frames: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frames.mean(axis=(1, 2, 3)).astype(np.float32))
