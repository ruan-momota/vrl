from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.video_io import (
    read_sampled_clip,
    read_video_frames,
    sample_frame_indices,
)


def test_sample_frame_indices_center_clip_for_long_video() -> None:
    assert sample_frame_indices(20, 6) == (7, 8, 9, 10, 11, 12)


def test_sample_frame_indices_repeats_short_video() -> None:
    assert sample_frame_indices(3, 8) == (0, 0, 0, 1, 1, 2, 2, 2)


def test_sample_frame_indices_uniform_strategy() -> None:
    assert sample_frame_indices(20, 5, strategy="deterministic_uniform") == (
        2,
        6,
        10,
        14,
        18,
    )


def test_read_video_frames_and_sample_clip(tmp_path: Path) -> None:
    video_path = _write_tiny_video(tmp_path / "sample.mp4", frame_count=5)

    decoded = read_video_frames(video_path, video_id="sample")
    assert decoded.frames.shape == (5, 12, 16, 3)
    assert decoded.frames.dtype == np.uint8
    assert decoded.metadata.video_id == "sample"
    assert decoded.metadata.frames_decoded == 5

    clip = read_sampled_clip(video_path, video_id="sample", num_frames=8)
    assert clip.frames.shape == (8, 12, 16, 3)
    assert clip.frame_indices == (0, 0, 1, 2, 2, 3, 4, 4)


def _write_tiny_video(path: Path, *, frame_count: int) -> Path:
    av = pytest.importorskip("av")
    try:
        container = av.open(str(path), mode="w")
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = 16
        stream.height = 12
        stream.pix_fmt = "yuv420p"
        for frame_index in range(frame_count):
            array = np.full((12, 16, 3), frame_index * 25, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception as error:  # noqa: BLE001 - codec support depends on FFmpeg build.
        pytest.skip(f"could not create tiny video fixture: {error}")
    return path
