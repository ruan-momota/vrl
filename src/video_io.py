from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


FrameSamplingStrategy = Literal[
    "deterministic_center_clip",
    "deterministic_uniform",
]


class VideoReadError(RuntimeError):
    """Raised when a video cannot be decoded into at least one RGB frame."""

    def __init__(
        self,
        message: str,
        *,
        video_path: str | Path,
        video_id: str | None = None,
    ) -> None:
        self.video_path = str(video_path)
        self.video_id = video_id
        prefix = f"video_id={video_id} " if video_id is not None else ""
        super().__init__(f"{prefix}path={self.video_path}: {message}")


@dataclass(frozen=True)
class VideoMetadata:
    video_path: str
    frames_decoded: int
    height: int
    width: int
    video_id: str | None = None
    fps: float | None = None
    duration_seconds: float | None = None
    stream_frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecodedVideo:
    frames: np.ndarray
    metadata: VideoMetadata


@dataclass(frozen=True)
class SampledClip:
    frames: np.ndarray
    frame_indices: tuple[int, ...]
    metadata: VideoMetadata
    sampling_strategy: str


def read_video_frames(video_path: str | Path, *, video_id: str | None = None) -> DecodedVideo:
    """Decode a video into RGB uint8 frames with shape [F, H, W, C]."""
    path = Path(video_path)
    try:
        import av
    except ImportError as error:  # pragma: no cover - dependency is declared.
        raise RuntimeError("PyAV is not installed; cannot decode video files") from error

    try:
        container = av.open(str(path))  # open a video file
    except Exception as error:  # noqa: BLE001 - wrapped with sample context.
        raise VideoReadError(
            f"failed to open video ({error})", video_path=path, video_id=video_id
        ) from error

    try:
        if not container.streams.video:
            raise VideoReadError(
                "no video stream found", video_path=path, video_id=video_id
            )

        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frame_list = [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]
        if not frame_list:
            raise VideoReadError(
                "decoded zero frames", video_path=path, video_id=video_id
            )

        frames = np.ascontiguousarray(np.stack(frame_list, axis=0))
        metadata = VideoMetadata(
            video_id=video_id,
            video_path=str(path),
            frames_decoded=int(frames.shape[0]),  # number of frames
            height=int(frames.shape[1]),
            width=int(frames.shape[2]),
            fps=_stream_fps(stream),
            duration_seconds=_duration_seconds(container, stream),
            stream_frames=int(stream.frames) if stream.frames else None,
        )
        return DecodedVideo(frames=frames, metadata=metadata)
    finally:
        container.close()


def sample_frame_indices(
    frame_count: int,
    num_frames: int,
    *,
    strategy: FrameSamplingStrategy = "deterministic_center_clip",
) -> tuple[int, ...]:
    """Return deterministic frame indices for a fixed-size clip.

    The MVP default keeps a contiguous center clip when the video is long enough.
    Short videos are padded by deterministic repeated sampling over the full video.
    """
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")

    if strategy == "deterministic_center_clip":
        if frame_count >= num_frames:
            start = (frame_count - num_frames) // 2
            return tuple(range(start, start + num_frames))
        return _uniform_repeated_indices(frame_count, num_frames)

    if strategy == "deterministic_uniform":
        return _uniform_repeated_indices(frame_count, num_frames)

    raise ValueError(f"Unsupported frame sampling strategy: {strategy}")


def read_sampled_clip(
    video_path: str | Path,
    *,
    num_frames: int,
    sampling_strategy: FrameSamplingStrategy = "deterministic_center_clip",
    video_id: str | None = None,
) -> SampledClip:
    decoded = read_video_frames(video_path, video_id=video_id)
    indices = sample_frame_indices(
        decoded.metadata.frames_decoded,
        num_frames,
        strategy=sampling_strategy,
    )
    clip = np.ascontiguousarray(decoded.frames[list(indices)])
    return SampledClip(
        frames=clip,
        frame_indices=indices,
        metadata=decoded.metadata,
        sampling_strategy=sampling_strategy,
    )


def _uniform_repeated_indices(frame_count: int, num_frames: int) -> tuple[int, ...]:
    positions = (np.arange(num_frames, dtype=np.float64) + 0.5) * (
        frame_count / num_frames
    )
    indices = np.floor(positions).astype(np.int64)
    indices = np.clip(indices, 0, frame_count - 1)
    return tuple(int(index) for index in indices)


def _stream_fps(stream: Any) -> float | None:
    if stream.average_rate is None:
        return None
    return float(stream.average_rate)


def _duration_seconds(container: Any, stream: Any) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None:
        return float(container.duration / 1_000_000)
    return None
