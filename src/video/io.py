"""Video decoding and deterministic frame sampling."""

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
    window_frames: int | None = None,
) -> tuple[int, ...]:
    """Return deterministic frame indices for a fixed-size clip.

    The MVP default keeps a contiguous center clip when the video is long enough.
    Short videos are padded by deterministic repeated sampling over the full video.

    ``window_frames``, when given, decouples "how many frames the model
    consumes" (``num_frames``) from "how much real time the clip spans"
    (``window_frames`` native frames, i.e. ``window_frames / fps`` seconds).
    Without it, a lower-``num_frames`` model implicitly samples a shorter
    contiguous slice of the source video than a higher-``num_frames`` model
    (stride is always 1), which means temporal perturbations like
    ``freeze_tail``/``temporal_shuffle`` are not operating on comparable
    real-world durations across models -- a fixed fraction of a 16-frame
    window and the same fraction of a 64-frame window cover very different
    amounts of actual motion. Passing the same ``window_frames`` to every
    model in a comparison normalizes the sampled duration: each model still
    gets its own native ``num_frames``, just spread evenly across the same
    center window instead of densely packed into a shorter one. Defaults to
    ``None`` (identical to the pre-existing behavior) so no existing config
    or already-extracted artifact is affected.
    """
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if window_frames is not None and window_frames <= 0:
        raise ValueError(f"window_frames must be positive, got {window_frames}")

    if strategy == "deterministic_center_clip":
        if window_frames is None:
            if frame_count >= num_frames:
                start = (frame_count - num_frames) // 2
                return tuple(range(start, start + num_frames))
            return _uniform_repeated_indices(frame_count, num_frames)

        effective_window = min(window_frames, frame_count)
        if effective_window >= num_frames:
            start = (frame_count - effective_window) // 2
            return _window_spaced_indices(start, effective_window, num_frames)
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
    window_frames: int | None = None,
) -> SampledClip:
    decoded = read_video_frames(video_path, video_id=video_id)
    indices = sample_frame_indices(
        decoded.metadata.frames_decoded,
        num_frames,
        strategy=sampling_strategy,
        window_frames=window_frames,
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


def _window_spaced_indices(start: int, window_frames: int, num_frames: int) -> tuple[int, ...]:
    """Evenly space ``num_frames`` bin-center indices across a native window.

    Degenerates to a contiguous ``range(start, start + num_frames)`` when
    ``window_frames == num_frames`` (i.e. when a model's own num_frames is
    used as the reference window), matching the no-``window_frames`` path
    exactly.
    """
    positions = (np.arange(num_frames, dtype=np.float64) + 0.5) * (
        window_frames / num_frames
    )
    offsets = np.floor(positions).astype(np.int64)
    offsets = np.clip(offsets, 0, window_frames - 1)
    return tuple(int(start + offset) for offset in offsets)


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
