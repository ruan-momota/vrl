"""Deterministic clip-level interventions used by all dataset adapters."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


PerturbationName = Literal[
    "none",
    "temporal_reverse",
    "temporal_shuffle",
    "freeze_tail",
    "single_frame",
    "grayscale",
    "center_occlusion",
    "color_transform",
    "spatial_blur",
]


@dataclass(frozen=True)
class VideoPerturbationConfig:
    name: PerturbationName = "none"
    seed: int = 0
    frame_index: int | None = None
    freeze_start_fraction: float = 0.5
    occlusion_size_fraction: float = 0.25
    occlusion_fill_value: int = 0
    color_strength: float = 0.0
    blur_kernel_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerturbationResult:
    frames: np.ndarray
    metadata: dict[str, Any]


class VideoPerturbation:
    def __init__(self, config: VideoPerturbationConfig) -> None:
        self.config = config

    def __call__(
        self,
        frames: np.ndarray,
        *,
        video_id: str | None = None,
    ) -> PerturbationResult:
        return apply_video_perturbation(frames, self.config, video_id=video_id)


def apply_video_perturbation(
    frames: np.ndarray,
    config: VideoPerturbationConfig,
    *,
    video_id: str | None = None,
) -> PerturbationResult:
    validate_rgb_frames(frames)

    if config.name == "none":
        return _result(frames.copy(), config, operation="identity")

    if config.name == "temporal_reverse":
        order = tuple(reversed(range(frames.shape[0])))
        return _result(
            frames[list(order)],
            config,
            operation="temporal_reverse",
            frame_order=list(order),
        )

    if config.name == "temporal_shuffle":
        order = _deterministic_permutation(
            frames.shape[0],
            seed=config.seed,
            video_id=video_id,
        )
        return _result(
            frames[list(order)],
            config,
            operation="temporal_shuffle",
            frame_order=list(order),
        )

    if config.name == "freeze_tail":
        return _freeze_tail(frames, config)

    if config.name == "single_frame":
        return _single_frame(frames, config)

    if config.name == "grayscale":
        return _grayscale(frames, config)

    if config.name == "center_occlusion":
        return _center_occlusion(frames, config)

    if config.name == "color_transform":
        return _color_transform(frames, config)

    if config.name == "spatial_blur":
        return _spatial_blur(frames, config)

    raise ValueError(f"Unsupported video perturbation: {config.name}")


def build_video_perturbation(
    config: VideoPerturbationConfig | None,
) -> VideoPerturbation | None:
    if config is None or config.name == "none":
        return None
    return VideoPerturbation(config)


def parse_perturbation_name(value: str) -> PerturbationName:
    names = {
        "none",
        "temporal_reverse",
        "temporal_shuffle",
        "freeze_tail",
        "single_frame",
        "grayscale",
        "center_occlusion",
        "color_transform",
        "spatial_blur",
    }
    if value not in names:
        raise ValueError(f"Unsupported video perturbation: {value}")
    return value  # type: ignore[return-value]


def validate_rgb_frames(frames: np.ndarray) -> None:
    if frames.ndim != 4:
        raise ValueError(f"frames must have shape [T, H, W, C], got {frames.shape}")
    if frames.shape[0] <= 0:
        raise ValueError("frames must contain at least one frame")
    if frames.shape[-1] != 3:
        raise ValueError(f"frames must be RGB with 3 channels, got {frames.shape}")


def _freeze_tail(frames: np.ndarray, config: VideoPerturbationConfig) -> PerturbationResult:
    if not 0.0 <= config.freeze_start_fraction <= 1.0:
        raise ValueError("freeze_start_fraction must be between 0.0 and 1.0")
    start = min(
        frames.shape[0] - 1,
        int(np.floor(frames.shape[0] * config.freeze_start_fraction)),
    )
    perturbed = frames.copy()
    perturbed[start:] = perturbed[start]
    return _result(
        perturbed,
        config,
        operation="freeze_tail",
        freeze_start_index=start,
    )


def _single_frame(frames: np.ndarray, config: VideoPerturbationConfig) -> PerturbationResult:
    frame_index = (
        frames.shape[0] // 2 if config.frame_index is None else int(config.frame_index)
    )
    if not 0 <= frame_index < frames.shape[0]:
        raise ValueError(
            f"frame_index must be in [0, {frames.shape[0] - 1}], got {frame_index}"
        )
    perturbed = np.repeat(frames[frame_index : frame_index + 1], frames.shape[0], axis=0)
    return _result(
        perturbed,
        config,
        operation="single_frame",
        source_frame_index=frame_index,
    )


def _grayscale(frames: np.ndarray, config: VideoPerturbationConfig) -> PerturbationResult:
    frames_float = frames.astype(np.float32)
    luminance = (
        frames_float[..., 0] * 0.299
        + frames_float[..., 1] * 0.587
        + frames_float[..., 2] * 0.114
    )
    grayscale = np.clip(np.rint(luminance), 0, 255).astype(frames.dtype)
    perturbed = np.repeat(grayscale[..., None], 3, axis=-1)
    return _result(perturbed, config, operation="grayscale")


def _center_occlusion(frames: np.ndarray, config: VideoPerturbationConfig) -> PerturbationResult:
    if not 0.0 < config.occlusion_size_fraction <= 1.0:
        raise ValueError("occlusion_size_fraction must be in (0.0, 1.0]")
    if not 0 <= config.occlusion_fill_value <= 255:
        raise ValueError("occlusion_fill_value must be in [0, 255]")

    height = frames.shape[1]
    width = frames.shape[2]
    box_h = max(1, int(round(height * config.occlusion_size_fraction)))
    box_w = max(1, int(round(width * config.occlusion_size_fraction)))
    y0 = (height - box_h) // 2
    x0 = (width - box_w) // 2
    perturbed = frames.copy()
    perturbed[:, y0 : y0 + box_h, x0 : x0 + box_w, :] = config.occlusion_fill_value
    return _result(
        perturbed,
        config,
        operation="center_occlusion",
        box={"x0": x0, "y0": y0, "width": box_w, "height": box_h},
    )


def _color_transform(frames: np.ndarray, config: VideoPerturbationConfig) -> PerturbationResult:
    """Apply one fixed color-affine transform to every frame in a clip.

    ``color_strength`` controls a blend between the original RGB signal and a
    luminance-preserving, channel-reweighted version.  The parameters are
    deliberately clip-global: the transform cannot create frame-to-frame
    flicker that would confound the appearance probe with a temporal signal.
    """
    if not 0.0 < config.color_strength <= 1.0:
        raise ValueError("color_strength must be in (0.0, 1.0]")

    strength = float(config.color_strength)
    frames_float = frames.astype(np.float32)
    luminance = (
        frames_float[..., 0] * 0.299
        + frames_float[..., 1] * 0.587
        + frames_float[..., 2] * 0.114
    )
    saturation_scale = 1.0 - strength
    desaturated = luminance[..., None] + saturation_scale * (
        frames_float - luminance[..., None]
    )
    channel_gains = np.array(
        (1.0 + 0.18 * strength, 1.0 - 0.06 * strength, 1.0 - 0.12 * strength),
        dtype=np.float32,
    )
    color_shifted = desaturated * channel_gains
    perturbed_float = (1.0 - strength) * frames_float + strength * color_shifted
    perturbed = _restore_frame_dtype(perturbed_float, frames.dtype)
    return _result(
        perturbed,
        config,
        operation="color_transform",
        constant_across_frames=True,
        color_strength=strength,
        saturation_scale=saturation_scale,
        channel_gains=[float(value) for value in channel_gains],
    )


def _spatial_blur(frames: np.ndarray, config: VideoPerturbationConfig) -> PerturbationResult:
    """Apply deterministic per-frame separable box blur without changing time."""
    kernel_size = int(config.blur_kernel_size)
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("blur_kernel_size must be an odd integer greater than or equal to 3")

    blurred = _box_blur(frames.astype(np.float32), kernel_size)
    perturbed = _restore_frame_dtype(blurred, frames.dtype)
    return _result(
        perturbed,
        config,
        operation="spatial_blur",
        blur_kernel_size=kernel_size,
        blur_type="separable_box",
        constant_across_frames=True,
    )


def _box_blur(frames: np.ndarray, kernel_size: int) -> np.ndarray:
    radius = kernel_size // 2
    horizontally_blurred = _box_blur_axis(frames, radius=radius, axis=2)
    return _box_blur_axis(horizontally_blurred, radius=radius, axis=1)


def _box_blur_axis(frames: np.ndarray, *, radius: int, axis: int) -> np.ndarray:
    kernel_size = radius * 2 + 1
    pad_width = [(0, 0)] * frames.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(frames, pad_width, mode="edge")
    cumulative = np.cumsum(padded, axis=axis, dtype=np.float32)
    zero_shape = list(cumulative.shape)
    zero_shape[axis] = 1
    cumulative = np.concatenate((np.zeros(zero_shape, dtype=np.float32), cumulative), axis=axis)
    upper = _slice_axis(cumulative, axis=axis, start=kernel_size, stop=None)
    lower = _slice_axis(cumulative, axis=axis, start=None, stop=-kernel_size)
    return (upper - lower) / float(kernel_size)


def _slice_axis(
    values: np.ndarray,
    *,
    axis: int,
    start: int | None,
    stop: int | None,
) -> np.ndarray:
    slices = [slice(None)] * values.ndim
    slices[axis] = slice(start, stop)
    return values[tuple(slices)]


def _restore_frame_dtype(values: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


def _deterministic_permutation(
    frame_count: int,
    *,
    seed: int,
    video_id: str | None,
) -> tuple[int, ...]:
    digest = hashlib.sha256(f"{seed}:{video_id or ''}".encode("utf-8")).digest()
    rng_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
    rng = np.random.default_rng(rng_seed)
    return tuple(int(index) for index in rng.permutation(frame_count))


def _result(
    frames: np.ndarray,
    config: VideoPerturbationConfig,
    *,
    operation: str,
    **metadata: Any,
) -> PerturbationResult:
    return PerturbationResult(
        frames=np.ascontiguousarray(frames),
        metadata={
            "name": config.name,
            "operation": operation,
            "config": config.to_dict(),
            **metadata,
        },
    )
