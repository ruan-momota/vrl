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
]


@dataclass(frozen=True)
class VideoPerturbationConfig:
    name: PerturbationName = "none"
    seed: int = 0
    frame_index: int | None = None
    freeze_start_fraction: float = 0.5
    occlusion_size_fraction: float = 0.25
    occlusion_fill_value: int = 0

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
