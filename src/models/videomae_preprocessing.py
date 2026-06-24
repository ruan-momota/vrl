"""VideoMAE-specific frame preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class VideoMAEPreprocessConfig:
    image_size: int = 224
    processor_checkpoint: str | None = None
    local_files_only: bool = False


class VideoMAEClipTransform:
    """Convert RGB frame clips to VideoMAE pixel_values tensors."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        processor_checkpoint: str | None = None,
        local_files_only: bool = False,
        image_processor: Any | None = None,
    ) -> None:
        self.config = VideoMAEPreprocessConfig(
            image_size=image_size,
            processor_checkpoint=processor_checkpoint,
            local_files_only=local_files_only,
        )
        self.image_processor = image_processor or build_videomae_image_processor(
            image_size=image_size,
            processor_checkpoint=processor_checkpoint,
            local_files_only=local_files_only,
        )

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"frames must have shape [T, H, W, C], got {frames.shape}")
        if frames.shape[-1] != 3:
            raise ValueError(f"frames must be RGB with 3 channels, got {frames.shape}")

        batch = self.image_processor(
            list(frames),
            return_tensors="pt",
            input_data_format="channels_last",
            data_format="channels_first",
        )
        pixel_values = batch["pixel_values"]
        if pixel_values.ndim != 5 or pixel_values.shape[0] != 1:
            raise ValueError(
                "VideoMAE image processor returned unexpected pixel_values shape: "
                f"{tuple(pixel_values.shape)}"
            )
        return pixel_values.squeeze(0).to(dtype=torch.float32).contiguous()


def build_videomae_image_processor(
    *,
    image_size: int,
    processor_checkpoint: str | None = None,
    local_files_only: bool = False,
) -> Any:
    size = {"shortest_edge": image_size}
    crop_size = {"height": image_size, "width": image_size}
    try:
        from transformers import AutoImageProcessor, VideoMAEImageProcessor
    except ImportError as error:  # pragma: no cover - dependency is declared.
        raise RuntimeError(
            "transformers with vision dependencies is required for VideoMAE preprocessing"
        ) from error

    try:
        if processor_checkpoint is not None:
            return AutoImageProcessor.from_pretrained(
                processor_checkpoint,
                size=size,
                crop_size=crop_size,
                local_files_only=local_files_only,
            )
        return VideoMAEImageProcessor(size=size, crop_size=crop_size)
    except ImportError as error:
        raise RuntimeError(
            "VideoMAE preprocessing requires Pillow; add the pillow dependency and "
            "sync the environment."
        ) from error
