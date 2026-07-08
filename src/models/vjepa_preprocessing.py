"""V-JEPA 2-specific frame preprocessing.

The official ``VJEPA2VideoProcessor`` in ``transformers`` requires
``torchvision``, which is not a dependency of this repo. Instead this mirrors
the manual PIL-based resize used for DisMo (``src/models/dismo.py``): resize
to a square crop, scale to ``[0, 1]``, then apply the standard ImageNet
mean/std normalization V-JEPA 2 was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class VJEPA2PreprocessConfig:
    image_size: int = 256
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


class VJEPA2ClipTransform:
    """Convert RGB frame clips to V-JEPA 2 pixel_values_videos tensors."""

    def __init__(
        self,
        *,
        image_size: int = 256,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        self.config = VJEPA2PreprocessConfig(
            image_size=image_size, mean=mean, std=std)
        self._mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"frames must have shape [T, H, W, C], got {frames.shape}")
        if frames.shape[-1] != 3:
            raise ValueError(f"frames must be RGB with 3 channels, got {frames.shape}")

        resized = np.stack(
            [_resize(frame, self.config.image_size) for frame in frames]
        )
        tensor = torch.from_numpy(resized).to(dtype=torch.float32)
        # [T, H, W, C] -> [T, C, H, W], scale to [0, 1], then normalize.
        tensor = tensor.permute(0, 3, 1, 2).div(255.0)
        tensor = tensor.sub(self._mean).div(self._std)
        return tensor.contiguous()


def _resize(frame: np.ndarray, image_size: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(resized)
