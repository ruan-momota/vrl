from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.vjepa_preprocessing import VJEPA2ClipTransform


def test_vjepa_preprocess_resizes_and_normalizes() -> None:
    frames = np.zeros((8, 12, 20, 3), dtype=np.uint8)
    transform = VJEPA2ClipTransform(image_size=16)

    pixel_values = transform(frames)

    assert pixel_values.shape == (8, 3, 16, 16)
    assert pixel_values.dtype == torch.float32


def test_vjepa_preprocess_zero_frame_matches_negative_mean_over_std() -> None:
    frames = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    transform = VJEPA2ClipTransform(
        image_size=4, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    pixel_values = transform(frames)

    # All-black input -> (0/255 - mean) / std == -1.0 for every channel.
    assert torch.allclose(pixel_values, torch.full_like(pixel_values, -1.0))


def test_vjepa_preprocess_rejects_non_rgb_input() -> None:
    transform = VJEPA2ClipTransform(image_size=16)

    with pytest.raises(ValueError, match="RGB"):
        transform(np.zeros((8, 16, 16, 4), dtype=np.uint8))


def test_vjepa_preprocess_rejects_wrong_ndim() -> None:
    transform = VJEPA2ClipTransform(image_size=16)

    with pytest.raises(ValueError, match=r"\[T, H, W, C\]"):
        transform(np.zeros((16, 16, 3), dtype=np.uint8))
