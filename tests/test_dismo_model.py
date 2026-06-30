from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.dismo import DisMoEncoder
from src.models.registry import supported_video_encoders


def test_dismo_registry_is_available_without_loading_weights() -> None:
    assert "dismo" in supported_video_encoders()


def test_dismo_preprocess_resizes_and_scales_to_unit_range() -> None:
    frames = np.zeros((8, 12, 20, 3), dtype=np.uint8)
    frames[..., 0] = 255
    encoder = DisMoEncoder(
        model=TinyDisMo(), checkpoint="tiny-local", image_size=16)

    pixel_values = encoder.preprocess(frames)

    assert pixel_values.shape == (8, 16, 16, 3)
    assert pixel_values.dtype == torch.float32
    assert torch.isclose(pixel_values.max(), torch.tensor(1.0))
    assert torch.isclose(pixel_values.min(), torch.tensor(-1.0))


def test_dismo_preprocess_rejects_non_rgb_input() -> None:
    encoder = DisMoEncoder(
        model=TinyDisMo(), checkpoint="tiny-local", image_size=16)

    with pytest.raises(ValueError, match="RGB frames"):
        encoder.preprocess(np.zeros((8, 16, 16), dtype=np.uint8))


def test_dismo_encoder_time_pools_sliding_features() -> None:
    encoder = DisMoEncoder(model=TinyDisMo(feature_dim=5),
                           checkpoint="tiny-local", image_size=16)

    embeddings = encoder.encode(torch.randn(2, 8, 16, 16, 3), device="cpu")

    assert embeddings.shape == (2, 5)
    assert embeddings.requires_grad is False
    assert torch.isfinite(embeddings).all()
    assert encoder.metadata()["encoder_name"] == "dismo"
    assert encoder.input_spec()["value_range"] == [-1.0, 1.0]


def test_dismo_encoder_rejects_wrong_layout() -> None:
    encoder = DisMoEncoder(
        model=TinyDisMo(), checkpoint="tiny-local", image_size=16)

    with pytest.raises(ValueError, match="\\[B, T, H, W, C\\]"):
        encoder.encode(torch.randn(2, 3, 8, 16, 16), device="cpu")


def test_dismo_local_files_only_has_actionable_error() -> None:
    with pytest.raises(RuntimeError, match="network access"):
        DisMoEncoder.from_pretrained(device="cpu", local_files_only=True)


class TinyDisMo(torch.nn.Module):
    def __init__(self, feature_dim: int = 4) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = torch.nn.Linear(3, feature_dim)

    def forward_sliding(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: (B, T, H, W, C) -> (B, windows, feature_dim)
        pooled = clip.mean(dim=(2, 3))  # (B, T, C)
        return self.proj(pooled)
