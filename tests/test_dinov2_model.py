from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import src.models.registry as registry
from src.models.dinov2 import DINOv2ClipTransform, DINOv2Encoder, mean_pool_frame_embeddings


def test_dinov2_registry_is_available_without_loading_weights() -> None:
    assert "dinov2" in registry.supported_video_encoders()


def test_dinov2_registry_dispatches_to_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    class Factory:
        @staticmethod
        def from_pretrained(**kwargs: object) -> str:
            assert kwargs["checkpoint"] == "facebook/dinov2-base"
            return "loaded-dinov2"

    monkeypatch.setattr(registry, "DINOv2Encoder", Factory)

    assert registry.load_video_encoder("dinov2", checkpoint="facebook/dinov2-base") == "loaded-dinov2"


def test_dinov2_transform_outputs_framewise_tensor() -> None:
    frames = np.zeros((16, 12, 20, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 255
    transform = DINOv2ClipTransform(image_size=14, num_frames=16)

    pixel_values = transform(frames)

    assert pixel_values.shape == (16, 3, 14, 14)
    assert pixel_values.dtype == torch.float32
    assert torch.isfinite(pixel_values).all()


def test_dinov2_transform_rejects_wrong_frame_count() -> None:
    transform = DINOv2ClipTransform(image_size=16, num_frames=16)

    with pytest.raises(ValueError, match="unexpected frame count"):
        transform(np.zeros((8, 12, 20, 3), dtype=np.uint8))


def test_mean_pool_frame_embeddings_is_order_invariant_for_frame_shuffle() -> None:
    frame_embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [8.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [0.0, 4.0],
            [0.0, 8.0],
        ]
    )
    shuffled = torch.cat([frame_embeddings[[2, 0, 3, 1]], frame_embeddings[[6, 4, 7, 5]]])

    original = mean_pool_frame_embeddings(frame_embeddings, batch_size=2, frame_count=4)
    perturbed = mean_pool_frame_embeddings(shuffled, batch_size=2, frame_count=4)

    assert torch.equal(original, perturbed)


def test_dinov2_encoder_returns_clip_embeddings() -> None:
    encoder = DINOv2Encoder(
        model=TinyDINOv2(),
        checkpoint="tiny-local",
        transform=DINOv2ClipTransform(image_size=8, num_frames=4),
        device="cpu",
    )
    pixel_values = torch.randn(2, 4, 3, 8, 8)

    embeddings = encoder.encode(pixel_values, device="cpu")

    assert embeddings.shape == (2, 3)
    assert embeddings.requires_grad is False
    assert torch.isfinite(embeddings).all()
    metadata = encoder.metadata()
    assert metadata["encoder_name"] == "dinov2"
    assert metadata["feature_source"] == "last_hidden_state_cls_token"
    assert metadata["frame_aggregation"] == "mean"
    assert metadata["input_spec"]["input_layout"] == "[B, T, C, H, W]"
    json.dumps(metadata)


class TinyDINOv2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(3, 3)
        self.config = SimpleNamespace(
            model_type="dinov2",
            hidden_size=3,
            image_size=8,
            patch_size=4,
            num_hidden_layers=1,
        )

    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        frame_features = pixel_values.mean(dim=(2, 3))
        cls_token = self.proj(frame_features).unsqueeze(1)
        patch_token = torch.zeros_like(cls_token)
        return SimpleNamespace(last_hidden_state=torch.cat([cls_token, patch_token], dim=1))
