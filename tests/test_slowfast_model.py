from __future__ import annotations

import numpy as np
import pytest
import torch

from src.models.registry import supported_video_encoders
from src.models.slowfast import SlowFastClipTransform, SlowFastEncoder


def test_slowfast_registry_is_available_without_loading_weights() -> None:
    assert "slowfast" in supported_video_encoders()


def test_slowfast_transform_outputs_fast_pathway_tensor() -> None:
    frames = np.zeros((32, 12, 20, 3), dtype=np.uint8)
    frames[:, :, :, 0] = 255
    transform = SlowFastClipTransform(image_size=16, num_frames=32, slowfast_alpha=4)

    pixel_values = transform(frames)

    assert pixel_values.shape == (3, 32, 16, 16)
    assert pixel_values.dtype == torch.float32
    assert torch.isfinite(pixel_values).all()


def test_slowfast_transform_rejects_wrong_frame_count() -> None:
    transform = SlowFastClipTransform(image_size=16, num_frames=32)

    with pytest.raises(ValueError, match="unexpected frame count"):
        transform(np.zeros((16, 12, 20, 3), dtype=np.uint8))


def test_slowfast_encoder_returns_pre_classifier_features() -> None:
    model = TinySlowFast()
    encoder = SlowFastEncoder(
        model=model,
        checkpoint="tiny-local",
        transform=SlowFastClipTransform(image_size=16, num_frames=32, slowfast_alpha=4),
    )

    embeddings = encoder.encode(torch.randn(2, 3, 32, 16, 16), device="cpu")

    assert embeddings.shape == (2, 3)
    assert embeddings.requires_grad is False
    assert torch.isfinite(embeddings).all()
    assert encoder.metadata()["encoder_name"] == "slowfast"
    assert encoder.input_spec()["pathways"]["alpha"] == 4


def test_slowfast_torch_hub_local_files_only_has_actionable_error() -> None:
    with pytest.raises(RuntimeError, match="cached repository"):
        SlowFastEncoder.from_pretrained(
            "facebookresearch/pytorchvideo:slowfast_r50",
            device="cpu",
            local_files_only=True,
        )


class TinySlowFast(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(3, 2)

    def forward(self, pathways: list[torch.Tensor]) -> torch.Tensor:
        _slow, fast = pathways
        features = fast.mean(dim=(2, 3, 4))
        return self.classifier(features)
