from __future__ import annotations

import pytest
import torch

from src.models.registry import supported_video_encoders
from src.models.vjepa import VJEPA2Encoder
from src.models.vjepa_model import (
    build_vjepa_metadata,
    forward_vjepa_embeddings,
    pool_vjepa_hidden_state,
    resolve_device,
)


def test_vjepa_is_registered() -> None:
    assert "vjepa" in supported_video_encoders()


def test_pool_vjepa_hidden_state_mean_pooling() -> None:
    hidden_state = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    embeddings = pool_vjepa_hidden_state(hidden_state)

    assert embeddings.shape == (2, 4)
    assert torch.equal(embeddings, hidden_state.mean(dim=1))


def test_forward_vjepa_embeddings_with_tiny_model() -> None:
    model = _build_tiny_vjepa_model()
    model.train()
    pixel_values = torch.randn(2, 4, 3, 16, 16)

    result = forward_vjepa_embeddings(model, pixel_values, device="cpu")

    assert result.embeddings.shape == (2, 24)
    assert result.embeddings.requires_grad is False
    assert result.embedding_type == "last_hidden_state_mean_pool"
    assert torch.isfinite(result.embeddings).all()
    assert model.training is False


def test_forward_vjepa_embeddings_rejects_wrong_layout() -> None:
    model = _build_tiny_vjepa_model()

    with pytest.raises(ValueError, match=r"\[B, T, C, H, W\]"):
        forward_vjepa_embeddings(model, torch.randn(4, 3, 16, 16), device="cpu")


def test_vjepa_metadata_reads_config_fields() -> None:
    model = _build_tiny_vjepa_model()

    metadata = build_vjepa_metadata(
        model,
        checkpoint="tiny-local",
        device="cpu",
        embedding_type="last_hidden_state_mean_pool",
    )

    assert metadata.checkpoint == "tiny-local"
    assert metadata.device == "cpu"
    assert metadata.model_type == "vjepa2"
    assert metadata.hidden_size == 24
    assert metadata.frames_per_clip == 4
    assert metadata.crop_size == 16
    assert metadata.patch_size == 8
    assert metadata.tubelet_size == 2
    assert metadata.to_dict()["embedding_type"] == "last_hidden_state_mean_pool"


def test_vjepa_encoder_adapter_wraps_existing_model() -> None:
    model = _build_tiny_vjepa_model()
    metadata = build_vjepa_metadata(
        model,
        checkpoint="tiny-local",
        device="cpu",
        embedding_type="last_hidden_state_mean_pool",
    )
    encoder = VJEPA2Encoder(
        model=model,
        model_metadata=metadata,
        transform=lambda frames: torch.from_numpy(frames).permute(0, 3, 1, 2).float(),
    )

    embeddings = encoder.encode(torch.randn(2, 4, 3, 16, 16), device="cpu")

    assert embeddings.shape == (2, 24)
    assert encoder.metadata()["encoder_name"] == "vjepa"
    assert encoder.input_spec()["input_layout"] == "[B, T, C, H, W]"


def test_resolve_device_auto_and_cpu() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda"}
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def _build_tiny_vjepa_model() -> torch.nn.Module:
    pytest.importorskip("transformers")
    from transformers import VJEPA2Config, VJEPA2Model

    config = VJEPA2Config(
        crop_size=16,
        patch_size=8,
        frames_per_clip=4,
        tubelet_size=2,
        hidden_size=24,
        num_hidden_layers=1,
        num_attention_heads=2,
        in_chans=3,
        pred_hidden_size=16,
        pred_num_hidden_layers=1,
        pred_num_attention_heads=2,
    )
    return VJEPA2Model(config)
