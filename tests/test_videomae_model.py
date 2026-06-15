from __future__ import annotations

import pytest
import torch

from src.videomae_model import (
    build_videomae_metadata,
    forward_videomae_embeddings,
    pool_videomae_hidden_state,
    resolve_device,
)


def test_pool_videomae_hidden_state_mean_pooling() -> None:
    hidden_state = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)

    embeddings = pool_videomae_hidden_state(hidden_state)

    assert embeddings.shape == (2, 4)
    assert torch.equal(embeddings, hidden_state.mean(dim=1))


def test_forward_videomae_embeddings_with_tiny_model() -> None:
    model = _build_tiny_videomae_model()
    model.train()
    pixel_values = torch.randn(2, 4, 3, 16, 16)

    result = forward_videomae_embeddings(model, pixel_values, device="cpu")

    assert result.embeddings.shape == (2, 16)
    assert result.embeddings.requires_grad is False
    assert result.embedding_type == "last_hidden_state_mean_pool"
    assert result.last_hidden_state_shape == (2, 8, 16)
    assert torch.isfinite(result.embeddings).all()
    assert model.training is False


def test_forward_videomae_embeddings_rejects_wrong_layout() -> None:
    model = _build_tiny_videomae_model()

    with pytest.raises(ValueError, match=r"\[B, T, C, H, W\]"):
        forward_videomae_embeddings(model, torch.randn(4, 3, 16, 16), device="cpu")


def test_videomae_metadata_reads_config_fields() -> None:
    model = _build_tiny_videomae_model()

    metadata = build_videomae_metadata(
        model,
        checkpoint="tiny-local",
        device="cpu",
        embedding_type="last_hidden_state_mean_pool",
    )

    assert metadata.checkpoint == "tiny-local"
    assert metadata.device == "cpu"
    assert metadata.model_type == "videomae"
    assert metadata.hidden_size == 16
    assert metadata.num_frames == 4
    assert metadata.image_size == 16
    assert metadata.patch_size == 8
    assert metadata.tubelet_size == 2
    assert metadata.to_dict()["embedding_type"] == "last_hidden_state_mean_pool"


def test_resolve_device_auto_and_cpu() -> None:
    assert resolve_device("auto").type in {"cpu", "cuda"}
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_rejects_unavailable_cuda() -> None:
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment")

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def _build_tiny_videomae_model() -> torch.nn.Module:
    pytest.importorskip("transformers")
    from transformers import VideoMAEConfig, VideoMAEModel

    config = VideoMAEConfig(
        image_size=16,
        patch_size=8,
        num_frames=4,
        tubelet_size=2,
        num_channels=3,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return VideoMAEModel(config)
