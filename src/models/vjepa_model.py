"""Low-level V-JEPA 2 loading and embedding helpers."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Literal

import torch


EmbeddingType = Literal["last_hidden_state_mean_pool"]


@dataclass(frozen=True)
class VJEPA2ModelMetadata:
    checkpoint: str
    device: str
    embedding_type: EmbeddingType
    model_type: str | None
    hidden_size: int | None
    frames_per_clip: int | None
    crop_size: int | None
    patch_size: int | None
    tubelet_size: int | None
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VJEPA2ForwardResult:
    embeddings: torch.Tensor
    last_hidden_state_shape: tuple[int, ...]
    embedding_type: EmbeddingType


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def load_vjepa_model(
    checkpoint: str,
    *,
    device: str = "auto",
    local_files_only: bool = False,
    revision: str | None = None,
) -> tuple[torch.nn.Module, VJEPA2ModelMetadata]:
    """Load a bare V-JEPA 2 encoder for embedding extraction."""
    try:
        from transformers import VJEPA2Model
    except ImportError as error:  # pragma: no cover - dependency is declared.
        raise RuntimeError("transformers is required to load V-JEPA 2 models") from error

    resolved_device = resolve_device(device)
    load_options: dict[str, Any] = {"local_files_only": local_files_only}
    if revision is not None:
        load_options["revision"] = revision
    model = VJEPA2Model.from_pretrained(checkpoint, **load_options)
    model.to(resolved_device)
    model.eval()
    metadata = build_vjepa_metadata(
        model,
        checkpoint=checkpoint,
        device=str(resolved_device),
        embedding_type="last_hidden_state_mean_pool",
        revision=revision or _config_value(getattr(model, "config", None), "_commit_hash"),
    )
    if resolved_device.type == "cuda" and os.environ.get("VRL_FORCE_FP32") != "1":
        model = torch.compile(model)
    return model, metadata


def build_vjepa_metadata(
    model: torch.nn.Module,
    *,
    checkpoint: str,
    device: str,
    embedding_type: EmbeddingType,
    revision: str | None = None,
) -> VJEPA2ModelMetadata:
    config = getattr(model, "config", None)
    return VJEPA2ModelMetadata(
        checkpoint=checkpoint,
        device=device,
        embedding_type=embedding_type,
        model_type=_config_value(config, "model_type"),
        hidden_size=_config_value(config, "hidden_size"),
        frames_per_clip=_config_value(config, "frames_per_clip"),
        crop_size=_config_value(config, "crop_size"),
        patch_size=_config_value(config, "patch_size"),
        tubelet_size=_config_value(config, "tubelet_size"),
        revision=revision or _config_value(config, "_commit_hash"),
    )


@torch.inference_mode()
def forward_vjepa_embeddings(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    *,
    embedding_type: EmbeddingType = "last_hidden_state_mean_pool",
    device: str | torch.device | None = None,
) -> VJEPA2ForwardResult:
    """Run V-JEPA 2 inference and return clip-level embeddings.

    The predictor head (``skip_predictor=False``) is only needed for the
    masked-prediction pretraining objective, not for embedding extraction, so
    it is always skipped here.
    """
    if pixel_values.ndim != 5:
        raise ValueError(
            "pixel_values must have shape [B, T, C, H, W], "
            f"got {tuple(pixel_values.shape)}"
        )

    resolved_device = torch.device(device) if device is not None else model_device(model)
    pixel_values = pixel_values.to(resolved_device)
    model.eval()
    outputs = model(pixel_values_videos=pixel_values, skip_predictor=True)
    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is None:
        raise RuntimeError("V-JEPA 2 output did not include last_hidden_state")

    embeddings = pool_vjepa_hidden_state(
        last_hidden_state,
        embedding_type=embedding_type,
    )
    if not torch.isfinite(embeddings).all():
        raise RuntimeError("V-JEPA 2 embeddings contain NaN or Inf values")

    return VJEPA2ForwardResult(
        embeddings=embeddings,
        last_hidden_state_shape=tuple(last_hidden_state.shape),
        embedding_type=embedding_type,
    )


def pool_vjepa_hidden_state(
    last_hidden_state: torch.Tensor,
    *,
    embedding_type: EmbeddingType = "last_hidden_state_mean_pool",
) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state must have shape [B, tokens, hidden], "
            f"got {tuple(last_hidden_state.shape)}"
        )
    if embedding_type != "last_hidden_state_mean_pool":
        raise ValueError(f"Unsupported embedding type: {embedding_type}")
    return last_hidden_state.mean(dim=1)


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _config_value(config: Any, name: str) -> Any:
    if config is None:
        return None
    return getattr(config, name, None)
