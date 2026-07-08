from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.models.vjepa_model import (
    EmbeddingType,
    VJEPA2ModelMetadata,
    forward_vjepa_embeddings,
    load_vjepa_model,
)
from src.models.vjepa_preprocessing import VJEPA2ClipTransform


DEFAULT_VJEPA_CHECKPOINT = "facebook/vjepa2-vitl-fpc64-256"


@dataclass
class VJEPA2Encoder:
    """Adapter exposing Meta's V-JEPA 2 (via ``transformers``) through ``VideoEncoder``."""

    model: torch.nn.Module
    model_metadata: VJEPA2ModelMetadata
    transform: VJEPA2ClipTransform
    embedding_type: EmbeddingType = "last_hidden_state_mean_pool"
    name: str = "vjepa"

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = DEFAULT_VJEPA_CHECKPOINT,
        *,
        device: str = "auto",
        image_size: int = 256,
        local_files_only: bool = False,
        embedding_type: EmbeddingType = "last_hidden_state_mean_pool",
        revision: str | None = None,
    ) -> "VJEPA2Encoder":
        model, metadata = load_vjepa_model(
            checkpoint,
            device=device,
            local_files_only=local_files_only,
            revision=revision,
        )
        return cls(
            model=model,
            model_metadata=metadata,
            transform=VJEPA2ClipTransform(image_size=image_size),
            embedding_type=embedding_type,
        )

    def input_spec(self) -> dict[str, Any]:
        return {
            "num_frames": self.model_metadata.frames_per_clip,
            "image_size": self.model_metadata.crop_size,
            "input_layout": "[B, T, C, H, W]",
            "embedding_type": self.embedding_type,
        }

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        return self.transform(frames)

    def encode(
        self,
        pixel_values: torch.Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        return forward_vjepa_embeddings(
            self.model,
            pixel_values,
            embedding_type=self.embedding_type,
            device=device,
        ).embeddings

    def metadata(self) -> dict[str, Any]:
        return {
            "encoder_name": self.name,
            **self.model_metadata.to_dict(),
            "input_spec": self.input_spec(),
        }
