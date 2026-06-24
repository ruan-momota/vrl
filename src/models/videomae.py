from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.models.videomae_model import (
    EmbeddingType,
    VideoMAEModelMetadata,
    forward_videomae_embeddings,
    load_videomae_model,
)
from src.models.videomae_preprocessing import VideoMAEClipTransform


@dataclass
class VideoMAEEncoder:
    """Adapter exposing the existing VideoMAE path through ``VideoEncoder``."""

    model: torch.nn.Module
    model_metadata: VideoMAEModelMetadata
    transform: VideoMAEClipTransform
    embedding_type: EmbeddingType = "last_hidden_state_mean_pool"
    name: str = "videomae"

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: str = "auto",
        image_size: int = 224,
        processor_checkpoint: str | None = None,
        local_files_only: bool = False,
        embedding_type: EmbeddingType = "last_hidden_state_mean_pool",
    ) -> "VideoMAEEncoder":
        model, metadata = load_videomae_model(
            checkpoint,
            device=device,
            local_files_only=local_files_only,
        )
        return cls(
            model=model,
            model_metadata=metadata,
            transform=VideoMAEClipTransform(
                image_size=image_size,
                processor_checkpoint=processor_checkpoint,
                local_files_only=local_files_only,
            ),
            embedding_type=embedding_type,
        )

    def input_spec(self) -> dict[str, Any]:
        return {
            "num_frames": self.model_metadata.num_frames,
            "image_size": self.model_metadata.image_size,
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
        return forward_videomae_embeddings(
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
