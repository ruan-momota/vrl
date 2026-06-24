from __future__ import annotations

from typing import Any, Protocol

import numpy as np
import torch


class VideoEncoder(Protocol):
    """Minimal model boundary used by the extraction pipeline."""

    name: str

    def input_spec(self) -> dict[str, Any]:
        """Return the model-specific frame and image input requirements."""

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """Convert sampled RGB frames to this model's input tensor layout."""

    def encode(
        self,
        pixel_values: torch.Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        """Return finite clip embeddings with shape ``[B, D]``."""

    def metadata(self) -> dict[str, Any]:
        """Return serializable checkpoint and feature-extraction metadata."""
