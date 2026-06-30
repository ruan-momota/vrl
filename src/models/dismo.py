from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


DISMO_IMAGE_SIZE = 256
DISMO_HUB_REPO = "CompVis/DisMo"
DISMO_HUB_ENTRYPOINT = "motion_extractor_large"


@dataclass
class DisMoEncoder:
    """Adapter exposing the DisMo motion extractor through ``VideoEncoder``.

    DisMo is a pure representation model (no classification head), loaded from
    ``torch.hub`` exactly as in the Kinetics notebook. Frames are resized to
    256x256, scaled to ``[-1, 1]`` and fed as ``THWC``; the sliding-window
    motion features are mean-pooled over time to produce one ``[B, D]``
    embedding per clip, matching the shared extraction contract.
    """

    model: torch.nn.Module
    checkpoint: str
    device: str = "cpu"
    image_size: int = DISMO_IMAGE_SIZE
    model_source: str = "torch_hub:CompVis/DisMo:motion_extractor_large"
    embedding_source: str = "forward_sliding_time_mean_pool"
    name: str = "dismo"

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str = DISMO_HUB_ENTRYPOINT,
        *,
        device: str = "auto",
        image_size: int = DISMO_IMAGE_SIZE,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> "DisMoEncoder":
        if local_files_only:
            raise RuntimeError(
                "DisMo torch.hub loading needs network access to fetch the "
                "CompVis/DisMo repository and weights. Run without "
                "--local-files-only on a compute node after preparing the cache."
            )
        resolved_device = _resolve_device(device)
        repo = DISMO_HUB_REPO if revision is None else f"{DISMO_HUB_REPO}:{revision}"
        model = torch.hub.load(repo, checkpoint, trust_repo=True)
        model.to(resolved_device)
        model.eval()
        source = "torch_hub:CompVis/DisMo:motion_extractor_large"
        if revision:
            source = f"{source}@{revision}"
        return cls(
            model=model,
            checkpoint=checkpoint,
            device=str(resolved_device),
            image_size=image_size,
            model_source=source,
        )

    def input_spec(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "input_layout": "[B, T, H, W, C]",
            "value_range": [-1.0, 1.0],
            "embedding_source": self.embedding_source,
        }

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(
                f"DisMo expects RGB frames with shape [T, H, W, C], got {frames.shape}"
            )
        resized = np.stack(
            [_resize(frame, self.image_size) for frame in frames]
        )
        tensor = torch.from_numpy(resized).to(dtype=torch.float32)
        # Match the notebook: uint8 -> [0, 1] -> [-1, 1], kept as THWC.
        return tensor.div(255.0).mul(2.0).sub(1.0)

    def encode(
        self,
        pixel_values: torch.Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(
                f"DisMo input must have shape [B, T, H, W, C], got {tuple(pixel_values.shape)}"
            )
        resolved_device = _resolve_device(device or self.device)
        clips = pixel_values.to(
            resolved_device, dtype=torch.float32, non_blocking=True)
        self.model.to(resolved_device)
        self.model.eval()
        with torch.inference_mode():
            embeddings: list[torch.Tensor] = []
            for clip in clips:
                # forward_sliding expects a leading batch dim: (1, T, H, W, C).
                sequence = self.model.forward_sliding(
                    clip.unsqueeze(0)).squeeze(0)
                pooled = sequence.reshape(-1, sequence.shape[-1]).mean(dim=0)
                embeddings.append(pooled)
            features = torch.stack(embeddings, dim=0)
        if features.ndim != 2:
            raise ValueError(
                f"DisMo features must have shape [B, D], got {tuple(features.shape)}"
            )
        return features.detach()

    def metadata(self) -> dict[str, Any]:
        return {
            "encoder_name": self.name,
            "model_type": "dismo",
            "checkpoint": self.checkpoint,
            "model_source": self.model_source,
            "device": self.device,
            "input_spec": self.input_spec(),
        }


def _resize(frame: np.ndarray, image_size: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    resized = image.resize((image_size, image_size), Image.BILINEAR)
    return np.asarray(resized)


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved
