from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DINOv2PreprocessConfig:
    image_size: int = 224
    num_frames: int = 16
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


class DINOv2ClipTransform:
    """Convert RGB clips to frame-wise DINOv2 image tensors."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        num_frames: int = 16,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        self.config = DINOv2PreprocessConfig(
            image_size=image_size,
            num_frames=num_frames,
            mean=mean,
            std=std,
        )

    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        if frames.ndim != 4:
            raise ValueError(f"frames must have shape [T, H, W, C], got {frames.shape}")
        if frames.shape[-1] != 3:
            raise ValueError(f"frames must be RGB with 3 channels, got {frames.shape}")
        if frames.shape[0] != self.config.num_frames:
            raise ValueError(
                "DINOv2 transform received an unexpected frame count: "
                f"expected {self.config.num_frames}, got {frames.shape[0]}"
            )

        resized = np.stack([_resize_center_crop(frame, self.config.image_size) for frame in frames])
        tensor = torch.from_numpy(resized).to(dtype=torch.float32).div(255.0)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        mean = torch.tensor(self.config.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.config.std, dtype=torch.float32).view(1, 3, 1, 1)
        return (tensor - mean) / std


@dataclass
class DINOv2Encoder:
    model: torch.nn.Module
    checkpoint: str
    transform: DINOv2ClipTransform
    device: str = "cpu"
    revision: str | None = None
    model_source: str = "transformers.Dinov2Model"
    feature_source: str = "last_hidden_state_cls_token"
    frame_aggregation: str = "mean"
    name: str = "dinov2"

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: str = "auto",
        image_size: int = 224,
        local_files_only: bool = False,
        revision: str | None = None,
        num_frames: int = 16,
    ) -> "DINOv2Encoder":
        try:
            from transformers import Dinov2Model
        except ImportError as error:  # pragma: no cover - dependency is declared.
            raise RuntimeError("transformers is required to load DINOv2 models") from error

        resolved_device = _resolve_device(device)
        load_options: dict[str, Any] = {"local_files_only": local_files_only}
        if revision is not None:
            load_options["revision"] = revision
        model = Dinov2Model.from_pretrained(checkpoint, **load_options)
        model.to(resolved_device)
        model.eval()
        resolved_revision = revision or _config_value(getattr(model, "config", None), "_commit_hash")
        return cls(
            model=model,
            checkpoint=checkpoint,
            transform=DINOv2ClipTransform(image_size=image_size, num_frames=num_frames),
            device=str(resolved_device),
            revision=resolved_revision,
        )

    def input_spec(self) -> dict[str, Any]:
        config = self.transform.config
        return {
            "num_frames": config.num_frames,
            "image_size": config.image_size,
            "input_layout": "[B, T, C, H, W]",
            "frame_encoder_layout": "[B*T, C, H, W]",
            "mean": list(config.mean),
            "std": list(config.std),
            "feature_source": self.feature_source,
            "frame_aggregation": self.frame_aggregation,
        }

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        return self.transform(frames)

    def encode(
        self,
        pixel_values: torch.Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        if pixel_values.ndim != 5:
            raise ValueError(
                "DINOv2 input must have shape [B, T, C, H, W], "
                f"got {tuple(pixel_values.shape)}"
            )
        resolved_device = _resolve_device(device or self.device)
        batch_size, frame_count = pixel_values.shape[:2]
        frame_values = pixel_values.to(resolved_device, dtype=torch.float32, non_blocking=True)
        frame_values = frame_values.reshape(batch_size * frame_count, *frame_values.shape[2:])
        self.model.to(resolved_device)
        self.model.eval()
        with torch.inference_mode():
            outputs = self.model(pixel_values=frame_values)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError("DINOv2 output did not include last_hidden_state")
        frame_embeddings = _cls_token_embeddings(last_hidden_state)
        embeddings = mean_pool_frame_embeddings(
            frame_embeddings,
            batch_size=batch_size,
            frame_count=frame_count,
        )
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("DINOv2 embeddings contain NaN or Inf values")
        return embeddings.detach()

    def metadata(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        return {
            "encoder_name": self.name,
            "model_type": _config_value(config, "model_type") or "dinov2",
            "checkpoint": self.checkpoint,
            "model_source": self.model_source,
            "device": self.device,
            "revision": self.revision,
            "feature_source": self.feature_source,
            "frame_aggregation": self.frame_aggregation,
            "hidden_size": _config_value(config, "hidden_size"),
            "image_size": _config_value(config, "image_size"),
            "patch_size": _config_value(config, "patch_size"),
            "num_hidden_layers": _config_value(config, "num_hidden_layers"),
            "preprocess": asdict(self.transform.config),
            "input_spec": self.input_spec(),
        }


def mean_pool_frame_embeddings(
    frame_embeddings: torch.Tensor,
    *,
    batch_size: int,
    frame_count: int,
) -> torch.Tensor:
    if frame_embeddings.ndim != 2:
        raise ValueError(
            "frame_embeddings must have shape [B*T, D], "
            f"got {tuple(frame_embeddings.shape)}"
        )
    if frame_embeddings.shape[0] != batch_size * frame_count:
        raise ValueError(
            "frame embedding count does not match batch_size * frame_count: "
            f"{frame_embeddings.shape[0]} != {batch_size} * {frame_count}"
        )
    return frame_embeddings.reshape(batch_size, frame_count, frame_embeddings.shape[1]).mean(dim=1)


def _cls_token_embeddings(last_hidden_state: torch.Tensor) -> torch.Tensor:
    if last_hidden_state.ndim != 3:
        raise ValueError(
            "last_hidden_state must have shape [B*T, tokens, hidden], "
            f"got {tuple(last_hidden_state.shape)}"
        )
    if last_hidden_state.shape[1] < 1:
        raise ValueError("last_hidden_state must include at least one token")
    return last_hidden_state[:, 0, :]


def _resize_center_crop(frame: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    width, height = image.size
    scale = image_size / float(min(width, height))
    resized_width = max(image_size, int(round(width * scale)))
    resized_height = max(image_size, int(round(height * scale)))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    image = image.resize((resized_width, resized_height), resampling)
    left = (resized_width - image_size) // 2
    top = (resized_height - image_size) // 2
    return np.asarray(image.crop((left, top, left + image_size, top + image_size)))


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _config_value(config: Any, name: str) -> Any:
    if config is None:
        return None
    return getattr(config, name, None)
