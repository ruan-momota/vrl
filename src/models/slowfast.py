from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


KINETICS_MEAN = (0.45, 0.45, 0.45)
KINETICS_STD = (0.225, 0.225, 0.225)


@dataclass(frozen=True)
class SlowFastPreprocessConfig:
    image_size: int = 256
    num_frames: int = 32
    slowfast_alpha: int = 4
    mean: tuple[float, float, float] = KINETICS_MEAN
    std: tuple[float, float, float] = KINETICS_STD


class SlowFastClipTransform:
    """Convert RGB clips to a SlowFast fast-pathway tensor.

    The dataset and collate path remain model-agnostic by carrying only the
    normalized fast pathway as ``[C, T, H, W]``. The encoder builds the slow
    pathway just before model forward.
    """

    def __init__(
        self,
        *,
        image_size: int = 256,
        num_frames: int = 32,
        slowfast_alpha: int = 4,
        mean: tuple[float, float, float] = KINETICS_MEAN,
        std: tuple[float, float, float] = KINETICS_STD,
    ) -> None:
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if slowfast_alpha <= 0:
            raise ValueError("slowfast_alpha must be positive")
        self.config = SlowFastPreprocessConfig(
            image_size=image_size,
            num_frames=num_frames,
            slowfast_alpha=slowfast_alpha,
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
                "SlowFast transform received an unexpected frame count: "
                f"expected {self.config.num_frames}, got {frames.shape[0]}"
            )

        resized = np.stack([_resize_center_crop(frame, self.config.image_size) for frame in frames])
        tensor = torch.from_numpy(resized).to(dtype=torch.float32).div(255.0)
        tensor = tensor.permute(3, 0, 1, 2).contiguous()
        mean = torch.tensor(self.config.mean, dtype=torch.float32).view(3, 1, 1, 1)
        std = torch.tensor(self.config.std, dtype=torch.float32).view(3, 1, 1, 1)
        return (tensor - mean) / std


@dataclass
class SlowFastEncoder:
    model: torch.nn.Module
    checkpoint: str
    transform: SlowFastClipTransform
    model_source: str = "pytorchvideo_torch_hub"
    embedding_source: str = "last_linear_input"
    device: str = "cpu"
    name: str = "slowfast"

    @classmethod
    def from_pretrained(
        cls,
        checkpoint: str,
        *,
        device: str = "auto",
        image_size: int = 256,
        local_files_only: bool = False,
        revision: str | None = None,
        num_frames: int = 32,
        slowfast_alpha: int = 4,
    ) -> "SlowFastEncoder":
        if local_files_only and _is_torch_hub_checkpoint(checkpoint):
            raise RuntimeError(
                "SlowFast torch.hub loading needs a cached repository and weights. "
                "Run without --local-files-only on a compute node after preparing the cache, "
                "or pass a local serialized model path."
            )
        model, source = _load_slowfast_model(checkpoint, revision=revision)
        resolved_device = _resolve_device(device)
        model.to(resolved_device)
        model.eval()
        return cls(
            model=model,
            checkpoint=checkpoint,
            transform=SlowFastClipTransform(
                image_size=image_size,
                num_frames=num_frames,
                slowfast_alpha=slowfast_alpha,
            ),
            model_source=source,
            device=str(resolved_device),
        )

    def input_spec(self) -> dict[str, Any]:
        config = self.transform.config
        return {
            "num_frames": config.num_frames,
            "image_size": config.image_size,
            "input_layout": "[B, C, T, H, W]",
            "pathways": {
                "fast": "[B, C, T, H, W]",
                "slow": "[B, C, T/alpha, H, W]",
                "alpha": config.slowfast_alpha,
            },
            "mean": list(config.mean),
            "std": list(config.std),
            "embedding_source": self.embedding_source,
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
            raise ValueError(f"SlowFast input must have shape [B, C, T, H, W], got {tuple(pixel_values.shape)}")
        resolved_device = _resolve_device(device or self.device)
        fast_pathway = pixel_values.to(resolved_device, dtype=torch.float32, non_blocking=True)
        slow_pathway = _build_slow_pathway(fast_pathway, alpha=self.transform.config.slowfast_alpha)
        self.model.to(resolved_device)
        self.model.eval()

        captured: dict[str, torch.Tensor] = {}
        hook = _last_linear(self.model).register_forward_hook(
            lambda _module, inputs, _output: captured.setdefault("features", inputs[0])
        )
        try:
            with torch.inference_mode():
                output = self.model([slow_pathway, fast_pathway])
        finally:
            hook.remove()

        features = captured.get("features")
        if features is None:
            features = output if isinstance(output, torch.Tensor) else None
        if features is None:
            raise RuntimeError("SlowFast forward did not return or expose tensor features")
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        if features.ndim != 2:
            raise ValueError(f"SlowFast features must have shape [B, D], got {tuple(features.shape)}")
        return features.detach()

    def metadata(self) -> dict[str, Any]:
        return {
            "encoder_name": self.name,
            "model_type": "slowfast",
            "checkpoint": self.checkpoint,
            "model_source": self.model_source,
            "device": self.device,
            "input_spec": self.input_spec(),
        }


def _resize_center_crop(frame: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    width, height = image.size
    scale = image_size / float(min(width, height))
    resized_width = max(image_size, int(round(width * scale)))
    resized_height = max(image_size, int(round(height * scale)))
    image = image.resize((resized_width, resized_height), Image.BILINEAR)
    left = (resized_width - image_size) // 2
    top = (resized_height - image_size) // 2
    return np.asarray(image.crop((left, top, left + image_size, top + image_size)))


def _build_slow_pathway(fast_pathway: torch.Tensor, *, alpha: int) -> torch.Tensor:
    frame_count = fast_pathway.shape[2]
    slow_count = max(1, frame_count // alpha)
    indices = torch.linspace(0, frame_count - 1, slow_count, device=fast_pathway.device).long()
    return torch.index_select(fast_pathway, dim=2, index=indices)


def _last_linear(model: torch.nn.Module) -> torch.nn.Linear:
    last: torch.nn.Linear | None = None
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            last = module
    if last is None:
        raise RuntimeError("SlowFast model does not contain a Linear classifier to hook")
    return last


def _load_slowfast_model(
    checkpoint: str,
    *,
    revision: str | None = None,
) -> tuple[torch.nn.Module, str]:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return _load_local_slowfast_checkpoint(checkpoint_path)

    if not _is_torch_hub_checkpoint(checkpoint):
        raise ValueError(
            "Unsupported SlowFast checkpoint. Use 'facebookresearch/pytorchvideo:slowfast_r50', "
            "'pytorchvideo/slowfast_r50', 'pytorchvideo/slowfast_r50_8x8', 'slowfast_r50', "
            "or a local PyTorchVideo SlowFast state-dict / serialized module path."
        )

    direct_error: Exception | None = None
    try:
        from pytorchvideo.models.hub import slowfast_r50

        return slowfast_r50(pretrained=True), "pytorchvideo.models.hub:slowfast_r50"
    except Exception as error:  # noqa: BLE001 - optional dependency/backend discovery.
        direct_error = error

    hub_error: Exception | None = None
    try:
        model = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=True,
            force_reload=False,
            trust_repo=True,
            skip_validation=True,
            source="github",
        )
    except Exception as error:  # noqa: BLE001 - surface actionable optional dependency setup.
        hub_error = error
        raise RuntimeError(
            "Failed to load SlowFast. First tried installed pytorchvideo, then torch.hub. "
            "On a compute node, install/cache pytorchvideo and allow pretrained weights "
            "to be downloaded, or provide a local PyTorchVideo SlowFast checkpoint path. "
            f"Installed-package error: {direct_error!r}. torch.hub error: {hub_error!r}."
        ) from error
    source = "torch_hub:facebookresearch/pytorchvideo:slowfast_r50"
    if revision:
        source = f"{source}@{revision}"
    return model, source


def _load_local_slowfast_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, str]:
    loaded = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(loaded, torch.nn.Module):
        return loaded, f"local_module:{checkpoint_path}"

    try:
        from pytorchvideo.models.hub import slowfast_r50
    except Exception as error:  # noqa: BLE001 - optional dependency discovery.
        raise RuntimeError(
            "Loading a local SlowFast state dict requires the pytorchvideo package so the "
            "matching architecture can be constructed."
        ) from error

    model = slowfast_r50(pretrained=False)
    state_dict = _extract_state_dict(loaded)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise RuntimeError(
            "Local SlowFast checkpoint did not match PyTorchVideo slowfast_r50. "
            "Use a PyTorchVideo slowfast_r50 state dict, a serialized torch.nn.Module, "
            "or switch the adapter to the checkpoint's native implementation."
        ) from error
    return model, f"local_pytorchvideo_state_dict:{checkpoint_path}"


def _extract_state_dict(loaded: Any) -> dict[str, torch.Tensor]:
    if not isinstance(loaded, dict):
        raise TypeError(
            "Local SlowFast checkpoint must be a serialized torch.nn.Module or a state-dict mapping"
        )
    candidates = (
        loaded.get("model_state"),
        loaded.get("state_dict"),
        loaded.get("model"),
        loaded,
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and all(
            isinstance(key, str) for key in candidate.keys()
        ):
            return {
                key.removeprefix("module."): value
                for key, value in candidate.items()
                if isinstance(value, torch.Tensor)
            }
    raise TypeError("Could not find a tensor state_dict in local SlowFast checkpoint")


def _is_torch_hub_checkpoint(checkpoint: str) -> bool:
    return checkpoint in {
        "slowfast_r50",
        "pytorchvideo/slowfast_r50",
        "pytorchvideo/slowfast_r50_8x8",
        "facebookresearch/pytorchvideo:slowfast_r50",
    }


def _resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved
