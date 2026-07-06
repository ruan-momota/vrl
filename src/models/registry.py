from __future__ import annotations

from typing import Any

from src.models.base import VideoEncoder
from src.models.dinov2 import DINOv2Encoder
from src.models.slowfast import SlowFastEncoder
from src.models.videomae import VideoMAEEncoder


def load_video_encoder(name: str, **kwargs: Any) -> VideoEncoder:
    if name == "dinov2":
        return DINOv2Encoder.from_pretrained(**kwargs)
    if name == "videomae":
        return VideoMAEEncoder.from_pretrained(**kwargs)
    if name == "slowfast":
        return SlowFastEncoder.from_pretrained(**kwargs)
    supported = ", ".join(supported_video_encoders())
    raise ValueError(f"Unsupported video encoder {name!r}; supported: {supported}")


def supported_video_encoders() -> tuple[str, ...]:
    return ("dinov2", "slowfast", "videomae")
