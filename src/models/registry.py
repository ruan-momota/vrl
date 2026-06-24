from __future__ import annotations

from typing import Any

from src.models.base import VideoEncoder
from src.models.videomae import VideoMAEEncoder


def load_video_encoder(name: str, **kwargs: Any) -> VideoEncoder:
    if name == "videomae":
        return VideoMAEEncoder.from_pretrained(**kwargs)
    raise ValueError(f"Unsupported video encoder {name!r}; supported: videomae")


def supported_video_encoders() -> tuple[str, ...]:
    return ("videomae",)
