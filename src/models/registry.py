from __future__ import annotations

from typing import Any

from src.models.base import VideoEncoder
from src.models.dismo import DisMoEncoder
from src.models.slowfast import SlowFastEncoder
from src.models.videomae import VideoMAEEncoder


def load_video_encoder(name: str, **kwargs: Any) -> VideoEncoder:
    if name == "videomae":
        return VideoMAEEncoder.from_pretrained(**kwargs)
    if name == "slowfast":
        return SlowFastEncoder.from_pretrained(**kwargs)
    if name == "dismo":
        return DisMoEncoder.from_pretrained(**kwargs)
    raise ValueError(
        f"Unsupported video encoder {name!r}; supported: dismo, slowfast, videomae")


def supported_video_encoders() -> tuple[str, ...]:
    return ("dismo", "slowfast", "videomae")
