"""Encoder adapters for pretrained video models."""

from src.models.base import VideoEncoder
from src.models.videomae import VideoMAEEncoder

__all__ = ["VideoEncoder", "VideoMAEEncoder"]
