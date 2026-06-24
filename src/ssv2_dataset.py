"""Backward-compatible SSV2 dataset import path."""

from src.data.indexed_dataset import collate_video_batch, load_index_jsonl
from src.data.ssv2 import SSV2ClipDataset

__all__ = ["SSV2ClipDataset", "collate_video_batch", "load_index_jsonl"]
