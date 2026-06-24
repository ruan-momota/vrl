"""Dataset adapters and generic indexed-video dataset utilities."""

from src.data.base import DatasetAdapter
from src.data.indexed_dataset import IndexedVideoDataset, collate_video_batch, load_index_jsonl
from src.data.records import VideoRecord

__all__ = [
    "DatasetAdapter",
    "IndexedVideoDataset",
    "VideoRecord",
    "collate_video_batch",
    "load_index_jsonl",
]
