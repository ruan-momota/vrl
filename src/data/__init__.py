"""Dataset adapters and generic indexed-video dataset utilities."""

from src.data.base import DatasetAdapter
from src.data.diving48 import Diving48DatasetAdapter
from src.data.indexed_dataset import IndexedVideoDataset, collate_video_batch, load_index_jsonl
from src.data.records import VideoRecord
from src.data.ucf101 import UCF101DatasetAdapter

__all__ = [
    "DatasetAdapter",
    "Diving48DatasetAdapter",
    "IndexedVideoDataset",
    "UCF101DatasetAdapter",
    "VideoRecord",
    "collate_video_batch",
    "load_index_jsonl",
]
