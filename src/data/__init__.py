"""Dataset adapters and generic indexed-video dataset utilities."""

from src.data.base import DatasetAdapter
from src.data.hmdb51 import HMDB51DatasetAdapter
from src.data.indexed_dataset import IndexedVideoDataset, collate_video_batch, load_index_jsonl
from src.data.kinetics import KineticsDatasetAdapter
from src.data.records import VideoRecord
from src.data.ucf101 import UCF101DatasetAdapter

__all__ = [
    "DatasetAdapter",
    "HMDB51DatasetAdapter",
    "IndexedVideoDataset",
    "KineticsDatasetAdapter",
    "UCF101DatasetAdapter",
    "VideoRecord",
    "collate_video_batch",
    "load_index_jsonl",
]
