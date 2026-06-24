from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.records import VideoRecord
from src.video.io import FrameSamplingStrategy, read_sampled_clip
from src.video.perturbations import VideoPerturbation


def load_index_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    """Load a normalized video index without assuming a source dataset."""
    index_path = Path(path)
    samples: list[dict[str, Any]] = []
    with index_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "video_id" not in item or "video_path" not in item:
                raise ValueError(
                    f"{index_path}:{line_number} must include video_id and video_path"
                )
            samples.append(item)
    return tuple(samples)


class IndexedVideoDataset(Dataset[dict[str, Any]]):
    """Dataset backed by a normalized JSONL video index.

    The dataset owns no SSV2-, UCF101-, or model-specific logic. A dataset
    adapter supplies the normalized index and an encoder supplies ``transform``.
    """

    def __init__(
        self,
        index_path: str | Path,
        *,
        num_frames: int = 16,
        sampling_strategy: FrameSamplingStrategy = "deterministic_center_clip",
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
        perturbation: VideoPerturbation | None = None,
        include_original_clip: bool = False,
        source_dataset: str | None = None,
        subset_id: str | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.samples = load_index_jsonl(self.index_path)
        self.records = tuple(
            VideoRecord.from_mapping(
                sample,
                source_dataset=source_dataset,
                subset_id=subset_id,
            )
            for sample in self.samples
        )
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.transform = transform
        self.perturbation = perturbation
        self.include_original_clip = include_original_clip

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            clip = read_sampled_clip(
                record.video_path,
                num_frames=self.num_frames,
                sampling_strategy=self.sampling_strategy,
                video_id=record.video_id,
            )
            frames = clip.frames
            perturbation_metadata = {"name": "none", "operation": "identity"}
            if self.perturbation is not None:
                perturbation_result = self.perturbation(frames, video_id=record.video_id)
                frames = perturbation_result.frames
                perturbation_metadata = perturbation_result.metadata

            metadata = {
                **record.to_mapping(),
                "decode": clip.metadata.to_dict(),
                "frame_indices": list(clip.frame_indices),
                "sampling_strategy": clip.sampling_strategy,
                "perturbation": perturbation_metadata,
            }
            item = {
                "pixel_values": self._to_pixel_values(frames),
                "label_id": record.label_id,
                "video_id": record.video_id,
                "metadata": metadata,
                "frame_indices": torch.tensor(clip.frame_indices, dtype=torch.long),
            }
            if self.include_original_clip:
                item["original_pixel_values"] = self._to_pixel_values(clip.frames)
            return item
        except Exception as error:  # noqa: BLE001 - attach sample context before failing fast.
            perturbation_config = (
                self.perturbation.config.to_dict()
                if self.perturbation is not None
                else {"name": "none"}
            )
            raise RuntimeError(
                "Failed to build indexed video clip sample "
                f"video_id={record.video_id} "
                f"path={record.video_path} "
                f"perturbation={perturbation_config}"
            ) from error

    def _to_pixel_values(self, frames: np.ndarray) -> torch.Tensor:
        if self.transform is None:
            return torch.from_numpy(np.ascontiguousarray(frames))
        return self.transform(frames)


def collate_video_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")

    pixel_values = torch.stack([sample["pixel_values"] for sample in samples], dim=0)
    label_values = [sample["label_id"] for sample in samples]
    label_ids = (
        torch.tensor(label_values, dtype=torch.long)
        if all(label is not None for label in label_values)
        else None
    )
    frame_indices = torch.stack([sample["frame_indices"] for sample in samples], dim=0)
    batch = {
        "pixel_values": pixel_values,
        "label_ids": label_ids,
        "video_ids": [sample["video_id"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "frame_indices": frame_indices,
    }
    has_original = ["original_pixel_values" in sample for sample in samples]
    if any(has_original) and not all(has_original):
        raise ValueError("cannot mix samples with and without original_pixel_values")
    if all(has_original):
        batch["original_pixel_values"] = torch.stack(
            [sample["original_pixel_values"] for sample in samples], dim=0
        )
    return batch
