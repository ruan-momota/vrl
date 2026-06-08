from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.video_io import (
    FrameSamplingStrategy,
    read_sampled_clip,
)


def load_index_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
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


class SSV2ClipDataset(Dataset[dict[str, Any]]):
    """Dataset backed by the normalized SSV2 JSONL index."""

    def __init__(
        self,
        index_path: str | Path,
        *,
        num_frames: int = 16,
        sampling_strategy: FrameSamplingStrategy = "deterministic_center_clip",
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.samples = load_index_jsonl(self.index_path)
        self.num_frames = num_frames
        self.sampling_strategy = sampling_strategy
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        video_id = str(sample["video_id"])
        clip = read_sampled_clip(
            sample["video_path"],
            num_frames=self.num_frames,
            sampling_strategy=self.sampling_strategy,
            video_id=video_id,
        )
        if self.transform is None:
            pixel_values = torch.from_numpy(np.ascontiguousarray(clip.frames))
        else:
            pixel_values = self.transform(clip.frames)

        metadata = {
            **sample,
            "decode": clip.metadata.to_dict(),
            "frame_indices": list(clip.frame_indices),
            "sampling_strategy": clip.sampling_strategy,
        }
        return {
            "pixel_values": pixel_values,
            "label_id": sample.get("label_id"),
            "video_id": video_id,
            "metadata": metadata,
            "frame_indices": torch.tensor(clip.frame_indices, dtype=torch.long),
        }


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
    return {
        "pixel_values": pixel_values,
        "label_ids": label_ids,
        "video_ids": [sample["video_id"] for sample in samples],
        "metadata": [sample["metadata"] for sample in samples],
        "frame_indices": frame_indices,
    }
