from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.data.indexed_dataset import IndexedVideoDataset
from src.video.io import FrameSamplingStrategy
from src.video.perturbations import VideoPerturbation


@dataclass(frozen=True)
class Diving48DatasetAdapter:
    """Diving48 adapter for the normalized JSONL index format."""

    name: str = "diving48"

    def build_dataset(
        self,
        index_path: str | Path,
        *,
        num_frames: int,
        sampling_strategy: FrameSamplingStrategy,
        transform: Callable[[np.ndarray], torch.Tensor] | None,
        perturbation: VideoPerturbation | None,
        subset_id: str | None = None,
        window_frames: int | None = None,
    ) -> IndexedVideoDataset:
        return IndexedVideoDataset(
            index_path,
            num_frames=num_frames,
            sampling_strategy=sampling_strategy,
            transform=transform,
            perturbation=perturbation,
            source_dataset=self.name,
            subset_id=subset_id,
            window_frames=window_frames,
        )
