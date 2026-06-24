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
class SSV2DatasetAdapter:
    """SSV2 adapter for the already-normalized JSONL index format."""

    name: str = "ssv2"

    def build_dataset(
        self,
        index_path: str | Path,
        *,
        num_frames: int,
        sampling_strategy: FrameSamplingStrategy,
        transform: Callable[[np.ndarray], torch.Tensor] | None,
        perturbation: VideoPerturbation | None,
    ) -> IndexedVideoDataset:
        return IndexedVideoDataset(
            index_path,
            num_frames=num_frames,
            sampling_strategy=sampling_strategy,
            transform=transform,
            perturbation=perturbation,
            source_dataset=self.name,
        )


class SSV2ClipDataset(IndexedVideoDataset):
    """Compatibility name for the normalized SSV2 indexed dataset.

    New pipeline code should construct datasets through ``SSV2DatasetAdapter``.
    The class stays in the SSV2 adapter module so old callers do not reintroduce
    a root-level dataset implementation.
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
    ) -> None:
        super().__init__(
            index_path,
            num_frames=num_frames,
            sampling_strategy=sampling_strategy,
            transform=transform,
            perturbation=perturbation,
            include_original_clip=include_original_clip,
            source_dataset="ssv2",
        )
