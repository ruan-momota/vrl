from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from src.video.io import FrameSamplingStrategy
from src.video.perturbations import VideoPerturbation


class DatasetAdapter(Protocol):
    """Build model-agnostic clip datasets from a normalized dataset index."""

    name: str

    def build_dataset(
        self,
        index_path: str | Path,
        *,
        num_frames: int,
        sampling_strategy: FrameSamplingStrategy,
        transform: Callable[[np.ndarray], torch.Tensor] | None,
        perturbation: VideoPerturbation | None,
        subset_id: str | None = None,
    ) -> Dataset[dict[str, object]]: ...
