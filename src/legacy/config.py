"""Configuration schema for pre-run-config experiment commands."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str
    split: str
    video_root: str
    annotation_path: str
    label_mapping_path: str | None
    model_checkpoint: str
    num_frames: int
    sampling_strategy: str
    image_size: int
    batch_size: int
    num_workers: int
    device: str
    embedding_type: str
    video_decoder: str
    output_path: str
    seed: int
    deterministic: bool

    @classmethod
    def from_file(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls.from_mapping(data)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "ExperimentConfig":
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
