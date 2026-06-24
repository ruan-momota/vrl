"""Compatibility CLI for the pre-Phase-1 run configuration format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, cast

import torch
from torch.utils.data import Dataset, Subset

from src.artifacts import (
    EmbeddingExtractionResult,
    load_embedding_artifact,
    save_embedding_artifact,
    validate_embedding_artifact,
    validate_embedding_result,
)
from src.legacy.config import ExperimentConfig
from src.data.indexed_dataset import IndexedVideoDataset
from src.data.registry import get_dataset_adapter
from src.models.base import VideoEncoder
from src.models.videomae import VideoMAEEncoder
from src.pipeline.extraction import extract_embeddings as _extract_embeddings
from src.pipeline.extraction import resolve_device
from src.reproducibility import seed_everything
from src.models.videomae_model import EmbeddingType, forward_videomae_embeddings
from src.models.videomae_preprocessing import VideoMAEClipTransform
from src.video.perturbations import (
    VideoPerturbationConfig,
    build_video_perturbation,
    parse_perturbation_name,
)


class _LegacyVideoMAEModelEncoder:
    """Adapt the historical raw VideoMAE model argument to the encoder contract."""

    name = "legacy-videomae-model"

    def __init__(self, model: torch.nn.Module, embedding_type: EmbeddingType) -> None:
        self.model = model
        self.embedding_type = embedding_type

    def encode(
        self,
        pixel_values: torch.Tensor,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor:
        return forward_videomae_embeddings(
            self.model,
            pixel_values,
            embedding_type=self.embedding_type,
            device=device,
        ).embeddings


def extract_embeddings(
    *,
    model: torch.nn.Module | None = None,
    encoder: VideoEncoder | None = None,
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
    embedding_type: EmbeddingType = "last_hidden_state_mean_pool",
    pin_memory: bool = False,
    show_progress: bool = True,
) -> EmbeddingExtractionResult:
    """Historical extraction API forwarding to the generic extraction core."""
    if model is None and encoder is None:
        raise ValueError("Provide either model or encoder for embedding extraction")
    if model is not None and encoder is not None:
        raise ValueError("Provide model or encoder, not both")
    resolved_encoder: VideoEncoder
    if encoder is not None:
        resolved_encoder = encoder
    else:
        assert model is not None
        resolved_encoder = cast(
            VideoEncoder,
            _LegacyVideoMAEModelEncoder(model, embedding_type),
        )
    return _extract_embeddings(
        encoder=resolved_encoder,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        pin_memory=pin_memory,
        show_progress=show_progress,
    )


def build_extraction_dataset(
    *,
    index_path: str | Path,
    num_frames: int,
    sampling_strategy: str,
    image_size: int,
    processor_checkpoint: str | None = None,
    local_files_only: bool = False,
    limit: int | None = None,
    perturbation_config: VideoPerturbationConfig | None = None,
    dataset_name: str = "ssv2",
    transform: Callable[[Any], torch.Tensor] | None = None,
) -> IndexedVideoDataset | Subset[dict[str, Any]]:
    """Build a dataset from the historical CLI options."""
    resolved_transform = transform or VideoMAEClipTransform(
        image_size=image_size,
        processor_checkpoint=processor_checkpoint,
        local_files_only=local_files_only,
    )
    adapter = get_dataset_adapter(dataset_name)
    dataset: IndexedVideoDataset | Subset[dict[str, Any]] = adapter.build_dataset(
        index_path,
        num_frames=num_frames,
        sampling_strategy=sampling_strategy,
        transform=resolved_transform,
        perturbation=build_video_perturbation(perturbation_config),
    )
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility entry point for split-level VideoMAE embeddings."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/ssv2_videomae_smoke.json"))
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--processor-checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument(
        "--perturbation",
        default="none",
        choices=[
            "none",
            "temporal_reverse",
            "temporal_shuffle",
            "freeze_tail",
            "single_frame",
            "grayscale",
            "center_occlusion",
        ],
    )
    parser.add_argument("--perturbation-seed", type=int, default=0)
    parser.add_argument("--perturbation-frame-index", type=int, default=None)
    parser.add_argument("--freeze-start-fraction", type=float, default=0.5)
    parser.add_argument("--occlusion-size-fraction", type=float, default=0.25)
    parser.add_argument("--occlusion-fill-value", type=int, default=0)
    return parser.parse_args()


def run_extraction_from_args(args: argparse.Namespace) -> dict[str, Any]:
    config = ExperimentConfig.from_file(args.config)
    seed_everything(config.seed, deterministic=config.deterministic)

    split = args.split or config.split
    index_path = args.index_path or Path("data/ssv2/index") / f"{split}.jsonl"
    checkpoint = args.checkpoint or config.model_checkpoint
    device_name = args.device or config.device
    batch_size = args.batch_size or config.batch_size
    num_workers = config.num_workers if args.num_workers is None else args.num_workers
    output_path = args.output_path or Path(config.output_path)
    resolved_device = resolve_device(device_name)
    perturbation_config = VideoPerturbationConfig(
        name=parse_perturbation_name(args.perturbation),
        seed=args.perturbation_seed,
        frame_index=args.perturbation_frame_index,
        freeze_start_fraction=args.freeze_start_fraction,
        occlusion_size_fraction=args.occlusion_size_fraction,
        occlusion_fill_value=args.occlusion_fill_value,
    )

    encoder = VideoMAEEncoder.from_pretrained(
        checkpoint,
        device=str(resolved_device),
        image_size=config.image_size,
        processor_checkpoint=args.processor_checkpoint,
        local_files_only=args.local_files_only,
        embedding_type=config.embedding_type,  # type: ignore[arg-type]
    )
    dataset = build_extraction_dataset(
        index_path=index_path,
        num_frames=config.num_frames,
        sampling_strategy=config.sampling_strategy,
        image_size=config.image_size,
        processor_checkpoint=args.processor_checkpoint,
        local_files_only=args.local_files_only,
        limit=args.limit,
        perturbation_config=perturbation_config,
        dataset_name=config.dataset_name,
        transform=encoder.preprocess,
    )
    result = extract_embeddings(
        encoder=encoder,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=resolved_device,
        pin_memory=args.pin_memory,
        show_progress=not args.no_progress,
    )
    config_snapshot = {
        **config.to_dict(),
        "split": split,
        "index_path": str(index_path),
        "model_checkpoint": checkpoint,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "device": str(resolved_device),
        "output_path": str(output_path),
        "perturbation": perturbation_config.to_dict(),
    }
    artifact = save_embedding_artifact(
        result,
        output_path,
        config_snapshot=config_snapshot,
        model_metadata=encoder.metadata(),
        extraction_options={
            "limit": args.limit,
            "pin_memory": args.pin_memory,
            "local_files_only": args.local_files_only,
            "processor_checkpoint": args.processor_checkpoint,
            "perturbation": perturbation_config.to_dict(),
        },
        overwrite=args.overwrite,
    )
    reloaded = load_embedding_artifact(output_path)
    return {
        "output_path": str(output_path),
        "reloaded_ok": True,
        "summary": reloaded["summary"],
        "model_metadata": artifact["model_metadata"],
    }


def main() -> int:
    summary = run_extraction_from_args(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


__all__ = [
    "build_extraction_dataset",
    "extract_embeddings",
    "load_embedding_artifact",
    "save_embedding_artifact",
    "validate_embedding_artifact",
    "validate_embedding_result",
]
