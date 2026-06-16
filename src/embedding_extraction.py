from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from src.config import ExperimentConfig
from src.reproducibility import seed_everything
from src.ssv2_dataset import SSV2ClipDataset, collate_video_batch
from src.videomae_model import (
    EmbeddingType,
    VideoMAEModelMetadata,
    forward_videomae_embeddings,
    load_videomae_model,
    resolve_device,
)
from src.videomae_preprocessing import VideoMAEClipTransform
from src.video_perturbations import (
    VideoPerturbationConfig,
    build_video_perturbation,
    parse_perturbation_name,
)


@dataclass(frozen=True)
class EmbeddingExtractionSummary:
    dataset_size: int
    successful_samples: int
    failed_samples: int
    batch_count: int
    total_seconds: float
    average_batch_seconds: float
    embeddings_shape: tuple[int, ...]
    embeddings_dtype: str
    labels_shape: tuple[int, ...] | None
    output_path: str | None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["embeddings_shape"] = list(self.embeddings_shape)
        data["labels_shape"] = None if self.labels_shape is None else list(self.labels_shape)
        return data


@dataclass(frozen=True)
class EmbeddingExtractionResult:
    embeddings: torch.Tensor
    label_ids: torch.Tensor | None
    video_ids: list[str]
    sample_metadata: list[dict[str, Any]]
    frame_indices: torch.Tensor
    summary: EmbeddingExtractionSummary


def extract_embeddings(
    *,
    model: torch.nn.Module,
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
    embedding_type: EmbeddingType = "last_hidden_state_mean_pool",
    pin_memory: bool = False,
    show_progress: bool = True,
) -> EmbeddingExtractionResult:
    """Extract clip embeddings for every sample in a dataset.

    The MVP path is intentionally fail-fast: a decode, preprocessing, or forward
    error aborts the run before any output artifact is written.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_video_batch,
    )
    resolved_device = torch.device(device)
    embeddings: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    saw_unlabeled_batch = False
    video_ids: list[str] = []
    sample_metadata: list[dict[str, Any]] = []
    frame_index_chunks: list[torch.Tensor] = []
    batch_seconds: list[float] = []

    loader_iter = iter(loader)
    progress = tqdm(
        range(len(loader)),
        total=len(loader),
        desc="extract embeddings",
        disable=not show_progress,
    )
    total_start = time.perf_counter()
    for _ in progress:
        batch_start = time.perf_counter()
        batch = next(loader_iter)
        result = forward_videomae_embeddings(
            model,
            batch["pixel_values"],
            embedding_type=embedding_type,
            device=resolved_device,
        )
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        batch_seconds.append(time.perf_counter() - batch_start)

        embeddings.append(result.embeddings.detach().cpu().to(dtype=torch.float32))
        label_ids = batch["label_ids"]
        if label_ids is None:
            saw_unlabeled_batch = True
        else:
            label_chunks.append(label_ids.detach().cpu())
        video_ids.extend(str(video_id) for video_id in batch["video_ids"])
        sample_metadata.extend(batch["metadata"])
        frame_index_chunks.append(batch["frame_indices"].detach().cpu())

    total_seconds = time.perf_counter() - total_start
    embedding_tensor = (
        torch.cat(embeddings, dim=0) if embeddings else torch.empty((0, 0), dtype=torch.float32)
    )
    label_tensor = None
    if label_chunks and not saw_unlabeled_batch:
        label_tensor = torch.cat(label_chunks, dim=0)
    elif label_chunks and saw_unlabeled_batch:
        raise RuntimeError("Cannot mix labeled and unlabeled batches in one extraction run")

    frame_indices = (
        torch.cat(frame_index_chunks, dim=0)
        if frame_index_chunks
        else torch.empty((0, 0), dtype=torch.long)
    )
    summary = EmbeddingExtractionSummary(
        dataset_size=len(dataset),
        successful_samples=len(video_ids),
        failed_samples=0,
        batch_count=len(batch_seconds),
        total_seconds=total_seconds,
        average_batch_seconds=sum(batch_seconds) / len(batch_seconds)
        if batch_seconds
        else 0.0,
        embeddings_shape=tuple(embedding_tensor.shape),
        embeddings_dtype=str(embedding_tensor.dtype),
        labels_shape=None if label_tensor is None else tuple(label_tensor.shape),
        output_path=None,
    )
    artifact = EmbeddingExtractionResult(
        embeddings=embedding_tensor,
        label_ids=label_tensor,
        video_ids=video_ids,
        sample_metadata=sample_metadata,
        frame_indices=frame_indices,
        summary=summary,
    )
    validate_embedding_result(artifact)
    return artifact


def save_embedding_artifact(
    result: EmbeddingExtractionResult,
    output_path: str | Path,
    *,
    config_snapshot: dict[str, Any],
    model_metadata: VideoMAEModelMetadata | dict[str, Any],
    extraction_options: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Embedding output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    model_metadata_dict = (
        model_metadata.to_dict()
        if isinstance(model_metadata, VideoMAEModelMetadata)
        else dict(model_metadata)
    )
    summary = EmbeddingExtractionSummary(
        **{
            **result.summary.to_dict(),
            "embeddings_shape": tuple(result.summary.embeddings_shape),
            "labels_shape": None
            if result.summary.labels_shape is None
            else tuple(result.summary.labels_shape),
            "output_path": str(path),
        }
    )
    artifact = {
        "format_version": 1,
        "embeddings": result.embeddings,
        "label_ids": result.label_ids,
        "video_ids": result.video_ids,
        "sample_metadata": result.sample_metadata,
        "frame_indices": result.frame_indices,
        "config": config_snapshot,
        "model_metadata": model_metadata_dict,
        "extraction_options": extraction_options or {},
        "summary": summary.to_dict(),
    }
    validate_embedding_artifact(artifact)
    torch.save(artifact, path)
    return artifact


def load_embedding_artifact(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location="cpu")
    validate_embedding_artifact(artifact)
    return artifact


def validate_embedding_result(result: EmbeddingExtractionResult) -> None:
    artifact = {
        "embeddings": result.embeddings,
        "label_ids": result.label_ids,
        "video_ids": result.video_ids,
        "sample_metadata": result.sample_metadata,
        "frame_indices": result.frame_indices,
    }
    validate_embedding_artifact(artifact)


def validate_embedding_artifact(artifact: dict[str, Any]) -> None:
    embeddings = artifact.get("embeddings")
    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("Embedding artifact must contain a tensor at key 'embeddings'")
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must have shape [N, D], got {tuple(embeddings.shape)}")
    if not torch.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or Inf values")

    sample_count = embeddings.shape[0]
    video_ids = artifact.get("video_ids")
    if not isinstance(video_ids, list) or len(video_ids) != sample_count:
        raise ValueError("video_ids must be a list with one item per embedding")

    label_ids = artifact.get("label_ids")
    if label_ids is not None:
        if not isinstance(label_ids, torch.Tensor):
            raise TypeError("label_ids must be a tensor or None")
        if label_ids.shape != (sample_count,):
            raise ValueError(
                f"label_ids must have shape [{sample_count}], got {tuple(label_ids.shape)}"
            )

    sample_metadata = artifact.get("sample_metadata")
    if not isinstance(sample_metadata, list) or len(sample_metadata) != sample_count:
        raise ValueError("sample_metadata must be a list with one item per embedding")

    frame_indices = artifact.get("frame_indices")
    if not isinstance(frame_indices, torch.Tensor) or frame_indices.ndim != 2:
        raise ValueError("frame_indices must be a rank-2 tensor")
    if frame_indices.shape[0] != sample_count:
        raise ValueError("frame_indices must have one row per embedding")


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
) -> SSV2ClipDataset | Subset[dict[str, Any]]:
    transform = VideoMAEClipTransform(
        image_size=image_size,
        processor_checkpoint=processor_checkpoint,
        local_files_only=local_files_only,
    )
    dataset: SSV2ClipDataset | Subset[dict[str, Any]] = SSV2ClipDataset(
        index_path,
        num_frames=num_frames,
        sampling_strategy=sampling_strategy,
        transform=transform,
        perturbation=build_video_perturbation(perturbation_config),
    )
    if limit is not None:
        dataset = Subset(dataset, range(min(limit, len(dataset))))
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and save split-level VideoMAE embeddings."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ssv2_videomae_smoke.json"),
        help="Experiment config with dataset, preprocessing, and model settings.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=None,
        help="Normalized SSV2 JSONL index. Defaults to data/ssv2/index/{split}.jsonl.",
    )
    parser.add_argument("--split", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--processor-checkpoint",
        default=None,
        help="Optional HF image processor checkpoint. Omit to use VideoMAE defaults.",
    )
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
        help="Optional deterministic frame-level perturbation before VideoMAE preprocessing.",
    )
    parser.add_argument("--perturbation-seed", type=int, default=0)
    parser.add_argument(
        "--perturbation-frame-index",
        type=int,
        default=None,
        help="Source frame for single_frame perturbation. Defaults to the center frame.",
    )
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

    dataset = build_extraction_dataset(
        index_path=index_path,
        num_frames=config.num_frames,
        sampling_strategy=config.sampling_strategy,
        image_size=config.image_size,
        processor_checkpoint=args.processor_checkpoint,
        local_files_only=args.local_files_only,
        limit=args.limit,
        perturbation_config=perturbation_config,
    )
    model, model_metadata = load_videomae_model(
        checkpoint,
        device=str(resolved_device),
        local_files_only=args.local_files_only,
    )
    result = extract_embeddings(
        model=model,
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        device=resolved_device,
        embedding_type=config.embedding_type,  # type: ignore[arg-type]
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
        model_metadata=model_metadata,
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


if __name__ == "__main__":
    raise SystemExit(main())
