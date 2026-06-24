"""Dataset- and model-agnostic embedding extraction core."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.artifacts import (
    EmbeddingExtractionResult,
    EmbeddingExtractionSummary,
    validate_embedding_result,
)
from src.data.indexed_dataset import collate_video_batch
from src.models.base import VideoEncoder


def resolve_device(device: str | torch.device) -> torch.device:
    """Resolve the project-wide ``auto`` device spelling without model coupling."""
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def extract_embeddings(
    *,
    encoder: VideoEncoder,
    dataset: Dataset[dict[str, Any]],
    batch_size: int,
    num_workers: int,
    device: str | torch.device,
    pin_memory: bool = False,
    show_progress: bool = True,
) -> EmbeddingExtractionResult:
    """Extract one finite ``[B, D]`` embedding for every indexed video clip."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_video_batch,
    )
    resolved_device = resolve_device(device)
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
        result_embeddings = encoder.encode(
            batch["pixel_values"],
            device=resolved_device,
        )
        if result_embeddings.ndim != 2:
            raise ValueError(
                "Video encoder must return embeddings with shape [B, D], "
                f"got {tuple(result_embeddings.shape)}"
            )
        if not torch.isfinite(result_embeddings).all():
            raise RuntimeError("Video encoder embeddings contain NaN or Inf values")
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        batch_seconds.append(time.perf_counter() - batch_start)

        embeddings.append(result_embeddings.detach().cpu().to(dtype=torch.float32))
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
    result = EmbeddingExtractionResult(
        embeddings=embedding_tensor,
        label_ids=label_tensor,
        video_ids=video_ids,
        sample_metadata=sample_metadata,
        frame_indices=frame_indices,
        summary=summary,
    )
    validate_embedding_result(result)
    return result
