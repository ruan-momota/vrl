from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.config import ExperimentConfig
from src.ssv2_dataset import SSV2ClipDataset, collate_video_batch
from src.videomae_model import (
    forward_videomae_embeddings,
    load_videomae_model,
    resolve_device,
)
from src.videomae_preprocessing import VideoMAEClipTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VideoMAE model loading and embedding sanity checks."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ssv2_videomae_smoke.json"),
        help="Experiment config containing checkpoint and input shape settings.",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/ssv2/index/debug_train.jsonl"),
        help="Normalized SSV2 JSONL index for the real-batch check.",
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--skip-real-batch", action="store_true")
    return parser.parse_args()


def run_sanity_check(args: argparse.Namespace) -> dict[str, Any]:
    config = ExperimentConfig.from_file(args.config)
    checkpoint = args.checkpoint or config.model_checkpoint
    device_name = args.device or config.device
    batch_size = args.batch_size or config.batch_size
    num_workers = config.num_workers if args.num_workers is None else args.num_workers
    device = resolve_device(device_name)

    model, model_metadata = load_videomae_model(
        checkpoint,
        device=str(device),
        local_files_only=args.local_files_only,
    )
    summary: dict[str, Any] = {
        "config_path": str(args.config),
        "checkpoint": checkpoint,
        "device": str(device),
        "model_training": bool(model.training),
        "model_metadata": model_metadata.to_dict(),
    }

    if not args.skip_random:
        random_pixel_values = torch.randn(
            batch_size,
            config.num_frames,
            3,
            config.image_size,
            config.image_size,
            dtype=torch.float32,
        )
        random_result = forward_videomae_embeddings(
            model,
            random_pixel_values,
            embedding_type="last_hidden_state_mean_pool",
            device=device,
        )
        summary["random_forward"] = summarize_forward_result(
            random_result.embeddings,
            pixel_values=random_pixel_values,
            last_hidden_state_shape=random_result.last_hidden_state_shape,
        )

    if not args.skip_real_batch:
        transform = VideoMAEClipTransform(image_size=config.image_size)
        dataset: SSV2ClipDataset | Subset[dict[str, Any]] = SSV2ClipDataset(
            args.index_path,
            num_frames=config.num_frames,
            sampling_strategy=config.sampling_strategy,
            transform=transform,
        )
        if args.limit is not None:
            dataset = Subset(dataset, range(min(args.limit, len(dataset))))

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_video_batch,
        )
        batch = next(iter(loader))
        real_result = forward_videomae_embeddings(
            model,
            batch["pixel_values"],
            embedding_type="last_hidden_state_mean_pool",
            device=device,
        )
        summary["real_batch_forward"] = {
            **summarize_forward_result(
                real_result.embeddings,
                pixel_values=batch["pixel_values"],
                last_hidden_state_shape=real_result.last_hidden_state_shape,
            ),
            "index_path": str(args.index_path),
            "dataset_size": len(dataset),
            "video_ids": batch["video_ids"],
            "label_ids": None
            if batch["label_ids"] is None
            else batch["label_ids"].tolist(),
        }

    return summary


def summarize_forward_result(
    embeddings: torch.Tensor,
    *,
    pixel_values: torch.Tensor,
    last_hidden_state_shape: tuple[int, ...],
) -> dict[str, Any]:
    embeddings_cpu = embeddings.detach().cpu()
    return {
        "pixel_values_shape": list(pixel_values.shape),
        "last_hidden_state_shape": list(last_hidden_state_shape),
        "embeddings_shape": list(embeddings_cpu.shape),
        "embeddings_dtype": str(embeddings_cpu.dtype),
        "embeddings_finite": bool(torch.isfinite(embeddings_cpu).all().item()),
        "embeddings_min": float(embeddings_cpu.min().item()),
        "embeddings_max": float(embeddings_cpu.max().item()),
    }


def main() -> int:
    summary = run_sanity_check(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
