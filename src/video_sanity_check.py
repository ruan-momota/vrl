from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

from src.ssv2_dataset import SSV2ClipDataset, collate_video_batch
from src.videomae_preprocessing import VideoMAEClipTransform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-batch video decode and VideoMAE preprocessing sanity check."
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/ssv2/index/debug_train.jsonl"),
        help="Normalized SSV2 JSONL index to read.",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--sampling-strategy",
        default="deterministic_center_clip",
        choices=("deterministic_center_clip", "deterministic_uniform"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--processor-checkpoint",
        default=None,
        help="Optional HF image processor checkpoint. Omit to use VideoMAE defaults.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not attempt network access when --processor-checkpoint is set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of leading indexed samples to expose to the DataLoader.",
    )
    return parser.parse_args()


def run_sanity_check(args: argparse.Namespace) -> dict[str, Any]:
    transform = VideoMAEClipTransform(
        image_size=args.image_size,
        processor_checkpoint=args.processor_checkpoint,
        local_files_only=args.local_files_only,
    )
    dataset: SSV2ClipDataset | Subset[dict[str, Any]] = SSV2ClipDataset(
        args.index_path,
        num_frames=args.num_frames,
        sampling_strategy=args.sampling_strategy,
        transform=transform,
    )
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_video_batch,
    )
    batch = next(iter(loader))
    pixel_values = batch["pixel_values"]
    return {
        "index_path": str(args.index_path),
        "dataset_size": len(dataset),
        "pixel_values_shape": list(pixel_values.shape),
        "pixel_values_dtype": str(pixel_values.dtype),
        "pixel_values_min": float(pixel_values.min().item()),
        "pixel_values_max": float(pixel_values.max().item()),
        "pixel_values_finite": bool(torch.isfinite(pixel_values).all().item()),
        "label_ids": None
        if batch["label_ids"] is None
        else batch["label_ids"].tolist(),
        "video_ids": batch["video_ids"],
        "frame_indices_shape": list(batch["frame_indices"].shape),
        "first_frame_indices": batch["frame_indices"][0].tolist(),
    }


def main() -> int:
    summary = run_sanity_check(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
