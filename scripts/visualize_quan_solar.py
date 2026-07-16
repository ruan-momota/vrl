#!/usr/bin/env python3
"""Visualize one sampled frame under RGB quantization and solarization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from src.data.indexed_dataset import load_index_jsonl
from src.video.io import read_sampled_clip
from src.video.perturbations import VideoPerturbationConfig, apply_video_perturbation


PERTURBATIONS = (
    (
        "rgb-quantization-low",
        "RGB quantization low (16 levels)",
        VideoPerturbationConfig(name="rgb_quantization", rgb_levels=16),
    ),
    (
        "rgb-quantization-mid",
        "RGB quantization mid (8 levels)",
        VideoPerturbationConfig(name="rgb_quantization", rgb_levels=8),
    ),
    (
        "rgb-quantization-high",
        "RGB quantization high (4 levels)",
        VideoPerturbationConfig(name="rgb_quantization", rgb_levels=4),
    ),
    (
        "solarization-low",
        "Solarization low (threshold 192)",
        VideoPerturbationConfig(name="solarization", solarization_threshold=192),
    ),
    (
        "solarization-mid",
        "Solarization mid (threshold 128)",
        VideoPerturbationConfig(name="solarization", solarization_threshold=128),
    ),
    (
        "solarization-high",
        "Solarization high (threshold 64)",
        VideoPerturbationConfig(name="solarization", solarization_threshold=64),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode one deterministic clip and save the same frame before and after "
            "RGB quantization and solarization. No model is loaded."
        )
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=Path("data/ssv2/index/validation.jsonl"),
        help="Normalized JSONL index. Default: SSV2 validation index.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based row in the index when --video-id is omitted. Default: 0.",
    )
    parser.add_argument(
        "--video-id",
        default=None,
        help="Optional video ID; takes precedence over --sample-index.",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--frame-position",
        type=int,
        default=None,
        help="Position in the sampled clip. Default: middle frame.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/examples/quan_solar"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_index_jsonl(args.index_path)
    record = _select_record(records, sample_index=args.sample_index, video_id=args.video_id)
    video_id = str(record["video_id"])
    clip = read_sampled_clip(
        record["video_path"],
        num_frames=args.num_frames,
        sampling_strategy="deterministic_center_clip",
        video_id=video_id,
    )
    frame_position = args.num_frames // 2 if args.frame_position is None else args.frame_position
    if not 0 <= frame_position < args.num_frames:
        raise ValueError(f"frame_position must be in [0, {args.num_frames - 1}]")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = _safe_name(video_id)
    original_frame = clip.frames[frame_position]
    original_path = output_dir / f"{prefix}_original.png"
    Image.fromarray(original_frame).save(original_path)

    panels: dict[str, tuple[str, np.ndarray]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    output_paths: dict[str, str] = {"original": str(original_path)}
    for artifact_label, display_label, config in PERTURBATIONS:
        result = apply_video_perturbation(clip.frames, config, video_id=video_id)
        frame = result.frames[frame_position]
        path = output_dir / f"{prefix}_{artifact_label}.png"
        Image.fromarray(frame).save(path)
        panels[artifact_label] = (display_label, frame)
        output_paths[artifact_label] = str(path)
        difference = np.abs(frame.astype(np.int16) - original_frame.astype(np.int16))
        metrics[artifact_label] = {
            "mean_absolute_rgb_difference": float(difference.mean()),
            "normalized_mean_absolute_rgb_difference": float(difference.mean() / 255.0),
            "changed_value_ratio": float(np.mean(difference != 0)),
            "perturbation": result.metadata,
        }

    comparison_path = output_dir / f"{prefix}_comparison.png"
    write_comparison(original_frame, panels, comparison_path)
    output_paths["comparison"] = str(comparison_path)

    metadata = {
        "index_path": str(args.index_path),
        "sample_index": args.sample_index,
        "video_id": video_id,
        "video_path": str(record["video_path"]),
        "label_id": record.get("label_id"),
        "label_name": record.get("label_name"),
        "num_frames": args.num_frames,
        "sampling_strategy": clip.sampling_strategy,
        "sampled_frame_indices": list(clip.frame_indices),
        "frame_position": frame_position,
        "source_frame_index": clip.frame_indices[frame_position],
        "frame_shape": list(original_frame.shape),
        "metrics": metrics,
        "outputs": output_paths,
    }
    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"video_id={video_id}")
    print(f"comparison={comparison_path}")
    print(f"metadata={metadata_path}")
    return 0


def _select_record(
    records: tuple[dict[str, Any], ...],
    *,
    sample_index: int,
    video_id: str | None,
) -> dict[str, Any]:
    if not records:
        raise ValueError("index contains no samples")
    if video_id is not None:
        for record in records:
            if str(record["video_id"]) == video_id:
                return record
        raise ValueError(f"video_id not found in index: {video_id}")
    if not 0 <= sample_index < len(records):
        raise ValueError(f"sample_index must be in [0, {len(records) - 1}]")
    return records[sample_index]


def write_comparison(
    original: np.ndarray,
    panels: dict[str, tuple[str, np.ndarray]],
    output_path: Path,
) -> None:
    rows = (
        (
            ("Original", original),
            panels["rgb-quantization-low"],
            panels["rgb-quantization-mid"],
            panels["rgb-quantization-high"],
        ),
        (
            ("Original", original),
            panels["solarization-low"],
            panels["solarization-mid"],
            panels["solarization-high"],
        ),
    )
    frame_height, frame_width = original.shape[:2]
    label_height = 28
    canvas = Image.new(
        "RGB",
        (frame_width * 4, (frame_height + label_height) * 2),
        color="white",
    )
    draw = ImageDraw.Draw(canvas)
    for row_index, row in enumerate(rows):
        y = row_index * (frame_height + label_height)
        for column_index, (label, frame) in enumerate(row):
            x = column_index * frame_width
            draw.text((x + 8, y + 7), label, fill="black")
            canvas.paste(Image.fromarray(frame), (x, y + label_height))
    canvas.save(output_path)


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "-" for character in value)


if __name__ == "__main__":
    raise SystemExit(main())
