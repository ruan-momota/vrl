#!/usr/bin/env python3
"""Copy a limited subset of videos listed by label split files.

The default layout matches the local SSV2 sample:

  data/ssv2/labels/train.json
  data/ssv2/labels/validation.json
  data/ssv2/labels/test.json
  data/ssv2/videos/<id>.webm

Output videos are copied into one directory per label split:

  <output_dir>/train/<id>.webm
  <output_dir>/validation/<id>.webm
  <output_dir>/test/<id>.webm
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_LABEL_FILES = ("train.json", "validation.json", "test.json")


@dataclass(frozen=True)
class CopyStats:
    split: str
    requested: int
    copied: int
    skipped_existing: int
    missing: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy videos referenced by label JSON files into split-specific "
            "directories, with an optional limit per split."
        )
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/ssv2/labels"),
        help="Directory containing split JSON files. Default: data/ssv2/labels",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("data/ssv2/videos"),
        help="Directory containing source videos named like <id>.<ext>. Default: data/ssv2/videos",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination root. The script creates one subdirectory per split.",
    )
    parser.add_argument(
        "--label-files",
        nargs="+",
        default=list(DEFAULT_LABEL_FILES),
        help=(
            "Split label JSON files to process, relative to --labels-dir. "
            "Default: train.json validation.json test.json"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of videos to copy from each split. Default: copy all.",
    )
    parser.add_argument(
        "--video-ext",
        default=".webm",
        help="Video filename extension, with or without leading dot. Default: .webm",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace destination files when they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without creating directories or copying files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any requested videos are missing.",
    )
    return parser.parse_args()


def normalize_ext(video_ext: str) -> str:
    return video_ext if video_ext.startswith(".") else f".{video_ext}"


def load_video_ids(label_path: Path) -> list[str]:
    with label_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{label_path} must contain a JSON list of records")

    video_ids: list[str] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict) or "id" not in item:
            raise ValueError(f"{label_path} item {index} does not contain an id field")
        video_ids.append(str(item["id"]))

    return video_ids


def limited(video_ids: Iterable[str], limit: int | None) -> list[str]:
    if limit is None:
        return list(video_ids)
    if limit < 0:
        raise ValueError("--limit must be non-negative")
    return list(video_ids)[:limit]


def copy_split_videos(
    *,
    split: str,
    video_ids: list[str],
    videos_dir: Path,
    output_dir: Path,
    video_ext: str,
    overwrite: bool,
    dry_run: bool,
) -> CopyStats:
    split_dir = output_dir / split
    copied = 0
    skipped_existing = 0
    missing = 0

    if not dry_run:
        split_dir.mkdir(parents=True, exist_ok=True)

    for video_id in video_ids:
        source = videos_dir / f"{video_id}{video_ext}"
        destination = split_dir / source.name

        if not source.exists():
            missing += 1
            print(f"[missing] {split}: {source}")
            continue

        if destination.exists() and not overwrite:
            skipped_existing += 1
            print(f"[exists]  {split}: {destination}")
            continue

        copied += 1
        print(f"[copy]    {split}: {source} -> {destination}")
        if not dry_run:
            shutil.copy2(source, destination)

    return CopyStats(
        split=split,
        requested=len(video_ids),
        copied=copied,
        skipped_existing=skipped_existing,
        missing=missing,
    )


def main() -> int:
    args = parse_args()
    labels_dir = args.labels_dir
    videos_dir = args.videos_dir
    output_dir = args.output_dir
    video_ext = normalize_ext(args.video_ext)

    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory does not exist: {labels_dir}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory does not exist: {videos_dir}")

    stats: list[CopyStats] = []
    for label_file in args.label_files:
        label_path = labels_dir / label_file
        split = Path(label_file).stem

        if not label_path.exists():
            raise FileNotFoundError(f"Label file does not exist: {label_path}")

        video_ids = limited(load_video_ids(label_path), args.limit)
        stats.append(
            copy_split_videos(
                split=split,
                video_ids=video_ids,
                videos_dir=videos_dir,
                output_dir=output_dir,
                video_ext=video_ext,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )

    print("\nSummary")
    total_missing = 0
    for item in stats:
        total_missing += item.missing
        print(
            f"{item.split}: requested={item.requested}, copied={item.copied}, "
            f"skipped_existing={item.skipped_existing}, missing={item.missing}"
        )

    if args.strict and total_missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
