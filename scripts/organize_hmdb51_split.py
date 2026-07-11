#!/usr/bin/env python3
"""Organize flat, per-class HMDB51 clips into a <split>/<class>/ tree.

HMDB51 ships as one .avi archive per class plus a separate split-files
archive (``testTrainMulti_7030_splits``) with one text file per class per
fold, e.g. ``brush_hair_test_split1.txt``. Each line is
``<video_filename>.avi <label>`` where label is ``0`` (unused), ``1``
(train), or ``2`` (test).

This script reads one fold's split files and copies (or symlinks) each
class's clips into ``<output_root>/<train|test>/<class>/<video>.avi``, the
layout ``src.data.hmdb51_index`` expects.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

SPLIT_LABELS = {"1": "train", "2": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clips-root",
        type=Path,
        required=True,
        help="Directory containing one subfolder of .avi files per class "
        "(the extracted contents of hmdb51_org.rar).",
    )
    parser.add_argument(
        "--splits-root",
        type=Path,
        required=True,
        help="Directory containing the *_test_split<fold>.txt files "
        "(the extracted contents of test_train_splits.rar).",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument(
        "--link",
        action="store_true",
        help="Symlink instead of copying (saves disk space).",
    )
    return parser.parse_args()


def organize(
    *,
    clips_root: Path,
    splits_root: Path,
    output_root: Path,
    fold: int,
    link: bool,
) -> dict[str, int]:
    counts = {"train": 0, "test": 0, "skipped": 0}
    split_files = sorted(splits_root.glob(f"*_test_split{fold}.txt"))
    if not split_files:
        raise FileNotFoundError(
            f"No *_test_split{fold}.txt files found under {splits_root}")

    for split_file in split_files:
        class_name = split_file.name[: -len(f"_test_split{fold}.txt")]
        class_clip_dir = clips_root / class_name
        if not class_clip_dir.exists():
            raise FileNotFoundError(
                f"Split file {split_file.name} references class "
                f"{class_name!r} but {class_clip_dir} does not exist")

        # Some class .rar archives extract into a nested <class>/<class>/
        # subfolder rather than dropping .avi files directly into
        # <class>/, so search recursively rather than assuming a flat dir.
        clips_by_name = {path.name: path for path in class_clip_dir.rglob("*.avi")}

        for line in split_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            video_name, label = line.rsplit(maxsplit=1)
            split = SPLIT_LABELS.get(label)
            if split is None:
                counts["skipped"] += 1
                continue

            source = clips_by_name.get(video_name)
            if source is None:
                raise FileNotFoundError(
                    f"Missing clip: {video_name} under {class_clip_dir}")

            dest_dir = output_root / split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / video_name
            if dest.exists():
                continue
            if link:
                dest.symlink_to(source.resolve())
            else:
                shutil.copy2(source, dest)
            counts[split] += 1

    return counts


def main() -> int:
    args = parse_args()
    counts = organize(
        clips_root=args.clips_root,
        splits_root=args.splits_root,
        output_root=args.output_root,
        fold=args.fold,
        link=args.link,
    )
    print(f"train={counts['train']} test={counts['test']} skipped={counts['skipped']}")
    print(f"wrote tree under {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
