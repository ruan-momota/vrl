"""Kinetics directory-tree to normalized-subset-index conversion.

Kinetics is distributed as ``<split>/<class>/<video>.mp4`` (or a flat
``<class>/<video>.mp4`` tree). The parent-folder name is the class label, which
matches the ``index_videos`` convention used in the original Kinetics notebook.
This module turns that tree into the same normalized JSONL artifacts the other
dataset cells use:

- ``label_mapping.json``
- ``train.jsonl`` / ``heldout.jsonl``
- ``selected_samples.jsonl``
- ``summary.json``

Each JSONL record is a canonical ``VideoRecord`` mapping understood by
``src.data.indexed_dataset.IndexedVideoDataset``.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".avi")


def discover_class_videos(
    video_root: Path,
    *,
    video_extensions: Sequence[str] = VIDEO_EXTENSIONS,
) -> dict[str, list[Path]]:
    """Group videos by their parent-directory class name.

    Mirrors the notebook's ``index_videos`` indexing: the directory immediately
    containing a video file is treated as its class label.
    """
    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")

    normalized_exts = {
        ext if ext.startswith(".") else f".{ext}" for ext in video_extensions
    }
    by_class: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(video_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in normalized_exts:
            by_class[path.parent.name].append(path)
    return {class_name: sorted(paths) for class_name, paths in sorted(by_class.items())}


def build_label_mapping(class_names: Sequence[str]) -> dict[str, Any]:
    classes = sorted(class_names)
    return {
        "classes": classes,
        "class_to_id": {name: index for index, name in enumerate(classes)},
    }


def _video_id(path: Path) -> str:
    return path.stem


def _record(
    *,
    path: Path,
    label_id: int,
    label_name: str,
    split: str,
    subset_id: str,
) -> dict[str, Any]:
    return {
        "video_id": _video_id(path),
        "video_path": str(path),
        "label_id": label_id,
        "label_name": label_name,
        "split": split,
        "subset_id": subset_id,
        "source_dataset": "kinetics",
    }


def split_class_videos(
    paths: Sequence[Path],
    *,
    train_per_class: int | None,
    heldout_per_class: int | None,
    val_frac: float,
    rng: random.Random,
) -> tuple[list[Path], list[Path]]:
    """Stratified per-class train/held-out split.

    When ``train_per_class``/``heldout_per_class`` are given they cap the counts
    directly; otherwise ``val_frac`` reproduces the notebook's behaviour of
    holding out ``max(1, floor(n * val_frac))`` clips per class.
    """
    shuffled = list(paths)
    rng.shuffle(shuffled)

    if heldout_per_class is not None:
        heldout = shuffled[:heldout_per_class]
        remaining = shuffled[heldout_per_class:]
    else:
        n_val = max(1, int(len(shuffled) * val_frac))
        heldout = shuffled[:n_val]
        remaining = shuffled[n_val:]

    train = remaining if train_per_class is None else remaining[:train_per_class]
    return sorted(train, key=_video_id), sorted(heldout, key=_video_id)


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def build_subset_index(
    *,
    video_root: Path,
    output_dir: Path,
    subset_id: str,
    train_per_class: int | None = None,
    heldout_per_class: int | None = None,
    max_classes: int | None = None,
    val_frac: float = 0.2,
    seed: int = 0,
    video_extensions: Sequence[str] = VIDEO_EXTENSIONS,
) -> dict[str, Any]:
    """Build a normalized Kinetics subset index from a class-directory tree."""
    rng = random.Random(seed)
    by_class = discover_class_videos(
        video_root, video_extensions=video_extensions)
    if not by_class:
        raise ValueError(
            f"No videos with {tuple(video_extensions)} found under {video_root}")

    class_names = sorted(by_class)
    if max_classes is not None:
        class_names = class_names[:max_classes]
    label_mapping = build_label_mapping(class_names)
    class_to_id = label_mapping["class_to_id"]

    train_records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []
    per_class_summary: list[dict[str, Any]] = []
    for class_name in class_names:
        label_id = class_to_id[class_name]
        train_paths, heldout_paths = split_class_videos(
            by_class[class_name],
            train_per_class=train_per_class,
            heldout_per_class=heldout_per_class,
            val_frac=val_frac,
            rng=rng,
        )
        for path in train_paths:
            train_records.append(
                _record(
                    path=path,
                    label_id=label_id,
                    label_name=class_name,
                    split="train",
                    subset_id=subset_id,
                )
            )
        for path in heldout_paths:
            heldout_records.append(
                _record(
                    path=path,
                    label_id=label_id,
                    label_name=class_name,
                    split="heldout",
                    subset_id=subset_id,
                )
            )
        per_class_summary.append(
            {
                "label_id": label_id,
                "label_name": class_name,
                "available": len(by_class[class_name]),
                "train": len(train_paths),
                "heldout": len(heldout_paths),
            }
        )

    train_records.sort(key=lambda record: (
        record["label_id"], record["video_id"]))
    heldout_records.sort(key=lambda record: (
        record["label_id"], record["video_id"]))
    selected_records = train_records + heldout_records

    write_json(output_dir / "label_mapping.json", label_mapping)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "heldout.jsonl", heldout_records)
    write_jsonl(output_dir / "selected_samples.jsonl", selected_records)

    def _per_class_counts(records: Sequence[Mapping[str, Any]]) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for record in records:
            counts[int(record["label_id"])] += 1
        return dict(counts)

    train_counts = _per_class_counts(train_records)
    heldout_counts = _per_class_counts(heldout_records)
    summary = {
        "dataset": "kinetics",
        "subset_id": subset_id,
        "video_root": str(video_root),
        "seed": seed,
        "class_count": len(class_names),
        "splits": {
            "train": {
                "count": len(train_records),
                "per_class_min": min(train_counts.values()) if train_counts else 0,
                "per_class_max": max(train_counts.values()) if train_counts else 0,
            },
            "heldout": {
                "count": len(heldout_records),
                "per_class_min": min(heldout_counts.values()) if heldout_counts else 0,
                "per_class_max": max(heldout_counts.values()) if heldout_counts else 0,
            },
        },
        "per_class": per_class_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized Kinetics subset index from a class-directory tree."
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
        help="Root containing <class>/<video> or <split>/<class>/<video> trees.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the normalized index artifacts are written.",
    )
    parser.add_argument("--subset-id", required=True,
                        help="Subset identifier, e.g. c20_train20.")
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--heldout-per-class", type=int, default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_subset_index(
        video_root=args.video_root,
        output_dir=args.output_dir,
        subset_id=args.subset_id,
        train_per_class=args.train_per_class,
        heldout_per_class=args.heldout_per_class,
        max_classes=args.max_classes,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(
        f"classes={summary['class_count']} "
        f"train={summary['splits']['train']['count']} "
        f"heldout={summary['splits']['heldout']['count']}"
    )
    print(f"wrote index files under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
