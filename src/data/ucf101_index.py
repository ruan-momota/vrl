"""UCF101 directory-tree to normalized-subset-index conversion.

UCF101 is distributed (once organized locally) as
``<split>/<class>/<video>.avi`` with ``split`` in ``{train, test}``. Unlike
Kinetics, UCF101 ships its own held-out split, so this module reads the raw
``train``/``test`` directories directly instead of re-splitting a flat pool:
raw ``train`` videos become ``split="train"`` records and raw ``test`` videos
become ``split="heldout"`` records (tagged with ``original_split="test"``),
matching the "UCF101 `test` is reported as held-out" convention used
elsewhere in this repo. Output artifacts mirror the other dataset cells:

- ``label_mapping.json``
- ``train.jsonl`` / ``heldout.jsonl``
- ``selected_samples.jsonl``
- ``summary.json`` (includes a ``path_audit`` block)
- ``decode_failures.jsonl``

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


VIDEO_EXTENSIONS = (".avi", ".mp4", ".webm", ".mkv")
RAW_SPLIT_DIRS = ("train", "test")


def discover_class_videos(
    video_root: Path,
    *,
    video_extensions: Sequence[str] = VIDEO_EXTENSIONS,
) -> dict[str, dict[str, list[Path]]]:
    """Group videos by raw split directory, then by class.

    Expects ``<video_root>/<train|test>/<class>/<video>`` and mirrors the
    Kinetics indexer's parent-directory-is-class-label convention within
    each raw split.
    """
    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")

    normalized_exts = {
        ext if ext.startswith(".") else f".{ext}" for ext in video_extensions
    }
    by_split: dict[str, dict[str, list[Path]]] = {
        split: defaultdict(list) for split in RAW_SPLIT_DIRS
    }
    for split in RAW_SPLIT_DIRS:
        split_root = video_root / split
        if not split_root.exists():
            continue
        for path in sorted(split_root.rglob("*")):
            if path.is_file() and path.suffix.lower() in normalized_exts:
                by_split[split][path.parent.name].append(path)

    return {
        split: {class_name: sorted(paths) for class_name, paths in sorted(classes.items())}
        for split, classes in by_split.items()
    }


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
    original_split: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "video_id": _video_id(path),
        "video_path": str(path),
        "label_id": label_id,
        "label_name": label_name,
        "split": split,
        "subset_id": subset_id,
        "source_dataset": "ucf101",
    }
    if original_split is not None:
        record["original_split"] = original_split
    return record


def _select(
    paths: Sequence[Path],
    *,
    per_class: int | None,
    rng: random.Random,
) -> list[Path]:
    shuffled = list(paths)
    rng.shuffle(shuffled)
    selected = shuffled if per_class is None else shuffled[:per_class]
    return sorted(selected, key=_video_id)


def audit_paths(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    missing = [
        record["video_path"] for record in records if not Path(record["video_path"]).exists()
    ]
    return {"missing_path_count": len(missing), "missing_paths": missing}


def probe_decode_failures(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Attempt to decode every record's video, returning failure entries.

    This is a separate, opt-in pass (``--probe-decode``) rather than part of
    the default index build, since it requires real, decodable video files
    and is comparatively slow.
    """
    from src.video.io import VideoReadError, read_video_frames

    failures: list[dict[str, Any]] = []
    for record in records:
        try:
            read_video_frames(record["video_path"], video_id=record.get("video_id"))
        except VideoReadError as error:
            failures.append(
                {
                    "video_id": record.get("video_id"),
                    "video_path": record["video_path"],
                    "error": str(error),
                }
            )
    return failures


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
    seed: int = 0,
    video_extensions: Sequence[str] = VIDEO_EXTENSIONS,
    probe_decode: bool = False,
) -> dict[str, Any]:
    """Build a normalized UCF101 subset index from a train/test class tree."""
    rng = random.Random(seed)
    by_split = discover_class_videos(video_root, video_extensions=video_extensions)
    train_by_class = by_split["train"]
    test_by_class = by_split["test"]
    if not train_by_class and not test_by_class:
        raise ValueError(
            f"No videos with {tuple(video_extensions)} found under {video_root}")

    class_names = sorted(set(train_by_class) | set(test_by_class))
    if max_classes is not None:
        class_names = class_names[:max_classes]
    label_mapping = build_label_mapping(class_names)
    class_to_id = label_mapping["class_to_id"]

    train_records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []
    per_class_summary: list[dict[str, Any]] = []
    for class_name in class_names:
        label_id = class_to_id[class_name]
        train_paths = _select(
            train_by_class.get(class_name, []), per_class=train_per_class, rng=rng)
        heldout_paths = _select(
            test_by_class.get(class_name, []), per_class=heldout_per_class, rng=rng)

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
                    original_split="test",
                )
            )
        per_class_summary.append(
            {
                "label_id": label_id,
                "label_name": class_name,
                "available_train": len(train_by_class.get(class_name, [])),
                "available_test": len(test_by_class.get(class_name, [])),
                "train": len(train_paths),
                "heldout": len(heldout_paths),
            }
        )

    train_records.sort(key=lambda record: (record["label_id"], record["video_id"]))
    heldout_records.sort(key=lambda record: (record["label_id"], record["video_id"]))
    selected_records = train_records + heldout_records

    write_json(output_dir / "label_mapping.json", label_mapping)
    write_jsonl(output_dir / "train.jsonl", train_records)
    write_jsonl(output_dir / "heldout.jsonl", heldout_records)
    write_jsonl(output_dir / "selected_samples.jsonl", selected_records)

    decode_failures = probe_decode_failures(
        selected_records) if probe_decode else []
    write_jsonl(output_dir / "decode_failures.jsonl", decode_failures)

    def _per_class_counts(records: Sequence[Mapping[str, Any]]) -> dict[int, int]:
        counts: dict[int, int] = defaultdict(int)
        for record in records:
            counts[int(record["label_id"])] += 1
        return dict(counts)

    train_counts = _per_class_counts(train_records)
    heldout_counts = _per_class_counts(heldout_records)
    summary = {
        "dataset": "ucf101",
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
        "path_audit": audit_paths(selected_records),
        "decode_failures": {
            "probed": probe_decode,
            "failure_count": len(decode_failures),
        },
        "per_class": per_class_summary,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a normalized UCF101 subset index from a train/test class-directory tree."
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        required=True,
        help="Root containing <train|test>/<class>/<video> trees.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the normalized index artifacts are written.",
    )
    parser.add_argument("--subset-id", required=True,
                        help="Subset identifier, e.g. c50_train100_heldout30.")
    parser.add_argument("--train-per-class", type=int, default=None)
    parser.add_argument("--heldout-per-class", type=int, default=None)
    parser.add_argument("--max-classes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--probe-decode",
        action="store_true",
        help="Attempt to decode every selected video and record failures.",
    )
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
        seed=args.seed,
        probe_decode=args.probe_decode,
    )
    print(
        f"classes={summary['class_count']} "
        f"train={summary['splits']['train']['count']} "
        f"heldout={summary['splits']['heldout']['count']} "
        f"missing_paths={summary['path_audit']['missing_path_count']}"
    )
    print(f"wrote index files under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
