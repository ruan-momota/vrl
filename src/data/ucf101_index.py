"""UCF101 directory-to-normalized-index conversion for the selected C50 subset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm import tqdm

from src.video.io import FrameSamplingStrategy, read_sampled_clip


DEFAULT_SUBSET_ID = "c50_train100_heldout30"
DEFAULT_VIDEO_EXTS = (".avi",)


@dataclass(frozen=True)
class UCF101Sample:
    video_id: str
    video_path: str
    label_id: int
    label_name: str
    split: str
    subset_id: str
    source_dataset: str = "ucf101"
    original_split: str = "train"
    filename: str = ""
    class_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_subset_index(
    *,
    video_root: Path,
    output_dir: Path,
    subset_id: str = DEFAULT_SUBSET_ID,
    train_per_class: int = 100,
    heldout_per_class: int = 30,
    video_exts: Sequence[str] = DEFAULT_VIDEO_EXTS,
    decode_audit: bool = False,
    decode_progress: bool = True,
    num_frames: int = 16,
    sampling_strategy: FrameSamplingStrategy = "deterministic_center_clip",
    selection_note: str = "preselected directory snapshot",
) -> dict[str, Any]:
    """Write label mapping, normalized indexes, selected samples and summary."""
    video_root = Path(video_root)
    output_dir = Path(output_dir)
    normalized_exts = tuple(_normalize_ext(ext) for ext in video_exts)
    split_dirs = {
        "train": _class_dirs(video_root / "train"),
        "heldout": _class_dirs(video_root / "test"),
    }
    _validate_class_sets(split_dirs)

    class_names = sorted(split_dirs["train"])
    label_mapping = _label_mapping(subset_id, class_names)
    train_samples = _build_split_samples(
        split_dirs["train"],
        label_mapping=label_mapping["class_to_id"],
        split="train",
        original_split="train",
        subset_id=subset_id,
        expected_per_class=train_per_class,
        video_exts=normalized_exts,
    )
    heldout_samples = _build_split_samples(
        split_dirs["heldout"],
        label_mapping=label_mapping["class_to_id"],
        split="heldout",
        original_split="test",
        subset_id=subset_id,
        expected_per_class=heldout_per_class,
        video_exts=normalized_exts,
    )
    all_samples = tuple(train_samples) + tuple(heldout_samples)
    _validate_unique_video_ids(all_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "label_mapping.json", label_mapping)
    write_jsonl(output_dir / "train.jsonl", train_samples)
    write_jsonl(output_dir / "heldout.jsonl", heldout_samples)
    write_jsonl(output_dir / "selected_samples.jsonl", all_samples)

    decode_failures = (
        audit_decode_samples(
            all_samples,
            num_frames=num_frames,
            sampling_strategy=sampling_strategy,
            show_progress=decode_progress,
        )
        if decode_audit
        else []
    )
    write_jsonl(output_dir / "decode_failures.jsonl", decode_failures)

    summary = {
        "dataset": "ucf101",
        "subset_id": subset_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "video_root": str(video_root),
        "video_exts": list(normalized_exts),
        "selection": {
            "note": selection_note,
            "class_order": "alphabetical_by_class_directory_name",
            "train_per_class": train_per_class,
            "heldout_per_class": heldout_per_class,
        },
        "files": {
            "label_mapping": str(output_dir / "label_mapping.json"),
            "train_index": str(output_dir / "train.jsonl"),
            "heldout_index": str(output_dir / "heldout.jsonl"),
            "selected_samples": str(output_dir / "selected_samples.jsonl"),
            "decode_failures": str(output_dir / "decode_failures.jsonl"),
        },
        "label_mapping": {
            "class_count": len(class_names),
            "ids_contiguous_from_zero": True,
            "classes": class_names,
        },
        "splits": {
            "train": _split_summary(train_samples),
            "heldout": _split_summary(heldout_samples),
        },
        "path_audit": _path_audit(all_samples),
        "decode_audit": {
            "performed": decode_audit,
            "num_frames": num_frames,
            "sampling_strategy": sampling_strategy,
            "attempted_count": len(all_samples) if decode_audit else 0,
            "failure_count": len(decode_failures) if decode_audit else None,
        },
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def audit_decode_samples(
    samples: Sequence[UCF101Sample],
    *,
    num_frames: int = 16,
    sampling_strategy: FrameSamplingStrategy = "deterministic_center_clip",
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    iterator = tqdm(
        samples,
        desc="decode audit",
        unit="video",
        total=len(samples),
        disable=not show_progress,
    )
    for sample in iterator:
        try:
            read_sampled_clip(
                sample.video_path,
                num_frames=num_frames,
                sampling_strategy=sampling_strategy,
                video_id=sample.video_id,
            )
        except Exception as error:  # noqa: BLE001 - persisted for dataset audit.
            failures.append(
                {
                    "video_id": sample.video_id,
                    "video_path": sample.video_path,
                    "label_id": sample.label_id,
                    "label_name": sample.label_name,
                    "split": sample.split,
                    "error": str(error),
                }
            )
    return failures


def write_jsonl(path: Path, rows: Iterable[UCF101Sample | Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            data = row.to_dict() if isinstance(row, UCF101Sample) else dict(row)
            file.write(json.dumps(data, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _class_dirs(split_root: Path) -> dict[str, Path]:
    if not split_root.exists():
        raise FileNotFoundError(f"UCF101 split directory does not exist: {split_root}")
    return {
        path.name: path
        for path in sorted(split_root.iterdir(), key=lambda item: item.name)
        if path.is_dir()
    }


def _validate_class_sets(split_dirs: Mapping[str, Mapping[str, Path]]) -> None:
    class_sets = {split: set(dirs) for split, dirs in split_dirs.items()}
    if not class_sets["train"]:
        raise ValueError("UCF101 train split contains no class directories")
    if class_sets["train"] != class_sets["heldout"]:
        missing_from_heldout = sorted(class_sets["train"] - class_sets["heldout"])
        missing_from_train = sorted(class_sets["heldout"] - class_sets["train"])
        raise ValueError(
            "UCF101 train/heldout class sets differ: "
            f"missing_from_heldout={missing_from_heldout}, "
            f"missing_from_train={missing_from_train}"
        )


def _label_mapping(subset_id: str, class_names: Sequence[str]) -> dict[str, Any]:
    class_to_id = {class_name: index for index, class_name in enumerate(class_names)}
    return {
        "dataset": "ucf101",
        "subset_id": subset_id,
        "class_order": "alphabetical_by_class_directory_name",
        "classes": list(class_names),
        "class_to_id": class_to_id,
        "id_to_class": {str(index): class_name for class_name, index in class_to_id.items()},
    }


def _build_split_samples(
    class_dirs: Mapping[str, Path],
    *,
    label_mapping: Mapping[str, int],
    split: str,
    original_split: str,
    subset_id: str,
    expected_per_class: int,
    video_exts: Sequence[str],
) -> tuple[UCF101Sample, ...]:
    samples: list[UCF101Sample] = []
    for class_name in sorted(class_dirs):
        paths = _video_paths(class_dirs[class_name], video_exts)
        if len(paths) != expected_per_class:
            raise ValueError(
                f"{original_split}/{class_name} expected {expected_per_class} videos, "
                f"found {len(paths)}"
            )
        for path in paths:
            if path.stat().st_size <= 0:
                raise ValueError(f"Video file is empty: {path}")
            samples.append(
                UCF101Sample(
                    video_id=path.stem,
                    video_path=str(path),
                    label_id=int(label_mapping[class_name]),
                    label_name=class_name,
                    split=split,
                    subset_id=subset_id,
                    original_split=original_split,
                    filename=path.name,
                    class_dir=class_name,
                )
            )
    return tuple(samples)


def _video_paths(class_dir: Path, video_exts: Sequence[str]) -> tuple[Path, ...]:
    ext_set = {ext.lower() for ext in video_exts}
    paths = [
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ext_set
    ]
    return tuple(sorted(paths, key=lambda path: path.name))


def _validate_unique_video_ids(samples: Sequence[UCF101Sample]) -> None:
    counts = Counter(sample.video_id for sample in samples)
    duplicates = sorted(video_id for video_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"UCF101 selected subset contains duplicate video_id values: {duplicates}")


def _split_summary(samples: Sequence[UCF101Sample]) -> dict[str, Any]:
    counts = Counter(sample.label_name for sample in samples)
    return {
        "total_count": len(samples),
        "class_count": len(counts),
        "per_class_min": min(counts.values()) if counts else 0,
        "per_class_max": max(counts.values()) if counts else 0,
        "per_class_counts": dict(sorted(counts.items())),
    }


def _path_audit(samples: Sequence[UCF101Sample]) -> dict[str, Any]:
    missing = []
    empty = []
    for sample in samples:
        path = Path(sample.video_path)
        if not path.exists():
            missing.append(sample.video_path)
        elif path.stat().st_size <= 0:
            empty.append(sample.video_path)
    return {
        "missing_path_count": len(missing),
        "empty_file_count": len(empty),
        "missing_paths_preview": missing[:20],
        "empty_files_preview": empty[:20],
    }


def _normalize_ext(value: str) -> str:
    return value if value.startswith(".") else f".{value}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the selected UCF101 C50 index.")
    parser.add_argument("--video-root", type=Path, default=Path("data/ucf101/videos"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ucf101/subsets/c50_train100_heldout30"),
    )
    parser.add_argument("--subset-id", default=DEFAULT_SUBSET_ID)
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--heldout-per-class", type=int, default=30)
    parser.add_argument("--video-exts", nargs="+", default=list(DEFAULT_VIDEO_EXTS))
    parser.add_argument("--decode-audit", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the decode-audit progress bar.",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument(
        "--sampling-strategy",
        choices=["deterministic_center_clip", "deterministic_uniform"],
        default="deterministic_center_clip",
    )
    parser.add_argument(
        "--selection-note",
        default="50 classes preselected by the project owner before embedding extraction",
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
        video_exts=args.video_exts,
        decode_audit=args.decode_audit,
        decode_progress=not args.no_progress,
        num_frames=args.num_frames,
        sampling_strategy=args.sampling_strategy,
        selection_note=args.selection_note,
    )
    for split, split_summary in summary["splits"].items():
        print(
            f"{split}: total={split_summary['total_count']} "
            f"classes={split_summary['class_count']} "
            f"per_class={split_summary['per_class_min']}..{split_summary['per_class_max']}"
        )
    decode_summary = summary["decode_audit"]
    if decode_summary["performed"]:
        print(
            "decode audit: "
            f"attempted={decode_summary['attempted_count']} "
            f"failures={decode_summary['failure_count']}"
        )
    print(f"wrote UCF101 subset files under {args.output_dir}")
    return 1 if decode_summary["performed"] and decode_summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
