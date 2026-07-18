"""Diving48 selected-subset to normalized-index conversion."""

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


DEFAULT_SUBSET_ID = "c32_train50_heldout15"
DEFAULT_VIDEO_EXT = ".mp4"


@dataclass(frozen=True)
class Diving48Class:
    new_label: int
    old_label: int
    class_name: str
    folder_name: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "Diving48Class":
        return cls(
            new_label=int(data["new_label"]),
            old_label=int(data["old_label"]),
            class_name=str(data["class_name"]),
            folder_name=str(data["folder_name"]),
        )


@dataclass(frozen=True)
class Diving48Sample:
    video_id: str
    video_path: str
    label_id: int
    label_name: str
    split: str
    subset_id: str
    source_dataset: str = "diving48"
    original_split: str = "train"
    old_label: int = -1
    class_folder: str = ""
    vid_name: str = ""
    filename: str = ""
    start_frame: int = 0
    end_frame: int = 0
    num_frames: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_subset_index(
    *,
    label_dir: Path,
    video_root: Path,
    output_dir: Path,
    subset_id: str = DEFAULT_SUBSET_ID,
    train_per_class: int = 50,
    heldout_per_class: int = 15,
    video_ext: str = DEFAULT_VIDEO_EXT,
    decode_audit: bool = False,
    decode_progress: bool = True,
    num_frames: int = 16,
    sampling_strategy: FrameSamplingStrategy = "deterministic_center_clip",
    selection_note: str = "quota3 class-balanced subset selected before embedding extraction",
) -> dict[str, Any]:
    """Write label mapping, normalized indexes, selected samples, and summary."""
    label_dir = Path(label_dir)
    video_root = Path(video_root)
    output_dir = Path(output_dir)
    normalized_ext = _normalize_ext(video_ext)

    classes = _load_class_mapping(label_dir / "class_mapping.json")
    classes_by_label = {item.new_label: item for item in classes}
    label_mapping = _label_mapping(subset_id, classes)
    train_rows = _load_json_list(label_dir / "selected_train.json")
    heldout_rows = _load_json_list(label_dir / "selected_test.json")
    train_samples = _build_split_samples(
        train_rows,
        classes_by_label=classes_by_label,
        video_root=video_root,
        split="train",
        original_split="train",
        subset_id=subset_id,
        expected_per_class=train_per_class,
        video_ext=normalized_ext,
    )
    heldout_samples = _build_split_samples(
        heldout_rows,
        classes_by_label=classes_by_label,
        video_root=video_root,
        split="heldout",
        original_split="test",
        subset_id=subset_id,
        expected_per_class=heldout_per_class,
        video_ext=normalized_ext,
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
        "dataset": "diving48",
        "subset_id": subset_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "label_dir": str(label_dir),
        "video_root": str(video_root),
        "video_ext": normalized_ext,
        "selection": {
            "note": selection_note,
            "source": "selected_train.json and selected_test.json",
            "train_per_class": train_per_class,
            "heldout_per_class": heldout_per_class,
        },
        "files": {
            "class_mapping": str(label_dir / "class_mapping.json"),
            "selected_train": str(label_dir / "selected_train.json"),
            "selected_test": str(label_dir / "selected_test.json"),
            "label_mapping": str(output_dir / "label_mapping.json"),
            "train_index": str(output_dir / "train.jsonl"),
            "heldout_index": str(output_dir / "heldout.jsonl"),
            "selected_samples": str(output_dir / "selected_samples.jsonl"),
            "decode_failures": str(output_dir / "decode_failures.jsonl"),
        },
        "label_mapping": {
            "class_count": len(classes),
            "ids_contiguous_from_zero": True,
            "classes": [item.class_name for item in classes],
            "old_labels": [item.old_label for item in classes],
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
    samples: Sequence[Diving48Sample],
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


def write_jsonl(
    path: Path,
    rows: Iterable[Diving48Sample | Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            data = row.to_dict() if isinstance(row, Diving48Sample) else dict(row)
            file.write(json.dumps(data, ensure_ascii=False, sort_keys=True))
            file.write("\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {index} must be an object")
        rows.append(item)
    return rows


def _load_class_mapping(path: Path) -> tuple[Diving48Class, ...]:
    rows = _load_json_list(path)
    classes = tuple(Diving48Class.from_mapping(row) for row in rows)
    ids = [item.new_label for item in classes]
    if sorted(ids) != list(range(len(ids))):
        raise ValueError(f"{path} new_label values must be contiguous from 0")
    _validate_unique(path, "folder_name", [item.folder_name for item in classes])
    _validate_unique(path, "class_name", [item.class_name for item in classes])
    return tuple(sorted(classes, key=lambda item: item.new_label))


def _build_split_samples(
    rows: Sequence[Mapping[str, Any]],
    *,
    classes_by_label: Mapping[int, Diving48Class],
    video_root: Path,
    split: str,
    original_split: str,
    subset_id: str,
    expected_per_class: int,
    video_ext: str,
) -> tuple[Diving48Sample, ...]:
    counts = Counter(int(row["new_label"]) for row in rows)
    missing_labels = sorted(set(classes_by_label) - set(counts))
    extra_labels = sorted(set(counts) - set(classes_by_label))
    if missing_labels or extra_labels:
        raise ValueError(
            f"Diving48 {original_split} labels do not match class mapping: "
            f"missing={missing_labels}, extra={extra_labels}"
        )
    bad_counts = {
        label: count
        for label, count in sorted(counts.items())
        if count != expected_per_class
    }
    if bad_counts:
        raise ValueError(
            f"Diving48 {original_split} expected {expected_per_class} samples per class, "
            f"found {bad_counts}"
        )

    samples: list[Diving48Sample] = []
    for row in rows:
        label_id = int(row["new_label"])
        class_info = classes_by_label[label_id]
        _validate_sample_matches_class(row, class_info)
        vid_name = str(row["vid_name"])
        class_folder = str(row["class_folder"])
        filename = f"{vid_name}{video_ext}"
        video_path = video_root / original_split / class_folder / filename
        if not video_path.exists():
            raise FileNotFoundError(f"Diving48 video does not exist: {video_path}")
        if video_path.stat().st_size <= 0:
            raise ValueError(f"Diving48 video file is empty: {video_path}")

        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        num_frames = int(row["num_frames"])
        if start_frame > end_frame:
            raise ValueError(f"Diving48 sample has start_frame > end_frame: {vid_name}")
        if num_frames <= 0:
            raise ValueError(f"Diving48 sample has non-positive num_frames: {vid_name}")

        samples.append(
            Diving48Sample(
                video_id=f"{split}_{label_id:02d}_{vid_name}",
                video_path=str(video_path),
                label_id=label_id,
                label_name=class_info.class_name,
                split=split,
                subset_id=subset_id,
                original_split=original_split,
                old_label=class_info.old_label,
                class_folder=class_folder,
                vid_name=vid_name,
                filename=filename,
                start_frame=start_frame,
                end_frame=end_frame,
                num_frames=num_frames,
            )
        )
    return tuple(samples)


def _validate_sample_matches_class(
    row: Mapping[str, Any],
    class_info: Diving48Class,
) -> None:
    mismatches = {
        "old_label": int(row["old_label"]) != class_info.old_label,
        "class_name": str(row["class_name"]) != class_info.class_name,
        "class_folder": str(row["class_folder"]) != class_info.folder_name,
    }
    bad_fields = sorted(field for field, bad in mismatches.items() if bad)
    if bad_fields:
        raise ValueError(
            "Diving48 selected sample does not match class mapping: "
            f"vid_name={row.get('vid_name')!r}, new_label={class_info.new_label}, "
            f"bad_fields={bad_fields}"
        )


def _label_mapping(subset_id: str, classes: Sequence[Diving48Class]) -> dict[str, Any]:
    class_to_id = {item.class_name: item.new_label for item in classes}
    return {
        "dataset": "diving48",
        "subset_id": subset_id,
        "class_order": "new_label_ascending",
        "classes": [item.class_name for item in classes],
        "class_to_id": class_to_id,
        "id_to_class": {str(item.new_label): item.class_name for item in classes},
        "old_label_by_new_label": {
            str(item.new_label): item.old_label for item in classes
        },
        "folder_by_new_label": {
            str(item.new_label): item.folder_name for item in classes
        },
    }


def _validate_unique(path: Path, field_name: str, values: Sequence[str]) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"{path} contains duplicate {field_name}: {duplicates}")


def _validate_unique_video_ids(samples: Sequence[Diving48Sample]) -> None:
    counts = Counter(sample.video_id for sample in samples)
    duplicates = sorted(video_id for video_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"Diving48 selected subset contains duplicate video_id values: {duplicates}")


def _split_summary(samples: Sequence[Diving48Sample]) -> dict[str, Any]:
    counts = Counter(sample.label_name for sample in samples)
    return {
        "total_count": len(samples),
        "class_count": len(counts),
        "per_class_min": min(counts.values()) if counts else 0,
        "per_class_max": max(counts.values()) if counts else 0,
        "per_class_counts": dict(sorted(counts.items())),
    }


def _path_audit(samples: Sequence[Diving48Sample]) -> dict[str, Any]:
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
    parser = argparse.ArgumentParser(description="Build the selected Diving48 C32 index.")
    parser.add_argument("--label-dir", type=Path, default=Path("data/diving48/labels"))
    parser.add_argument("--video-root", type=Path, default=Path("data/diving48/videos"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/diving48/subsets/c32_train50_heldout15"),
    )
    parser.add_argument("--subset-id", default=DEFAULT_SUBSET_ID)
    parser.add_argument("--train-per-class", type=int, default=50)
    parser.add_argument("--heldout-per-class", type=int, default=15)
    parser.add_argument("--video-ext", default=DEFAULT_VIDEO_EXT)
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
        default="quota3 32-class balanced subset selected before embedding extraction",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_subset_index(
        label_dir=args.label_dir,
        video_root=args.video_root,
        output_dir=args.output_dir,
        subset_id=args.subset_id,
        train_per_class=args.train_per_class,
        heldout_per_class=args.heldout_per_class,
        video_ext=args.video_ext,
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
    print(f"wrote Diving48 subset files under {args.output_dir}")
    return 1 if decode_summary["performed"] and decode_summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
