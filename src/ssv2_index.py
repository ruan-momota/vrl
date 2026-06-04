from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SSV2Sample:
    video_id: str
    video_path: str
    label_id: int | None
    label_name: str | None
    split: str
    caption: str | None = None
    template: str | None = None
    placeholders: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["placeholders"] = list(self.placeholders)
        return data


@dataclass(frozen=True)
class SplitIndexResult:
    split: str
    annotation_count: int
    samples: tuple[SSV2Sample, ...]
    missing_video_ids: tuple[str, ...]
    duplicate_annotation_ids: tuple[str, ...]

    @property
    def indexed_count(self) -> int:
        return len(self.samples)

    @property
    def missing_video_count(self) -> int:
        return len(self.missing_video_ids)

    def to_summary(self) -> dict[str, Any]:
        label_ids = sorted(
            {sample.label_id for sample in self.samples if sample.label_id is not None}
        )
        return {
            "annotation_count": self.annotation_count,
            "indexed_count": self.indexed_count,
            "missing_video_count": self.missing_video_count,
            "missing_video_ids_preview": list(self.missing_video_ids[:20]),
            "duplicate_annotation_id_count": len(self.duplicate_annotation_ids),
            "duplicate_annotation_ids_preview": list(self.duplicate_annotation_ids[:20]),
            "labeled_sample_count": sum(
                sample.label_id is not None for sample in self.samples
            ),
            "class_count": len(label_ids),
            "label_ids_present": label_ids,
            "label_ids_present_contiguous": ids_are_contiguous(label_ids),
        }


def canonical_label_name(template: str) -> str:
    """Convert SSV2 bracketed templates into labels.json class names."""
    without_brackets = re.sub(r"\[([^\]]+)\]", r"\1", template)
    return re.sub(r"\s+", " ", without_brackets).strip()


def ids_are_contiguous(ids: Sequence[int]) -> bool:
    if not ids:
        return True
    return sorted(ids) == list(range(min(ids), max(ids) + 1))


def normalize_video_ext(video_ext: str) -> str:
    return video_ext if video_ext.startswith(".") else f".{video_ext}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_annotations(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")

    annotations: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"{path} item {index} must be an object")
        if "id" not in item:
            raise ValueError(f"{path} item {index} is missing the id field")
        annotations.append(item)
    return annotations


def load_label_mapping(path: Path) -> dict[str, int]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")

    label_mapping: dict[str, int] = {}
    for label_name, label_id in data.items():
        label_mapping[str(label_name)] = int(label_id)

    ids = sorted(label_mapping.values())
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path} contains duplicate label ids")
    if ids != list(range(len(ids))):
        raise ValueError(f"{path} label ids must be contiguous from 0")
    return label_mapping


def build_video_lookup(
    video_root: Path, video_ext: str = ".webm"
) -> dict[str, tuple[Path, ...]]:
    if not video_root.exists():
        raise FileNotFoundError(f"Video root does not exist: {video_root}")

    normalized_ext = normalize_video_ext(video_ext)
    lookup: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(video_root.rglob(f"*{normalized_ext}")):
        if path.is_file():
            lookup[path.stem].append(path)
    return {video_id: tuple(paths) for video_id, paths in lookup.items()}


def resolve_video_path(
    video_lookup: Mapping[str, Sequence[Path]], video_id: str, split: str
) -> Path | None:
    paths = list(video_lookup.get(video_id, ()))
    if not paths:
        return None

    split_matches = [path for path in paths if path.parent.name == split]
    if split_matches:
        return sorted(split_matches)[0]
    return sorted(paths, key=lambda path: (len(path.parts), str(path)))[0]


def build_split_index(
    *,
    annotation_path: Path,
    split: str,
    video_lookup: Mapping[str, Sequence[Path]],
    label_mapping: Mapping[str, int] | None,
) -> SplitIndexResult:
    annotations = load_annotations(annotation_path)
    samples: list[SSV2Sample] = []
    missing_video_ids: list[str] = []
    seen_ids: set[str] = set()
    duplicate_ids: list[str] = []

    for item in annotations:
        video_id = str(item["id"])
        if video_id in seen_ids:
            duplicate_ids.append(video_id)
        seen_ids.add(video_id)

        video_path = resolve_video_path(video_lookup, video_id, split)
        if video_path is None:
            missing_video_ids.append(video_id)
            continue

        template = str(item["template"]) if "template" in item else None
        label_name = canonical_label_name(template) if template is not None else None
        label_id = None
        if label_name is not None and label_mapping is not None:
            if label_name not in label_mapping:
                raise ValueError(
                    f"{annotation_path} video {video_id} has unmapped template: "
                    f"{template!r} -> {label_name!r}"
                )
            label_id = label_mapping[label_name]

        placeholders = tuple(str(value) for value in item.get("placeholders", ()))
        caption = str(item["label"]) if "label" in item else None
        samples.append(
            SSV2Sample(
                video_id=video_id,
                video_path=str(video_path),
                label_id=label_id,
                label_name=label_name,
                split=split,
                caption=caption,
                template=template,
                placeholders=placeholders,
            )
        )

    return SplitIndexResult(
        split=split,
        annotation_count=len(annotations),
        samples=tuple(samples),
        missing_video_ids=tuple(missing_video_ids),
        duplicate_annotation_ids=tuple(duplicate_ids),
    )


def write_jsonl(path: Path, samples: Iterable[SSV2Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample.to_dict(), ensure_ascii=False, sort_keys=True))
            file.write("\n")


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")


def group_labeled_by_class(samples: Iterable[SSV2Sample]) -> dict[int, list[SSV2Sample]]:
    grouped: dict[int, list[SSV2Sample]] = defaultdict(list)
    for sample in samples:
        if sample.label_id is None:
            continue
        grouped[sample.label_id].append(sample)

    for label_samples in grouped.values():
        label_samples.sort(key=sample_sort_key)
    return dict(grouped)


def sample_sort_key(sample: SSV2Sample) -> tuple[int, str]:
    try:
        numeric_id = int(sample.video_id)
    except ValueError:
        numeric_id = 0
    return numeric_id, sample.video_id


def select_overlapping_debug_subset(
    train_samples: Sequence[SSV2Sample],
    validation_samples: Sequence[SSV2Sample],
    *,
    target_per_split: int = 32,
    max_classes: int | None = None,
) -> tuple[dict[str, tuple[SSV2Sample, ...]], dict[str, Any]]:
    train_by_class = group_labeled_by_class(train_samples)
    validation_by_class = group_labeled_by_class(validation_samples)
    common_label_ids = sorted(set(train_by_class) & set(validation_by_class))

    class_infos = []
    label_names = {
        sample.label_id: sample.label_name
        for sample in tuple(train_samples) + tuple(validation_samples)
        if sample.label_id is not None
    }
    for label_id in common_label_ids:
        train_count = len(train_by_class[label_id])
        validation_count = len(validation_by_class[label_id])
        class_infos.append(
            {
                "label_id": label_id,
                "label_name": label_names[label_id],
                "train_count": train_count,
                "validation_count": validation_count,
            }
        )
    class_infos.sort(
        key=lambda item: (
            -min(int(item["train_count"]), int(item["validation_count"])),
            -(int(item["train_count"]) + int(item["validation_count"])),
            str(item["label_name"]),
        )
    )

    selected_class_infos: list[dict[str, Any]] = []
    selected_train: list[SSV2Sample] = []
    selected_validation: list[SSV2Sample] = []
    for class_info in class_infos:
        if max_classes is not None and len(selected_class_infos) >= max_classes:
            break

        label_id = int(class_info["label_id"])
        selected_class_infos.append(class_info)
        selected_train.extend(train_by_class[label_id])
        selected_validation.extend(validation_by_class[label_id])

        if (
            len(selected_train) >= target_per_split
            and len(selected_validation) >= target_per_split
        ):
            break

    selected_train.sort(key=sample_sort_key)
    selected_validation.sort(key=sample_sort_key)
    metadata = {
        "strategy": "overlapping_classes_until_target",
        "target_per_split": target_per_split,
        "max_classes": max_classes,
        "selected_class_count": len(selected_class_infos),
        "selected_classes": selected_class_infos,
        "train_count": len(selected_train),
        "validation_count": len(selected_validation),
        "target_reached": (
            len(selected_train) >= target_per_split
            and len(selected_validation) >= target_per_split
        ),
    }
    return {
        "train": tuple(selected_train),
        "validation": tuple(selected_validation),
    }, metadata


def count_videos_by_directory(
    video_lookup: Mapping[str, Sequence[Path]], video_root: Path
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for paths in video_lookup.values():
        for path in paths:
            try:
                relative_parent = path.parent.relative_to(video_root)
            except ValueError:
                relative_parent = path.parent
            directory = "." if str(relative_parent) == "." else str(relative_parent)
            counts[directory] += 1
    return dict(sorted(counts.items()))


def probe_video(path: Path, *, decode_frames: bool = True) -> dict[str, Any]:
    try:
        import av
    except ImportError as error:
        raise RuntimeError("PyAV is not installed; cannot probe video files") from error

    container = av.open(str(path))
    try:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {path}")
        stream = container.streams.video[0]
        duration_seconds = None
        if stream.duration is not None and stream.time_base is not None:
            duration_seconds = float(stream.duration * stream.time_base)
        elif container.duration is not None:
            duration_seconds = float(container.duration / 1_000_000)

        fps = float(stream.average_rate) if stream.average_rate is not None else None
        metadata = {
            "frames_reported": int(stream.frames) if stream.frames else None,
            "frames_decoded": None,
            "duration_seconds": duration_seconds,
            "fps": fps,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
        }
        if decode_frames:
            metadata["frames_decoded"] = sum(1 for _ in container.decode(stream))
        return metadata
    finally:
        container.close()


def probe_samples(
    samples: Sequence[SSV2Sample], *, count: int, seed: int
) -> tuple[list[dict[str, Any]], str | None]:
    if count <= 0 or not samples:
        return [], None

    rng = random.Random(seed)
    selected = rng.sample(list(samples), k=min(count, len(samples)))
    probes: list[dict[str, Any]] = []
    for sample in selected:
        try:
            video_metadata = probe_video(Path(sample.video_path))
        except Exception as error:  # noqa: BLE001 - recorded in summary for inspection.
            probes.append(
                {
                    "video_id": sample.video_id,
                    "split": sample.split,
                    "video_path": sample.video_path,
                    "label_id": sample.label_id,
                    "label_name": sample.label_name,
                    "error": str(error),
                }
            )
            continue

        probes.append(
            {
                "video_id": sample.video_id,
                "split": sample.split,
                "video_path": sample.video_path,
                "label_id": sample.label_id,
                "label_name": sample.label_name,
                **video_metadata,
            }
        )
    errors = [probe for probe in probes if "error" in probe]
    error_message = f"{len(errors)} of {len(probes)} video probes failed" if errors else None
    return probes, error_message


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build local Something-Something-V2 dataset indexes."
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("data/ssv2/labels"),
        help="Directory containing train.json, validation.json, test.json.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=Path("data/ssv2/videos"),
        help="Root containing SSV2 videos, either flat or split into subdirectories.",
    )
    parser.add_argument(
        "--label-mapping-path",
        type=Path,
        default=Path("data/ssv2/labels/labels.json"),
        help="Path to labels.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ssv2/index"),
        help="Directory where split JSONL indexes and summary.json are written.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Splits to index. Default: train validation test.",
    )
    parser.add_argument(
        "--video-ext",
        default=".webm",
        help="Video extension to index. Default: .webm.",
    )
    parser.add_argument(
        "--probe-count",
        type=int,
        default=5,
        help="Number of indexed videos to probe with PyAV. Use 0 to skip.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Probe sample seed.")
    parser.add_argument(
        "--debug-target-per-split",
        type=int,
        default=32,
        help="Minimum train and validation samples for the overlapping debug subset.",
    )
    parser.add_argument(
        "--debug-max-classes",
        type=int,
        default=None,
        help="Optional cap on overlapping classes used for the debug subset.",
    )
    parser.add_argument(
        "--no-debug-subset",
        action="store_true",
        help="Do not write debug_train.jsonl/debug_validation.jsonl.",
    )
    parser.add_argument(
        "--strict-video-probe",
        action="store_true",
        help="Exit non-zero when any requested video probe fails.",
    )
    return parser.parse_args()


def build_indexes_from_args(args: argparse.Namespace) -> dict[str, Any]:
    label_mapping = load_label_mapping(args.label_mapping_path)
    video_lookup = build_video_lookup(args.video_root, args.video_ext)
    results: dict[str, SplitIndexResult] = {}

    for split in args.splits:
        annotation_path = args.labels_dir / f"{split}.json"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file does not exist: {annotation_path}")
        result = build_split_index(
            annotation_path=annotation_path,
            split=split,
            video_lookup=video_lookup,
            label_mapping=label_mapping,
        )
        results[split] = result
        write_jsonl(args.output_dir / f"{split}.jsonl", result.samples)

    summary: dict[str, Any] = {
        "dataset": "ssv2",
        "video_root": str(args.video_root),
        "video_ext": normalize_video_ext(args.video_ext),
        "video_count": sum(len(paths) for paths in video_lookup.values()),
        "video_count_by_directory": count_videos_by_directory(
            video_lookup, args.video_root
        ),
        "label_mapping": {
            "path": str(args.label_mapping_path),
            "class_count": len(label_mapping),
            "ids_contiguous_from_zero": ids_are_contiguous(
                sorted(label_mapping.values())
            )
            and min(label_mapping.values()) == 0,
            "id_min": min(label_mapping.values()),
            "id_max": max(label_mapping.values()),
        },
        "splits": {
            split: result.to_summary() for split, result in sorted(results.items())
        },
    }

    all_samples: list[SSV2Sample] = []
    for result in results.values():
        all_samples.extend(result.samples)
    probes, probe_error = probe_samples(all_samples, count=args.probe_count, seed=args.seed)
    summary["video_probes"] = probes
    if probe_error is not None:
        summary["video_probe_error"] = probe_error

    if (
        not args.no_debug_subset
        and "train" in results
        and "validation" in results
    ):
        debug_subset, debug_summary = select_overlapping_debug_subset(
            results["train"].samples,
            results["validation"].samples,
            target_per_split=args.debug_target_per_split,
            max_classes=args.debug_max_classes,
        )
        write_jsonl(args.output_dir / "debug_train.jsonl", debug_subset["train"])
        write_jsonl(
            args.output_dir / "debug_validation.jsonl", debug_subset["validation"]
        )
        summary["debug_subset"] = debug_summary

    write_json(args.output_dir / "summary.json", summary)
    return summary


def main() -> int:
    args = parse_args()
    summary = build_indexes_from_args(args)

    for split, split_summary in summary["splits"].items():
        print(
            f"{split}: indexed={split_summary['indexed_count']} "
            f"annotation={split_summary['annotation_count']} "
            f"missing={split_summary['missing_video_count']} "
            f"classes={split_summary['class_count']}"
        )

    if "debug_subset" in summary:
        debug_summary = summary["debug_subset"]
        print(
            "debug subset: "
            f"train={debug_summary['train_count']} "
            f"validation={debug_summary['validation_count']} "
            f"classes={debug_summary['selected_class_count']} "
            f"target_reached={debug_summary['target_reached']}"
        )

    if args.strict_video_probe and summary.get("video_probe_error"):
        print(summary["video_probe_error"])
        return 1
    print(f"wrote index files under {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
