#!/usr/bin/env python3
"""Run a train-only pixel audit for quantization and solarization strengths."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.visualize_quan_solar import PERTURBATIONS, write_comparison
from src.data.indexed_dataset import load_index_jsonl
from src.video.io import read_video_frames, sample_frame_indices
from src.video.perturbations import apply_video_perturbation


DATASETS = {
    "ssv2": Path("data/ssv2/index/train.jsonl"),
    "ucf101": Path("data/ucf101/subsets/c50_train100_heldout30/train.jsonl"),
    "diving48": Path("data/diving48/subsets/c32_train50_heldout15/train.jsonl"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit RGB quantization and solarization on deterministic train clips. "
            "No model is loaded and no held-out result is read."
        )
    )
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--num-frames", nargs="+", type=int, default=[16, 32])
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("outputs/reports/diving48_3x3/quan_solar_pixel_audit.csv"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/reports/diving48_3x3/quan_solar_pixel_audit.json"),
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=Path("outputs/plots/diving48_3x3"),
    )
    parser.add_argument("--visual-samples-per-dataset", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if any(value <= 0 for value in args.num_frames):
        raise ValueError("num_frames values must be positive")

    aggregates: dict[tuple[str, int, str], dict[str, Any]] = {}
    selected_samples: dict[str, list[dict[str, Any]]] = {}
    decode_failures: list[dict[str, str]] = []
    args.plot_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, index_path in DATASETS.items():
        records = sorted(
            load_index_jsonl(index_path),
            key=lambda item: (int(item["label_id"]), str(item["video_id"])),
        )
        label_ids = sorted({int(record["label_id"]) for record in records})
        counts: dict[int, int] = defaultdict(int)
        selected_samples[dataset_name] = []
        visual_labels: set[int] = set()

        for record in records:
            label_id = int(record["label_id"])
            if counts[label_id] >= args.samples_per_class:
                continue
            try:
                decoded = read_video_frames(record["video_path"], video_id=str(record["video_id"]))
            except Exception as error:  # noqa: BLE001 - audit records failures and continues.
                decode_failures.append(
                    {
                        "dataset": dataset_name,
                        "video_id": str(record["video_id"]),
                        "video_path": str(record["video_path"]),
                        "error": str(error),
                    }
                )
                continue

            counts[label_id] += 1
            selected_samples[dataset_name].append(
                {
                    "video_id": str(record["video_id"]),
                    "label_id": label_id,
                    "video_path": str(record["video_path"]),
                }
            )
            make_visual = (
                len(visual_labels) < args.visual_samples_per_dataset
                and label_id not in visual_labels
            )
            for num_frames in args.num_frames:
                indices = sample_frame_indices(
                    decoded.metadata.frames_decoded,
                    num_frames,
                    strategy="deterministic_center_clip",
                )
                frames = np.ascontiguousarray(decoded.frames[list(indices)])
                frame_position = num_frames // 2
                panels: dict[str, tuple[str, np.ndarray]] = {}
                for artifact_label, display_label, config in PERTURBATIONS:
                    result = apply_video_perturbation(
                        frames,
                        config,
                        video_id=str(record["video_id"]),
                    )
                    key = (dataset_name, num_frames, artifact_label)
                    aggregate = aggregates.setdefault(key, _empty_aggregate(config.name))
                    _update_aggregate(aggregate, frames, result.frames, config)
                    panels[artifact_label] = (display_label, result.frames[frame_position])

                if make_visual:
                    plot_name = (
                        f"quan_solar_audit_{dataset_name}_{record['video_id']}_{num_frames}f.png"
                    )
                    write_comparison(
                        frames[frame_position],
                        panels,
                        args.plot_dir / plot_name,
                    )
            if make_visual:
                visual_labels.add(label_id)

        incomplete = {
            label_id: counts[label_id]
            for label_id in label_ids
            if counts[label_id] != args.samples_per_class
        }
        if incomplete:
            raise RuntimeError(f"could not audit the requested samples for {dataset_name}: {incomplete}")

    rows = [
        _finalize_row(dataset_name, num_frames, artifact_label, aggregate)
        for (dataset_name, num_frames, artifact_label), aggregate in sorted(aggregates.items())
    ]
    _validate_strength_order(rows)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, args.csv_output)
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(
            {
                "policy": {
                    "split": "train-only",
                    "selection": "first decodable samples by stable label_id/video_id order",
                    "samples_per_class": args.samples_per_class,
                    "num_frames": args.num_frames,
                    "parameters_frozen_before_model_inference": True,
                },
                "selected_samples": selected_samples,
                "decode_failures": decode_failures,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"csv={args.csv_output}")
    print(f"json={args.json_output}")
    print(f"plots={args.plot_dir}")
    return 0


def _empty_aggregate(perturbation_name: str) -> dict[str, Any]:
    return {
        "perturbation": perturbation_name,
        "clips": 0,
        "absolute_difference_sum": 0,
        "changed_values": 0,
        "total_values": 0,
        "affected_values": 0,
        "output_values": [set(), set(), set()],
        "shape_dtype_timeline_ok": True,
        "config": None,
    }


def _update_aggregate(
    aggregate: dict[str, Any],
    original: np.ndarray,
    perturbed: np.ndarray,
    config: Any,
) -> None:
    difference = np.abs(perturbed.astype(np.int16) - original.astype(np.int16))
    aggregate["clips"] += 1
    aggregate["absolute_difference_sum"] += int(difference.sum(dtype=np.int64))
    aggregate["changed_values"] += int(np.count_nonzero(difference))
    aggregate["total_values"] += int(difference.size)
    aggregate["shape_dtype_timeline_ok"] &= bool(
        perturbed.shape == original.shape
        and perturbed.dtype == original.dtype
        and perturbed.flags.c_contiguous
    )
    aggregate["config"] = config.to_dict()
    if config.name == "solarization":
        aggregate["affected_values"] += int(
            np.count_nonzero(original >= config.solarization_threshold)
        )
    for channel in range(3):
        aggregate["output_values"][channel].update(
            int(value) for value in np.unique(perturbed[..., channel])
        )


def _finalize_row(
    dataset_name: str,
    num_frames: int,
    artifact_label: str,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    total = int(aggregate["total_values"])
    mean_difference = float(aggregate["absolute_difference_sum"] / total)
    config = dict(aggregate["config"])
    return {
        "dataset": dataset_name,
        "num_frames": num_frames,
        "artifact_label": artifact_label,
        "perturbation": aggregate["perturbation"],
        "strength": artifact_label.rsplit("-", maxsplit=1)[-1],
        "rgb_levels": config.get("rgb_levels"),
        "solarization_threshold": config.get("solarization_threshold"),
        "clips": aggregate["clips"],
        "mean_absolute_rgb_difference": mean_difference,
        "normalized_mean_absolute_rgb_difference": mean_difference / 255.0,
        "changed_value_ratio": aggregate["changed_values"] / total,
        "affected_value_ratio": aggregate["affected_values"] / total,
        "output_levels_r": len(aggregate["output_values"][0]),
        "output_levels_g": len(aggregate["output_values"][1]),
        "output_levels_b": len(aggregate["output_values"][2]),
        "shape_dtype_timeline_ok": aggregate["shape_dtype_timeline_ok"],
    }


def _validate_strength_order(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, int, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        grouped[(row["dataset"], row["num_frames"], row["perturbation"])][
            row["strength"]
        ] = float(row["normalized_mean_absolute_rgb_difference"])
        if not row["shape_dtype_timeline_ok"]:
            raise RuntimeError(f"shape/dtype/timeline audit failed: {row}")
        if row["perturbation"] == "rgb_quantization":
            maximum = int(row["rgb_levels"])
            if max(row["output_levels_r"], row["output_levels_g"], row["output_levels_b"]) > maximum:
                raise RuntimeError(f"quantization level audit failed: {row}")
    for key, values in grouped.items():
        if set(values) != {"low", "mid", "high"}:
            raise RuntimeError(f"missing strength in audit group {key}: {values}")
        if not values["low"] < values["mid"] < values["high"]:
            raise RuntimeError(f"non-monotonic pixel MAD in audit group {key}: {values}")


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
