from __future__ import annotations

import json
from pathlib import Path

from src.data.indexed_dataset import load_index_jsonl
from src.data.kinetics import KineticsDatasetAdapter
from src.data.kinetics_index import build_subset_index
from src.data.registry import get_dataset_adapter, supported_dataset_adapters


def test_kinetics_index_builder_writes_stable_mapping_and_indexes(tmp_path: Path) -> None:
    video_root = _write_kinetics_tree(tmp_path)
    output_dir = tmp_path / "subset"

    summary = build_subset_index(
        video_root=video_root,
        output_dir=output_dir,
        subset_id="tiny",
        train_per_class=2,
        heldout_per_class=1,
        seed=0,
    )

    label_mapping = json.loads(
        (output_dir / "label_mapping.json").read_text(encoding="utf-8"))
    train_index = load_index_jsonl(output_dir / "train.jsonl")
    heldout_index = load_index_jsonl(output_dir / "heldout.jsonl")
    selected = load_index_jsonl(output_dir / "selected_samples.jsonl")

    assert label_mapping["classes"] == ["abseiling", "archery"]
    assert label_mapping["class_to_id"] == {"abseiling": 0, "archery": 1}
    assert len(train_index) == 4
    assert len(heldout_index) == 2
    assert len(selected) == 6
    assert train_index[0]["source_dataset"] == "kinetics"
    assert train_index[0]["split"] == "train"
    assert heldout_index[0]["split"] == "heldout"
    assert summary["splits"]["train"]["per_class_min"] == 2
    assert summary["splits"]["heldout"]["per_class_max"] == 1
    assert summary["class_count"] == 2


def test_kinetics_index_builder_is_deterministic(tmp_path: Path) -> None:
    video_root = _write_kinetics_tree(tmp_path)

    first = build_subset_index(
        video_root=video_root,
        output_dir=tmp_path / "a",
        subset_id="tiny",
        train_per_class=2,
        heldout_per_class=1,
        seed=42,
    )
    second = build_subset_index(
        video_root=video_root,
        output_dir=tmp_path / "b",
        subset_id="tiny",
        train_per_class=2,
        heldout_per_class=1,
        seed=42,
    )

    assert (tmp_path / "a" / "train.jsonl").read_text(encoding="utf-8") == (
        tmp_path / "b" / "train.jsonl"
    ).read_text(encoding="utf-8")
    assert first["per_class"] == second["per_class"]


def test_kinetics_adapter_reads_normalized_index(tmp_path: Path) -> None:
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "video_id": "abseiling_0001",
                "video_path": "data/kinetics/videos/train/abseiling/abseiling_0001.mp4",
                "label_id": 0,
                "label_name": "abseiling",
                "split": "train",
                "subset_id": "tiny",
                "source_dataset": "kinetics",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = KineticsDatasetAdapter().build_dataset(
        index_path,
        num_frames=16,
        sampling_strategy="deterministic_center_clip",
        transform=None,
        perturbation=None,
        subset_id="tiny",
    )

    assert len(dataset) == 1
    assert dataset.records[0].source_dataset == "kinetics"
    assert dataset.records[0].subset_id == "tiny"


def test_kinetics_index_builder_drops_undecodable_clips_when_probing(tmp_path: Path) -> None:
    video_root = _write_kinetics_tree(tmp_path)
    output_dir = tmp_path / "subset"

    summary = build_subset_index(
        video_root=video_root,
        output_dir=output_dir,
        subset_id="tiny",
        train_per_class=2,
        heldout_per_class=1,
        seed=0,
        probe_decode=True,
    )

    train_index = load_index_jsonl(output_dir / "train.jsonl")
    heldout_index = load_index_jsonl(output_dir / "heldout.jsonl")
    decode_failures = json.loads(
        "[" + ",".join((output_dir / "decode_failures.jsonl").read_text(
            encoding="utf-8").splitlines()) + "]"
    )

    assert summary["decode_failures"]["probed"] is True
    assert summary["decode_failures"]["failure_count"] == 6
    assert len(decode_failures) == 6
    assert len(train_index) == 0
    assert len(heldout_index) == 0
    assert set(summary["short_classes"]) == {"abseiling", "archery"}


def test_kinetics_is_registered() -> None:
    assert "kinetics" in supported_dataset_adapters()
    assert get_dataset_adapter("kinetics").name == "kinetics"


def _write_kinetics_tree(tmp_path: Path) -> Path:
    root = tmp_path / "videos"
    for class_name in ("archery", "abseiling"):
        class_dir = root / class_name
        class_dir.mkdir(parents=True)
        for index in range(1, 4):
            path = class_dir / f"{class_name}_{index:04d}.mp4"
            path.write_bytes(b"not-empty")
    return root
