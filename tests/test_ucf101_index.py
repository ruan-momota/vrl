from __future__ import annotations

import json
from pathlib import Path

from src.data.indexed_dataset import load_index_jsonl
from src.data.registry import get_dataset_adapter, supported_dataset_adapters
from src.data.ucf101 import UCF101DatasetAdapter
from src.data.ucf101_index import build_subset_index


def test_ucf101_index_builder_writes_stable_mapping_and_indexes(tmp_path: Path) -> None:
    video_root = _write_ucf101_tree(tmp_path)
    output_dir = tmp_path / "subset"

    summary = build_subset_index(
        video_root=video_root,
        output_dir=output_dir,
        subset_id="tiny",
        train_per_class=2,
        heldout_per_class=1,
    )

    label_mapping = json.loads((output_dir / "label_mapping.json").read_text(encoding="utf-8"))
    train_index = load_index_jsonl(output_dir / "train.jsonl")
    heldout_index = load_index_jsonl(output_dir / "heldout.jsonl")
    selected = load_index_jsonl(output_dir / "selected_samples.jsonl")

    assert label_mapping["classes"] == ["ApplyEyeMakeup", "Archery"]
    assert label_mapping["class_to_id"] == {"ApplyEyeMakeup": 0, "Archery": 1}
    assert len(train_index) == 4
    assert len(heldout_index) == 2
    assert len(selected) == 6
    assert train_index[0]["source_dataset"] == "ucf101"
    assert train_index[0]["split"] == "train"
    assert heldout_index[0]["split"] == "heldout"
    assert heldout_index[0]["original_split"] == "test"
    assert summary["splits"]["train"]["per_class_min"] == 2
    assert summary["splits"]["heldout"]["per_class_max"] == 1
    assert summary["path_audit"]["missing_path_count"] == 0
    assert (output_dir / "decode_failures.jsonl").read_text(encoding="utf-8") == ""


def test_ucf101_adapter_reads_normalized_index(tmp_path: Path) -> None:
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "video_id": "v_ApplyEyeMakeup_g01_c01",
                "video_path": "data/ucf101/videos/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01.avi",
                "label_id": 0,
                "label_name": "ApplyEyeMakeup",
                "split": "train",
                "subset_id": "tiny",
                "source_dataset": "ucf101",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = UCF101DatasetAdapter().build_dataset(
        index_path,
        num_frames=16,
        sampling_strategy="deterministic_center_clip",
        transform=None,
        perturbation=None,
        subset_id="tiny",
    )

    assert len(dataset) == 1
    assert dataset.records[0].source_dataset == "ucf101"
    assert dataset.records[0].subset_id == "tiny"


def test_ucf101_is_registered() -> None:
    assert "ucf101" in supported_dataset_adapters()
    assert get_dataset_adapter("ucf101").name == "ucf101"


def _write_ucf101_tree(tmp_path: Path) -> Path:
    root = tmp_path / "videos"
    for split, count in (("train", 2), ("test", 1)):
        for class_name in ("Archery", "ApplyEyeMakeup"):
            class_dir = root / split / class_name
            class_dir.mkdir(parents=True)
            for index in range(1, count + 1):
                group = "01" if split == "train" else "02"
                path = class_dir / f"v_{class_name}_g{group}_c{index:02d}.avi"
                path.write_bytes(b"not-empty")
    return root
