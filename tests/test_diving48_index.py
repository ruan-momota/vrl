from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.diving48 import Diving48DatasetAdapter
from src.data.diving48_index import build_subset_index
from src.data.indexed_dataset import load_index_jsonl
from src.data.records import VideoRecord
from src.data.registry import get_dataset_adapter, supported_dataset_adapters


def test_diving48_index_builder_writes_balanced_normalized_subset(tmp_path: Path) -> None:
    label_dir, video_root = _write_diving48_tree(tmp_path)
    output_dir = tmp_path / "subset"

    summary = build_subset_index(
        label_dir=label_dir,
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

    assert label_mapping["classes"] == ["Back_15som_05Twis_FREE", "Forward_Dive_NoTwis_PIKE"]
    assert label_mapping["class_to_id"] == {
        "Back_15som_05Twis_FREE": 0,
        "Forward_Dive_NoTwis_PIKE": 1,
    }
    assert label_mapping["old_label_by_new_label"] == {"0": 0, "1": 29}
    assert len(train_index) == 4
    assert len(heldout_index) == 2
    assert len(selected) == 6
    assert train_index[0]["source_dataset"] == "diving48"
    assert train_index[0]["split"] == "train"
    assert train_index[0]["original_split"] == "train"
    assert train_index[0]["video_id"].startswith("train_00_")
    assert train_index[0]["filename"].endswith(".mp4")
    assert train_index[0]["start_frame"] == 0
    assert heldout_index[0]["split"] == "heldout"
    assert heldout_index[0]["original_split"] == "test"
    assert heldout_index[0]["video_id"].startswith("heldout_00_")
    assert summary["splits"]["train"]["per_class_min"] == 2
    assert summary["splits"]["heldout"]["per_class_max"] == 1
    assert summary["path_audit"]["missing_path_count"] == 0
    assert (output_dir / "decode_failures.jsonl").read_text(encoding="utf-8") == ""

    record = VideoRecord.from_mapping(train_index[0], source_dataset="diving48")
    assert record.source_dataset == "diving48"
    assert record.extra_metadata["class_folder"] == "00_oldlabel_00_Back_15som_05Twis_FREE"


def test_diving48_index_builder_rejects_unbalanced_selected_split(tmp_path: Path) -> None:
    label_dir, video_root = _write_diving48_tree(tmp_path)
    selected_train = json.loads((label_dir / "selected_train.json").read_text(encoding="utf-8"))
    (label_dir / "selected_train.json").write_text(
        json.dumps(selected_train[:-1], indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 2 samples per class"):
        build_subset_index(
            label_dir=label_dir,
            video_root=video_root,
            output_dir=tmp_path / "subset",
            subset_id="tiny",
            train_per_class=2,
            heldout_per_class=1,
        )


def test_diving48_adapter_reads_normalized_index(tmp_path: Path) -> None:
    index_path = tmp_path / "index.jsonl"
    index_path.write_text(
        json.dumps(
            {
                "video_id": "train_00_example",
                "video_path": "data/diving48/videos/train/class/example.mp4",
                "label_id": 0,
                "label_name": "Back_15som_05Twis_FREE",
                "split": "train",
                "subset_id": "tiny",
                "source_dataset": "diving48",
                "old_label": 0,
                "class_folder": "class",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = Diving48DatasetAdapter().build_dataset(
        index_path,
        num_frames=16,
        sampling_strategy="deterministic_center_clip",
        transform=None,
        perturbation=None,
        subset_id="tiny",
    )

    assert len(dataset) == 1
    assert dataset.records[0].source_dataset == "diving48"
    assert dataset.records[0].subset_id == "tiny"


def test_diving48_is_registered() -> None:
    assert "diving48" in supported_dataset_adapters()
    assert get_dataset_adapter("diving48").name == "diving48"
    with pytest.raises(ValueError, match="diving48"):
        get_dataset_adapter("not-a-dataset")


def _write_diving48_tree(tmp_path: Path) -> tuple[Path, Path]:
    label_dir = tmp_path / "labels"
    video_root = tmp_path / "videos"
    label_dir.mkdir()
    classes = [
        {
            "new_label": 0,
            "old_label": 0,
            "class_name": "Back_15som_05Twis_FREE",
            "folder_name": "00_oldlabel_00_Back_15som_05Twis_FREE",
        },
        {
            "new_label": 1,
            "old_label": 29,
            "class_name": "Forward_Dive_NoTwis_PIKE",
            "folder_name": "01_oldlabel_29_Forward_Dive_NoTwis_PIKE",
        },
    ]
    (label_dir / "class_mapping.json").write_text(
        json.dumps(classes, indent=2) + "\n",
        encoding="utf-8",
    )

    for split, count in (("train", 2), ("test", 1)):
        rows = []
        for class_info in classes:
            class_dir = video_root / split / class_info["folder_name"]
            class_dir.mkdir(parents=True)
            for index in range(count):
                vid_name = f"{class_info['class_name']}_{split}_{index:02d}"
                (class_dir / f"{vid_name}.mp4").write_bytes(b"not-empty")
                rows.append(
                    {
                        "split": split,
                        "vid_name": vid_name,
                        "old_label": class_info["old_label"],
                        "new_label": class_info["new_label"],
                        "class_name": class_info["class_name"],
                        "class_folder": class_info["folder_name"],
                        "start_frame": 0,
                        "end_frame": 15,
                        "num_frames": 16,
                    }
                )
        name = "selected_train.json" if split == "train" else "selected_test.json"
        (label_dir / name).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    return label_dir, video_root
