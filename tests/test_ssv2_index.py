from pathlib import Path

from src.ssv2_index import (
    build_split_index,
    build_video_lookup,
    canonical_label_name,
    select_overlapping_debug_subset,
)


def test_canonical_label_name_removes_template_brackets() -> None:
    assert (
        canonical_label_name("Holding [something] next to [something]")
        == "Holding something next to something"
    )
    assert (
        canonical_label_name("[Something] falling like a feather or paper")
        == "Something falling like a feather or paper"
    )
    assert (
        canonical_label_name("Taking [one of many similar things on the table]")
        == "Taking one of many similar things on the table"
    )


def test_build_split_index_prefers_matching_split_directory(tmp_path: Path) -> None:
    video_root = tmp_path / "videos"
    train_dir = video_root / "train"
    validation_dir = video_root / "validation"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    (train_dir / "1.webm").write_bytes(b"")
    (validation_dir / "1.webm").write_bytes(b"")

    annotation_path = tmp_path / "train.json"
    annotation_path.write_text(
        '[{"id": "1", "label": "holding apple next to cup", '
        '"template": "Holding [something] next to [something]", '
        '"placeholders": ["apple", "cup"]}, {"id": "2"}]',
        encoding="utf-8",
    )

    result = build_split_index(
        annotation_path=annotation_path,
        split="train",
        video_lookup=build_video_lookup(video_root),
        label_mapping={"Holding something next to something": 19},
    )

    assert result.annotation_count == 2
    assert result.indexed_count == 1
    assert result.missing_video_ids == ("2",)
    sample = result.samples[0]
    assert sample.video_id == "1"
    assert sample.video_path == str(train_dir / "1.webm")
    assert sample.label_id == 19
    assert sample.label_name == "Holding something next to something"
    assert sample.placeholders == ("apple", "cup")


def test_select_overlapping_debug_subset_uses_common_classes(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "items.json"
    annotation_path.write_text(
        "["
        '{"id": "1", "template": "Moving [something] up"},'
        '{"id": "2", "template": "Moving [something] up"},'
        '{"id": "3", "template": "Folding [something]"}'
        "]",
        encoding="utf-8",
    )
    video_root = tmp_path / "videos"
    for split in ("train", "validation"):
        split_dir = video_root / split
        split_dir.mkdir(parents=True)
        for video_id in ("1", "2", "3"):
            (split_dir / f"{video_id}.webm").write_bytes(b"")

    label_mapping = {"Moving something up": 45, "Folding something": 14}
    lookup = build_video_lookup(video_root)
    train = build_split_index(
        annotation_path=annotation_path,
        split="train",
        video_lookup=lookup,
        label_mapping=label_mapping,
    )
    validation = build_split_index(
        annotation_path=annotation_path,
        split="validation",
        video_lookup=lookup,
        label_mapping=label_mapping,
    )

    subset, summary = select_overlapping_debug_subset(
        train.samples, validation.samples, target_per_split=2
    )

    assert summary["target_reached"] is True
    assert summary["selected_class_count"] == 1
    assert [sample.video_id for sample in subset["train"]] == ["1", "2"]
    assert [sample.video_id for sample in subset["validation"]] == ["1", "2"]
