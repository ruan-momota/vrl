from __future__ import annotations

from src.data.records import VideoRecord


def test_video_record_round_trips_legacy_index_fields() -> None:
    record = VideoRecord.from_mapping(
        {
            "video_id": "12",
            "video_path": "videos/12.webm",
            "label_id": 7,
            "label_name": "Example",
            "split": "train",
            "placeholders": ["cup"],
        },
        source_dataset="ssv2",
        subset_id="c50",
    )

    assert record.video_id == "12"
    assert record.source_dataset == "ssv2"
    assert record.subset_id == "c50"
    assert record.extra_metadata == {"placeholders": ["cup"]}
    assert record.to_mapping()["label_name"] == "Example"
    assert record.to_mapping()["placeholders"] == ["cup"]
