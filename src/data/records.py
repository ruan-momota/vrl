from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class VideoRecord:
    """Canonical video record shared by every dataset adapter.

    Dataset-specific fields remain in ``extra_metadata`` so adapters can retain
    useful provenance without forcing the extraction pipeline to know them.
    """

    video_id: str
    video_path: str
    label_id: int | None = None
    label_name: str | None = None
    split: str | None = None
    subset_id: str | None = None
    source_dataset: str | None = None
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, Any],
        *,
        source_dataset: str | None = None,
        subset_id: str | None = None,
    ) -> "VideoRecord":
        for field_name in ("video_id", "video_path"):
            if field_name not in data:
                raise ValueError(f"Video record must include {field_name}")

        label_id = data.get("label_id")
        if label_id is not None and not isinstance(label_id, int):
            raise TypeError("label_id must be an integer or None")

        canonical = {
            "video_id",
            "video_path",
            "label_id",
            "label_name",
            "split",
            "subset_id",
            "source_dataset",
        }
        return cls(
            video_id=str(data["video_id"]),
            video_path=str(data["video_path"]),
            label_id=label_id,
            label_name=_optional_string(data.get("label_name")),
            split=_optional_string(data.get("split")),
            subset_id=subset_id or _optional_string(data.get("subset_id")),
            source_dataset=source_dataset or _optional_string(data.get("source_dataset")),
            extra_metadata={key: value for key, value in data.items() if key not in canonical},
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            **dict(self.extra_metadata),
            "video_id": self.video_id,
            "video_path": self.video_path,
            "label_id": self.label_id,
            "label_name": self.label_name,
            "split": self.split,
            "subset_id": self.subset_id,
            "source_dataset": self.source_dataset,
        }


def _optional_string(value: Any) -> str | None:
    return None if value is None else str(value)
