from __future__ import annotations

from src.data.base import DatasetAdapter
from src.data.diving48 import Diving48DatasetAdapter
from src.data.ssv2 import SSV2DatasetAdapter
from src.data.ucf101 import UCF101DatasetAdapter


_DATASET_ADAPTERS: dict[str, DatasetAdapter] = {
    "diving48": Diving48DatasetAdapter(),
    "ssv2": SSV2DatasetAdapter(),
    "ucf101": UCF101DatasetAdapter(),
}


def get_dataset_adapter(name: str) -> DatasetAdapter:
    try:
        return _DATASET_ADAPTERS[name]
    except KeyError as error:
        supported = ", ".join(sorted(_DATASET_ADAPTERS))
        raise ValueError(f"Unsupported dataset adapter {name!r}; supported: {supported}") from error


def supported_dataset_adapters() -> tuple[str, ...]:
    return tuple(sorted(_DATASET_ADAPTERS))
