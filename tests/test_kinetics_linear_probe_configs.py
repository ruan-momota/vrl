from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.evaluate import RunEvaluationConfig
from src.pipeline.run import RunConfig


VIDEOMAE_CONFIG_DIR = Path("configs/runs/kinetics_videomae_linear_probe")
VIDEOMAE_RUN_ID = "kinetics-c50-train100-heldout30-videomae-base-frozen-linear-probe"
DISMO_CONFIG_DIR = Path("configs/runs/kinetics_dismo_linear_probe")
DISMO_RUN_ID = "kinetics-c50-train100-heldout30-dismo-motion-extractor-large-frozen-linear-probe"
VJEPA_CONFIG_DIR = Path("configs/runs/kinetics_vjepa_linear_probe")
VJEPA_RUN_ID = "kinetics-c50-train100-heldout30-vjepa2-vitl-fpc64-256-frozen-linear-probe"

SUBSET_SUMMARY = "data/kinetics/subsets/c50_train100_heldout30/summary.json"


@pytest.mark.parametrize(
    ("config_dir", "prefix", "run_id", "model_name"),
    [
        (VIDEOMAE_CONFIG_DIR, "kinetics_videomae_c50_", VIDEOMAE_RUN_ID, "videomae"),
        (DISMO_CONFIG_DIR, "kinetics_dismo_c50_", DISMO_RUN_ID, "dismo"),
        (VJEPA_CONFIG_DIR, "kinetics_vjepa_c50_", VJEPA_RUN_ID, "vjepa"),
    ],
)
def test_extraction_configs_share_one_run_identity(
    config_dir: Path, prefix: str, run_id: str, model_name: str
) -> None:
    configs = [
        RunConfig.from_file(path)
        for path in sorted(config_dir.glob(f"{prefix}*.json"))
        if "evaluation" not in path.name and "smoke" not in path.name
    ]

    assert len(configs) == 10
    assert {config.resolved_run_id for config in configs} == {run_id}
    assert {config.split for config in configs} == {"train", "heldout"}
    assert {config.model_name for config in configs} == {model_name}
    assert {config.dataset_name for config in configs} == {"kinetics"}
    assert sum(config.perturbation["name"] !=
               "none" for config in configs) == 8
    assert all(config.subset_summary_path ==
               SUBSET_SUMMARY for config in configs)


@pytest.mark.parametrize(
    ("config_path", "run_id"),
    [
        (VIDEOMAE_CONFIG_DIR /
         "kinetics_videomae_c50_linear_probe_evaluation.json", VIDEOMAE_RUN_ID),
        (DISMO_CONFIG_DIR / "kinetics_dismo_c50_linear_probe_evaluation.json", DISMO_RUN_ID),
        (VJEPA_CONFIG_DIR / "kinetics_vjepa_c50_linear_probe_evaluation.json", VJEPA_RUN_ID),
    ],
)
def test_evaluation_config_covers_motion_and_appearance_probes(
    config_path: Path, run_id: str
) -> None:
    config = RunEvaluationConfig.from_file(config_path)

    assert config.run_id == run_id
    assert len(config.perturbations) == 8
    assert config.knn == {"metric": "cosine", "k_values": [5]}
    assert {spec.group for spec in config.perturbations} == {
        "motion", "appearance"}
    assert {spec.name for spec in config.perturbations} == {
        "temporal_shuffle", "freeze_tail", "color_transform", "spatial_blur"}
