from __future__ import annotations

from pathlib import Path

import pytest

from src.pipeline.evaluate import RunEvaluationConfig
from src.pipeline.run import RunConfig


VIDEOMAE_CONFIG_DIR = Path("configs/runs/kinetics_videomae_linear_probe")
VIDEOMAE_RUN_ID = "kinetics-c20-train16-heldout4-videomae-base-frozen-linear-probe"
DISMO_CONFIG_DIR = Path("configs/runs/kinetics_dismo_linear_probe")
DISMO_RUN_ID = "kinetics-c20-train16-heldout4-dismo-motion-extractor-large-frozen-linear-probe"

SUBSET_SUMMARY = "data/kinetics/subsets/c20_train16_heldout4/summary.json"


@pytest.mark.parametrize(
    ("config_dir", "prefix", "run_id", "model_name"),
    [
        (VIDEOMAE_CONFIG_DIR, "kinetics_videomae_c20_", VIDEOMAE_RUN_ID, "videomae"),
        (DISMO_CONFIG_DIR, "kinetics_dismo_c20_", DISMO_RUN_ID, "dismo"),
    ],
)
def test_extraction_configs_share_one_run_identity(
    config_dir: Path, prefix: str, run_id: str, model_name: str
) -> None:
    configs = [
        RunConfig.from_file(path)
        for path in sorted(config_dir.glob(f"{prefix}*.json"))
        if "evaluation" not in path.name
    ]

    assert len(configs) == 4
    assert {config.resolved_run_id for config in configs} == {run_id}
    assert {config.split for config in configs} == {"train", "heldout"}
    assert {config.model_name for config in configs} == {model_name}
    assert {config.dataset_name for config in configs} == {"kinetics"}
    assert sum(config.perturbation["name"] !=
               "none" for config in configs) == 2
    assert all(config.subset_summary_path ==
               SUBSET_SUMMARY for config in configs)


@pytest.mark.parametrize(
    ("config_path", "run_id"),
    [
        (VIDEOMAE_CONFIG_DIR /
         "kinetics_videomae_c20_linear_probe_evaluation.json", VIDEOMAE_RUN_ID),
        (DISMO_CONFIG_DIR / "kinetics_dismo_c20_linear_probe_evaluation.json", DISMO_RUN_ID),
    ],
)
def test_evaluation_config_covers_motion_and_appearance_probes(
    config_path: Path, run_id: str
) -> None:
    config = RunEvaluationConfig.from_file(config_path)

    assert config.run_id == run_id
    assert len(config.perturbations) == 2
    assert config.knn == {"metric": "cosine", "k_values": [5]}
    assert {spec.group for spec in config.perturbations} == {
        "motion", "appearance"}
    assert {spec.name for spec in config.perturbations} == {
        "temporal_shuffle", "spatial_blur"}
