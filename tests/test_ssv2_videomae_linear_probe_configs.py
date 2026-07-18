from __future__ import annotations

from pathlib import Path

from src.pipeline.evaluate import RunEvaluationConfig
from src.pipeline.run import RunConfig


LINEAR_PROBE_CONFIG_DIR = Path("configs/runs/ssv2_videomae_linear_probe")
LINEAR_PROBE_RUN_ID = "ssv2-c50-train100-heldout30-videomae-base-frozen-linear-probe"


def test_linear_probe_extraction_configs_share_one_run_identity() -> None:
    configs = [
        RunConfig.from_file(path)
        for path in sorted(LINEAR_PROBE_CONFIG_DIR.glob("ssv2_videomae_c50_*.json"))
        if "smoke" not in path.name and "evaluation" not in path.name
    ]

    assert len(configs) == 16
    assert {config.resolved_run_id for config in configs} == {LINEAR_PROBE_RUN_ID}
    assert {config.split for config in configs} == {"train", "validation"}
    assert sum(config.perturbation["name"] != "none" for config in configs) == 14
    assert all(config.subset_summary_path == "data/ssv2/index/summary.json" for config in configs)


def test_linear_probe_evaluation_config_matches_the_heldout_artifacts() -> None:
    config = RunEvaluationConfig.from_file(
        LINEAR_PROBE_CONFIG_DIR / "ssv2_videomae_c50_linear_probe_evaluation.json"
    )

    assert config.run_id == LINEAR_PROBE_RUN_ID
    assert len(config.perturbations) == 14
    assert config.knn == {"metric": "cosine", "k_values": [5]}
    assert {spec.group for spec in config.perturbations} == {"motion", "appearance"}
    assert {spec.name for spec in config.perturbations} == {
        "temporal_shuffle",
        "freeze_tail",
        "color_transform",
        "spatial_blur",
        "rgb_quantization",
        "solarization",
    }
