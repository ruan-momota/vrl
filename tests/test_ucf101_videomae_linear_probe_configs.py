from __future__ import annotations

from pathlib import Path

from src.pipeline.evaluate import RunEvaluationConfig
from src.pipeline.run import RunConfig


CONFIG_DIR = Path("configs/runs/ucf101_videomae_linear_probe")
RUN_ID = "ucf101-c50-train100-heldout30-videomae-base-frozen-linear-probe"


def test_ucf101_extraction_configs_share_one_run_identity() -> None:
    configs = [
        RunConfig.from_file(path)
        for path in sorted(CONFIG_DIR.glob("ucf101_videomae_c50_*.json"))
        if "smoke" not in path.name and "evaluation" not in path.name
    ]

    assert len(configs) == 16
    assert {config.dataset_name for config in configs} == {"ucf101"}
    assert {config.resolved_run_id for config in configs} == {RUN_ID}
    assert {config.split for config in configs} == {"train", "heldout"}
    assert sum(config.perturbation["name"] != "none" for config in configs) == 14
    assert all(
        config.subset_summary_path
        == "data/ucf101/subsets/c50_train100_heldout30/summary.json"
        for config in configs
    )


def test_ucf101_evaluation_config_matches_the_heldout_artifacts() -> None:
    config = RunEvaluationConfig.from_file(
        CONFIG_DIR / "ucf101_videomae_c50_linear_probe_evaluation.json"
    )

    assert config.run_id == RUN_ID
    assert config.train_original == "embeddings/train/original.pt"
    assert config.heldout_original == "embeddings/heldout/original.pt"
    assert len(config.perturbations) == 14
    assert config.knn == {"metric": "cosine", "k_values": [5]}
    assert {spec.group for spec in config.perturbations} == {"motion", "appearance"}
