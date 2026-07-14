from __future__ import annotations

from pathlib import Path

from src.pipeline.evaluate import RunEvaluationConfig
from src.pipeline.run import RunConfig


CONFIG_DIR = Path("configs/runs/hmdb51_videomae_linear_probe")
RUN_ID = "hmdb51-full-split1-videomae-base-frozen-linear-probe"


def test_hmdb51_videomae_extraction_configs_share_one_run_identity() -> None:
    configs = [
        RunConfig.from_file(path)
        for path in sorted(CONFIG_DIR.glob("hmdb51_videomae_full_split1_*.json"))
        if "smoke" not in path.name and "evaluation" not in path.name
    ]

    assert len(configs) == 10
    assert {config.dataset_name for config in configs} == {"hmdb51"}
    assert {config.resolved_run_id for config in configs} == {RUN_ID}
    assert {config.split for config in configs} == {"train", "heldout"}
    assert sum(config.perturbation["name"] != "none" for config in configs) == 8
    assert all(
        config.subset_summary_path == "data/hmdb51/subsets/full_split1/summary.json"
        for config in configs
    )
    assert all(config.model_name == "videomae" for config in configs)
    assert all(config.num_frames == 16 for config in configs)
    assert all(config.image_size == 224 for config in configs)


def test_hmdb51_videomae_evaluation_config_matches_the_eight_heldout_artifacts() -> None:
    config = RunEvaluationConfig.from_file(
        CONFIG_DIR / "hmdb51_videomae_full_split1_linear_probe_evaluation.json"
    )

    assert config.run_id == RUN_ID
    assert config.train_original == "embeddings/train/original.pt"
    assert config.heldout_original == "embeddings/heldout/original.pt"
    assert len(config.perturbations) == 8
    assert config.knn == {"metric": "cosine", "k_values": [5]}
    assert {spec.group for spec in config.perturbations} == {"motion", "appearance"}
