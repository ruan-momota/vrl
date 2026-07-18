from __future__ import annotations

from pathlib import Path

from src.pipeline.evaluate import RunEvaluationConfig
from src.pipeline.run import RunConfig


SSV2_CONFIG_DIR = Path("configs/runs/ssv2_slowfast_linear_probe")
SSV2_RUN_ID = "ssv2-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe"
UCF101_CONFIG_DIR = Path("configs/runs/ucf101_slowfast_linear_probe")
UCF101_RUN_ID = "ucf101-c50-train100-heldout30-slowfast-r50-8x8-frozen-linear-probe"


def test_ssv2_slowfast_extraction_configs_share_one_run_identity() -> None:
    configs = _extraction_configs(SSV2_CONFIG_DIR, "ssv2_slowfast_c50_*.json")

    assert len(configs) == 16
    assert {config.dataset_name for config in configs} == {"ssv2"}
    assert {config.model_name for config in configs} == {"slowfast"}
    assert {config.model_checkpoint for config in configs} == {
        "facebookresearch/pytorchvideo:slowfast_r50"
    }
    assert {config.resolved_run_id for config in configs} == {SSV2_RUN_ID}
    assert {config.split for config in configs} == {"train", "validation"}
    assert {config.num_frames for config in configs} == {32}
    assert {config.image_size for config in configs} == {256}
    assert sum(config.perturbation["name"] != "none" for config in configs) == 14
    assert all(config.subset_summary_path == "data/ssv2/index/summary.json" for config in configs)


def test_ucf101_slowfast_extraction_configs_share_one_run_identity() -> None:
    configs = _extraction_configs(UCF101_CONFIG_DIR, "ucf101_slowfast_c50_*.json")

    assert len(configs) == 16
    assert {config.dataset_name for config in configs} == {"ucf101"}
    assert {config.model_name for config in configs} == {"slowfast"}
    assert {config.model_checkpoint for config in configs} == {
        "facebookresearch/pytorchvideo:slowfast_r50"
    }
    assert {config.resolved_run_id for config in configs} == {UCF101_RUN_ID}
    assert {config.split for config in configs} == {"train", "heldout"}
    assert {config.num_frames for config in configs} == {32}
    assert {config.image_size for config in configs} == {256}
    assert sum(config.perturbation["name"] != "none" for config in configs) == 14
    assert all(
        config.subset_summary_path
        == "data/ucf101/subsets/c50_train100_heldout30/summary.json"
        for config in configs
    )


def test_slowfast_evaluation_configs_match_the_heldout_artifacts() -> None:
    ssv2 = RunEvaluationConfig.from_file(
        SSV2_CONFIG_DIR / "ssv2_slowfast_c50_linear_probe_evaluation.json"
    )
    ucf101 = RunEvaluationConfig.from_file(
        UCF101_CONFIG_DIR / "ucf101_slowfast_c50_linear_probe_evaluation.json"
    )

    assert ssv2.run_id == SSV2_RUN_ID
    assert ssv2.train_original == "embeddings/train/original.pt"
    assert ssv2.heldout_original == "embeddings/validation/original.pt"
    assert ucf101.run_id == UCF101_RUN_ID
    assert ucf101.train_original == "embeddings/train/original.pt"
    assert ucf101.heldout_original == "embeddings/heldout/original.pt"
    for config in (ssv2, ucf101):
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


def _extraction_configs(directory: Path, pattern: str) -> list[RunConfig]:
    return [
        RunConfig.from_file(path)
        for path in sorted(directory.glob(pattern))
        if "smoke" not in path.name and "evaluation" not in path.name
    ]
