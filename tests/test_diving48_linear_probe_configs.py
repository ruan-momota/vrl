from __future__ import annotations

from pathlib import Path

from src.pipeline.evaluate import RunEvaluationConfig
from src.pipeline.run import RunConfig


SUBSET_ID = "c32_train50_heldout15"
SUBSET_SUMMARY = "data/diving48/subsets/c32_train50_heldout15/summary.json"
TRAIN_INDEX = "data/diving48/subsets/c32_train50_heldout15/train.jsonl"
HELDOUT_INDEX = "data/diving48/subsets/c32_train50_heldout15/heldout.jsonl"

MODEL_CONFIGS = {
    "videomae": {
        "directory": Path("configs/runs/diving48_videomae_linear_probe"),
        "run_id": "diving48-c32-train50-heldout15-videomae-base-frozen-linear-probe",
        "checkpoint": "MCG-NJU/videomae-base",
        "input_profile": "videomae_16f_224_deterministic_center_clip",
        "num_frames": 16,
        "image_size": 224,
    },
    "slowfast": {
        "directory": Path("configs/runs/diving48_slowfast_linear_probe"),
        "run_id": "diving48-c32-train50-heldout15-slowfast-r50-8x8-frozen-linear-probe",
        "checkpoint": "facebookresearch/pytorchvideo:slowfast_r50",
        "input_profile": "slowfast_r50_8x8_32f_256_deterministic_center_clip_alpha4",
        "num_frames": 32,
        "image_size": 256,
    },
    "dinov2": {
        "directory": Path("configs/runs/diving48_dinov2_linear_probe"),
        "run_id": "diving48-c32-train50-heldout15-dinov2-base-frame-mean-frozen-linear-probe",
        "checkpoint": "facebook/dinov2-base",
        "input_profile": "dinov2_base_frame_mean_16f_224_deterministic_center_clip",
        "num_frames": 16,
        "image_size": 224,
    },
}

PERTURBATION_LABELS = {
    "temporal-shuffle-mid",
    "freeze-tail-low",
    "freeze-tail-mid",
    "freeze-tail-high",
    "color-low",
    "color-mid",
    "color-high",
    "spatial-blur-mid",
    "rgb-quantization-low",
    "rgb-quantization-mid",
    "rgb-quantization-high",
    "solarization-low",
    "solarization-mid",
    "solarization-high",
}


def test_diving48_extraction_configs_are_complete_and_model_specific() -> None:
    for model_name, expected in MODEL_CONFIGS.items():
        configs = _extraction_configs(expected["directory"], model_name)

        assert len(configs) == 16
        assert {config.dataset_name for config in configs} == {"diving48"}
        assert {config.subset_id for config in configs} == {SUBSET_ID}
        assert {config.subset_summary_path for config in configs} == {SUBSET_SUMMARY}
        assert {config.model_name for config in configs} == {model_name}
        assert {config.model_checkpoint for config in configs} == {expected["checkpoint"]}
        assert {config.input_profile for config in configs} == {expected["input_profile"]}
        assert {config.num_frames for config in configs} == {expected["num_frames"]}
        assert {config.image_size for config in configs} == {expected["image_size"]}
        assert {config.sampling_strategy for config in configs} == {
            "deterministic_center_clip"
        }
        assert {config.batch_size for config in configs} == {1}
        assert {config.seed for config in configs} == {42}
        assert {config.resolved_run_id for config in configs} == {expected["run_id"]}
        assert {config.output_root for config in configs} == {"outputs/runs"}
        assert {config.split for config in configs} == {"train", "heldout"}

        train_configs = [config for config in configs if config.split == "train"]
        heldout_configs = [config for config in configs if config.split == "heldout"]
        assert len(train_configs) == 1
        assert len(heldout_configs) == 15
        assert train_configs[0].index_path == TRAIN_INDEX
        assert {config.index_path for config in heldout_configs} == {HELDOUT_INDEX}

        perturbation_configs = [
            config for config in configs if config.perturbation["name"] != "none"
        ]
        assert len(perturbation_configs) == 14
        assert {
            config.perturbation["artifact_label"] for config in perturbation_configs
        } == PERTURBATION_LABELS


def test_diving48_smoke_configs_are_separate_from_full_runs() -> None:
    for model_name, expected in MODEL_CONFIGS.items():
        smoke_configs = _smoke_configs(expected["directory"], model_name)

        assert len(smoke_configs) == 6
        assert {config.dataset_name for config in smoke_configs} == {"diving48"}
        assert {config.subset_id for config in smoke_configs} == {
            "c32_train50_heldout15_smoke"
        }
        assert {config.output_root for config in smoke_configs} == {"outputs/runs-smoke"}
        assert {config.resolved_run_id for config in smoke_configs} == {
            f"{expected['run_id']}-smoke"
        }
        assert {config.index_path for config in smoke_configs if config.split == "train"} == {
            TRAIN_INDEX
        }
        assert {
            config.index_path for config in smoke_configs if config.split == "heldout"
        } == {HELDOUT_INDEX}


def test_diving48_evaluation_configs_match_the_heldout_artifacts() -> None:
    for model_name, expected in MODEL_CONFIGS.items():
        config = RunEvaluationConfig.from_file(
            expected["directory"] / f"diving48_{model_name}_c32_linear_probe_evaluation.json"
        )

        assert config.run_id == expected["run_id"]
        assert config.output_root == "outputs/runs"
        assert config.train_original == "embeddings/train/original.pt"
        assert config.heldout_original == "embeddings/heldout/original.pt"
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
        assert {spec.role for spec in config.perturbations} == {"fixed_mid", "curve"}
        assert {
            Path(spec.artifact).stem for spec in config.perturbations
        } == PERTURBATION_LABELS


def _extraction_configs(directory: Path, model_name: str) -> list[RunConfig]:
    return [
        RunConfig.from_file(path)
        for path in sorted(directory.glob(f"diving48_{model_name}_c32_*.json"))
        if "smoke" not in path.name and "evaluation" not in path.name
    ]


def _smoke_configs(directory: Path, model_name: str) -> list[RunConfig]:
    return [
        RunConfig.from_file(path)
        for path in sorted(directory.glob(f"diving48_{model_name}_c32_smoke_*.json"))
    ]
