from __future__ import annotations

from src.perturbation_sweeps import (
    build_sweep_summary,
    expand_sweep_cases,
)


def test_expand_sweep_cases_applies_defaults_and_formats_paths() -> None:
    matrix = {
        "first_round": [
            {
                "name": "freeze_tail",
                "default_parameters": {
                    "seed": 9,
                    "frame_index": None,
                    "freeze_start_fraction": 0.5,
                    "occlusion_size_fraction": 0.25,
                    "occlusion_fill_value": 0,
                },
            },
            {
                "name": "single_frame",
                "default_parameters": {
                    "seed": 0,
                    "frame_index": None,
                    "freeze_start_fraction": 0.5,
                    "occlusion_size_fraction": 0.25,
                    "occlusion_fill_value": 0,
                },
            },
        ],
        "sweeps": [
            {
                "sweep_name": "freeze_tail_start_fraction",
                "base_perturbation": "freeze_tail",
                "group": "motion",
                "parameter": "freeze_start_fraction",
                "values": [0.25, 0.5],
                "output_path_template": "out/freeze_tail_start-{value}.pt",
                "interpretation": "smaller freezes more",
            },
            {
                "sweep_name": "single_frame_position",
                "base_perturbation": "single_frame",
                "group": "motion",
                "parameter": "frame_index",
                "values": [
                    {"label": "first", "value": 0},
                    {"label": "center", "value": None},
                ],
                "output_path_template": "out/single_frame_{label}.pt",
                "interpretation": "position",
            },
        ],
    }

    cases = expand_sweep_cases(matrix)

    assert len(cases) == 4
    assert cases[0].output_path == "out/freeze_tail_start-0.25.pt"
    assert cases[0].perturbation_parameters["seed"] == 9
    assert cases[0].perturbation_parameters["freeze_start_fraction"] == 0.25
    assert cases[2].label == "first"
    assert cases[2].output_path == "out/single_frame_first.pt"
    assert cases[3].value is None
    assert cases[3].perturbation_parameters["frame_index"] is None


def test_build_sweep_summary_reports_monotonic_trends() -> None:
    entries = [
        _entry(
            value=0.15,
            mean_cosine_distance=0.01,
            all_drop=0.0,
            train_seen_drop=0.0,
            prediction_change_rate=0.1,
        ),
        _entry(
            value=0.25,
            mean_cosine_distance=0.02,
            all_drop=0.01,
            train_seen_drop=0.02,
            prediction_change_rate=0.2,
        ),
        _entry(
            value=0.4,
            mean_cosine_distance=0.03,
            all_drop=0.02,
            train_seen_drop=0.04,
            prediction_change_rate=0.3,
        ),
    ]

    summary = build_sweep_summary(
        matrix={"matrix_name": "tiny"},
        sweep_name="center_occlusion_size_fraction",
        entries=entries,
        metric="cosine",
    )

    assert summary["case_count"] == 3
    assert summary["trend_analysis"]["mean_cosine_distance_direction"] == "increasing"
    assert summary["trend_analysis"]["k1_all_accuracy_drop_direction"] == "increasing"
    assert summary["cases"][2]["knn_by_k"]["1"]["prediction_change_rate"] == 0.3


def test_build_sweep_summary_reports_seed_repeat_stats() -> None:
    entries = [
        _entry(value=0, mean_cosine_distance=0.01, all_drop=0.0, train_seen_drop=0.0),
        _entry(value=1, mean_cosine_distance=0.03, all_drop=0.02, train_seen_drop=0.04),
        _entry(value=2, mean_cosine_distance=0.02, all_drop=0.01, train_seen_drop=0.02),
    ]
    for entry in entries:
        entry["case"]["sweep_name"] = "temporal_shuffle_seed_repeat"
        entry["case"]["parameter"] = "seed"

    summary = build_sweep_summary(
        matrix={"matrix_name": "tiny"},
        sweep_name="temporal_shuffle_seed_repeat",
        entries=entries,
        metric="cosine",
    )

    assert summary["seed_repeat_summary"]["seed_count"] == 3
    assert summary["seed_repeat_summary"]["mean_cosine_distance"]["mean"] > 0.0
    assert summary["trend_analysis"]["mean_cosine_distance_direction"] == "non_monotonic"


def _entry(
    *,
    value,
    mean_cosine_distance: float,
    all_drop: float,
    train_seen_drop: float,
    prediction_change_rate: float = 0.0,
) -> dict:
    return {
        "case": {
            "sweep_name": "center_occlusion_size_fraction",
            "base_perturbation": "center_occlusion",
            "group": "appearance",
            "parameter": "occlusion_size_fraction",
            "value": value,
            "label": str(value),
            "output_path": f"out/{value}.pt",
            "interpretation": "larger removes more",
            "perturbation_parameters": {
                "occlusion_size_fraction": value,
            },
        },
        "sensitivity_report_path": f"out/{value}_sensitivity.json",
        "knn_report_path": f"out/{value}_knn.json",
        "sensitivity": {
            "summary": {
                "metrics": {
                    "cosine_distance": {
                        "mean": mean_cosine_distance,
                        "median": mean_cosine_distance,
                    },
                    "l2_distance": {"mean": mean_cosine_distance * 10.0},
                    "relative_l2_distance": {"mean": mean_cosine_distance * 2.0},
                }
            }
        },
        "knn": {
            "results": [
                {
                    "k": 1,
                    "all": {
                        "perturbed_accuracy": 0.02 - all_drop,
                        "absolute_accuracy_drop": all_drop,
                    },
                    "train_seen": {
                        "perturbed_accuracy": 0.04 - train_seen_drop,
                        "absolute_accuracy_drop": train_seen_drop,
                    },
                    "prediction_changes": {
                        "all": {
                            "prediction_change_rate": prediction_change_rate,
                        }
                    },
                }
            ]
        },
    }
