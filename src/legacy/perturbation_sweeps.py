"""Legacy SSV2×VideoMAE sweep runner retained for the anchor experiment only."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.legacy.config import ExperimentConfig
from src.artifacts import load_embedding_artifact, save_embedding_artifact
from src.legacy.embedding_extraction import (
    build_extraction_dataset,
    extract_embeddings,
)
from src.evaluation.knn import DistanceMetric
from src.evaluation.knn_perturbation import run_knn_perturbation_drop
from src.evaluation.sensitivity import run_embedding_sensitivity, save_json_report
from src.models.videomae_model import load_videomae_model, resolve_device
from src.reproducibility import seed_everything
from src.video.perturbations import VideoPerturbationConfig, parse_perturbation_name


DEFAULT_MATRIX_PATH = Path("configs/ssv2_videomae_50c_perturbation_matrix.json")
DEFAULT_CONFIG_PATH = Path("configs/ssv2_videomae_50c_validation.json")
DEFAULT_OUTPUT_DIR = Path("outputs/logs")
DEFAULT_REPORT_BASENAME = "ssv2_50c_train100_val30_videomae_base_16f"


@dataclass(frozen=True)
class SweepCase:
    sweep_name: str
    base_perturbation: str
    group: str
    parameter: str
    value: Any
    label: str
    output_path: str
    interpretation: str
    perturbation_parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def expand_sweep_cases(matrix: dict[str, Any]) -> list[SweepCase]:
    defaults_by_name = {
        str(entry["name"]): dict(entry.get("default_parameters", {}))
        for entry in matrix.get("first_round", [])
    }
    cases: list[SweepCase] = []
    for sweep in matrix.get("sweeps", []):
        base_perturbation = str(sweep["base_perturbation"])
        parameter = str(sweep["parameter"])
        defaults = {
            "seed": 0,
            "frame_index": None,
            "freeze_start_fraction": 0.5,
            "occlusion_size_fraction": 0.25,
            "occlusion_fill_value": 0,
            **defaults_by_name.get(base_perturbation, {}),
        }
        for raw_value in sweep.get("values", []):
            label, value = _sweep_label_and_value(raw_value)
            parameters = dict(defaults)
            parameters[parameter] = value
            cases.append(
                SweepCase(
                    sweep_name=str(sweep["sweep_name"]),
                    base_perturbation=base_perturbation,
                    group=str(sweep["group"]),
                    parameter=parameter,
                    value=value,
                    label=label,
                    output_path=_format_output_path(
                        str(sweep["output_path_template"]),
                        label=label,
                        value=value,
                    ),
                    interpretation=str(sweep.get("interpretation", "")),
                    perturbation_parameters=parameters,
                )
            )
    return cases


def generate_missing_sweep_artifacts(
    *,
    cases: list[SweepCase],
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    matrix: dict[str, Any],
    overwrite: bool = False,
    local_files_only: bool = True,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    config = ExperimentConfig.from_file(config_path)
    seed_everything(config.seed, deterministic=config.deterministic)
    scope = matrix.get("scope", {})
    split = str(scope.get("apply_perturbations_to_split", "validation"))
    index_path = Path(scope.get("validation_index_path", f"data/ssv2/index/{split}.jsonl"))
    device = resolve_device(config.device)
    model, model_metadata = load_videomae_model(
        config.model_checkpoint,
        device=str(device),
        local_files_only=local_files_only,
    )

    statuses: list[dict[str, Any]] = []
    for case in cases:
        output_path = Path(case.output_path)
        if output_path.exists() and not overwrite:
            artifact = load_embedding_artifact(output_path)
            statuses.append(
                {
                    "sweep_name": case.sweep_name,
                    "label": case.label,
                    "output_path": str(output_path),
                    "status": "existing",
                    "embeddings_shape": artifact["summary"]["embeddings_shape"],
                }
            )
            continue

        perturbation_config = _case_perturbation_config(case)
        dataset = build_extraction_dataset(
            index_path=index_path,
            num_frames=config.num_frames,
            sampling_strategy=config.sampling_strategy,
            image_size=config.image_size,
            local_files_only=local_files_only,
            perturbation_config=perturbation_config,
        )
        result = extract_embeddings(
            model=model,
            dataset=dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            device=device,
            embedding_type=config.embedding_type,  # type: ignore[arg-type]
            show_progress=show_progress,
        )
        config_snapshot = {
            **config.to_dict(),
            "split": split,
            "index_path": str(index_path),
            "model_checkpoint": config.model_checkpoint,
            "device": str(device),
            "output_path": str(output_path),
            "perturbation": perturbation_config.to_dict(),
            "sweep": case.to_dict(),
        }
        artifact = save_embedding_artifact(
            result,
            output_path,
            config_snapshot=config_snapshot,
            model_metadata=model_metadata,
            extraction_options={
                "limit": None,
                "pin_memory": False,
                "local_files_only": local_files_only,
                "processor_checkpoint": None,
                "perturbation": perturbation_config.to_dict(),
                "sweep": case.to_dict(),
            },
            overwrite=overwrite,
        )
        statuses.append(
            {
                "sweep_name": case.sweep_name,
                "label": case.label,
                "output_path": str(output_path),
                "status": "generated",
                "embeddings_shape": artifact["summary"]["embeddings_shape"],
            }
        )
    return statuses


def run_sweep_reports(
    *,
    matrix: dict[str, Any],
    cases: list[SweepCase],
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    metric: DistanceMetric | None = None,
    k_values: list[int] | tuple[int, ...] | None = None,
    query_batch_size: int = 1024,
    overwrite: bool = False,
) -> dict[str, Any]:
    scope = matrix.get("scope", {})
    original_path = Path(scope["validation_original_artifact"])
    train_path = Path(scope["train_reference_artifact"])
    resolved_metric = metric or scope.get("primary_knn_metric", "cosine")
    if resolved_metric not in {"cosine", "l2"}:
        raise ValueError(f"Unsupported KNN metric: {resolved_metric}")
    resolved_k_values = k_values or tuple(int(k) for k in scope.get("knn_k_values", [1, 5, 10]))
    output_path = Path(output_dir)

    sweep_entries: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        artifact_path = Path(case.output_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Sweep artifact does not exist: {artifact_path}")
        sensitivity = run_embedding_sensitivity(
            original_artifact_path=original_path,
            perturbed_artifact_path=artifact_path,
            perturbation_name=case.base_perturbation,
            perturbation_group=case.group,
        )
        _attach_sweep_metadata(sensitivity, case)
        sensitivity_path = _sensitivity_report_path(artifact_path, output_path)
        save_json_report(sensitivity, sensitivity_path, overwrite=overwrite)

        knn = run_knn_perturbation_drop(
            train_artifact_path=train_path,
            original_validation_artifact_path=original_path,
            perturbed_validation_artifact_path=artifact_path,
            k_values=resolved_k_values,
            metric=resolved_metric,  # type: ignore[arg-type]
            query_batch_size=query_batch_size,
            perturbation_name=case.base_perturbation,
            perturbation_group=case.group,
        )
        _attach_sweep_metadata(knn, case)
        knn_path = _knn_report_path(artifact_path, output_path, metric=resolved_metric)
        save_json_report(knn, knn_path, overwrite=overwrite)

        sweep_entries.setdefault(case.sweep_name, []).append(
            {
                "case": case.to_dict(),
                "sensitivity_report_path": str(sensitivity_path),
                "knn_report_path": str(knn_path),
                "sensitivity": sensitivity,
                "knn": knn,
            }
        )

    sweep_summaries: list[dict[str, Any]] = []
    report_basename = _report_basename(matrix)
    for sweep_name, entries in sweep_entries.items():
        summary = build_sweep_summary(
            matrix=matrix,
            sweep_name=sweep_name,
            entries=entries,
            metric=resolved_metric,
        )
        summary_path = output_path / f"{report_basename}_{sweep_name}_sweep_summary.json"
        save_json_report(summary, summary_path, overwrite=overwrite)
        sweep_summaries.append(
            {
                "sweep_name": sweep_name,
                "summary_report_path": str(summary_path),
            }
        )

    all_summary = {
        "format_version": 1,
        "report_type": "all_sweep_summary",
        "matrix_name": matrix.get("matrix_name"),
        "sweep_count": len(sweep_summaries),
        "sweeps": sweep_summaries,
    }
    all_summary_path = output_path / f"{report_basename}_all_sweeps_summary.json"
    save_json_report(all_summary, all_summary_path, overwrite=overwrite)
    return {
        "sweeps": sweep_summaries,
        "all_sweeps_summary_path": str(all_summary_path),
    }


def build_sweep_summary(
    *,
    matrix: dict[str, Any],
    sweep_name: str,
    entries: list[dict[str, Any]],
    metric: str,
) -> dict[str, Any]:
    if not entries:
        raise ValueError("At least one sweep entry is required")
    first_case = entries[0]["case"]
    cases = [_compact_case_entry(entry) for entry in entries]
    numeric_values = [
        case["value"]
        for case in cases
        if isinstance(case["value"], int | float) and not isinstance(case["value"], bool)
    ]
    trend_analysis = _build_trend_analysis(cases) if len(numeric_values) == len(cases) else None
    seed_repeat_summary = (
        _build_seed_repeat_summary(cases)
        if first_case["parameter"] == "seed"
        else None
    )
    return {
        "format_version": 1,
        "report_type": "perturbation_sweep_summary",
        "matrix_name": matrix.get("matrix_name"),
        "sweep_name": sweep_name,
        "base_perturbation": first_case["base_perturbation"],
        "group": first_case["group"],
        "parameter": first_case["parameter"],
        "interpretation": first_case["interpretation"],
        "metric": metric,
        "case_count": len(cases),
        "cases": cases,
        "trend_analysis": trend_analysis,
        "seed_repeat_summary": seed_repeat_summary,
    }


def run_matrix_sweeps(
    *,
    matrix_path: str | Path = DEFAULT_MATRIX_PATH,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    generate_missing_artifacts: bool = False,
    overwrite_artifacts: bool = False,
    overwrite_reports: bool = False,
    local_files_only: bool = True,
    no_progress: bool = False,
    metric: DistanceMetric | None = None,
    k_values: list[int] | tuple[int, ...] | None = None,
) -> dict[str, Any]:
    matrix = _load_json(Path(matrix_path))
    cases = expand_sweep_cases(matrix)
    artifact_statuses: list[dict[str, Any]] = []
    if generate_missing_artifacts:
        artifact_statuses = generate_missing_sweep_artifacts(
            cases=cases,
            config_path=config_path,
            matrix=matrix,
            overwrite=overwrite_artifacts,
            local_files_only=local_files_only,
            show_progress=not no_progress,
        )
    report_result = run_sweep_reports(
        matrix=matrix,
        cases=cases,
        output_dir=output_dir,
        metric=metric,
        k_values=k_values,
        overwrite=overwrite_reports,
    )
    return {
        "matrix_path": str(matrix_path),
        "case_count": len(cases),
        "artifact_statuses": artifact_statuses,
        **report_result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run perturbation strength/seed/position sweep reports."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--generate-missing-artifacts", action="store_true")
    parser.add_argument("--overwrite-artifacts", action="store_true")
    parser.add_argument("--overwrite-reports", action="store_true")
    parser.add_argument("--allow-online-files", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--metric", choices=["cosine", "l2"], default=None)
    parser.add_argument("--k", type=int, nargs="+", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_matrix_sweeps(
        matrix_path=args.matrix,
        config_path=args.config,
        output_dir=args.output_dir,
        generate_missing_artifacts=args.generate_missing_artifacts,
        overwrite_artifacts=args.overwrite_artifacts,
        overwrite_reports=args.overwrite_reports,
        local_files_only=not args.allow_online_files,
        no_progress=args.no_progress,
        metric=args.metric,
        k_values=args.k,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _compact_case_entry(entry: dict[str, Any]) -> dict[str, Any]:
    case = dict(entry["case"])
    sensitivity_summary = entry["sensitivity"]["summary"]["metrics"]
    k_results = {
        str(result["k"]): {
            "all_perturbed_accuracy": result["all"]["perturbed_accuracy"],
            "all_accuracy_drop": result["all"]["absolute_accuracy_drop"],
            "train_seen_perturbed_accuracy": result["train_seen"]["perturbed_accuracy"],
            "train_seen_accuracy_drop": result["train_seen"]["absolute_accuracy_drop"],
            "prediction_change_rate": result["prediction_changes"]["all"][
                "prediction_change_rate"
            ],
        }
        for result in entry["knn"]["results"]
    }
    return {
        **case,
        "sensitivity_report_path": entry["sensitivity_report_path"],
        "knn_report_path": entry["knn_report_path"],
        "mean_cosine_distance": sensitivity_summary["cosine_distance"]["mean"],
        "median_cosine_distance": sensitivity_summary["cosine_distance"]["median"],
        "mean_l2_distance": sensitivity_summary["l2_distance"]["mean"],
        "mean_relative_l2_distance": sensitivity_summary["relative_l2_distance"]["mean"],
        "knn_by_k": k_results,
    }


def _build_trend_analysis(cases: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(cases, key=lambda item: float(item["value"]))
    return {
        "ordered_values": [case["value"] for case in ordered],
        "mean_cosine_distance_direction": _monotonic_direction(
            [case["mean_cosine_distance"] for case in ordered]
        ),
        "k1_all_accuracy_drop_direction": _monotonic_direction(
            [case["knn_by_k"]["1"]["all_accuracy_drop"] for case in ordered]
        )
        if "1" in ordered[0]["knn_by_k"]
        else None,
        "k1_train_seen_accuracy_drop_direction": _monotonic_direction(
            [case["knn_by_k"]["1"]["train_seen_accuracy_drop"] for case in ordered]
        )
        if "1" in ordered[0]["knn_by_k"]
        else None,
    }


def _build_seed_repeat_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "seed_count": len(cases),
        "mean_cosine_distance": _summary_stats(
            [case["mean_cosine_distance"] for case in cases]
        ),
        "k1_all_accuracy_drop": _summary_stats(
            [case["knn_by_k"]["1"]["all_accuracy_drop"] for case in cases]
        )
        if "1" in cases[0]["knn_by_k"]
        else None,
        "k1_prediction_change_rate": _summary_stats(
            [case["knn_by_k"]["1"]["prediction_change_rate"] for case in cases]
        )
        if "1" in cases[0]["knn_by_k"]
        else None,
    }


def _summary_stats(values: list[float]) -> dict[str, float | int]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def _monotonic_direction(values: list[float]) -> str:
    if len(values) < 2:
        return "not_applicable"
    nondecreasing = all(right >= left for left, right in zip(values, values[1:]))
    nonincreasing = all(right <= left for left, right in zip(values, values[1:]))
    if nondecreasing and nonincreasing:
        return "flat"
    if nondecreasing:
        return "increasing"
    if nonincreasing:
        return "decreasing"
    return "non_monotonic"


def _case_perturbation_config(case: SweepCase) -> VideoPerturbationConfig:
    params = case.perturbation_parameters
    return VideoPerturbationConfig(
        name=parse_perturbation_name(case.base_perturbation),
        seed=int(params.get("seed", 0)),
        frame_index=params.get("frame_index"),
        freeze_start_fraction=float(params.get("freeze_start_fraction", 0.5)),
        occlusion_size_fraction=float(params.get("occlusion_size_fraction", 0.25)),
        occlusion_fill_value=int(params.get("occlusion_fill_value", 0)),
    )


def _attach_sweep_metadata(report: dict[str, Any], case: SweepCase) -> None:
    report["sweep"] = case.to_dict()


def _sensitivity_report_path(artifact_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{artifact_path.stem}_sensitivity.json"


def _knn_report_path(artifact_path: Path, output_dir: Path, *, metric: str) -> Path:
    return output_dir / f"{artifact_path.stem}_knn_{metric}.json"


def _sweep_label_and_value(raw_value: Any) -> tuple[str, Any]:
    if isinstance(raw_value, dict):
        return str(raw_value["label"]), raw_value.get("value")
    return str(raw_value), raw_value


def _format_output_path(template: str, *, label: str, value: Any) -> str:
    value_text = "none" if value is None else str(value)
    return template.format(label=label, value=value_text)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_basename(matrix: dict[str, Any]) -> str:
    return str(matrix.get("report_basename", DEFAULT_REPORT_BASENAME))


if __name__ == "__main__":
    raise SystemExit(main())
