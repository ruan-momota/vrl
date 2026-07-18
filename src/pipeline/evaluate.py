"""Run-scoped frozen-linear-probe sensitivity evaluation for video embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts import load_embedding_artifact
from src.evaluation.alignment import check_paired_embedding_alignment
from src.evaluation.bootstrap import BootstrapConfig
from src.evaluation.knn import evaluate_knn_baseline
from src.evaluation.knn_perturbation import evaluate_knn_perturbation_drop
from src.evaluation.linear_probe import (
    LinearProbeConfig,
    build_paired_linear_probe_report,
    evaluate_frozen_linear_probe,
    fit_frozen_linear_probe,
    save_frozen_linear_probe,
    save_linear_probe_predictions,
)
from src.evaluation.reporting import render_markdown_table, write_bar_chart_svg, write_line_chart_svg
from src.evaluation.run_reporting import (
    build_failure_summary,
    build_legacy_anchor_comparison,
    build_qualitative_records,
    build_qualitative_summary,
    build_run_provenance,
)
from src.evaluation.sensitivity import build_embedding_sensitivity_report


_BASE_LIMITATIONS = [
    "Each perturbation measures sensitivity to a specific intervention; it does not by itself prove human-like action understanding.",
    "Temporal shuffle and freeze-tail also alter clip naturalness and temporal redundancy, so they are temporal-dependence probes rather than isolated causal motion interventions.",
    "The fixed color transform and spatial blur preserve frame order but can still alter normalization-sensitive statistics, object visibility, and texture cues; they are not perfectly isolated appearance interventions.",
]


@dataclass(frozen=True)
class PerturbationEvaluationSpec:
    name: str
    artifact: str
    group: str
    role: str
    strength: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "PerturbationEvaluationSpec":
        return cls(
            name=str(data["name"]),
            artifact=str(data["artifact"]),
            group=str(data["group"]),
            role=str(data["role"]),
            strength=None if data.get("strength") is None else str(data["strength"]),
        )


@dataclass(frozen=True)
class RunEvaluationConfig:
    run_id: str
    output_root: str
    train_original: str
    heldout_original: str
    perturbations: tuple[PerturbationEvaluationSpec, ...]
    linear_probe: dict[str, Any]
    bootstrap: dict[str, Any]
    knn: dict[str, Any]
    hardware: dict[str, Any]
    legacy_anchor_comparison: dict[str, Any] | None

    @classmethod
    def from_file(cls, path: str | Path) -> "RunEvaluationConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise TypeError("Run evaluation config must be a JSON object")
        perturbations = data.get("perturbations")
        if not isinstance(perturbations, list) or not perturbations:
            raise ValueError("Run evaluation config requires non-empty perturbations")
        return cls(
            run_id=str(data["run_id"]),
            output_root=str(data.get("output_root", "outputs/runs")),
            train_original=str(data["train_original"]),
            heldout_original=str(data["heldout_original"]),
            perturbations=tuple(
                PerturbationEvaluationSpec.from_mapping(item)
                for item in perturbations
                if isinstance(item, dict)
            ),
            linear_probe=_mapping(data.get("linear_probe"), name="linear_probe"),
            bootstrap=_mapping(data.get("bootstrap"), name="bootstrap"),
            knn=_mapping(data.get("knn"), name="knn"),
            hardware=_optional_mapping(data.get("hardware")),
            legacy_anchor_comparison=_optional_mapping(data.get("legacy_anchor_comparison"))
            or None,
        )

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root) / self.run_id


def run_linear_probe_evaluation(
    config: RunEvaluationConfig,
    *,
    overwrite: bool = False,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Fit one train-only probe and evaluate all configured held-out interventions."""
    directories = _ensure_run_output_directories(config.run_dir)
    _record_evaluation_command(config, command=command)
    train_path = _artifact_path(config, config.train_original)
    original_path = _artifact_path(config, config.heldout_original)
    train_artifact = _load_run_artifact(train_path, config.run_id)
    original_artifact = _load_run_artifact(original_path, config.run_id)
    artifact_paths: dict[str, Path] = {
        "train_original": train_path,
        "heldout_original": original_path,
    }
    artifacts: dict[str, dict[str, Any]] = {
        "train_original": train_artifact,
        "heldout_original": original_artifact,
    }

    probe_config = _linear_probe_config(config.linear_probe)
    bootstrap_config = BootstrapConfig(**config.bootstrap)
    bootstrap_config.validate()
    knn_metric = str(config.knn.get("metric", "cosine"))
    knn_k_values = tuple(int(value) for value in config.knn.get("k_values", [5]))

    probe = fit_frozen_linear_probe(train_artifact, config=probe_config)
    probe_path = directories["metrics"] / "frozen_linear_probe.pt"
    _ensure_writable(probe_path, overwrite=overwrite)
    save_frozen_linear_probe(probe, probe_path)

    original_evaluation = evaluate_frozen_linear_probe(
        probe,
        original_artifact,
        artifact_path=original_path,
    )
    original_predictions_path = directories["metrics"] / "predictions" / "heldout_original.pt"
    _ensure_writable(original_predictions_path, overwrite=overwrite)
    save_linear_probe_predictions(original_evaluation, original_predictions_path)

    original_knn = evaluate_knn_baseline(
        train_artifact=train_artifact,
        validation_artifact=original_artifact,
        k_values=knn_k_values,
        metric=knn_metric,  # type: ignore[arg-type]
        train_artifact_path=str(train_path),
        validation_artifact_path=str(original_path),
    )
    original_knn_path = directories["metrics"] / "knn_original.json"
    _write_json(original_knn.to_dict(), original_knn_path, overwrite=overwrite)
    experiment = _experiment_labels(train_artifact)

    rows: list[dict[str, Any]] = []
    sensitivity_paths: list[str] = []
    linear_probe_paths: list[str] = []
    knn_paths: list[str] = []
    quality_records: list[dict[str, Any]] = []
    qualitative_records: list[dict[str, Any]] = []
    for spec in config.perturbations:
        perturbed_path = _artifact_path(config, spec.artifact)
        perturbed_artifact = _load_run_artifact(perturbed_path, config.run_id)
        alignment = check_paired_embedding_alignment(original_artifact, perturbed_artifact)
        label = _artifact_label(spec)
        artifact_paths[label] = perturbed_path
        artifacts[label] = perturbed_artifact

        sensitivity = build_embedding_sensitivity_report(
            original_artifact=original_artifact,
            perturbed_artifact=perturbed_artifact,
            original_artifact_path=original_path,
            perturbed_artifact_path=perturbed_path,
            perturbation_name=spec.name,
            perturbation_group=spec.group,
            bootstrap_config=bootstrap_config,
        )
        sensitivity_path = directories["logs"] / "sensitivity" / f"{label}.json"
        _write_json(sensitivity, sensitivity_path, overwrite=overwrite)
        sensitivity_paths.append(str(sensitivity_path))

        perturbed_evaluation = evaluate_frozen_linear_probe(
            probe,
            perturbed_artifact,
            artifact_path=perturbed_path,
        )
        predictions_path = directories["metrics"] / "predictions" / f"{label}.pt"
        _ensure_writable(predictions_path, overwrite=overwrite)
        save_linear_probe_predictions(perturbed_evaluation, predictions_path)
        linear_probe_report = build_paired_linear_probe_report(
            original_evaluation,
            perturbed_evaluation,
            bootstrap_config=bootstrap_config,
            perturbation_name=spec.name,
            perturbation_group=spec.group,
        )
        linear_probe_path = directories["metrics"] / "linear_probe" / f"{label}.json"
        _write_json(linear_probe_report, linear_probe_path, overwrite=overwrite)
        linear_probe_paths.append(str(linear_probe_path))

        knn_report = evaluate_knn_perturbation_drop(
            train_artifact=train_artifact,
            original_validation_artifact=original_artifact,
            perturbed_validation_artifact=perturbed_artifact,
            k_values=knn_k_values,
            metric=knn_metric,  # type: ignore[arg-type]
            train_artifact_path=train_path,
            original_validation_artifact_path=original_path,
            perturbed_validation_artifact_path=perturbed_path,
            perturbation_name=spec.name,
            perturbation_group=spec.group,
        )
        knn_path = directories["logs"] / "knn" / f"{label}.json"
        _write_json(knn_report, knn_path, overwrite=overwrite)
        knn_paths.append(str(knn_path))

        rows.append(
            _summary_row(
                spec=spec,
                sensitivity=sensitivity,
                linear_probe_report=linear_probe_report,
                knn_report=knn_report,
                knn_k=knn_k_values[0],
            )
        )
        quality_records.extend(
            _quality_audit_records(
                original_artifact=original_artifact,
                perturbed_artifact=perturbed_artifact,
                perturbation_label=label,
                seed=int(config.bootstrap.get("seed", 42)),
            )
        )
        qualitative_records.extend(
            build_qualitative_records(
                sensitivity,
                linear_probe_report,
                artifact_label=label,
            )
        )

    qualitative_summary = build_qualitative_summary(qualitative_records)
    failure_summary = build_failure_summary(artifacts, quality_records)
    legacy_comparison = _build_legacy_comparison(
        config=config,
        rows=rows,
        original_knn=original_knn.to_dict(),
    )
    provenance = build_run_provenance(
        run_dir=config.run_dir,
        artifact_paths=artifact_paths,
        artifacts=artifacts,
        hardware=config.hardware,
    )

    summary = {
        "format_version": 1,
        "report_type": "frozen_linear_probe_sensitivity_summary",
        "run_id": config.run_id,
        "command": command or [],
        "train_original_artifact": str(train_path),
        "heldout_original_artifact": str(original_path),
        "linear_probe_artifact": str(probe_path),
        "linear_probe_metadata": probe.metadata,
        "original_linear_probe_accuracy": original_evaluation.accuracy,
        "original_knn": original_knn.to_dict(),
        "bootstrap": bootstrap_config.to_dict(),
        "rows": rows,
        "qualitative_samples": qualitative_summary,
        "failure_summary": failure_summary,
        "legacy_anchor_comparison": legacy_comparison,
        "experiment": experiment,
        "limitations": _limitations_for_experiment(experiment),
    }
    summary_json = directories["reports"] / "linear_probe_sensitivity_summary.json"
    summary_csv = directories["reports"] / "linear_probe_perturbation_summary.csv"
    report_markdown = directories["reports"] / "linear_probe_sensitivity_report.md"
    quality_path = directories["reports"] / "quality_audit.json"
    provenance_path = directories["reports"] / "run_provenance.json"
    legacy_comparison_json = directories["reports"] / "legacy_anchor_comparison.json"
    legacy_comparison_markdown = directories["reports"] / "legacy_anchor_comparison.md"
    _write_json(summary, summary_json, overwrite=overwrite)
    _write_csv(rows, summary_csv, overwrite=overwrite)
    _write_markdown_report(summary, report_markdown, overwrite=overwrite)
    _write_json(
        {
            "format_version": 1,
            "run_id": config.run_id,
            "fixed_seed": int(config.bootstrap.get("seed", 42)),
            "records": quality_records,
            "failure_summary": failure_summary,
        },
        quality_path,
        overwrite=overwrite,
    )
    _write_linear_probe_plots(
        rows,
        directories["plots"],
        experiment=experiment,
        overwrite=overwrite,
    )
    _write_json(provenance, provenance_path, overwrite=overwrite)
    if legacy_comparison is not None:
        _write_json(legacy_comparison, legacy_comparison_json, overwrite=overwrite)
        _write_legacy_comparison_markdown(
            legacy_comparison,
            legacy_comparison_markdown,
            overwrite=overwrite,
        )
    return {
        "run_id": config.run_id,
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "report_markdown": str(report_markdown),
        "quality_audit": str(quality_path),
        "provenance": str(provenance_path),
        "legacy_anchor_comparison": None
        if legacy_comparison is None
        else str(legacy_comparison_json),
        "linear_probe": str(probe_path),
        "sensitivity_reports": sensitivity_paths,
        "linear_probe_reports": linear_probe_paths,
        "knn_reports": knn_paths,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one run-scoped frozen-embedding experiment.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_linear_probe_evaluation(
        RunEvaluationConfig.from_file(args.config),
        overwrite=args.overwrite,
        command=[sys.executable, "-m", "src.pipeline.evaluate", "--config", str(args.config)],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"Run evaluation config {name} must be an object")
    return dict(value)


def _optional_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("Optional run evaluation configuration sections must be objects")
    return dict(value)


def _artifact_path(config: RunEvaluationConfig, artifact: str) -> Path:
    path = Path(artifact)
    return path if path.is_absolute() else config.run_dir / path


def _load_run_artifact(path: Path, run_id: str) -> dict[str, Any]:
    artifact = load_embedding_artifact(path)
    if artifact.get("run_id") != run_id:
        raise ValueError(
            f"Artifact {path} does not belong to expected run {run_id!r}: {artifact.get('run_id')!r}"
        )
    return artifact


def _linear_probe_config(mapping: dict[str, Any]) -> LinearProbeConfig:
    resolved = dict(mapping)
    if "l2_values" in resolved:
        resolved["l2_values"] = tuple(float(value) for value in resolved["l2_values"])
    return LinearProbeConfig(**resolved)


def _ensure_run_output_directories(run_dir: Path) -> dict[str, Path]:
    directories = {
        "metrics": run_dir / "metrics",
        "reports": run_dir / "reports",
        "plots": run_dir / "plots",
        "logs": run_dir / "logs",
    }
    for directory in (
        *directories.values(),
        directories["metrics"] / "predictions",
        directories["metrics"] / "linear_probe",
        directories["logs"] / "sensitivity",
        directories["logs"] / "knn",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _record_evaluation_command(config: RunEvaluationConfig, *, command: list[str] | None) -> None:
    if command is None:
        return
    manifest_path = config.run_dir / "manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("run_id") != config.run_id:
        raise ValueError(f"Run manifest does not match evaluation run ID: {manifest_path}")
    commands = manifest.setdefault("evaluation_commands", [])
    if not isinstance(commands, list):
        raise TypeError(f"Run manifest has invalid evaluation_commands: {manifest_path}")
    if command not in commands:
        commands.append(command)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_label(spec: PerturbationEvaluationSpec) -> str:
    name = Path(spec.artifact).stem
    return name or spec.name.replace("_", "-")


def _summary_row(
    *,
    spec: PerturbationEvaluationSpec,
    sensitivity: dict[str, Any],
    linear_probe_report: dict[str, Any],
    knn_report: dict[str, Any],
    knn_k: int,
) -> dict[str, Any]:
    sensitivity_metrics = sensitivity["summary"]["metrics"]
    sensitivity_bootstrap = sensitivity["bootstrap"]["cosine_distance"]["statistics"]
    knn_by_k = {int(result["k"]): result for result in knn_report["results"]}
    knn = knn_by_k[knn_k]
    accuracy_bootstrap = linear_probe_report["accuracy_drop_bootstrap"]["statistics"]["mean"]
    c2i_bootstrap = linear_probe_report["correct_to_incorrect_bootstrap"]["statistics"]["mean"]
    return {
        "perturbation": spec.name,
        "artifact_label": _artifact_label(spec),
        "group": spec.group,
        "role": spec.role,
        "strength": spec.strength,
        "mean_cosine_distance": sensitivity_metrics["cosine_distance"]["mean"],
        "median_cosine_distance": sensitivity_metrics["cosine_distance"]["median"],
        "cosine_distance_ci_lower": sensitivity_bootstrap["mean"]["ci_lower"],
        "cosine_distance_ci_upper": sensitivity_bootstrap["mean"]["ci_upper"],
        "linear_probe_original_accuracy": linear_probe_report["original_accuracy"],
        "linear_probe_perturbed_accuracy": linear_probe_report["perturbed_accuracy"],
        "linear_probe_accuracy_drop": linear_probe_report["accuracy_drop"],
        "linear_probe_accuracy_drop_ci_lower": accuracy_bootstrap["ci_lower"],
        "linear_probe_accuracy_drop_ci_upper": accuracy_bootstrap["ci_upper"],
        "correct_to_incorrect_rate": linear_probe_report["correct_to_incorrect_rate"],
        "correct_to_incorrect_ci_lower": c2i_bootstrap["ci_lower"],
        "correct_to_incorrect_ci_upper": c2i_bootstrap["ci_upper"],
        "knn_k": knn_k,
        "knn_accuracy_drop": knn["all"]["absolute_accuracy_drop"],
        "knn_prediction_change_rate": knn["prediction_changes"]["all"]["prediction_change_rate"],
    }


def _quality_audit_records(
    *,
    original_artifact: dict[str, Any],
    perturbed_artifact: dict[str, Any],
    perturbation_label: str,
    seed: int,
    sample_count: int = 5,
) -> list[dict[str, Any]]:
    total = len(original_artifact["video_ids"])
    selected = sorted(random.Random(f"{seed}:{perturbation_label}").sample(range(total), min(total, sample_count)))
    records: list[dict[str, Any]] = []
    for index in selected:
        original_metadata = original_artifact["sample_metadata"][index]
        perturbed_metadata = perturbed_artifact["sample_metadata"][index]
        perturbation = perturbed_metadata.get("perturbation", {})
        records.append(
            {
                "perturbation_artifact": perturbation_label,
                "sample_index": index,
                "video_id": original_artifact["video_ids"][index],
                "video_path": original_metadata.get("video_path"),
                "frame_indices": original_metadata.get("frame_indices"),
                "frame_indices_match": original_metadata.get("frame_indices")
                == perturbed_metadata.get("frame_indices"),
                "sampling_strategy_match": original_metadata.get("sampling_strategy")
                == perturbed_metadata.get("sampling_strategy"),
                "perturbation_metadata": perturbation,
                "constant_across_frames": perturbation.get("constant_across_frames"),
            }
        )
    return records


def _write_linear_probe_plots(
    rows: list[dict[str, Any]],
    plot_dir: Path,
    *,
    experiment: dict[str, str],
    overwrite: bool,
) -> None:
    title_prefix = f"{experiment['dataset_label']} {experiment['model_label']}"
    bar_path = plot_dir / "linear_probe_mean_cosine_distance.svg"
    _ensure_writable(bar_path, overwrite=overwrite)
    write_bar_chart_svg(
        rows,
        label_key="artifact_label",
        value_key="mean_cosine_distance",
        title=f"{title_prefix} frozen-linear-probe perturbation mean cosine distance",
        path=bar_path,
    )
    curve_rows = [row for row in rows if row["role"] == "curve"]
    series: list[dict[str, Any]] = []
    for perturbation in sorted({str(row["perturbation"]) for row in curve_rows}):
        points = [
            {
                "x": _strength_position(row["strength"]),
                "y": row["mean_cosine_distance"],
                "label": str(row["strength"]),
            }
            for row in curve_rows
            if row["perturbation"] == perturbation
        ]
        if points:
            series.append({"name": perturbation, "points": sorted(points, key=lambda point: point["x"])})
    if series:
        line_path = plot_dir / "linear_probe_strength_curves_mean_cosine_distance.svg"
        _ensure_writable(line_path, overwrite=overwrite)
        write_line_chart_svg(
            series,
            title=f"{title_prefix} frozen-linear-probe strength curves: mean cosine distance",
            path=line_path,
        )


def _strength_position(value: Any) -> float:
    positions = {"low": 1.0, "mid": 2.0, "high": 3.0}
    return positions.get(str(value), 0.0)


def _build_legacy_comparison(
    *,
    config: RunEvaluationConfig,
    rows: list[dict[str, Any]],
    original_knn: dict[str, Any],
) -> dict[str, Any] | None:
    comparison_config = config.legacy_anchor_comparison
    if comparison_config is None:
        return None
    matches = comparison_config.get("matches")
    if not isinstance(matches, list):
        raise TypeError("legacy_anchor_comparison.matches must be a list")
    legacy_summary = comparison_config.get("legacy_perturbation_summary")
    legacy_baseline = comparison_config.get("legacy_baseline_table")
    if legacy_summary is None or legacy_baseline is None:
        raise ValueError(
            "legacy_anchor_comparison requires legacy_perturbation_summary and legacy_baseline_table"
        )
    return build_legacy_anchor_comparison(
        legacy_perturbation_summary=legacy_summary,
        legacy_baseline_table=legacy_baseline,
        current_rows=rows,
        current_original_knn=original_knn,
        matches=[dict(item) for item in matches if isinstance(item, dict)],
    )


def _write_markdown_report(summary: dict[str, Any], path: Path, *, overwrite: bool) -> None:
    _ensure_writable(path, overwrite=overwrite)
    rows = summary["rows"]
    experiment = summary["experiment"]
    columns = [
        "artifact_label",
        "group",
        "role",
        "strength",
        "mean_cosine_distance",
        "cosine_distance_ci_lower",
        "cosine_distance_ci_upper",
        "linear_probe_accuracy_drop",
        "linear_probe_accuracy_drop_ci_lower",
        "linear_probe_accuracy_drop_ci_upper",
        "correct_to_incorrect_rate",
        "knn_accuracy_drop",
    ]
    body = [
        (
            "# "
            f"{experiment['dataset_label']} × frozen {experiment['model_label']} "
            "linear-probe sensitivity experiment"
        ),
        "",
        "This report measures sensitivity to specific interventions. It does not by itself establish human-like action understanding.",
        "",
        "## Perturbation summary",
        "",
        render_markdown_table(rows, columns),
        "",
        "KNN is reported only as an auxiliary cosine k=5 neighbourhood diagnostic. The frozen linear probe is the primary label-related metric.",
        "",
        "## Representative qualitative samples",
        "",
        "Largest representation shifts:",
        "",
        render_markdown_table(
            summary["qualitative_samples"]["largest_embedding_shift"][:5],
            [
                "artifact_label",
                "video_id",
                "label_name",
                "cosine_distance",
                "original_prediction",
                "perturbed_prediction",
                "correct_to_incorrect",
            ],
        ),
        "",
        "Correct-to-incorrect examples:",
        "",
        render_markdown_table(
            summary["qualitative_samples"]["correct_to_incorrect"][:5],
            [
                "artifact_label",
                "video_id",
                "label_name",
                "cosine_distance",
                "original_prediction",
                "perturbed_prediction",
            ],
        ),
        "",
        "## Data quality and failures",
        "",
        f"Fail-fast extraction was used. All extraction artifacts succeeded: {summary['failure_summary']['all_extractions_succeeded']}. ",
        f"All sampled frame-index and sampling-strategy checks passed: {summary['failure_summary']['all_sampled_quality_checks_passed']}.",
        "",
        render_markdown_table(
            summary["failure_summary"]["per_artifact"],
            ["artifact_label", "dataset_size", "successful_samples", "failed_samples"],
        ),
        "",
        "## Interpretation boundaries",
        "",
        *[f"- {limitation}" for limitation in summary["limitations"]],
        "",
    ]
    legacy_comparison = summary.get("legacy_anchor_comparison")
    if legacy_comparison is not None:
        body.extend(
            [
                "## Legacy-anchor comparison",
                "",
                str(legacy_comparison["method_boundary"]),
                "",
                render_markdown_table(
                    legacy_comparison["shared_perturbation_comparisons"],
                    [
                        "legacy_perturbation",
                        "current_artifact_label",
                        "legacy_mean_cosine_distance",
                        "current_mean_cosine_distance",
                        "mean_cosine_distance_delta_current_minus_legacy",
                        "comparability_note",
                    ],
                ),
                "",
            ]
        )
    path.write_text("\n".join(body), encoding="utf-8")


def _experiment_labels(artifact: dict[str, Any]) -> dict[str, str]:
    config = artifact.get("config", {})
    model_metadata = artifact.get("model_metadata", {})
    sample_metadata = artifact.get("sample_metadata", [])
    dataset_name = str(
        config.get("dataset_name")
        or _first_sample_value(sample_metadata, "source_dataset")
        or "dataset"
    )
    model_name = str(
        config.get("model_name")
        or model_metadata.get("encoder_name")
        or model_metadata.get("model_type")
        or "model"
    )
    return {
        "dataset_name": dataset_name,
        "dataset_label": _dataset_label(dataset_name),
        "dataset_description": _dataset_description(dataset_name),
        "model_name": model_name,
        "model_label": _model_label(model_name),
    }


def _first_sample_value(samples: Any, key: str) -> Any:
    if isinstance(samples, list) and samples and isinstance(samples[0], dict):
        return samples[0].get(key)
    return None


def _dataset_label(dataset_name: str) -> str:
    labels = {
        "ssv2": "SSV2",
        "ucf101": "UCF101",
    }
    return labels.get(dataset_name.lower(), dataset_name)


def _dataset_description(dataset_name: str) -> str:
    descriptions = {
        "ssv2": "motion-oriented SSV2 dataset cell",
        "ucf101": "appearance-rich/context-correlated UCF101 dataset cell",
    }
    return descriptions.get(dataset_name.lower(), "dataset cell")


def _model_label(model_name: str) -> str:
    labels = {
        "videomae": "VideoMAE",
        "slowfast": "SlowFast",
    }
    return labels.get(model_name.lower(), model_name)


def _limitations_for_experiment(experiment: dict[str, str]) -> list[str]:
    return [
        *_BASE_LIMITATIONS,
        (
            "This report covers one frozen model × one "
            f"{experiment['dataset_description']}. It must not be generalized to "
            "another model or dataset before the remaining matrix cells are evaluated."
        ),
    ]


def _write_legacy_comparison_markdown(
    comparison: dict[str, Any],
    path: Path,
    *,
    overwrite: bool,
) -> None:
    _ensure_writable(path, overwrite=overwrite)
    lines = [
        "# Legacy-anchor comparison",
        "",
        str(comparison["method_boundary"]),
        "",
        render_markdown_table(
            comparison["shared_perturbation_comparisons"],
            [
                "legacy_perturbation",
                "current_artifact_label",
                "legacy_mean_cosine_distance",
                "current_mean_cosine_distance",
                "mean_cosine_distance_delta_current_minus_legacy",
                "legacy_knn_k",
                "current_knn_k",
                "comparability_note",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(data: dict[str, Any], path: Path, *, overwrite: bool) -> None:
    _ensure_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(rows: list[dict[str, Any]], path: Path, *, overwrite: bool) -> None:
    _ensure_writable(path, overwrite=overwrite)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _ensure_writable(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
