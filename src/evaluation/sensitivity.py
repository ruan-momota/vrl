"""Representation-shift metrics for paired original/perturbed artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.artifacts import load_embedding_artifact
from src.evaluation.alignment import check_paired_embedding_alignment
from src.evaluation.bootstrap import BootstrapConfig, paired_bootstrap_summary


EPSILON = 1e-12


def compute_embedding_distances(
    original_embeddings: torch.Tensor,
    perturbed_embeddings: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute row-wise embedding sensitivity metrics for aligned embeddings."""
    if original_embeddings.ndim != 2:
        raise ValueError("original_embeddings must have shape [N, D]")
    if perturbed_embeddings.ndim != 2:
        raise ValueError("perturbed_embeddings must have shape [N, D]")
    if original_embeddings.shape != perturbed_embeddings.shape:
        raise ValueError(
            "original and perturbed embeddings must have the same shape, "
            f"got {tuple(original_embeddings.shape)} and {tuple(perturbed_embeddings.shape)}"
        )

    original = original_embeddings.detach().cpu().float()
    perturbed = perturbed_embeddings.detach().cpu().float()
    delta = original - perturbed
    cosine_similarity = F.cosine_similarity(original, perturbed, dim=1, eps=EPSILON).clamp(
        -1.0,
        1.0,
    )
    l2_distance = torch.linalg.vector_norm(delta, ord=2, dim=1)
    original_norm = torch.linalg.vector_norm(original, ord=2, dim=1).clamp_min(EPSILON)
    return {
        "cosine_similarity": cosine_similarity,
        "cosine_distance": 1.0 - cosine_similarity,
        "l2_distance": l2_distance,
        "relative_l2_distance": l2_distance / original_norm,
    }


def summarize_values(values: torch.Tensor | list[float]) -> dict[str, float | int]:
    tensor = (
        values.detach().cpu().float().reshape(-1)
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float32)
    )
    if tensor.numel() == 0:
        raise ValueError("Cannot summarize an empty metric sequence")
    return {
        "count": int(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "median": float(torch.quantile(tensor, 0.5).item()),
        "std": float(tensor.std(unbiased=False).item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def build_embedding_sensitivity_report(
    *,
    original_artifact: dict[str, Any],
    perturbed_artifact: dict[str, Any],
    original_artifact_path: str | Path | None = None,
    perturbed_artifact_path: str | Path | None = None,
    perturbation_name: str | None = None,
    perturbation_group: str | None = None,
    bootstrap_config: BootstrapConfig | None = None,
) -> dict[str, Any]:
    alignment = check_paired_embedding_alignment(original_artifact, perturbed_artifact)
    original_embeddings = original_artifact["embeddings"]
    perturbed_embeddings = perturbed_artifact["embeddings"]
    distances = compute_embedding_distances(original_embeddings, perturbed_embeddings)
    resolved_name = perturbation_name or _infer_perturbation_name(perturbed_artifact)
    perturbation_config = _perturbation_config(perturbed_artifact)

    sample_metrics = _build_sample_metrics(
        original_artifact=original_artifact,
        distances=distances,
        perturbation_name=resolved_name,
        perturbation_group=perturbation_group,
        perturbation_config=perturbation_config,
    )
    summary = _summarize_sample_metrics(
        sample_metrics,
        perturbation_name=resolved_name,
        perturbation_group=perturbation_group,
        sample_count=int(original_embeddings.shape[0]),
        embedding_dim=int(original_embeddings.shape[1]),
    )

    return {
        "format_version": 1,
        "report_type": "embedding_sensitivity",
        "dataset_name": _nested_value(original_artifact, "config", "dataset_name"),
        "split": _nested_value(original_artifact, "config", "split"),
        "model_checkpoint": _nested_value(original_artifact, "model_metadata", "checkpoint"),
        "embedding_type": _nested_value(
            original_artifact,
            "model_metadata",
            "embedding_type",
        ),
        "original_artifact_path": None
        if original_artifact_path is None
        else str(original_artifact_path),
        "perturbed_artifact_path": None
        if perturbed_artifact_path is None
        else str(perturbed_artifact_path),
        "perturbation": resolved_name,
        "perturbation_group": perturbation_group,
        "perturbation_config": perturbation_config,
        "alignment": alignment.to_dict(),
        "summary": summary,
        "bootstrap": None
        if bootstrap_config is None
        else {
            metric_name: paired_bootstrap_summary(
                distances[metric_name],
                config=bootstrap_config,
            )
            for metric_name in (
                "cosine_distance",
                "l2_distance",
                "relative_l2_distance",
            )
        },
        "sample_metrics": sample_metrics,
    }


def run_embedding_sensitivity(
    *,
    original_artifact_path: str | Path,
    perturbed_artifact_path: str | Path,
    perturbation_name: str | None = None,
    perturbation_group: str | None = None,
    bootstrap_config: BootstrapConfig | None = None,
) -> dict[str, Any]:
    original_path = Path(original_artifact_path)
    perturbed_path = Path(perturbed_artifact_path)
    return build_embedding_sensitivity_report(
        original_artifact=load_embedding_artifact(original_path),
        perturbed_artifact=load_embedding_artifact(perturbed_path),
        original_artifact_path=original_path,
        perturbed_artifact_path=perturbed_path,
        perturbation_name=perturbation_name,
        perturbation_group=perturbation_group,
        bootstrap_config=bootstrap_config,
    )


def build_all_perturbations_summary(
    reports: list[dict[str, Any]],
    *,
    matrix: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one sensitivity report is required")

    compact_summaries = [_compact_perturbation_summary(report) for report in reports]
    return {
        "format_version": 1,
        "report_type": "embedding_sensitivity_summary",
        "matrix_name": None if matrix is None else matrix.get("matrix_name"),
        "dataset_name": reports[0].get("dataset_name"),
        "split": reports[0].get("split"),
        "model_checkpoint": reports[0].get("model_checkpoint"),
        "embedding_type": reports[0].get("embedding_type"),
        "original_artifact_path": reports[0].get("original_artifact_path"),
        "perturbation_count": len(reports),
        "sample_count_per_perturbation": reports[0]["summary"]["sample_count"],
        "perturbations": compact_summaries,
        "group_summaries": _build_group_summaries(reports),
        "ranked_by_mean_cosine_distance": sorted(
            compact_summaries,
            key=lambda item: item["mean_cosine_distance"],
            reverse=True,
        ),
        "ranked_by_mean_l2_distance": sorted(
            compact_summaries,
            key=lambda item: item["mean_l2_distance"],
            reverse=True,
        ),
        "group_comparison": _build_group_comparison(reports),
    }


def build_class_sensitivity_report(
    reports: list[dict[str, Any]],
    *,
    matrix: dict[str, Any] | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one sensitivity report is required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    class_records: dict[tuple[int | None, str | None], dict[str, Any]] = {}
    for report in reports:
        perturbation = str(report["perturbation"])
        group = report.get("perturbation_group")
        grouped = _group_sample_metrics_by_class(report["sample_metrics"])
        perturbation_class_metrics: list[tuple[tuple[int | None, str | None], dict[str, Any]]] = []
        for class_key, records in grouped.items():
            class_metric = _summarize_class_records(records)
            perturbation_class_metrics.append((class_key, class_metric))
            class_entry = class_records.setdefault(
                class_key,
                {
                    "label_id": class_key[0],
                    "label_name": class_key[1],
                    "sample_count": class_metric["sample_count"],
                    "perturbations": {},
                },
            )
            class_entry["sample_count"] = max(
                int(class_entry["sample_count"]),
                int(class_metric["sample_count"]),
            )
            class_entry["perturbations"][perturbation] = {
                "perturbation_group": group,
                **class_metric,
            }

        ranked = sorted(
            perturbation_class_metrics,
            key=lambda item: (
                item[1]["mean_cosine_distance"],
                -_rank_tie_label_id(item[0][0]),
                item[0][1] or "",
            ),
            reverse=True,
        )
        for rank, (class_key, _) in enumerate(ranked, start=1):
            class_records[class_key]["perturbations"][perturbation][
                "rank_by_mean_cosine_distance"
            ] = rank

    classes = [_finalize_class_entry(entry) for entry in class_records.values()]
    classes = sorted(
        classes,
        key=lambda item: (
            item["mean_cosine_distance_across_perturbations"],
            -_rank_tie_label_id(item["label_id"]),
            item["label_name"] or "",
        ),
        reverse=True,
    )
    return {
        "format_version": 1,
        "report_type": "class_embedding_sensitivity",
        "matrix_name": None if matrix is None else matrix.get("matrix_name"),
        "dataset_name": reports[0].get("dataset_name"),
        "split": reports[0].get("split"),
        "model_checkpoint": reports[0].get("model_checkpoint"),
        "embedding_type": reports[0].get("embedding_type"),
        "perturbations": [str(report["perturbation"]) for report in reports],
        "class_count": len(classes),
        "classes": classes,
        "most_sensitive_classes_by_mean_cosine_distance": classes[:top_k],
        "most_insensitive_classes_by_mean_cosine_distance": list(reversed(classes[-top_k:])),
    }


def run_matrix_sensitivity(
    *,
    matrix_path: str | Path,
    original_artifact_path: str | Path | None = None,
    output_dir: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    matrix = _load_json(Path(matrix_path))
    scope = matrix.get("scope", {})
    original_path = Path(
        original_artifact_path or scope["validation_original_artifact"],
    )
    output_path = Path(output_dir)

    reports: list[dict[str, Any]] = []
    per_perturbation_outputs: list[dict[str, str]] = []
    for entry in matrix.get("first_round", []):
        perturbation_name = str(entry["name"])
        perturbed_path = Path(entry["output_path"])
        report = run_embedding_sensitivity(
            original_artifact_path=original_path,
            perturbed_artifact_path=perturbed_path,
            perturbation_name=perturbation_name,
            perturbation_group=entry.get("group"),
        )
        report_path = _sensitivity_report_path(perturbed_path, output_path)
        save_json_report(report, report_path, overwrite=overwrite)
        reports.append(report)
        per_perturbation_outputs.append(
            {
                "perturbation": perturbation_name,
                "report_path": str(report_path),
            }
        )

    summary = build_all_perturbations_summary(reports, matrix=matrix)
    class_report = build_class_sensitivity_report(reports, matrix=matrix)
    report_basename = _report_basename(matrix)
    summary_path = output_path / f"{report_basename}_all_perturbations_sensitivity_summary.json"
    class_report_path = output_path / f"{report_basename}_class_sensitivity.json"
    save_json_report(summary, summary_path, overwrite=overwrite)
    save_json_report(class_report, class_report_path, overwrite=overwrite)
    return {
        "matrix_path": str(matrix_path),
        "original_artifact_path": str(original_path),
        "per_perturbation_reports": per_perturbation_outputs,
        "summary_report_path": str(summary_path),
        "class_report_path": str(class_report_path),
    }


def save_json_report(
    report: dict[str, Any],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> None:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _report_basename(matrix: dict[str, Any]) -> str:
    value = matrix.get("report_basename") or matrix.get("matrix_name")
    if value is None:
        raise ValueError("Matrix must define report_basename or matrix_name")
    return str(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute original-vs-perturbed embedding sensitivity reports."
    )
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--original-artifact", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_matrix_sensitivity(
        matrix_path=args.matrix,
        original_artifact_path=args.original_artifact,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _build_sample_metrics(
    *,
    original_artifact: dict[str, Any],
    distances: dict[str, torch.Tensor],
    perturbation_name: str,
    perturbation_group: str | None,
    perturbation_config: dict[str, Any],
) -> list[dict[str, Any]]:
    video_ids = original_artifact["video_ids"]
    label_ids = original_artifact.get("label_ids")
    metadata = original_artifact.get("sample_metadata", [])
    sample_metrics: list[dict[str, Any]] = []
    for index, video_id in enumerate(video_ids):
        sample_metadata = metadata[index] if index < len(metadata) else {}
        label_id = _label_id_at(label_ids, sample_metadata, index)
        sample_metrics.append(
            {
                "sample_index": index,
                "video_id": str(video_id),
                "label_id": label_id,
                "label_name": _metadata_str(sample_metadata, "label_name"),
                "perturbation": perturbation_name,
                "perturbation_group": perturbation_group,
                "perturbation_config": perturbation_config,
                "cosine_similarity": float(distances["cosine_similarity"][index].item()),
                "cosine_distance": float(distances["cosine_distance"][index].item()),
                "l2_distance": float(distances["l2_distance"][index].item()),
                "relative_l2_distance": float(
                    distances["relative_l2_distance"][index].item()
                ),
            }
        )
    return sample_metrics


def _summarize_sample_metrics(
    sample_metrics: list[dict[str, Any]],
    *,
    perturbation_name: str,
    perturbation_group: str | None,
    sample_count: int,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "perturbation": perturbation_name,
        "perturbation_group": perturbation_group,
        "sample_count": sample_count,
        "embedding_dim": embedding_dim,
        "metrics": {
            metric_name: summarize_values(
                [float(record[metric_name]) for record in sample_metrics]
            )
            for metric_name in (
                "cosine_similarity",
                "cosine_distance",
                "l2_distance",
                "relative_l2_distance",
            )
        },
    }


def _compact_perturbation_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["summary"]["metrics"]
    return {
        "perturbation": report["perturbation"],
        "perturbation_group": report.get("perturbation_group"),
        "sample_count": report["summary"]["sample_count"],
        "mean_cosine_distance": metrics["cosine_distance"]["mean"],
        "median_cosine_distance": metrics["cosine_distance"]["median"],
        "std_cosine_distance": metrics["cosine_distance"]["std"],
        "min_cosine_distance": metrics["cosine_distance"]["min"],
        "max_cosine_distance": metrics["cosine_distance"]["max"],
        "mean_l2_distance": metrics["l2_distance"]["mean"],
        "median_l2_distance": metrics["l2_distance"]["median"],
        "mean_relative_l2_distance": metrics["relative_l2_distance"]["mean"],
        "median_relative_l2_distance": metrics["relative_l2_distance"]["median"],
    }


def _build_group_summaries(reports: list[dict[str, Any]]) -> dict[str, Any]:
    groups = sorted({str(report.get("perturbation_group")) for report in reports})
    group_summaries: dict[str, Any] = {}
    for group in groups:
        group_reports = [
            report for report in reports if str(report.get("perturbation_group")) == group
        ]
        records = [
            record
            for report in group_reports
            for record in report["sample_metrics"]
        ]
        metric_summaries: dict[str, Any] = {}
        for metric_name in (
            "cosine_distance",
            "l2_distance",
            "relative_l2_distance",
        ):
            metric_summaries[metric_name] = {
                "sample_weighted": summarize_values(
                    [float(record[metric_name]) for record in records]
                ),
                "mean_of_perturbation_means": summarize_values(
                    [
                        float(report["summary"]["metrics"][metric_name]["mean"])
                        for report in group_reports
                    ]
                ),
            }
        group_summaries[group] = {
            "perturbation_count": len(group_reports),
            "sample_metric_count": len(records),
            "perturbations": [str(report["perturbation"]) for report in group_reports],
            "metrics": metric_summaries,
        }
    return group_summaries


def _build_group_comparison(reports: list[dict[str, Any]]) -> dict[str, Any]:
    group_summaries = _build_group_summaries(reports)
    motion = group_summaries.get("motion")
    appearance = group_summaries.get("appearance")
    if motion is None or appearance is None:
        return {
            "comparison_available": False,
            "reason": "motion and appearance groups are both required",
        }

    motion_mean = float(
        motion["metrics"]["cosine_distance"]["mean_of_perturbation_means"]["mean"]
    )
    appearance_mean = float(
        appearance["metrics"]["cosine_distance"]["mean_of_perturbation_means"]["mean"]
    )
    return {
        "comparison_available": True,
        "metric": "cosine_distance.mean_of_perturbation_means.mean",
        "motion_mean_cosine_distance": motion_mean,
        "appearance_mean_cosine_distance": appearance_mean,
        "motion_minus_appearance": motion_mean - appearance_mean,
        "larger_shift_group": "motion"
        if motion_mean > appearance_mean
        else "appearance"
        if appearance_mean > motion_mean
        else "tie",
    }


def _group_sample_metrics_by_class(
    sample_metrics: list[dict[str, Any]],
) -> dict[tuple[int | None, str | None], list[dict[str, Any]]]:
    grouped: dict[tuple[int | None, str | None], list[dict[str, Any]]] = {}
    for record in sample_metrics:
        key = (record.get("label_id"), record.get("label_name"))
        grouped.setdefault(key, []).append(record)
    return grouped


def _summarize_class_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(records),
        "mean_cosine_distance": summarize_values(
            [float(record["cosine_distance"]) for record in records]
        )["mean"],
        "median_cosine_distance": summarize_values(
            [float(record["cosine_distance"]) for record in records]
        )["median"],
        "mean_l2_distance": summarize_values(
            [float(record["l2_distance"]) for record in records]
        )["mean"],
        "median_l2_distance": summarize_values(
            [float(record["l2_distance"]) for record in records]
        )["median"],
        "mean_relative_l2_distance": summarize_values(
            [float(record["relative_l2_distance"]) for record in records]
        )["mean"],
    }


def _finalize_class_entry(entry: dict[str, Any]) -> dict[str, Any]:
    perturbation_metrics = entry["perturbations"]
    mean_cosine_by_perturbation = {
        name: float(metrics["mean_cosine_distance"])
        for name, metrics in perturbation_metrics.items()
    }
    strongest_name = max(mean_cosine_by_perturbation, key=mean_cosine_by_perturbation.get)
    weakest_name = min(mean_cosine_by_perturbation, key=mean_cosine_by_perturbation.get)

    group_values: dict[str, list[float]] = {}
    for metrics in perturbation_metrics.values():
        group = str(metrics.get("perturbation_group"))
        group_values.setdefault(group, []).append(float(metrics["mean_cosine_distance"]))

    return {
        **entry,
        "mean_cosine_distance_across_perturbations": sum(
            mean_cosine_by_perturbation.values()
        )
        / len(mean_cosine_by_perturbation),
        "strongest_perturbation": {
            "name": strongest_name,
            "mean_cosine_distance": mean_cosine_by_perturbation[strongest_name],
        },
        "weakest_perturbation": {
            "name": weakest_name,
            "mean_cosine_distance": mean_cosine_by_perturbation[weakest_name],
        },
        "group_mean_cosine_distance": {
            group: sum(values) / len(values) for group, values in group_values.items()
        },
    }


def _sensitivity_report_path(perturbed_artifact_path: Path, output_dir: Path) -> Path:
    name = perturbed_artifact_path.stem + "_sensitivity.json"
    return output_dir / name


def _infer_perturbation_name(artifact: dict[str, Any]) -> str:
    config = _perturbation_config(artifact)
    value = config.get("name")
    if value is not None:
        return str(value)
    metadata = artifact.get("sample_metadata")
    if isinstance(metadata, list) and metadata:
        perturbation = metadata[0].get("perturbation") if isinstance(metadata[0], dict) else None
        if isinstance(perturbation, dict) and perturbation.get("name") is not None:
            return str(perturbation["name"])
    return "unknown"


def _perturbation_config(artifact: dict[str, Any]) -> dict[str, Any]:
    config = artifact.get("config")
    if isinstance(config, dict) and isinstance(config.get("perturbation"), dict):
        return dict(config["perturbation"])
    options = artifact.get("extraction_options")
    if isinstance(options, dict) and isinstance(options.get("perturbation"), dict):
        return dict(options["perturbation"])
    return {}


def _label_id_at(
    label_ids: Any,
    sample_metadata: Any,
    index: int,
) -> int | None:
    if isinstance(label_ids, torch.Tensor):
        return int(label_ids[index].item())
    if isinstance(sample_metadata, dict) and sample_metadata.get("label_id") is not None:
        return int(sample_metadata["label_id"])
    return None


def _metadata_str(sample_metadata: Any, key: str) -> str | None:
    if not isinstance(sample_metadata, dict):
        return None
    value = sample_metadata.get(key)
    return None if value is None else str(value)


def _nested_value(
    artifact: dict[str, Any],
    section: str,
    key: str,
) -> str | None:
    section_value = artifact.get(section)
    if not isinstance(section_value, dict):
        return None
    value = section_value.get(key)
    return None if value is None else str(value)


def _rank_tie_label_id(label_id: int | None) -> int:
    return -1 if label_id is None else int(label_id)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
