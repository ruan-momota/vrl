"""Frozen linear-probe fitting and paired held-out evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F

from src.artifacts import validate_embedding_artifact
from src.evaluation.bootstrap import BootstrapConfig, paired_bootstrap_summary


FeatureNormalization = Literal["none", "l2"]


@dataclass(frozen=True)
class LinearProbeConfig:
    """Train-only selection protocol for a frozen embedding linear probe."""

    validation_fraction: float = 0.2
    split_seed: int = 42
    training_seed: int = 42
    l2_values: tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3, 1e-2)
    max_iterations: int = 100
    learning_rate: float = 1.0
    feature_normalization: FeatureNormalization = "l2"
    device: str = "cpu"

    def validate(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0.0, 1.0)")
        if not self.l2_values:
            raise ValueError("l2_values must not be empty")
        if any(value < 0.0 for value in self.l2_values):
            raise ValueError("l2_values must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.feature_normalization not in {"none", "l2"}:
            raise ValueError(f"Unsupported feature_normalization: {self.feature_normalization}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["l2_values"] = list(self.l2_values)
        return data


@dataclass(frozen=True)
class FrozenLinearProbe:
    """A fitted single-layer classifier over precomputed frozen embeddings."""

    weight: torch.Tensor
    bias: torch.Tensor
    class_label_ids: torch.Tensor
    feature_normalization: FeatureNormalization
    metadata: dict[str, Any]

    def predict_logits(self, embeddings: torch.Tensor) -> torch.Tensor:
        features = _normalize_features(embeddings, self.feature_normalization)
        if features.ndim != 2 or features.shape[1] != self.weight.shape[1]:
            raise ValueError(
                "Embedding dimension is incompatible with linear probe: "
                f"expected {self.weight.shape[1]}, got {tuple(features.shape)}"
            )
        return features @ self.weight.T + self.bias

    def predict(self, embeddings: torch.Tensor) -> torch.Tensor:
        logits = self.predict_logits(embeddings)
        class_indices = logits.argmax(dim=1)
        return self.class_label_ids[class_indices]

    def to_artifact(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "artifact_type": "frozen_linear_probe",
            "weight": self.weight.detach().cpu(),
            "bias": self.bias.detach().cpu(),
            "class_label_ids": self.class_label_ids.detach().cpu(),
            "feature_normalization": self.feature_normalization,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class LinearProbeEvaluation:
    """Per-video predictions of one probe over one embedding artifact."""

    video_ids: list[str]
    label_ids: torch.Tensor
    predicted_label_ids: torch.Tensor
    correct: torch.Tensor
    artifact_run_id: str | None
    artifact_path: str | None

    @property
    def accuracy(self) -> float:
        return float(self.correct.float().mean().item())

    def to_prediction_artifact(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "artifact_type": "linear_probe_predictions",
            "video_ids": self.video_ids,
            "label_ids": self.label_ids.detach().cpu(),
            "predicted_label_ids": self.predicted_label_ids.detach().cpu(),
            "correct": self.correct.detach().cpu(),
            "accuracy": self.accuracy,
            "run_id": self.artifact_run_id,
            "source_artifact_path": self.artifact_path,
        }


def fit_frozen_linear_probe(
    train_artifact: dict[str, Any],
    *,
    config: LinearProbeConfig,
) -> FrozenLinearProbe:
    """Select L2 on a stratified original-train split, then refit on all train rows."""
    config.validate()
    train_embeddings, train_labels = _labeled_embeddings(train_artifact, artifact_name="train")
    class_label_ids = torch.unique(train_labels, sorted=True)
    if class_label_ids.numel() < 2:
        raise ValueError("Linear probe requires at least two classes")

    features = _normalize_features(train_embeddings, config.feature_normalization)
    class_indices = _encode_labels(train_labels, class_label_ids)
    split = stratified_train_validation_split(
        train_labels,
        validation_fraction=config.validation_fraction,
        seed=config.split_seed,
    )

    candidates: list[dict[str, Any]] = []
    for candidate_index, l2_value in enumerate(sorted(set(float(value) for value in config.l2_values))):
        candidate = _fit_linear_layer(
            features[split["train_indices"]],
            class_indices[split["train_indices"]],
            class_count=int(class_label_ids.numel()),
            l2_value=l2_value,
            max_iterations=config.max_iterations,
            learning_rate=config.learning_rate,
            seed=config.training_seed + candidate_index,
            device=config.device,
        )
        validation_predictions = candidate(features[split["validation_indices"]].to(candidate.weight.device)).argmax(dim=1)
        validation_accuracy = float(
            (validation_predictions.cpu() == class_indices[split["validation_indices"]]).float().mean().item()
        )
        candidates.append(
            {
                "l2_value": l2_value,
                "validation_accuracy": validation_accuracy,
                "train_count": int(split["train_indices"].numel()),
                "validation_count": int(split["validation_indices"].numel()),
            }
        )

    selected = max(candidates, key=lambda item: (item["validation_accuracy"], -item["l2_value"]))
    final_layer = _fit_linear_layer(
        features,
        class_indices,
        class_count=int(class_label_ids.numel()),
        l2_value=float(selected["l2_value"]),
        max_iterations=config.max_iterations,
        learning_rate=config.learning_rate,
        seed=config.training_seed + 10_000,
        device=config.device,
    )
    return FrozenLinearProbe(
        weight=final_layer.weight.detach().cpu(),
        bias=final_layer.bias.detach().cpu(),
        class_label_ids=class_label_ids.detach().cpu(),
        feature_normalization=config.feature_normalization,
        metadata={
            "training_protocol": "stratified_train_validation_l2_selection_then_full_train_refit",
            "config": config.to_dict(),
            "selected_l2": float(selected["l2_value"]),
            "selection_candidates": candidates,
            "class_count": int(class_label_ids.numel()),
            "embedding_dim": int(features.shape[1]),
            "train_artifact_run_id": train_artifact.get("run_id"),
            "train_artifact_config": dict(train_artifact.get("config", {})),
        },
    )


def evaluate_frozen_linear_probe(
    probe: FrozenLinearProbe,
    artifact: dict[str, Any],
    *,
    artifact_path: str | Path | None = None,
) -> LinearProbeEvaluation:
    """Apply a fitted probe without altering it or using query labels for fitting."""
    embeddings, labels = _labeled_embeddings(artifact, artifact_name="evaluation")
    predictions = probe.predict(embeddings).detach().cpu().to(dtype=torch.long)
    labels = labels.detach().cpu().to(dtype=torch.long)
    return LinearProbeEvaluation(
        video_ids=[str(video_id) for video_id in artifact["video_ids"]],
        label_ids=labels,
        predicted_label_ids=predictions,
        correct=predictions.eq(labels),
        artifact_run_id=artifact.get("run_id"),
        artifact_path=None if artifact_path is None else str(artifact_path),
    )


def build_paired_linear_probe_report(
    original: LinearProbeEvaluation,
    perturbed: LinearProbeEvaluation,
    *,
    bootstrap_config: BootstrapConfig,
    perturbation_name: str,
    perturbation_group: str | None = None,
) -> dict[str, Any]:
    """Report paired original/perturbed accuracy and prediction changes."""
    _validate_paired_evaluations(original, perturbed)
    accuracy_drop = original.correct.float() - perturbed.correct.float()
    correct_to_incorrect = original.correct & ~perturbed.correct
    prediction_changed = original.predicted_label_ids.ne(perturbed.predicted_label_ids)
    sample_records = [
        {
            "sample_index": index,
            "video_id": video_id,
            "label_id": int(original.label_ids[index].item()),
            "original_prediction": int(original.predicted_label_ids[index].item()),
            "perturbed_prediction": int(perturbed.predicted_label_ids[index].item()),
            "original_correct": bool(original.correct[index].item()),
            "perturbed_correct": bool(perturbed.correct[index].item()),
            "correct_to_incorrect": bool(correct_to_incorrect[index].item()),
            "prediction_changed": bool(prediction_changed[index].item()),
        }
        for index, video_id in enumerate(original.video_ids)
    ]
    return {
        "format_version": 1,
        "report_type": "paired_frozen_linear_probe",
        "perturbation": perturbation_name,
        "perturbation_group": perturbation_group,
        "sample_count": len(original.video_ids),
        "original_accuracy": original.accuracy,
        "perturbed_accuracy": perturbed.accuracy,
        "accuracy_drop": float(accuracy_drop.mean().item()),
        "correct_to_incorrect_count": int(correct_to_incorrect.sum().item()),
        "correct_to_incorrect_rate": float(correct_to_incorrect.float().mean().item()),
        "prediction_change_rate": float(prediction_changed.float().mean().item()),
        "accuracy_drop_bootstrap": paired_bootstrap_summary(
            accuracy_drop,
            config=bootstrap_config,
            statistics=("mean", "median"),
        ),
        "correct_to_incorrect_bootstrap": paired_bootstrap_summary(
            correct_to_incorrect.float(),
            config=bootstrap_config,
            statistics=("mean", "median"),
        ),
        "sample_predictions": sample_records,
    }


def save_frozen_linear_probe(probe: FrozenLinearProbe, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(probe.to_artifact(), output_path)


def load_frozen_linear_probe(path: str | Path) -> FrozenLinearProbe:
    artifact = torch.load(Path(path), map_location="cpu")
    if not isinstance(artifact, dict) or artifact.get("artifact_type") != "frozen_linear_probe":
        raise TypeError("Not a frozen linear probe artifact")
    weight = artifact.get("weight")
    bias = artifact.get("bias")
    class_label_ids = artifact.get("class_label_ids")
    if not all(isinstance(value, torch.Tensor) for value in (weight, bias, class_label_ids)):
        raise TypeError("Frozen linear probe artifact has invalid tensor fields")
    normalization = artifact.get("feature_normalization")
    if normalization not in {"none", "l2"}:
        raise ValueError("Frozen linear probe artifact has invalid feature_normalization")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("Frozen linear probe artifact has invalid metadata")
    return FrozenLinearProbe(
        weight=weight.detach().cpu().float(),
        bias=bias.detach().cpu().float(),
        class_label_ids=class_label_ids.detach().cpu().long(),
        feature_normalization=normalization,
        metadata=metadata,
    )


def save_linear_probe_predictions(evaluation: LinearProbeEvaluation, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(evaluation.to_prediction_artifact(), output_path)


def stratified_train_validation_split(
    labels: torch.Tensor,
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Create a deterministic, train-only stratified split for probe selection."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0.0, 1.0)")
    label_vector = labels.detach().cpu().long().reshape(-1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_indices: list[torch.Tensor] = []
    validation_indices: list[torch.Tensor] = []
    for label_id in torch.unique(label_vector, sorted=True):
        class_indices = torch.nonzero(label_vector.eq(label_id), as_tuple=False).reshape(-1)
        if class_indices.numel() < 2:
            raise ValueError(
                "Each class needs at least two original train samples for a stratified probe split"
            )
        permutation = torch.randperm(class_indices.numel(), generator=generator)
        validation_count = max(1, int(round(class_indices.numel() * validation_fraction)))
        validation_count = min(validation_count, class_indices.numel() - 1)
        validation_indices.append(class_indices[permutation[:validation_count]])
        train_indices.append(class_indices[permutation[validation_count:]])
    return {
        "train_indices": torch.cat(train_indices).sort().values,
        "validation_indices": torch.cat(validation_indices).sort().values,
    }


def _fit_linear_layer(
    features: torch.Tensor,
    class_indices: torch.Tensor,
    *,
    class_count: int,
    l2_value: float,
    max_iterations: int,
    learning_rate: float,
    seed: int,
    device: str,
) -> torch.nn.Linear:
    resolved_device = _resolve_device(device)
    features = features.detach().to(resolved_device, dtype=torch.float32)
    class_indices = class_indices.detach().to(resolved_device, dtype=torch.long)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layer = torch.nn.Linear(features.shape[1], class_count, device=resolved_device)
        optimizer = torch.optim.LBFGS(
            layer.parameters(),
            lr=learning_rate,
            max_iter=max_iterations,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            logits = layer(features)
            loss = F.cross_entropy(logits, class_indices)
            if l2_value > 0.0:
                loss = loss + 0.5 * l2_value * layer.weight.square().sum()
            loss.backward()
            return loss

        optimizer.step(closure)
    return layer


def _labeled_embeddings(
    artifact: dict[str, Any],
    *,
    artifact_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    validate_embedding_artifact(artifact)
    embeddings = artifact["embeddings"]
    labels = artifact.get("label_ids")
    if not isinstance(labels, torch.Tensor):
        raise ValueError(f"{artifact_name} artifact requires label_ids for linear probe evaluation")
    if labels.shape != (embeddings.shape[0],):
        raise ValueError(f"{artifact_name} artifact label_ids do not match embeddings")
    return embeddings.detach().cpu().float(), labels.detach().cpu().long()


def _normalize_features(
    embeddings: torch.Tensor,
    normalization: FeatureNormalization,
) -> torch.Tensor:
    features = embeddings.detach().cpu().float()
    if normalization == "none":
        return features
    if normalization == "l2":
        return F.normalize(features, p=2, dim=1, eps=1e-12)
    raise ValueError(f"Unsupported feature_normalization: {normalization}")


def _encode_labels(labels: torch.Tensor, class_label_ids: torch.Tensor) -> torch.Tensor:
    class_indices = torch.searchsorted(class_label_ids, labels)
    if not torch.equal(class_label_ids[class_indices], labels):
        raise ValueError("Labels are not represented by class_label_ids")
    return class_indices.long()


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _validate_paired_evaluations(
    original: LinearProbeEvaluation,
    perturbed: LinearProbeEvaluation,
) -> None:
    if original.artifact_run_id != perturbed.artifact_run_id:
        raise ValueError("Linear probe evaluations have different run IDs")
    if original.video_ids != perturbed.video_ids:
        raise ValueError("Linear probe evaluations have different video ID order")
    if not torch.equal(original.label_ids, perturbed.label_ids):
        raise ValueError("Linear probe evaluations have different labels")
