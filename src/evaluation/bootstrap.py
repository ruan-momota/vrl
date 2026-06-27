"""Deterministic video-level paired bootstrap summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch


BootstrapStatistic = Literal["mean", "median"]


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for one reproducible non-parametric bootstrap."""

    resamples: int = 1_000
    seed: int = 42
    confidence_level: float = 0.95

    def validate(self) -> None:
        if self.resamples <= 0:
            raise ValueError("resamples must be positive")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0.0, 1.0)")

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def paired_bootstrap_summary(
    values: torch.Tensor | list[float],
    *,
    config: BootstrapConfig,
    statistics: tuple[BootstrapStatistic, ...] = ("mean", "median"),
) -> dict[str, object]:
    """Summarize a per-video paired quantity with percentile confidence intervals.

    ``values`` must already represent a paired quantity, for example one
    original-vs-perturbed cosine distance or correctness difference per video.
    Each bootstrap draw resamples those rows, so the original/perturbed pairing
    is retained by construction.
    """
    config.validate()
    tensor = _as_nonempty_finite_vector(values)
    if not statistics:
        raise ValueError("At least one bootstrap statistic is required")

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    indices = torch.randint(
        tensor.numel(),
        (config.resamples, tensor.numel()),
        generator=generator,
        device="cpu",
    )
    bootstrap_samples = tensor[indices]
    summaries = {
        statistic: _bootstrap_statistic(
            tensor,
            bootstrap_samples=bootstrap_samples,
            statistic=statistic,
            config=config,
        )
        for statistic in statistics
    }
    return {
        "unit": "video",
        "paired": True,
        "sample_count": int(tensor.numel()),
        "method": "nonparametric_percentile_bootstrap",
        "config": config.to_dict(),
        "statistics": summaries,
    }


def _bootstrap_statistic(
    values: torch.Tensor,
    *,
    bootstrap_samples: torch.Tensor,
    statistic: BootstrapStatistic,
    config: BootstrapConfig,
) -> dict[str, float | str]:
    point_estimate = _statistic(values, statistic)
    if statistic == "mean":
        draws = bootstrap_samples.mean(dim=1)
    elif statistic == "median":
        draws = torch.quantile(bootstrap_samples, 0.5, dim=1)
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")

    alpha = (1.0 - config.confidence_level) / 2.0
    interval = torch.quantile(draws, torch.tensor([alpha, 1.0 - alpha]))
    return {
        "statistic": statistic,
        "point_estimate": float(point_estimate.item()),
        "ci_lower": float(interval[0].item()),
        "ci_upper": float(interval[1].item()),
    }


def _statistic(values: torch.Tensor, statistic: BootstrapStatistic) -> torch.Tensor:
    if statistic == "mean":
        return values.mean()
    if statistic == "median":
        return torch.quantile(values, 0.5)
    raise ValueError(f"Unsupported bootstrap statistic: {statistic}")


def _as_nonempty_finite_vector(values: torch.Tensor | list[float]) -> torch.Tensor:
    tensor = (
        values.detach().cpu().float().reshape(-1)
        if isinstance(values, torch.Tensor)
        else torch.tensor(values, dtype=torch.float32).reshape(-1)
    )
    if tensor.numel() == 0:
        raise ValueError("Cannot bootstrap an empty value sequence")
    if not torch.isfinite(tensor).all():
        raise ValueError("Bootstrap values must be finite")
    return tensor
