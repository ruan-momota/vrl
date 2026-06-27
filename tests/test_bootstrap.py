from __future__ import annotations

import pytest
import torch

from src.evaluation.bootstrap import BootstrapConfig, paired_bootstrap_summary


def test_paired_bootstrap_is_reproducible_and_reports_percentile_interval() -> None:
    config = BootstrapConfig(resamples=200, seed=17)

    first = paired_bootstrap_summary(torch.tensor([0.0, 0.0, 1.0, 1.0]), config=config)
    second = paired_bootstrap_summary(torch.tensor([0.0, 0.0, 1.0, 1.0]), config=config)

    assert first == second
    assert first["unit"] == "video"
    assert first["paired"] is True
    mean = first["statistics"]["mean"]
    assert mean["point_estimate"] == pytest.approx(0.5)
    assert 0.0 <= mean["ci_lower"] <= mean["ci_upper"] <= 1.0


def test_paired_bootstrap_rejects_empty_or_nonfinite_values() -> None:
    config = BootstrapConfig(resamples=10, seed=1)

    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap_summary([], config=config)
    with pytest.raises(ValueError, match="finite"):
        paired_bootstrap_summary(torch.tensor([1.0, float("nan")]), config=config)
