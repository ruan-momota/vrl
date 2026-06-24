"""Backward-compatible baseline-interpretability CLI."""

from src.evaluation.baseline_interpretability import *  # noqa: F403
from src.evaluation.baseline_interpretability import main


if __name__ == "__main__":
    raise SystemExit(main())
