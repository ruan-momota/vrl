"""Backward-compatible CLI for the legacy SSV2×VideoMAE sweep runner."""

from src.legacy.perturbation_sweeps import *  # noqa: F403
from src.legacy.perturbation_sweeps import main


if __name__ == "__main__":
    raise SystemExit(main())
