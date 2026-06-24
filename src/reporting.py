"""Backward-compatible reporting CLI."""

from src.evaluation.reporting import *  # noqa: F403
from src.evaluation.reporting import main


if __name__ == "__main__":
    raise SystemExit(main())
