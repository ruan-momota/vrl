"""Backward-compatible sensitivity-analysis CLI."""

from src.evaluation.sensitivity import *  # noqa: F403
from src.evaluation.sensitivity import main


if __name__ == "__main__":
    raise SystemExit(main())
