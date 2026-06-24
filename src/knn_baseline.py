"""Backward-compatible KNN diagnostic CLI."""

from src.evaluation.knn import *  # noqa: F403
from src.evaluation.knn import main


if __name__ == "__main__":
    raise SystemExit(main())
