"""Backward-compatible perturbation KNN-analysis CLI."""

from src.evaluation.knn_perturbation import *  # noqa: F403
from src.evaluation.knn_perturbation import main


if __name__ == "__main__":
    raise SystemExit(main())
