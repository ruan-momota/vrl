"""Backward-compatible legacy VideoMAE diagnostic CLI."""

from src.legacy.videomae_model_sanity_check import main


if __name__ == "__main__":
    raise SystemExit(main())
