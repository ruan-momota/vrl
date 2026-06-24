"""Backward-compatible legacy video diagnostic CLI."""

from src.legacy.video_sanity_check import main


if __name__ == "__main__":
    raise SystemExit(main())
