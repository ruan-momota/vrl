"""Backward-compatible SSV2 index import and CLI entry point."""

from src.data.ssv2_index import *  # noqa: F403
from src.data.ssv2_index import main


if __name__ == "__main__":
    raise SystemExit(main())
