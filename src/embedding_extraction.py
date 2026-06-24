"""Backward-compatible import and CLI entry point for embedding extraction.

New code must use ``src.pipeline.extraction`` and ``src.artifacts``.  This
module preserves the historical import path and CLI while forwarding all work
to the generic or explicitly legacy implementations.
"""

from __future__ import annotations

from src.artifacts import (
    EmbeddingExtractionResult,
    EmbeddingExtractionSummary,
    load_embedding_artifact,
    save_embedding_artifact,
    validate_embedding_artifact,
    validate_embedding_result,
)
from src.legacy.embedding_extraction import (
    build_extraction_dataset,
    extract_embeddings,
    main,
    run_extraction_from_args,
)

__all__ = [
    "EmbeddingExtractionResult",
    "EmbeddingExtractionSummary",
    "build_extraction_dataset",
    "extract_embeddings",
    "load_embedding_artifact",
    "run_extraction_from_args",
    "save_embedding_artifact",
    "validate_embedding_artifact",
    "validate_embedding_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
