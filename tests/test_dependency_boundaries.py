"""Regression tests for the Phase 1.5 package dependency rules."""

from __future__ import annotations

import ast
from pathlib import Path

import src.artifact_alignment as compatibility_alignment
import src.embedding_extraction as compatibility_extraction
import src.knn_baseline as compatibility_knn
import src.ssv2_index as compatibility_ssv2_index
import src.ssv2_dataset as compatibility_ssv2_dataset
import src.video_io as compatibility_video_io
import src.videomae_model as compatibility_videomae_model
from src import artifacts
from src.data import ssv2, ssv2_index
from src.evaluation import alignment, knn
from src.models import videomae_model
from src.video import io


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPATIBILITY_IMPLEMENTATIONS = {
    "src.artifact_alignment",
    "src.embedding_extraction",
    "src.embedding_sensitivity",
    "src.knn_baseline",
    "src.knn_perturbation_analysis",
    "src.reporting",
    "src.baseline_interpretability",
    "src.ssv2_dataset",
    "src.ssv2_index",
    "src.videomae_model",
    "src.videomae_preprocessing",
    "src.video_io",
    "src.video_perturbations",
}


def test_pipeline_does_not_depend_on_compatibility_implementations() -> None:
    imports = _project_imports(PROJECT_ROOT / "src" / "pipeline")

    assert not imports & COMPATIBILITY_IMPLEMENTATIONS
    assert "src.models.dinov2" not in imports
    assert not any(module == "src.legacy" or module.startswith("src.legacy.") for module in imports)


def test_evaluation_depends_only_on_artifact_contract_and_evaluation_modules() -> None:
    imports = _project_imports(PROJECT_ROOT / "src" / "evaluation")
    forbidden_prefixes = ("src.data", "src.models", "src.video", "src.pipeline", "src.legacy")

    assert not any(module.startswith(forbidden_prefixes) for module in imports)
    assert "src.models.dinov2" not in imports
    assert not imports & COMPATIBILITY_IMPLEMENTATIONS


def test_selected_compatibility_imports_forward_to_the_new_implementation() -> None:
    assert compatibility_alignment.check_paired_embedding_alignment is alignment.check_paired_embedding_alignment
    assert compatibility_extraction.load_embedding_artifact is artifacts.load_embedding_artifact
    assert compatibility_knn.evaluate_knn_baseline is knn.evaluate_knn_baseline
    assert compatibility_ssv2_index.build_split_index is ssv2_index.build_split_index
    assert compatibility_ssv2_dataset.SSV2ClipDataset is ssv2.SSV2ClipDataset
    assert compatibility_video_io.read_sampled_clip is io.read_sampled_clip
    assert compatibility_videomae_model.forward_videomae_embeddings is (
        videomae_model.forward_videomae_embeddings
    )


def _project_imports(directory: Path) -> set[str]:
    modules: set[str] = set()
    for source_path in directory.glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
    return modules
