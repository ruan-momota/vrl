from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.artifacts import save_embedding_artifact
from src.data.base import DatasetAdapter
from src.data.registry import get_dataset_adapter
from src.models.base import VideoEncoder
from src.models.registry import load_video_encoder
from src.pipeline.extraction import extract_embeddings, resolve_device
from src.pipeline.run import RunConfig, RunPaths, write_manifest
from src.reproducibility import seed_everything
from src.video.perturbations import (
    VideoPerturbationConfig,
    build_video_perturbation,
    parse_perturbation_name,
)


def run_extraction(
    config: RunConfig,
    *,
    encoder: VideoEncoder | None = None,
    dataset_adapter: DatasetAdapter | None = None,
    limit: int | None = None,
    local_files_only: bool = False,
    overwrite: bool = False,
    show_progress: bool = True,
    command: list[str] | None = None,
) -> dict[str, Any]:
    """Extract one run-config-defined artifact through generic adapters."""
    seed_everything(config.seed, deterministic=config.deterministic)
    paths = RunPaths(config)
    manifest = write_manifest(paths, command=command, overwrite=overwrite)
    artifact_path = paths.artifact_path()

    perturbation_data = dict(config.perturbation)
    perturbation_data.pop("artifact_label", None)
    perturbation = VideoPerturbationConfig(
        name=parse_perturbation_name(str(perturbation_data.pop("name", "none"))),
        **perturbation_data,
    )
    resolved_encoder = encoder or load_video_encoder(
        config.model_name,
        checkpoint=config.model_checkpoint,
        device=config.device,
        image_size=config.image_size,
        local_files_only=local_files_only,
    )
    adapter = dataset_adapter or get_dataset_adapter(config.dataset_name)
    dataset = adapter.build_dataset(
        config.index_path,
        num_frames=config.num_frames,
        sampling_strategy=config.sampling_strategy,  # type: ignore[arg-type]
        transform=resolved_encoder.preprocess,
        perturbation=build_video_perturbation(perturbation),
    )
    if limit is not None:
        from torch.utils.data import Subset

        dataset = Subset(dataset, range(min(limit, len(dataset))))
    device = resolve_device(config.device)
    result = extract_embeddings(
        encoder=resolved_encoder,
        dataset=dataset,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        device=device,
        show_progress=show_progress,
    )
    artifact = save_embedding_artifact(
        result,
        artifact_path,
        config_snapshot={
            **config.to_dict(),
            "run_id": config.resolved_run_id,
            "manifest_path": str(paths.manifest_path),
        },
        model_metadata=resolved_encoder.metadata(),
        extraction_options={"limit": limit, "local_files_only": local_files_only},
        run_id=config.resolved_run_id,
        artifact_metadata={
            "artifact_kind": "original" if perturbation.name == "none" else "perturbed",
            "manifest_path": str(paths.manifest_path),
        },
        overwrite=overwrite,
    )
    return {
        "run_id": config.resolved_run_id,
        "manifest_path": str(paths.manifest_path),
        "artifact_path": str(artifact_path),
        "summary": artifact["summary"],
        "model_metadata": artifact["model_metadata"],
        "manifest": manifest,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract embeddings from a generic VRL run config.")
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_extraction(
        RunConfig.from_file(args.run_config),
        limit=args.limit,
        local_files_only=args.local_files_only,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
        command=[sys.executable, "-m", "src.pipeline.extract", "--run-config", str(args.run_config)],
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
