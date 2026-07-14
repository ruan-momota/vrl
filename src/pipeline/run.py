from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    """Minimal, serializable definition of one embedding extraction run."""

    dataset_name: str
    subset_id: str
    split: str
    index_path: str
    model_name: str
    model_checkpoint: str
    num_frames: int
    sampling_strategy: str
    image_size: int
    batch_size: int
    num_workers: int
    device: str
    seed: int
    deterministic: bool
    model_revision: str | None = None
    input_profile: str | None = None
    subset_summary_path: str | None = None
    window_frames: int | None = None
    perturbation: dict[str, Any] = field(default_factory=lambda: {"name": "none"})
    output_root: str = "outputs/runs"
    run_id: str | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "RunConfig":
        with Path(path).open("r", encoding="utf-8") as file:
            return cls.from_mapping(json.load(file))

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RunConfig":
        return cls(**data)

    @property
    def resolved_run_id(self) -> str:
        if self.run_id is not None:
            return self.run_id
        payload = self.experiment_definition()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:10]
        slug = "-".join(
            _slug_part(value)
            for value in (self.dataset_name, self.subset_id, self.model_name, f"{self.num_frames}f")
        )
        return f"{slug}-{digest}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def experiment_definition(self) -> dict[str, Any]:
        """Fields that must be shared by paired train/query artifacts.

        Split and perturbation identify an artifact within one experiment; they
        must not create a new run identity, otherwise original and perturbed
        held-out artifacts cannot be paired safely.
        """
        return {
            key: value
            for key, value in self.to_dict().items()
            if key
            not in {
                "split",
                "index_path",
                "perturbation",
                "output_root",
                "run_id",
            }
        }


@dataclass(frozen=True)
class RunPaths:
    config: RunConfig

    @property
    def run_dir(self) -> Path:
        return Path(self.config.output_root) / self.config.resolved_run_id

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def embeddings_dir(self) -> Path:
        return self.run_dir / "embeddings"

    @property
    def metrics_dir(self) -> Path:
        return self.run_dir / "metrics"

    @property
    def reports_dir(self) -> Path:
        return self.run_dir / "reports"

    @property
    def plots_dir(self) -> Path:
        return self.run_dir / "plots"

    @property
    def logs_dir(self) -> Path:
        return self.run_dir / "logs"

    def ensure_directories(self) -> None:
        for path in (
            self.embeddings_dir,
            self.metrics_dir,
            self.reports_dir,
            self.plots_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def artifact_path(self) -> Path:
        name = str(self.config.perturbation.get("name", "none"))
        if name == "none":
            return self.embeddings_dir / self.config.split / "original.pt"
        label = _slug_part(str(self.config.perturbation.get("artifact_label", name)))
        return self.embeddings_dir / self.config.split / "perturbations" / f"{label}.pt"


def write_manifest(
    paths: RunPaths,
    *,
    command: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write the resolved run definition before extraction begins."""
    paths.ensure_directories()
    manifest_path = paths.manifest_path
    artifact_key = str(paths.artifact_path().relative_to(paths.run_dir))
    manifest = {
        "format_version": 1,
        "run_id": paths.config.resolved_run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "experiment_config": paths.config.experiment_definition(),
        "artifact_configs": {artifact_key: paths.config.to_dict()},
        "artifact_commands": {artifact_key: command or []},
        "dataset_subset_summary": _load_optional_json(paths.config.subset_summary_path),
        "code_commit": _git_commit(),
        "command": command or [],
    }
    if manifest_path.exists() and not overwrite:
        existing = _load_json(manifest_path)
        if existing.get("experiment_config") != manifest["experiment_config"]:
            raise FileExistsError(
                f"Run manifest already exists with a different config: {manifest_path}"
            )
        artifact_configs = existing.setdefault("artifact_configs", {})
        if not isinstance(artifact_configs, dict):
            raise TypeError(f"Run manifest has invalid artifact_configs: {manifest_path}")
        artifact_commands = existing.setdefault("artifact_commands", {})
        if not isinstance(artifact_commands, dict):
            raise TypeError(f"Run manifest has invalid artifact_commands: {manifest_path}")
        existing_config = artifact_configs.get(artifact_key)
        if existing_config is not None and existing_config != paths.config.to_dict():
            raise FileExistsError(
                "Run manifest already contains a different config for artifact: "
                f"{artifact_key}"
            )
        if existing_config is None:
            artifact_configs[artifact_key] = paths.config.to_dict()
            artifact_commands[artifact_key] = command or []
            manifest_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return existing
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _slug_part(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return normalized.strip("-") or "run"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Run manifest must contain a JSON object: {path}")
    return data


def _load_optional_json(path_value: str | None) -> dict[str, Any] | None:
    if path_value is None:
        return None
    return _load_json(Path(path_value))
