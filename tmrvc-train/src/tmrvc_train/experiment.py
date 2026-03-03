"""Experiment management for reproducible training pipelines."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentMetadata:
    """Metadata for a training experiment."""

    experiment_id: str
    dataset: str
    created_at: str
    git_hash: str
    git_branch: str
    python_version: str
    config: dict[str, Any]
    seed: int
    workers: int
    status: str = "initialized"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentMetadata:
        return cls(**data)


class ExperimentManager:
    """Manages experiment lifecycle and metadata."""

    def __init__(self, experiment_dir: Path):
        self.experiment_dir = experiment_dir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = experiment_dir / "experiment.yaml"
        self.errors_file = experiment_dir / "errors.json"
        self.log_dir = experiment_dir / "logs"
        self.checkpoint_dir = experiment_dir / "checkpoints"

        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def create_experiment(
        self,
        experiment_id: str,
        dataset: str,
        config: dict[str, Any],
        seed: int,
        workers: int,
    ) -> ExperimentMetadata:
        """Create new experiment with full metadata."""

        git_hash = self._get_git_hash()
        git_branch = self._get_git_branch()

        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            dataset=dataset,
            created_at=datetime.now().isoformat(),
            git_hash=git_hash,
            git_branch=git_branch,
            python_version=sys.version,
            config=config,
            seed=seed,
            workers=workers,
            status="initialized",
        )

        self._save_metadata(metadata)
        return metadata

    def _get_git_hash(self) -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _get_git_branch(self) -> str:
        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def _save_metadata(self, metadata: ExperimentMetadata) -> None:
        with open(self.metadata_file, "w") as f:
            yaml.dump(metadata.to_dict(), f, default_flow_style=False)

    def load_metadata(self) -> ExperimentMetadata | None:
        if not self.metadata_file.exists():
            return None

        with open(self.metadata_file) as f:
            data = yaml.safe_load(f)

        return ExperimentMetadata.from_dict(data)

    def update_status(self, status: str) -> None:
        metadata = self.load_metadata()
        if metadata:
            metadata.status = status
            self._save_metadata(metadata)

    def log_errors(self, new_errors: list[dict[str, Any]]) -> None:
        """Log multiple processing errors at once."""
        errors = []
        if self.errors_file.exists():
            with open(self.errors_file) as f:
                errors = json.load(f)

        errors.extend(new_errors)

        with open(self.errors_file, "w") as f:
            json.dump(errors, f, indent=2)

    def log_error(
        self,
        utterance_id: str,
        error_type: str,
        error_message: str,
        stage: str = "preprocessing",
    ) -> None:
        """Log processing errors for later retry."""

        errors = []
        if self.errors_file.exists():
            with open(self.errors_file) as f:
                errors = json.load(f)

        errors.append(
            {
                "utterance_id": utterance_id,
                "error_type": error_type,
                "error_message": error_message,
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
            }
        )

        with open(self.errors_file, "w") as f:
            json.dump(errors, f, indent=2)

    def get_error_count(self) -> int:
        if not self.errors_file.exists():
            return 0

        with open(self.errors_file) as f:
            return len(json.load(f))
