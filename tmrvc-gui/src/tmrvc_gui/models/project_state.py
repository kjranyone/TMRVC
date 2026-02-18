"""Project state management: paths, checkpoint discovery, dependency checks."""

from __future__ import annotations

from pathlib import Path


class ProjectState:
    """Manages project paths and discovers artifacts (checkpoints, speakers, ONNX)."""

    def __init__(self, project_root: Path | None = None) -> None:
        self.project_root = project_root or Path.cwd()
        self.data_dir = self.project_root / "data"
        self.checkpoints_dir = self.project_root / "checkpoints"
        self.export_dir = self.project_root / "export"
        self.speakers_dir = self.project_root / "speakers"
        self.configs_dir = self.project_root / "configs"

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def find_checkpoints(self, pattern: str = "*.pt") -> list[Path]:
        """Find checkpoint files matching *pattern* under checkpoints_dir."""
        if not self.checkpoints_dir.exists():
            return []
        return sorted(self.checkpoints_dir.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    def find_speakers(self) -> list[Path]:
        """Find .tmrvc_speaker files under speakers_dir."""
        if not self.speakers_dir.exists():
            return []
        return sorted(self.speakers_dir.rglob("*.tmrvc_speaker"))

    def find_onnx_dirs(self) -> list[Path]:
        """Find directories containing ONNX models under export_dir."""
        if not self.export_dir.exists():
            return []
        dirs: list[Path] = []
        for onnx_file in self.export_dir.rglob("*.onnx"):
            d = onnx_file.parent
            if d not in dirs:
                dirs.append(d)
        return sorted(dirs)

    def find_eval_sets(self) -> list[str]:
        """Return available evaluation set names."""
        sets: list[str] = []
        test_dir = self.data_dir / "test"
        if test_dir.exists():
            for sub in sorted(test_dir.iterdir()):
                if sub.is_dir():
                    sets.append(sub.name)
        return sets if sets else ["VCTK-test", "JVS-test"]

    # ------------------------------------------------------------------
    # Prerequisite checks
    # ------------------------------------------------------------------

    def check_prerequisites(self, task: str) -> list[str]:
        """Check prerequisites for a given task. Returns list of missing items."""
        missing: list[str] = []

        if task == "train":
            if not self.data_dir.exists():
                missing.append("Data directory not found")
            if not list(self.data_dir.rglob("*.wav")) if self.data_dir.exists() else True:
                missing.append("No audio files found in data directory")

        elif task == "distill":
            if not self.find_checkpoints("teacher_*.pt"):
                missing.append("No teacher checkpoint found")

        elif task == "export":
            if not self.find_checkpoints("student_*.pt"):
                missing.append("No student checkpoint found")

        elif task == "realtime":
            if not self.find_onnx_dirs():
                missing.append("No ONNX models found")
            if not self.find_speakers():
                missing.append("No speaker files found")

        elif task == "evaluate":
            if not self.find_checkpoints():
                missing.append("No checkpoints found")

        elif task == "enroll":
            onnx_dirs = self.find_onnx_dirs()
            has_speaker_enc = any(
                (d / "speaker_encoder.onnx").exists() for d in onnx_dirs
            )
            if not has_speaker_enc:
                missing.append("speaker_encoder.onnx not found")

        return missing

    # ------------------------------------------------------------------
    # Directory creation
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create standard project directories if they don't exist."""
        for d in (self.data_dir, self.checkpoints_dir, self.export_dir,
                  self.speakers_dir, self.configs_dir):
            d.mkdir(parents=True, exist_ok=True)
