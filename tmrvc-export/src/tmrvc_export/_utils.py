"""Shared export utilities."""

from __future__ import annotations

from pathlib import Path


def prepare_output_path(output_path: Path) -> None:
    """Remove stale ONNX/external-data files before export."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for p in (output_path, Path(f"{output_path}.data")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
