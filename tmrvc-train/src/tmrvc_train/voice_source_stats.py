"""Voice source parameter statistics tracker and group preset computation."""

from __future__ import annotations

import fnmatch
import json
import logging
from pathlib import Path
from typing import Self

import numpy as np
import torch

from tmrvc_core.constants import N_IR_PARAMS, N_VOICE_SOURCE_PARAMS, VOICE_SOURCE_PARAM_NAMES

logger = logging.getLogger(__name__)


class VoiceSourceStatsTracker:
    """Per-speaker running mean tracker for voice source parameters.

    Accumulates voice source params (indices 24-31 of acoustic_params)
    using float64 running sums to prevent floating-point drift.
    """

    def __init__(self) -> None:
        self._sums: dict[str, np.ndarray] = {}   # speaker_id -> float64[8]
        self._counts: dict[str, int] = {}          # speaker_id -> count

    def update(self, acoustic_params: torch.Tensor, speaker_ids: list[str]) -> None:
        """Update running means with a batch of acoustic params.

        Args:
            acoustic_params: Tensor of shape ``[B, 32]`` (or ``[B, N_ACOUSTIC_PARAMS]``).
            speaker_ids: List of speaker IDs, length ``B``.
        """
        voice_source = acoustic_params[:, N_IR_PARAMS:].detach().cpu().numpy().astype(np.float64)
        B = voice_source.shape[0]
        assert len(speaker_ids) == B

        for i in range(B):
            sid = speaker_ids[i]
            if sid not in self._sums:
                self._sums[sid] = np.zeros(N_VOICE_SOURCE_PARAMS, dtype=np.float64)
                self._counts[sid] = 0
            self._sums[sid] += voice_source[i]
            self._counts[sid] += 1

    def get_speaker_mean(self, speaker_id: str) -> np.ndarray | None:
        """Return the running mean for a speaker, or None if not tracked."""
        if speaker_id not in self._sums or self._counts[speaker_id] == 0:
            return None
        return (self._sums[speaker_id] / self._counts[speaker_id]).astype(np.float32)

    def get_all_means(self) -> dict[str, list[float]]:
        """Return all speaker means as a dict of lists."""
        result = {}
        for sid in self._sums:
            mean = self.get_speaker_mean(sid)
            if mean is not None:
                result[sid] = mean.tolist()
        return result

    def save(self, path: Path) -> None:
        """Save tracker state to JSON."""
        data = {
            "param_names": list(VOICE_SOURCE_PARAM_NAMES),
            "speakers": {},
        }
        for sid in self._sums:
            data["speakers"][sid] = {
                "sum": self._sums[sid].tolist(),
                "count": self._counts[sid],
            }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Saved voice source stats to %s (%d speakers)", path, len(self._sums))

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load tracker state from JSON."""
        tracker = cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        for sid, info in data.get("speakers", {}).items():
            tracker._sums[sid] = np.array(info["sum"], dtype=np.float64)
            tracker._counts[sid] = info["count"]
        logger.info("Loaded voice source stats from %s (%d speakers)", path, len(tracker._sums))
        return tracker


def compute_group_preset(
    stats_path: str | Path,
    patterns: list[str],
    output_path: str | Path | None = None,
) -> dict:
    """Compute a group voice source preset by averaging matched speakers.

    Args:
        stats_path: Path to the JSON stats file saved by ``VoiceSourceStatsTracker``.
        patterns: List of fnmatch patterns to match speaker IDs.
        output_path: Optional path to save the result as JSON.

    Returns:
        Dict with keys ``preset`` (list of 8 floats), ``matched_speakers`` (list),
        ``n_speakers`` (int).

    Raises:
        ValueError: If no speakers match the given patterns.
    """
    tracker = VoiceSourceStatsTracker.load(Path(stats_path))
    all_means = tracker.get_all_means()

    matched = []
    for sid in all_means:
        for pat in patterns:
            if fnmatch.fnmatch(sid, pat):
                matched.append(sid)
                break

    if not matched:
        raise ValueError(
            f"No speakers matched patterns {patterns}. "
            f"Available speakers: {sorted(all_means.keys())}"
        )

    # Average the matched speaker means
    arrays = [np.array(all_means[sid], dtype=np.float64) for sid in matched]
    preset = (np.mean(arrays, axis=0)).astype(np.float32)

    result = {
        "preset": preset.tolist(),
        "matched_speakers": sorted(matched),
        "n_speakers": len(matched),
        "param_names": list(VOICE_SOURCE_PARAM_NAMES),
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("Saved group preset to %s (%d speakers)", output_path, len(matched))

    return result
