"""Stage 1: Audio normalization.

Applies loudness normalization (target -23 LUFS), DC offset removal,
and resampling to 24 kHz mono float32.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
)

logger = logging.getLogger(__name__)

# Canonical target sample rate for the v4 pipeline.
TARGET_SR = 24000


class NormalizeStage:
    """Loudness normalization, DC removal, and 24 kHz resampling.

    Writes the normalised waveform back to a staging directory so that
    downstream stages always operate on clean 24 kHz mono audio.
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()
        self.target_sr = self.config.target_sample_rate
        self.target_lufs = self.config.loudness_target_lufs

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Normalise audio for each utterance in-place.

        Updates ``audio_path`` to point at the normalised file (written
        next to the original with a ``_norm.wav`` suffix) and sets
        ``sample_rate`` to 24000.
        """
        result: List[BootstrapUtterance] = []

        for utt in utterances:
            if utt.audio_path is None:
                utt.warnings.append("normalize: no audio_path")
                result.append(utt)
                continue

            try:
                audio, sr = self._load_audio(Path(utt.audio_path))
            except Exception as exc:
                logger.warning(
                    "normalize: failed to load %s: %s", utt.audio_path, exc,
                )
                utt.errors.append(f"normalize_load_error:{exc}")
                result.append(utt)
                continue

            # 1. Mono downmix
            if audio.ndim > 1:
                audio = np.mean(audio, axis=0)

            # 2. DC offset removal
            audio = audio - np.mean(audio)

            # 3. Resample to target SR
            if sr != self.target_sr:
                audio = self._resample(audio, sr, self.target_sr)
                sr = self.target_sr

            # 4. Loudness normalization (-23 LUFS)
            audio = self._normalize_loudness(audio, sr, self.target_lufs)

            # 5. Clip guard
            peak = np.max(np.abs(audio))
            if peak > 1.0:
                audio = audio / peak
                utt.warnings.append(
                    f"normalize: clipped peak {peak:.3f} to 1.0"
                )

            # Write normalised file
            norm_path = self._write_normalized(
                audio, sr, Path(utt.audio_path),
            )

            utt.audio_path = norm_path
            utt.sample_rate = sr
            utt.duration_sec = len(audio) / sr
            utt.stage_completed = BootstrapStage.AUDIO_NORMALIZATION
            result.append(utt)

        logger.info("Normalize: processed %d utterances", len(result))
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(path: Path) -> tuple[np.ndarray, int]:
        """Load audio as float32 numpy array.

        Tries soundfile, then torchaudio, then scipy.
        """
        # soundfile
        try:
            import soundfile as sf

            data, sr = sf.read(str(path), dtype="float32")
            return data, sr
        except Exception:
            pass

        # torchaudio
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(path))
            return waveform.numpy().squeeze(), sr
        except Exception:
            pass

        # scipy (wav only)
        from scipy.io import wavfile

        sr, data = wavfile.read(str(path))
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        return data, sr

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Uses librosa.resample if available, else scipy.signal.resample_poly.
        """
        if orig_sr == target_sr:
            return audio

        try:
            import librosa

            return librosa.resample(
                audio, orig_sr=orig_sr, target_sr=target_sr,
            )
        except ImportError:
            pass

        # Fallback: scipy
        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        return resample_poly(audio, up, down).astype(np.float32)

    @staticmethod
    def _normalize_loudness(
        audio: np.ndarray, sr: int, target_lufs: float = -23.0,
    ) -> np.ndarray:
        """Normalise integrated loudness to *target_lufs*.

        Uses pyloudnorm if available.  Falls back to a simple RMS-based
        heuristic targeting an RMS that roughly corresponds to the
        requested LUFS level.
        """
        try:
            import pyloudnorm as pyln

            meter = pyln.Meter(sr)
            current_lufs = meter.integrated_loudness(audio)

            if np.isinf(current_lufs) or np.isnan(current_lufs):
                # Silence or near-silence -- skip
                return audio

            return pyln.normalize.loudness(audio, current_lufs, target_lufs)
        except ImportError:
            pass

        # Heuristic fallback: map target_lufs to target RMS.
        # LUFS -23 ~ RMS -23 dBFS ~ 0.0708
        target_rms = 10.0 ** (target_lufs / 20.0)
        current_rms = float(np.sqrt(np.mean(audio ** 2)))
        if current_rms < 1e-8:
            return audio

        gain = target_rms / current_rms
        return (audio * gain).astype(np.float32)

    @staticmethod
    def _write_normalized(
        audio: np.ndarray, sr: int, original_path: Path,
    ) -> Path:
        """Write the normalised waveform to a sibling ``_norm.wav`` file."""
        norm_path = original_path.with_name(
            original_path.stem + "_norm.wav",
        )

        try:
            import soundfile as sf

            sf.write(str(norm_path), audio, sr, subtype="FLOAT")
        except ImportError:
            from scipy.io import wavfile

            wavfile.write(str(norm_path), sr, audio)

        return norm_path
