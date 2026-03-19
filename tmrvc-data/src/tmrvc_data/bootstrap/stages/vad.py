"""Stage 2: Voice Activity Detection and utterance segmentation.

Uses Silero-VAD to detect speech regions, then splits long files into
utterance-level segments bounded by silence.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
    SegmentInfo,
)

logger = logging.getLogger(__name__)


class VADStage:
    """Silero-VAD based utterance segmentation.

    Splits each input file into utterance-level segments at silence
    boundaries detected by Silero-VAD.  Segments shorter than
    ``config.segment_min_sec`` or longer than ``config.segment_max_sec``
    are flagged accordingly.
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()
        self._vad_model = None
        self._vad_utils = None

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Run VAD on each utterance and split into segments.

        Each input utterance (one per file) may produce multiple output
        utterances (one per speech segment).
        """
        result: List[BootstrapUtterance] = []

        for utt in utterances:
            if utt.audio_path is None:
                utt.warnings.append("vad: no audio_path")
                result.append(utt)
                continue

            try:
                audio, sr = self._load_audio(Path(utt.audio_path))
            except Exception as exc:
                logger.warning("vad: failed to load %s: %s", utt.audio_path, exc)
                utt.errors.append(f"vad_load_error:{exc}")
                result.append(utt)
                continue

            segments = self._detect_speech_segments(audio, sr)

            if not segments:
                utt.warnings.append("vad: no speech detected")
                utt.stage_completed = BootstrapStage.VAD_SEGMENTATION
                result.append(utt)
                continue

            for seg_idx, (start_sec, end_sec) in enumerate(segments):
                duration = end_sec - start_sec

                if duration < self.config.segment_min_sec:
                    continue
                if duration > self.config.segment_max_sec:
                    sub_segments = self._split_long_segment(
                        start_sec, end_sec, self.config.segment_max_sec,
                    )
                    for sub_idx, (ss, se) in enumerate(sub_segments):
                        seg_utt = self._create_segment_utterance(
                            utt, ss, se, seg_idx * 100 + sub_idx, sr,
                        )
                        result.append(seg_utt)
                    continue

                seg_utt = self._create_segment_utterance(
                    utt, start_sec, end_sec, seg_idx, sr,
                )
                result.append(seg_utt)

        logger.info(
            "VAD: %d input utterances -> %d segments", len(utterances), len(result),
        )
        return result

    # ------------------------------------------------------------------
    # VAD implementation
    # ------------------------------------------------------------------

    def _get_vad_model(self):
        """Lazy-load Silero VAD model."""
        if self._vad_model is not None:
            return self._vad_model, self._vad_utils

        try:
            import torch

            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                trust_repo=True,
            )
            self._vad_model = model
            self._vad_utils = utils
            return model, utils
        except Exception as exc:
            raise ImportError(
                f"Silero-VAD is required for the VAD stage. "
                f"Install torch and ensure network access for torch.hub. "
                f"Error: {exc}"
            ) from exc

    def _detect_speech_segments(
        self, audio: np.ndarray, sr: int,
    ) -> List[Tuple[float, float]]:
        """Detect speech segments using Silero-VAD.

        Falls back to energy-based VAD if Silero is not available.
        """
        try:
            return self._silero_vad(audio, sr)
        except (ImportError, Exception) as exc:
            logger.warning(
                "Silero-VAD unavailable (%s), falling back to energy-based VAD",
                exc,
            )
            return self._energy_vad(audio, sr)

    def _silero_vad(
        self, audio: np.ndarray, sr: int,
    ) -> List[Tuple[float, float]]:
        """Run Silero-VAD and return speech segments."""
        import torch

        model, utils = self._get_vad_model()
        get_speech_timestamps = utils[0]

        # Silero VAD expects 16 kHz
        if sr != 16000:
            audio_16k = self._resample_for_vad(audio, sr, 16000)
        else:
            audio_16k = audio

        wav_tensor = torch.from_numpy(audio_16k).float()
        if wav_tensor.dim() == 0:
            return []

        speech_timestamps = get_speech_timestamps(
            wav_tensor,
            model,
            sampling_rate=16000,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=300,
            threshold=0.5,
        )

        segments: List[Tuple[float, float]] = []
        for ts in speech_timestamps:
            start_sec = ts["start"] / 16000.0
            end_sec = ts["end"] / 16000.0
            segments.append((start_sec, end_sec))

        return segments

    @staticmethod
    def _energy_vad(
        audio: np.ndarray, sr: int,
        frame_ms: int = 30,
        energy_threshold: float = 0.01,
        min_speech_frames: int = 8,
        min_silence_frames: int = 10,
    ) -> List[Tuple[float, float]]:
        """Simple energy-based VAD fallback."""
        frame_samples = int(sr * frame_ms / 1000)
        n_frames = len(audio) // frame_samples

        if n_frames == 0:
            return []

        energies = np.array([
            float(np.sqrt(np.mean(
                audio[i * frame_samples:(i + 1) * frame_samples] ** 2,
            )))
            for i in range(n_frames)
        ])

        positive_energies = energies[energies > 0]
        median_energy = float(np.median(positive_energies)) if len(positive_energies) > 0 else 0.0
        threshold = max(energy_threshold, 0.1 * median_energy)

        is_speech = energies > threshold

        segments: List[Tuple[float, float]] = []
        in_speech = False
        start_frame = 0
        silence_count = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    start_frame = i
                    in_speech = True
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        end_frame = i - silence_count + 1
                        if end_frame - start_frame >= min_speech_frames:
                            start_sec = start_frame * frame_ms / 1000.0
                            end_sec = end_frame * frame_ms / 1000.0
                            segments.append((start_sec, end_sec))
                        in_speech = False
                        silence_count = 0

        if in_speech:
            end_frame = n_frames
            if end_frame - start_frame >= min_speech_frames:
                start_sec = start_frame * frame_ms / 1000.0
                end_sec = end_frame * frame_ms / 1000.0
                segments.append((start_sec, end_sec))

        return segments

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_segment_utterance(
        self,
        parent: BootstrapUtterance,
        start_sec: float,
        end_sec: float,
        seg_idx: int,
        sr: int,
    ) -> BootstrapUtterance:
        """Create a new utterance for a detected speech segment."""
        file_hash = hashlib.sha256(parent.source_file.encode()).hexdigest()[:12]
        seg_id = f"{file_hash}_{seg_idx:04d}"

        return BootstrapUtterance(
            utterance_id=f"{parent.corpus_id}_{seg_id}",
            corpus_id=parent.corpus_id,
            source_file=parent.source_file,
            audio_path=parent.audio_path,
            sample_rate=sr,
            duration_sec=end_sec - start_sec,
            segment=SegmentInfo(
                segment_id=seg_id,
                source_file=parent.source_file,
                start_sec=round(start_sec, 4),
                end_sec=round(end_sec, 4),
                duration_sec=round(end_sec - start_sec, 4),
                sample_rate=sr,
            ),
            stage_completed=BootstrapStage.VAD_SEGMENTATION,
        )

    @staticmethod
    def _split_long_segment(
        start: float, end: float, max_dur: float,
    ) -> List[Tuple[float, float]]:
        """Split a long segment into chunks of at most *max_dur* seconds."""
        chunks: List[Tuple[float, float]] = []
        cursor = start
        while cursor < end:
            chunk_end = min(cursor + max_dur, end)
            chunks.append((cursor, chunk_end))
            cursor = chunk_end
        return chunks

    @staticmethod
    def _resample_for_vad(
        audio: np.ndarray, orig_sr: int, target_sr: int,
    ) -> np.ndarray:
        """Quick resample for VAD (quality not critical)."""
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            pass

        from scipy.signal import resample_poly
        from math import gcd

        g = gcd(orig_sr, target_sr)
        return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)

    @staticmethod
    def _load_audio(path: Path) -> tuple[np.ndarray, int]:
        """Load audio as float32 mono numpy array."""
        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=0)
            return data, sr
        except Exception:
            pass

        try:
            import torchaudio
            waveform, sr = torchaudio.load(str(path))
            data = waveform.mean(dim=0).numpy()
            return data, sr
        except Exception:
            pass

        from scipy.io import wavfile
        sr, data = wavfile.read(str(path))
        if data.dtype != np.float32:
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        if data.ndim > 1:
            data = np.mean(data, axis=0)
        return data, sr
