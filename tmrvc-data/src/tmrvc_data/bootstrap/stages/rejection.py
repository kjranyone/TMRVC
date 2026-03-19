"""Stage 3: Quality rejection — overlap, music, noise, BGM detection.

Rejects segments that would contaminate downstream speaker embedding
or physical feature extraction.  Segments are marked ``is_rejected=True``
with a rejection reason.
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


class RejectionStage:
    """Overlap / music / noise / BGM rejection.

    Rejection criteria:
    - Duration too short (< segment_min_sec)
    - Duration too long (> segment_max_sec) — split should have handled
    - Estimated SNR below threshold
    - Music-dominant content (spectral flux + harmonicity heuristic)
    - Speaker overlap detected (energy bimodality in diarization window)
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Evaluate each utterance and mark rejected ones.

        Rejected utterances are still returned (with ``is_rejected=True``)
        so that downstream reporting can account for them.
        """
        for utt in utterances:
            if utt.is_rejected:
                continue

            reasons = self._evaluate(utt)
            if reasons:
                utt.is_rejected = True
                utt.rejection_reason = "; ".join(reasons)
                logger.debug(
                    "Rejected %s: %s", utt.utterance_id, utt.rejection_reason,
                )

            utt.stage_completed = BootstrapStage.REJECTION

        n_rejected = sum(1 for u in utterances if u.is_rejected)
        logger.info(
            "Rejection: %d / %d utterances rejected",
            n_rejected, len(utterances),
        )
        return utterances

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _evaluate(self, utt: BootstrapUtterance) -> List[str]:
        """Return a list of rejection reasons (empty = accepted)."""
        reasons: List[str] = []

        # Duration check
        if utt.duration_sec < self.config.segment_min_sec:
            reasons.append(f"too_short:{utt.duration_sec:.2f}s")
            return reasons  # No point checking further

        if utt.duration_sec > self.config.segment_max_sec:
            reasons.append(f"too_long:{utt.duration_sec:.2f}s")

        # Audio-based checks require loading the segment
        if utt.audio_path is None:
            return reasons

        try:
            audio, sr = self._load_segment_audio(utt)
        except Exception as exc:
            reasons.append(f"load_error:{exc}")
            return reasons

        if len(audio) < 100:
            reasons.append("empty_audio")
            return reasons

        # SNR estimation
        snr_db = self._estimate_snr(audio, sr)
        if snr_db < self.config.snr_rejection_threshold_db:
            reasons.append(f"low_snr:{snr_db:.1f}dB")

        # Music detection
        music_score = self._estimate_music_score(audio, sr)
        if music_score > self.config.music_rejection_threshold:
            reasons.append(f"music:{music_score:.2f}")

        # Overlap detection
        overlap_score = self._estimate_overlap_score(audio, sr)
        if overlap_score > self.config.overlap_rejection_threshold:
            reasons.append(f"overlap:{overlap_score:.2f}")

        return reasons

    @staticmethod
    def _estimate_snr(audio: np.ndarray, sr: int) -> float:
        """Estimate SNR using a simple VAD-based approach.

        Splits the signal into speech and non-speech frames using
        energy thresholding, then computes the ratio.
        """
        frame_len = int(sr * 0.025)  # 25 ms frames
        hop = int(sr * 0.010)        # 10 ms hop
        n_frames = max(1, (len(audio) - frame_len) // hop)

        energies = np.array([
            np.mean(audio[i * hop:i * hop + frame_len] ** 2)
            for i in range(n_frames)
        ])

        if len(energies) == 0 or np.max(energies) < 1e-10:
            return 0.0

        # Simple threshold: median energy as boundary
        threshold = np.median(energies) * 0.5
        speech_energy = np.mean(energies[energies > threshold]) if np.any(energies > threshold) else 1e-10
        noise_energy = np.mean(energies[energies <= threshold]) if np.any(energies <= threshold) else 1e-10

        if noise_energy < 1e-10:
            return 60.0  # Effectively clean

        snr = 10.0 * np.log10(speech_energy / noise_energy)
        return float(snr)

    @staticmethod
    def _estimate_music_score(audio: np.ndarray, sr: int) -> float:
        """Estimate probability that the segment is music-dominant.

        Uses spectral flatness as a proxy: music tends to have lower
        spectral flatness (more harmonic) and higher rhythmic regularity.
        Returns a score in [0, 1].
        """
        try:
            import librosa

            # Spectral flatness: close to 0 = tonal/harmonic, close to 1 = noise-like
            spec_flat = librosa.feature.spectral_flatness(y=audio, hop_length=512)
            mean_flatness = float(np.mean(spec_flat))

            # Chroma-based harmonicity check
            chroma = librosa.feature.chroma_stft(y=audio, sr=sr, hop_length=512)
            chroma_peak = float(np.mean(np.max(chroma, axis=0)))

            # Music has low flatness + high chroma peaks
            # Score: higher = more likely music
            music_score = (1.0 - mean_flatness) * chroma_peak
            return float(np.clip(music_score, 0.0, 1.0))
        except ImportError:
            pass

        # Fallback: autocorrelation-based rhythmic regularity
        # Strong periodic peaks suggest music
        n = len(audio)
        if n < sr:
            return 0.0

        # Look at autocorrelation at typical beat frequencies (60-180 BPM)
        acf = np.correlate(audio[:sr], audio[:sr], mode="full")
        acf = acf[len(acf) // 2:]
        acf = acf / (acf[0] + 1e-10)

        # Check for peaks at beat intervals
        min_lag = int(sr / 3.0)  # 180 BPM
        max_lag = int(sr / 1.0)  # 60 BPM
        if max_lag > len(acf):
            max_lag = len(acf)
        if min_lag >= max_lag:
            return 0.0

        beat_region = acf[min_lag:max_lag]
        peak_score = float(np.max(beat_region)) if len(beat_region) > 0 else 0.0
        return float(np.clip(peak_score, 0.0, 1.0))

    @staticmethod
    def _estimate_overlap_score(audio: np.ndarray, sr: int) -> float:
        """Estimate speaker overlap probability.

        Uses energy variance in short windows: overlapping speech tends
        to show less energy modulation (flatter energy envelope) compared
        to single-speaker speech.
        """
        frame_len = int(sr * 0.050)  # 50 ms
        hop = int(sr * 0.025)
        n_frames = max(1, (len(audio) - frame_len) // hop)

        if n_frames < 4:
            return 0.0

        energies = np.array([
            float(np.sqrt(np.mean(audio[i * hop:i * hop + frame_len] ** 2)))
            for i in range(n_frames)
        ])

        # Coefficient of variation of energy
        mean_e = np.mean(energies)
        if mean_e < 1e-8:
            return 0.0

        cv = float(np.std(energies) / mean_e)

        # Single speaker: high CV (silences between words).
        # Overlap: low CV (continuous energy from multiple speakers).
        # Map: CV < 0.3 -> high overlap score, CV > 0.8 -> low overlap
        overlap_score = float(np.clip(1.0 - (cv - 0.3) / 0.5, 0.0, 1.0))
        return overlap_score

    @staticmethod
    def _load_segment_audio(utt: BootstrapUtterance) -> tuple[np.ndarray, int]:
        """Load the audio segment defined by utt.segment boundaries."""
        path = Path(utt.audio_path)

        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32")
        except Exception:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(str(path))
                data = waveform.numpy().squeeze()
            except Exception:
                from scipy.io import wavfile
                sr, data = wavfile.read(str(path))
                if data.dtype != np.float32:
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max

        if data.ndim > 1:
            data = np.mean(data, axis=0)

        # Extract segment if boundaries are set
        if utt.segment is not None and utt.segment.end_sec > 0:
            start_sample = int(utt.segment.start_sec * sr)
            end_sample = int(utt.segment.end_sec * sr)
            data = data[start_sample:end_sample]

        return data, sr
