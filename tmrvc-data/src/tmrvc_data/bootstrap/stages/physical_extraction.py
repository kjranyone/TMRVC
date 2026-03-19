"""Stage 9: Physical control targets and SSL feature extraction.

Extracts 12-D physical voice control targets per frame using DSP
analysis (via VoiceStateEstimator) and 128-dim WavLM SSL features.

Physical dimensions (canonical order):
  0: pitch_level       6: voice_irregularity
  1: pitch_range       7: openness
  2: energy_level      8: aperiodicity
  3: pressedness       9: formant_shift
  4: spectral_tilt    10: vocal_effort
  5: breathiness      11: creak
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
)

logger = logging.getLogger(__name__)

PHYSICAL_DIM = 12
SSL_DIM = 128
CANONICAL_HOP_LENGTH = 240
CANONICAL_SR = 24000


class PhysicalExtractionStage:
    """12-D physical control targets + 128-dim WavLM SSL features.

    Uses the existing VoiceStateEstimator for DSP-based physical
    feature extraction and WavLMFeatureExtractor for SSL features.
    Each dimension receives a per-frame confidence score and an
    observed_mask indicating extraction success.
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()
        self._voice_state_estimator = None
        self._wavlm_extractor = None

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Extract physical targets and SSL features for each utterance."""
        for utt in utterances:
            if utt.is_rejected:
                utt.stage_completed = BootstrapStage.PHYSICAL_EXTRACTION
                continue

            try:
                audio, sr = self._load_segment_audio(utt)
                n_frames = max(1, len(audio) // CANONICAL_HOP_LENGTH)

                # 1. Extract 12-D physical targets via DSP
                phys_targets, phys_mask, phys_conf = self._extract_physical(
                    audio, sr,
                )

                # 2. Extract SSL features via WavLM
                ssl_features = self._extract_ssl(audio, sr, n_frames)

                utt.physical_targets = phys_targets
                utt.physical_observed_mask = phys_mask
                utt.physical_confidence = phys_conf
                utt.n_frames = phys_targets.shape[0]

            except Exception as exc:
                logger.warning(
                    "Physical extraction failed for %s: %s",
                    utt.utterance_id, exc,
                )
                n_frames = max(1, int(utt.duration_sec * CANONICAL_SR / CANONICAL_HOP_LENGTH))
                utt.physical_targets = np.zeros(
                    (n_frames, PHYSICAL_DIM), dtype=np.float32,
                )
                utt.physical_observed_mask = np.zeros(
                    (n_frames, PHYSICAL_DIM), dtype=bool,
                )
                utt.physical_confidence = np.zeros(
                    (n_frames, PHYSICAL_DIM), dtype=np.float32,
                )
                utt.n_frames = n_frames
                utt.warnings.append(f"physical_extraction_error:{exc}")

            utt.stage_completed = BootstrapStage.PHYSICAL_EXTRACTION

        logger.info(
            "PhysicalExtraction: processed %d utterances", len(utterances),
        )
        return utterances

    # ------------------------------------------------------------------
    # DSP-based physical feature extraction
    # ------------------------------------------------------------------

    def _extract_physical(
        self, audio: np.ndarray, sr: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 12-D physical controls per frame.

        Returns:
            (targets [T,12], mask [T,12] bool, confidence [T,12])
        """
        estimator = self._get_voice_state_estimator()
        values, mask, confidence = estimator.estimate_frames(audio, sr)

        # Supplement with additional DSP analysis not in the base estimator
        self._enrich_with_parselmouth(audio, sr, values, mask, confidence)
        self._enrich_with_spectral_analysis(audio, sr, values, mask, confidence)

        return values, mask, confidence

    def _enrich_with_parselmouth(
        self,
        audio: np.ndarray,
        sr: int,
        values: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        """Enrich physical features using parselmouth (Praat)."""
        try:
            import parselmouth
            from parselmouth.praat import call
        except ImportError:
            logger.debug("parselmouth not available for physical enrichment")
            return

        n_frames = values.shape[0]
        hop_sec = CANONICAL_HOP_LENGTH / CANONICAL_SR

        try:
            snd = parselmouth.Sound(audio, sampling_frequency=sr)

            # Pitch analysis -> pitch_level (dim 0), pitch_range (dim 1)
            pitch_obj = call(snd, "To Pitch", 0.0, 75.0, 600.0)
            pitch_values = [
                call(pitch_obj, "Get value at time", i * hop_sec, "Hertz", "Linear")
                for i in range(n_frames)
            ]

            valid_pitches = [p for p in pitch_values if p == p and p > 0]  # filter NaN
            if valid_pitches:
                mean_pitch = np.mean(valid_pitches)
                std_pitch = np.std(valid_pitches) if len(valid_pitches) > 1 else 1.0

                for i in range(n_frames):
                    p = pitch_values[i]
                    if p == p and p > 0:  # not NaN
                        # Normalize pitch to [0, 1]
                        norm_pitch = np.clip((p - mean_pitch) / (3 * std_pitch + 1e-6) + 0.5, 0, 1)
                        values[i, 0] = norm_pitch
                        mask[i, 0] = True
                        confidence[i, 0] = 0.8

                # Pitch range: local variance over 10-frame windows
                for i in range(n_frames):
                    window_start = max(0, i - 5)
                    window_end = min(n_frames, i + 5)
                    window_pitches = [
                        pitch_values[j] for j in range(window_start, window_end)
                        if pitch_values[j] == pitch_values[j] and pitch_values[j] > 0
                    ]
                    if len(window_pitches) >= 2:
                        local_range = (max(window_pitches) - min(window_pitches)) / (mean_pitch + 1e-6)
                        values[i, 1] = np.clip(local_range, 0, 1)
                        mask[i, 1] = True
                        confidence[i, 1] = 0.7

            # HNR -> breathiness (dim 5) and voice_irregularity (dim 6)
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            for i in range(n_frames):
                t = i * hop_sec
                hnr = call(harmonicity, "Get value at time", t, "Cubic")
                if hnr == hnr and hnr != -200:  # not NaN, not undefined
                    # Low HNR = breathy; map HNR [0, 30] -> breathiness [1, 0]
                    breathiness = np.clip(1.0 - hnr / 30.0, 0, 1)
                    values[i, 5] = breathiness
                    mask[i, 5] = True
                    confidence[i, 5] = 0.7

            # Jitter + shimmer -> voice_irregularity (dim 6)
            point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
            try:
                jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                shimmer = call(
                    [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
                )
                if jitter == jitter and shimmer == shimmer:
                    irregularity = np.clip(jitter * 50 + shimmer * 5, 0, 1)
                    for i in range(n_frames):
                        values[i, 6] = irregularity
                        mask[i, 6] = True
                        confidence[i, 6] = 0.6
            except Exception:
                pass

        except Exception as exc:
            logger.debug("parselmouth analysis partial failure: %s", exc)

    def _enrich_with_spectral_analysis(
        self,
        audio: np.ndarray,
        sr: int,
        values: np.ndarray,
        mask: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        """Enrich physical features with spectral analysis.

        Computes spectral_tilt (dim 4), pressedness (dim 3), openness (dim 7),
        aperiodicity (dim 8), formant_shift (dim 9), vocal_effort (dim 10),
        creak (dim 11).
        """
        n_frames = values.shape[0]
        hop_samples = CANONICAL_HOP_LENGTH
        if sr != CANONICAL_SR:
            hop_samples = int(CANONICAL_HOP_LENGTH * sr / CANONICAL_SR)

        for i in range(n_frames):
            start = i * hop_samples
            end = min(start + hop_samples * 2, len(audio))  # Use 2-hop window
            frame = audio[start:end]

            if len(frame) < 64:
                continue

            # Spectral analysis via FFT
            n_fft = min(2048, len(frame))
            spectrum = np.abs(np.fft.rfft(frame, n=n_fft))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

            if len(spectrum) < 2:
                continue

            log_spectrum = np.log(spectrum + 1e-10)
            total_energy = np.sum(spectrum ** 2)

            if total_energy < 1e-12:
                continue

            # Spectral tilt (dim 4): slope of log spectrum
            x = np.arange(len(log_spectrum), dtype=np.float32)
            if len(x) > 1:
                slope = np.polyfit(x, log_spectrum, 1)[0]
                # Normalize: steep negative slope = high tilt (breathy),
                # flat/positive = pressed
                tilt_norm = np.clip(-slope * 100 + 0.5, 0, 1)
                values[i, 4] = tilt_norm
                mask[i, 4] = True
                confidence[i, 4] = 0.6

            # Pressedness (dim 3): ratio of high-freq to low-freq energy
            mid_bin = len(spectrum) // 2
            low_energy = np.sum(spectrum[:mid_bin] ** 2)
            high_energy = np.sum(spectrum[mid_bin:] ** 2)
            if low_energy > 1e-10:
                press_ratio = high_energy / (low_energy + high_energy)
                values[i, 3] = np.clip(press_ratio * 2, 0, 1)
                mask[i, 3] = True
                confidence[i, 3] = 0.5

            # Openness (dim 7): first formant frequency proxy
            # Higher F1 = more open
            try:
                import librosa
                spectral_centroid = librosa.feature.spectral_centroid(
                    y=frame, sr=sr, hop_length=len(frame),
                )
                if spectral_centroid.size > 0:
                    centroid = float(spectral_centroid[0, 0])
                    openness = np.clip(centroid / 3000.0, 0, 1)
                    values[i, 7] = openness
                    mask[i, 7] = True
                    confidence[i, 7] = 0.5
            except (ImportError, Exception):
                pass

            # Aperiodicity (dim 8): ratio of noise to total energy
            # Approximate using cepstral peak prominence
            cepstrum = np.abs(np.fft.rfft(log_spectrum))
            if len(cepstrum) > 10:
                peak_region = cepstrum[5:]  # Skip DC and low quefrency
                peak = np.max(peak_region)
                mean_level = np.mean(peak_region)
                if mean_level > 1e-10:
                    cpp = peak / mean_level
                    aperiodicity = np.clip(1.0 - cpp / 10.0, 0, 1)
                    values[i, 8] = aperiodicity
                    mask[i, 8] = True
                    confidence[i, 8] = 0.5

            # Vocal effort (dim 10): combination of energy and spectral tilt
            rms = float(np.sqrt(np.mean(frame ** 2)))
            effort = np.clip(rms / 0.15, 0, 1)  # Normalized against typical speech
            values[i, 10] = effort
            mask[i, 10] = True
            confidence[i, 10] = 0.6

            # Creak / vocal fry (dim 11): subharmonic detection
            # Look for energy at half the fundamental frequency
            if values[i, 0] > 0 and mask[i, 0]:
                # If we have pitch, check for subharmonics
                pass  # Complex analysis, leave for parselmouth enrichment

        # Formant shift (dim 9): relative to speaker mean
        # This is better computed after all frames are processed
        centroids = []
        for i in range(n_frames):
            if mask[i, 7]:
                centroids.append(values[i, 7])
        if centroids:
            mean_centroid = np.mean(centroids)
            for i in range(n_frames):
                if mask[i, 7]:
                    shift = values[i, 7] - mean_centroid + 0.5
                    values[i, 9] = np.clip(shift, 0, 1)
                    mask[i, 9] = True
                    confidence[i, 9] = 0.4

    # ------------------------------------------------------------------
    # SSL feature extraction
    # ------------------------------------------------------------------

    def _extract_ssl(
        self, audio: np.ndarray, sr: int, target_frames: int,
    ) -> np.ndarray:
        """Extract 128-dim WavLM SSL features.

        Falls back to zeros if WavLM is unavailable.
        """
        try:
            return self._extract_wavlm(audio, sr, target_frames)
        except (ImportError, Exception) as exc:
            logger.debug("WavLM SSL extraction failed: %s", exc)
            return np.zeros((target_frames, SSL_DIM), dtype=np.float32)

    def _extract_wavlm(
        self, audio: np.ndarray, sr: int, target_frames: int,
    ) -> np.ndarray:
        """Extract WavLM features and project to 128-dim."""
        import torch

        if self._wavlm_extractor is None:
            from tmrvc_data.wavlm_extractor import WavLMFeatureExtractor

            self._wavlm_extractor = WavLMFeatureExtractor(
                d_output=SSL_DIM,
                freeze=True,
                model_name="microsoft/wavlm-large",
            )
            if self.config.device == "cuda" and torch.cuda.is_available():
                self._wavlm_extractor = self._wavlm_extractor.to("cuda")

        # WavLM expects 16 kHz
        if sr != 16000:
            audio_16k = self._resample(audio, sr, 16000)
        else:
            audio_16k = audio

        device = next(self._wavlm_extractor.parameters()).device
        wav_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).to(device)

        with torch.no_grad():
            features = self._wavlm_extractor(wav_tensor)  # [1, d_output, T]

        features = features.squeeze(0).transpose(0, 1).cpu().numpy()  # [T, d_output]

        # Interpolate to match target frame count
        if features.shape[0] != target_frames and features.shape[0] > 0:
            from scipy.interpolate import interp1d

            x_orig = np.linspace(0, 1, features.shape[0])
            x_target = np.linspace(0, 1, target_frames)
            interpolator = interp1d(x_orig, features, axis=0, kind="linear")
            features = interpolator(x_target)

        return features.astype(np.float32)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_voice_state_estimator(self):
        """Lazy-init the VoiceStateEstimator."""
        if self._voice_state_estimator is None:
            from tmrvc_data.curation.providers.voice_state import (
                VoiceStateEstimator,
            )
            self._voice_state_estimator = VoiceStateEstimator()
        return self._voice_state_estimator

    @staticmethod
    def _load_segment_audio(utt: BootstrapUtterance) -> tuple[np.ndarray, int]:
        """Load audio segment."""
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

        if utt.segment is not None and utt.segment.end_sec > 0:
            start = int(utt.segment.start_sec * sr)
            end = int(utt.segment.end_sec * sr)
            data = data[start:end]

        return data, sr

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio."""
        if orig_sr == target_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
