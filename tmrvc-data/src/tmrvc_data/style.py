"""Style encoder: Prosody + Emotion → 128-d embedding.

Extracts style/prosody features from audio:
- F0 statistics (mean, std, range, slope)
- Energy statistics
- Speaking rate
- Voice quality features

For higher quality, can be replaced with emotion2vec or similar.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn

from tmrvc_core.constants import SAMPLE_RATE

logger = logging.getLogger(__name__)

D_STYLE = 128


class ProsodyExtractor:
    """Extract prosodic features from audio."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """Extract prosodic features.

        Args:
            waveform: Audio waveform [T].

        Returns:
            Prosody features [32].
        """
        import librosa

        # F0 extraction
        f0, voiced_flags = librosa.piptrack(
            y=waveform.astype(np.float32),
            sr=self.sample_rate,
            fmin=50,
            fmax=500,
        )
        f0_voiced = f0[voiced_flags] if voiced_flags.any() else np.array([0.0])

        # F0 statistics (8 features)
        f0_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0.0
        f0_std = np.std(f0_voiced) if len(f0_voiced) > 0 else 0.0
        f0_min = np.min(f0_voiced) if len(f0_voiced) > 0 else 0.0
        f0_max = np.max(f0_voiced) if len(f0_voiced) > 0 else 0.0
        f0_range = f0_max - f0_min
        f0_skew = self._skewness(f0_voiced) if len(f0_voiced) > 0 else 0.0
        f0_kurt = self._kurtosis(f0_voiced) if len(f0_voiced) > 0 else 0.0
        voiced_ratio = np.sum(voiced_flags) / np.prod(voiced_flags.shape)

        # Energy statistics (8 features)
        rms = librosa.feature.rms(y=waveform.astype(np.float32))[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_min = np.min(rms)
        energy_max = np.max(rms)
        energy_range = energy_max - energy_min
        energy_skew = self._skewness(rms)
        energy_kurt = self._kurtosis(rms)
        energy_dynamic_range = energy_max / (energy_mean + 1e-8)

        # Speaking rate features (8 features)
        tempo, beats = librosa.beat.beat_track(
            y=waveform.astype(np.float32),
            sr=self.sample_rate,
        )
        onset_env = librosa.onset.onset_strength(
            y=waveform.astype(np.float32),
            sr=self.sample_rate,
        )
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=self.sample_rate
        )
        n_onsets = len(onset_frames)
        duration_sec = len(waveform) / self.sample_rate
        onset_rate = n_onsets / duration_sec if duration_sec > 0 else 0.0
        tempo_val = (
            float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
        )

        # Spectral features (8 features)
        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform.astype(np.float32),
            sr=self.sample_rate,
        )[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=waveform.astype(np.float32),
            sr=self.sample_rate,
        )[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=waveform.astype(np.float32),
            sr=self.sample_rate,
        )[0]
        spectral_flatness = librosa.feature.spectral_flatness(
            y=waveform.astype(np.float32),
        )[0]

        features = np.array(
            [
                # F0 (8)
                f0_mean / 500.0,  # Normalized
                f0_std / 200.0,
                f0_min / 500.0,
                f0_max / 500.0,
                f0_range / 500.0,
                f0_skew,
                f0_kurt,
                voiced_ratio,
                # Energy (8)
                energy_mean,
                energy_std,
                energy_min,
                energy_max,
                energy_range,
                energy_skew,
                energy_kurt,
                energy_dynamic_range,
                # Rhythm (8)
                tempo_val / 200.0,  # Normalized
                onset_rate / 20.0,
                n_onsets / 100.0,
                duration_sec / 60.0,
                np.std(onset_env) if len(onset_env) > 0 else 0.0,
                np.mean(onset_env) if len(onset_env) > 0 else 0.0,
                len(beats) / duration_sec if duration_sec > 0 else 0.0,
                0.0,  # Padding
                # Spectral (8)
                np.mean(spectral_centroid) / 10000.0,
                np.std(spectral_centroid) / 5000.0,
                np.mean(spectral_bandwidth) / 5000.0,
                np.std(spectral_bandwidth) / 2000.0,
                np.mean(spectral_rolloff) / 10000.0,
                np.std(spectral_rolloff) / 5000.0,
                np.mean(spectral_flatness),
                np.std(spectral_flatness),
            ],
            dtype=np.float32,
        )

        return features

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        """Compute skewness."""
        if len(x) < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 3))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        """Compute kurtosis."""
        if len(x) < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-8:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4) - 3)


class ProsodyToEmbedding(nn.Module):
    """Convert prosody features to embedding."""

    def __init__(self, input_dim: int = 32, output_dim: int = D_STYLE) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StyleEncoder:
    """Extract style/prosody embeddings from audio.

    Combines:
    1. Prosody features (F0, energy, rhythm, spectral)
    2. Learned projection to 128-dim embedding
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._extractor = ProsodyExtractor()
        self._projector: ProsodyToEmbedding | None = None

    def _load_projector(self) -> None:
        """Load the prosody→embedding projector.

        If no pretrained weights, use random initialization.
        """
        if self._projector is not None:
            return

        self._projector = ProsodyToEmbedding()

        # Try to load pretrained weights
        from pathlib import Path

        weights_path = Path(__file__).parent / "weights" / "style_projector.pt"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self._projector.load_state_dict(state_dict)
            logger.info("Loaded style projector weights from %s", weights_path)
        else:
            logger.warning("No pretrained style projector found, using random weights")

        self._projector = self._projector.to(self.device)
        self._projector.eval()

    @torch.inference_mode()
    def extract(
        self,
        waveform: np.ndarray | torch.Tensor,
        sample_rate: int = SAMPLE_RATE,
    ) -> np.ndarray:
        """Extract style embedding.

        Args:
            waveform: Audio waveform [T] or [1, T].
            sample_rate: Audio sample rate.

        Returns:
            Style embedding [128].
        """
        self._load_projector()

        # Convert to numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy().squeeze()

        # Resample if needed
        if sample_rate != SAMPLE_RATE:
            import librosa

            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE
            )

        # Extract prosody features
        prosody = self._extractor.extract(waveform)  # [32]

        # Project to embedding
        prosody_tensor = (
            torch.from_numpy(prosody).unsqueeze(0).to(self.device)
        )  # [1, 32]
        embedding = self._projector(prosody_tensor)  # [1, 128]

        return embedding.squeeze(0).cpu().numpy()

    def extract_from_file(self, path: str) -> np.ndarray:
        """Convenience: load audio and extract style embedding."""
        import soundfile as sf

        waveform, sr = sf.read(path)
        if waveform.ndim > 1:
            waveform = waveform[:, 0]  # Take first channel

        return self.extract(waveform, sr)


def compute_style_from_files(paths: list[str], device: str = "cpu") -> np.ndarray:
    """Compute average style embedding from multiple audio files.

    Args:
        paths: List of audio file paths.
        device: Device for computation.

    Returns:
        Average style embedding [128].
    """
    encoder = StyleEncoder(device=device)
    embeddings = []

    for path in paths:
        try:
            emb = encoder.extract_from_file(path)
            embeddings.append(emb)
        except Exception as e:
            logger.warning("Failed to extract style from %s: %s", path, e)

    if not embeddings:
        logger.warning("No valid embeddings, returning zeros")
        return np.zeros(D_STYLE, dtype=np.float32)

    return np.mean(embeddings, axis=0).astype(np.float32)
