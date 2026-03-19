"""Stage 6: Speaker embedding extraction (ECAPA-TDNN 192-dim).

Extracts a fixed-dimensional speaker embedding per utterance using
ECAPA-TDNN (SpeechBrain or wespeaker).  Falls back to a simpler
embedding when the model is unavailable.
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

SPEAKER_EMBED_DIM = 192


class SpeakerEmbedStage:
    """ECAPA-TDNN 192-dim speaker embedding extraction.

    Lazy-loads the model on first use to avoid startup overhead.
    Supports SpeechBrain and wespeaker backends.
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()
        self._model = None
        self._backend: Optional[str] = None

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Extract speaker embeddings for all non-rejected utterances."""
        for utt in utterances:
            if utt.is_rejected:
                utt.stage_completed = BootstrapStage.SPEAKER_EMBEDDING
                continue

            try:
                audio, sr = self._load_segment_audio(utt)
                embedding = self._extract_embedding(audio, sr)
                utt.speaker_embed = embedding
            except Exception as exc:
                logger.warning(
                    "Speaker embed failed for %s: %s", utt.utterance_id, exc,
                )
                utt.speaker_embed = np.zeros(SPEAKER_EMBED_DIM, dtype=np.float32)
                utt.warnings.append(f"speaker_embed_error:{exc}")

            utt.stage_completed = BootstrapStage.SPEAKER_EMBEDDING

        logger.info("SpeakerEmbed: processed %d utterances", len(utterances))
        return utterances

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    def _extract_embedding(
        self, audio: np.ndarray, sr: int,
    ) -> np.ndarray:
        """Extract speaker embedding using the best available backend."""
        # Try SpeechBrain ECAPA-TDNN
        try:
            return self._extract_speechbrain(audio, sr)
        except (ImportError, Exception) as exc:
            logger.debug("SpeechBrain unavailable: %s", exc)

        # Try wespeaker
        try:
            return self._extract_wespeaker(audio, sr)
        except (ImportError, Exception) as exc:
            logger.debug("wespeaker unavailable: %s", exc)

        # Fallback to MFCC-based pseudo-embedding
        logger.info(
            "No speaker embedding model available, using MFCC fallback. "
            "Install speechbrain for better results: pip install speechbrain"
        )
        return self._extract_mfcc_fallback(audio, sr)

    def _extract_speechbrain(
        self, audio: np.ndarray, sr: int,
    ) -> np.ndarray:
        """Extract embedding using SpeechBrain ECAPA-TDNN."""
        import torch

        if self._model is None or self._backend != "speechbrain":
            from speechbrain.inference.speaker import EncoderClassifier

            self._model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                run_opts={"device": self.config.device},
            )
            self._backend = "speechbrain"

        # SpeechBrain expects 16 kHz
        if sr != 16000:
            audio = self._resample(audio, sr, 16000)

        wav_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        embeddings = self._model.encode_batch(wav_tensor)
        embedding = embeddings.squeeze().cpu().numpy()

        # ECAPA-TDNN outputs 192-dim
        if embedding.shape[0] != SPEAKER_EMBED_DIM:
            # Resize if needed (shouldn't happen with ECAPA-TDNN)
            if embedding.shape[0] > SPEAKER_EMBED_DIM:
                embedding = embedding[:SPEAKER_EMBED_DIM]
            else:
                padded = np.zeros(SPEAKER_EMBED_DIM, dtype=np.float32)
                padded[:embedding.shape[0]] = embedding
                embedding = padded

        return embedding.astype(np.float32)

    def _extract_wespeaker(
        self, audio: np.ndarray, sr: int,
    ) -> np.ndarray:
        """Extract embedding using wespeaker."""
        import torch

        if self._model is None or self._backend != "wespeaker":
            import wespeaker

            self._model = wespeaker.load_model("english")
            self._backend = "wespeaker"

        if sr != 16000:
            audio = self._resample(audio, sr, 16000)

        embedding = self._model.extract_embedding(audio)
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        if hasattr(embedding, "numpy"):
            embedding = embedding.numpy()

        embedding = embedding.flatten().astype(np.float32)

        if len(embedding) != SPEAKER_EMBED_DIM:
            result = np.zeros(SPEAKER_EMBED_DIM, dtype=np.float32)
            n = min(len(embedding), SPEAKER_EMBED_DIM)
            result[:n] = embedding[:n]
            return result

        return embedding

    @staticmethod
    def _extract_mfcc_fallback(
        audio: np.ndarray, sr: int,
    ) -> np.ndarray:
        """MFCC-based pseudo-embedding fallback (192-dim)."""
        try:
            import librosa

            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, hop_length=512)
            mean_mfcc = np.mean(mfcc, axis=1)   # [40]
            std_mfcc = np.std(mfcc, axis=1)      # [40]
            delta = np.diff(mfcc, axis=1)
            mean_delta = np.mean(delta, axis=1) if delta.shape[1] > 0 else np.zeros(40)
            std_delta = np.std(delta, axis=1) if delta.shape[1] > 0 else np.zeros(40)
            delta2 = np.diff(delta, axis=1) if delta.shape[1] > 1 else np.zeros_like(delta)
            mean_delta2 = np.mean(delta2, axis=1) if delta2.shape[1] > 0 else np.zeros(40)

            features = np.concatenate([
                mean_mfcc, std_mfcc, mean_delta, std_delta, mean_delta2[:32],
            ])  # 40*4 + 32 = 192
        except ImportError:
            # Ultra-fallback
            n_fft = min(2048, len(audio))
            spectrum = np.abs(np.fft.rfft(audio[:n_fft]))
            indices = np.linspace(0, len(spectrum) - 1, SPEAKER_EMBED_DIM).astype(int)
            features = spectrum[indices]

        features = features[:SPEAKER_EMBED_DIM].astype(np.float32)
        if len(features) < SPEAKER_EMBED_DIM:
            padded = np.zeros(SPEAKER_EMBED_DIM, dtype=np.float32)
            padded[:len(features)] = features
            features = padded

        # L2 normalize
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features /= norm

        return features

    # ------------------------------------------------------------------
    # Audio helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_segment_audio(utt: BootstrapUtterance) -> tuple[np.ndarray, int]:
        """Load the audio segment for an utterance."""
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
