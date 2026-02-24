"""TTS inference engine wrapping the full pipeline.

Loads TTS models (TextEncoder, DurationPredictor, F0Predictor, ContentSynthesizer)
and VC backend (Converter, Vocoder) to produce audio from text.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.constants import (
    D_CONTENT,
    D_SPEAKER,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_LENGTH,
)
from tmrvc_core.dialogue_types import CharacterProfile, DialogueTurn, StyleParams

logger = logging.getLogger(__name__)


class TTSEngine:
    """End-to-end TTS inference engine.

    Loads pre-trained TTS front-end and VC back-end models.
    Produces audio from text + speaker embedding + style params.

    Args:
        tts_checkpoint: Path to TTS checkpoint (.pt).
        vc_checkpoint: Path to VC/distill checkpoint (.pt) for Converter+Vocoder.
        device: Torch device string.
    """

    def __init__(
        self,
        tts_checkpoint: Path | str | None = None,
        vc_checkpoint: Path | str | None = None,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self._models_loaded = False
        self._tts_checkpoint = tts_checkpoint
        self._vc_checkpoint = vc_checkpoint

        # Models (lazily loaded)
        self._text_encoder: torch.nn.Module | None = None
        self._duration_predictor: torch.nn.Module | None = None
        self._f0_predictor: torch.nn.Module | None = None
        self._content_synthesizer: torch.nn.Module | None = None
        self._converter: torch.nn.Module | None = None
        self._vocoder: torch.nn.Module | None = None

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    def load_models(self) -> None:
        """Load all models from checkpoints."""
        from tmrvc_train.models.text_encoder import TextEncoder
        from tmrvc_train.models.duration_predictor import DurationPredictor
        from tmrvc_train.models.f0_predictor import F0Predictor
        from tmrvc_train.models.content_synthesizer import ContentSynthesizer

        self._text_encoder = TextEncoder().to(self.device).eval()
        self._duration_predictor = DurationPredictor().to(self.device).eval()
        self._f0_predictor = F0Predictor().to(self.device).eval()
        self._content_synthesizer = ContentSynthesizer().to(self.device).eval()

        if self._tts_checkpoint:
            ckpt = torch.load(self._tts_checkpoint, map_location=self.device, weights_only=False)
            state = ckpt if isinstance(ckpt, dict) and "text_encoder" not in ckpt else ckpt
            if "text_encoder" in state:
                self._text_encoder.load_state_dict(state["text_encoder"])
                self._duration_predictor.load_state_dict(state["duration_predictor"])
                self._f0_predictor.load_state_dict(state["f0_predictor"])
                self._content_synthesizer.load_state_dict(state["content_synthesizer"])
                logger.info("Loaded TTS models from %s", self._tts_checkpoint)

        # VC backend (Converter + Vocoder) — load if checkpoint provided
        if self._vc_checkpoint:
            self._load_vc_backend()

        self._models_loaded = True
        logger.info("TTS engine ready on %s", self.device)

    def _load_vc_backend(self) -> None:
        """Load Converter and Vocoder from VC checkpoint."""
        from tmrvc_train.models.converter import ConverterStudent
        from tmrvc_train.models.vocoder import VocoderStudent

        self._converter = ConverterStudent().to(self.device).eval()
        self._vocoder = VocoderStudent().to(self.device).eval()

        ckpt = torch.load(self._vc_checkpoint, map_location=self.device, weights_only=False)
        if "converter" in ckpt:
            self._converter.load_state_dict(ckpt["converter"])
        if "vocoder" in ckpt:
            self._vocoder.load_state_dict(ckpt["vocoder"])
        logger.info("Loaded VC backend from %s", self._vc_checkpoint)

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        language: str,
        spk_embed: torch.Tensor,
        style: StyleParams | None = None,
        speed: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Synthesize audio from text.

        Args:
            text: Input text.
            language: Language code ('ja' or 'en').
            spk_embed: ``[192]`` speaker embedding.
            style: Style parameters (None = neutral).
            speed: Speed factor (>1 = faster, <1 = slower).

        Returns:
            Tuple of (audio_samples [N], duration_sec).
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        from tmrvc_data.g2p import text_to_phonemes

        # G2P
        g2p_result = text_to_phonemes(text, language=language)
        phoneme_ids = g2p_result.phoneme_ids.unsqueeze(0).to(self.device)  # [1, L]

        # Language ID
        lang_id = 0 if language == "ja" else 1
        language_ids = torch.tensor([lang_id], dtype=torch.long, device=self.device)

        # Text encoding
        text_features = self._text_encoder(phoneme_ids, language_ids)  # [1, 256, L]

        # Style vector
        if style is None:
            style = StyleParams.neutral()
        style_vec = torch.tensor(
            style.to_vector(), dtype=torch.float32, device=self.device,
        ).unsqueeze(0)  # [1, 32]

        # Duration prediction
        durations = self._duration_predictor(text_features, style_vec)  # [1, L]
        durations = durations / speed
        durations = torch.round(durations).long().clamp(min=1)

        # Length regulate (expand phoneme features to frame-level)
        from tmrvc_train.models.f0_predictor import length_regulate
        expanded = length_regulate(text_features, durations.float())  # [1, 256, T]

        # F0 prediction
        f0, voiced_prob = self._f0_predictor(expanded, style_vec)  # [1, 1, T]

        # Content synthesis
        content = self._content_synthesizer(expanded)  # [1, 256, T]

        T = content.shape[-1]
        duration_sec = T * HOP_LENGTH / SAMPLE_RATE

        # If VC backend loaded, generate audio
        if self._converter is not None and self._vocoder is not None:
            spk = spk_embed.to(self.device).unsqueeze(0)  # [1, 192]
            # Zero acoustic params for now (TTS mode)
            from tmrvc_train.models.style_encoder import StyleEncoder
            acoustic_params = torch.zeros(1, 32, device=self.device)
            style_params = StyleEncoder.combine_style_params(acoustic_params, style_vec)

            # Converter: content + spk_embed + style_params → STFT features
            pred_features, _ = self._converter(content, spk, style_params)  # [1, 513, T]

            # Vocoder: STFT features → magnitude + phase
            stft_mag, stft_phase, _ = self._vocoder(pred_features)  # [1, 513, T]

            # iSTFT reconstruction
            stft_complex = stft_mag * torch.exp(1j * stft_phase)  # [1, 513, T]
            window = torch.hann_window(WINDOW_LENGTH, device=self.device)
            audio = torch.istft(
                stft_complex,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WINDOW_LENGTH,
                window=window,
                center=False,
            )  # [1, N_samples]
            audio_np = audio.squeeze(0).cpu().numpy()
        else:
            # No VC backend — return silence placeholder
            n_samples = T * HOP_LENGTH
            audio_np = np.zeros(n_samples, dtype=np.float32)
            logger.warning("No VC backend loaded; returning silence.")

        return audio_np, duration_sec
