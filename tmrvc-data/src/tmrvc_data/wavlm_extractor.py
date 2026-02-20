"""WavLM-large feature extractor for content representation.

Extracts features from WavLM-large layer 7 (1024-dim) for use as
content encoder teacher during distillation.
"""

from __future__ import annotations

import logging
from typing import Final

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

WAVLM_LARGE_DIM: Final[int] = 1024
WAVLM_LARGE_LAYERS: Final[int] = 24
DEFAULT_WAVLM_LAYER: Final[int] = 7


class WavLMFeatureExtractor(nn.Module):
    """WavLM-large feature extractor for content representation.

    Uses a pre-trained WavLM-large model from HuggingFace transformers.
    Extracts features from a specified intermediate layer (default: layer 7).

    The extracted features are projected to the target dimension for
    use as teacher features in content encoder distillation.

    Args:
        layer: Which intermediate layer to extract (0-24). Default: 7.
        d_output: Output dimension after projection. Default: 1024 (no projection).
        freeze: Whether to freeze the WavLM weights. Default: True.
        model_name: HuggingFace model identifier. Default: "microsoft/wavlm-large".
    """

    def __init__(
        self,
        layer: int = DEFAULT_WAVLM_LAYER,
        d_output: int = WAVLM_LARGE_DIM,
        freeze: bool = True,
        model_name: str = "microsoft/wavlm-large",
    ) -> None:
        super().__init__()
        self.layer = layer
        self.d_output = d_output
        self.model_name = model_name

        try:
            from transformers import WavLMModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for WavLMFeatureExtractor. "
                "Install with: pip install transformers"
            ) from e

        logger.info(f"Loading WavLM model: {model_name}")
        self.wavlm = WavLMModel.from_pretrained(model_name)

        if freeze:
            for param in self.wavlm.parameters():
                param.requires_grad = False
            self.wavlm.eval()

        self.projection: nn.Linear | None = None
        if d_output != WAVLM_LARGE_DIM:
            self.projection = nn.Linear(WAVLM_LARGE_DIM, d_output)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    @property
    def sample_rate(self) -> int:
        """WavLM expected sample rate (16kHz)."""
        return 16000

    @property
    def hop_length(self) -> int:
        """WavLM hop length in samples (320 = 20ms at 16kHz)."""
        return 320

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract WavLM features from audio.

        Args:
            audio: Audio waveform [B, T_audio] at 16kHz.

        Returns:
            Features [B, d_output, T_feat] where T_feat ≈ T_audio / 320.
        """
        if self.wavlm.training:
            self.wavlm.eval()

        with torch.no_grad():
            outputs = self.wavlm(
                audio,
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = outputs.hidden_states
        features = hidden_states[self.layer + 1]  # +1 because index 0 is embedding

        if self.projection is not None:
            features = self.projection(features)

        features = features.transpose(1, 2)

        return features

    @torch.no_grad()
    def extract_for_distillation(
        self,
        audio: torch.Tensor,
        audio_24k: torch.Tensor,
    ) -> torch.Tensor:
        """Extract and align features for distillation.

        WavLM operates at 16kHz with 20ms hop (320 samples).
        Our mel features are at 24kHz with 10ms hop (240 samples).

        This method extracts WavLM features and resamples them to match
        the mel frame rate.

        Args:
            audio: Audio at 16kHz [B, T_16k] for WavLM.
            audio_24k: Audio at 24kHz [B, T_24k] for reference.

        Returns:
            Aligned features [B, d_output, T_mel] matching mel frames.
        """
        features = self.forward(audio)

        T_mel = audio_24k.shape[-1] // 240
        T_feat = features.shape[-1]

        if T_mel == 0:
            raise ValueError(
                f"Audio too short for feature extraction: "
                f"{audio_24k.shape[-1]} samples at 24kHz = {audio_24k.shape[-1] / 24000 * 1000:.1f}ms "
                f"(need at least 240 samples / 10ms)"
            )

        if T_feat == 0:
            # Audio too short for WavLM (< 320 samples at 16kHz).
            # Return zeros aligned to mel frame count.
            return torch.zeros(
                features.shape[0], features.shape[1], T_mel,
                device=features.device, dtype=features.dtype,
            )

        if T_feat != T_mel:
            features = torch.nn.functional.interpolate(
                features,
                size=T_mel,
                mode="linear",
                align_corners=False,
            )

        return features


class ContentVecFeatureExtractor(nn.Module):
    """ContentVec feature extractor for backward compatibility.

    Uses ContentVec for Phase 0 training. WavLM is recommended for Phase 1+.

    Args:
        d_output: Output dimension after projection. Default: 768 (no projection).
        freeze: Whether to freeze the model. Default: True.
        checkpoint_path: Path to ContentVec checkpoint. If None, uses default.
    """

    def __init__(
        self,
        d_output: int = 768,
        freeze: bool = True,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        self.d_output = d_output

        try:
            from transformers import HubertModel
        except ImportError as e:
            raise ImportError(
                "transformers is required for ContentVecFeatureExtractor. "
                "Install with: pip install transformers"
            ) from e

        model_name = checkpoint_path or "Tkag/amazon-contentvec"
        logger.info(f"Loading ContentVec model: {model_name}")

        try:
            self.model = HubertModel.from_pretrained(model_name)
        except Exception:
            logger.warning(
                f"Failed to load {model_name}, falling back to hubert-base-ls960"
            )
            self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

        self.projection: nn.Linear | None = None
        if d_output != 768:
            self.projection = nn.Linear(768, d_output)
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def hop_length(self) -> int:
        return 320

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract ContentVec features from audio.

        Args:
            audio: Audio waveform [B, T_audio] at 16kHz.

        Returns:
            Features [B, d_output, T_feat].
        """
        if self.model.training:
            self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                audio,
                output_hidden_states=True,
                return_dict=True,
            )

        features = outputs.last_hidden_state

        if self.projection is not None:
            features = self.projection(features)

        features = features.transpose(1, 2)

        return features


def get_content_teacher(
    teacher_type: str = "wavlm",
    d_output: int = 1024,
    freeze: bool = True,
    **kwargs,
) -> nn.Module:
    """Factory function to create content feature extractor.

    Args:
        teacher_type: "wavlm" or "contentvec".
        d_output: Output dimension.
        freeze: Whether to freeze weights.
        **kwargs: Additional arguments passed to the extractor.

    Returns:
        Feature extractor module.
    """
    if teacher_type == "wavlm":
        return WavLMFeatureExtractor(d_output=d_output, freeze=freeze, **kwargs)
    elif teacher_type == "contentvec":
        return ContentVecFeatureExtractor(
            d_output=min(d_output, 768), freeze=freeze, **kwargs
        )
    else:
        raise ValueError(f"Unknown teacher_type: {teacher_type}")
