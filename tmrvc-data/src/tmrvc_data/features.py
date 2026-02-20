"""Feature extractors: ContentVec (768-d), WavLM-large (1024-d), and F0 (RMVPE / torchcrepe)."""

from __future__ import annotations

import abc
import logging
from typing import Protocol

import torch
import torch.nn.functional as F

from tmrvc_core.constants import D_CONTENT_VEC, D_WAVLM_LARGE, HOP_LENGTH, SAMPLE_RATE

logger = logging.getLogger(__name__)


# ============================================================================
# Content feature extraction
# ============================================================================


class ContentExtractor(Protocol):
    """Protocol for content feature extractors."""

    @property
    def output_dim(self) -> int:
        """Output dimension of the content features."""
        ...

    def extract(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Extract content features.

        Args:
            waveform: ``[1, T]`` audio.
            sample_rate: Audio sample rate.

        Returns:
            ``[output_dim, T_frames]`` content features at 10 ms hop.
        """
        ...


class ContentVecExtractor:
    """Extract ContentVec features (768-d) and interpolate to 10 ms hop.

    ContentVec operates at 16 kHz with hop=320 (20 ms).
    We interpolate 2x to match the TMRVC 10 ms hop.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._model = None

    @property
    def output_dim(self) -> int:
        return D_CONTENT_VEC  # 768

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import HubertModel

        self._model = HubertModel.from_pretrained("lengyue233/content-vec-best")
        self._model.eval().to(self.device)
        logger.info("ContentVec model loaded on %s", self.device)

    @torch.inference_mode()
    def extract(
        self,
        waveform: torch.Tensor,
        sample_rate: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """Extract and interpolate ContentVec features.

        Args:
            waveform: ``[1, T]`` audio at any sample rate (will be resampled to 16 kHz).
            sample_rate: Input sample rate.

        Returns:
            ``[768, T_frames]`` features at 10 ms hop.
        """
        self._load_model()

        # Resample to 16 kHz (ContentVec native rate)
        if sample_rate != 16000:
            import torchaudio.functional as AF

            waveform = AF.resample(waveform, sample_rate, 16000)

        waveform = waveform.to(self.device)

        # ContentVec expects [B, T]
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            input_wav = waveform
        else:
            input_wav = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform

        outputs = self._model(input_wav, output_hidden_states=True)
        # Use last hidden state: [B, T_cv, 768]
        features = outputs.last_hidden_state.squeeze(0)  # [T_cv, 768]
        features = features.transpose(0, 1)  # [768, T_cv]

        # Interpolate 20 ms → 10 ms (2x upsample)
        features = F.interpolate(
            features.unsqueeze(0),
            scale_factor=2.0,
            mode="linear",
            align_corners=False,
        ).squeeze(0)  # [768, T_out]

        return features.cpu()


class WavLMExtractor:
    """Extract WavLM-large layer 7 features (1024-d) and interpolate to 10 ms hop.

    WavLM-large operates at 16 kHz with hop=320 (20 ms).
    We interpolate 2x to match the TMRVC 10 ms hop.
    """

    def __init__(self, device: str = "cpu", layer: int = 7) -> None:
        self.device = torch.device(device)
        self.layer = layer
        self._model = None

    @property
    def output_dim(self) -> int:
        return D_WAVLM_LARGE  # 1024

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from transformers import WavLMModel

        self._model = WavLMModel.from_pretrained("microsoft/wavlm-large")
        self._model.eval().to(self.device)
        logger.info(
            "WavLM-large model loaded on %s (layer %d)", self.device, self.layer
        )

    @torch.inference_mode()
    def extract(
        self,
        waveform: torch.Tensor,
        sample_rate: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """Extract and interpolate WavLM-large layer 7 features.

        Args:
            waveform: ``[1, T]`` audio at any sample rate (will be resampled to 16 kHz).
            sample_rate: Input sample rate.

        Returns:
            ``[1024, T_frames]`` features at 10 ms hop.
        """
        self._load_model()

        # Resample to 16 kHz (WavLM native rate)
        if sample_rate != 16000:
            import torchaudio.functional as AF

            waveform = AF.resample(waveform, sample_rate, 16000)

        waveform = waveform.to(self.device)

        # WavLM expects [B, T]
        if waveform.dim() == 2 and waveform.shape[0] == 1:
            input_wav = waveform
        else:
            input_wav = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform

        outputs = self._model(input_wav, output_hidden_states=True, return_dict=True)
        # hidden_states[0] is embedding, hidden_states[i] is output of layer i-1
        # So hidden_states[layer+1] is output of layer `layer`
        features = outputs.hidden_states[self.layer + 1]  # [B, T_wavlm, 1024]
        features = features.squeeze(0)  # [T_wavlm, 1024]
        features = features.transpose(0, 1)  # [1024, T_wavlm]

        # Interpolate 20 ms → 10 ms (2x upsample)
        features = F.interpolate(
            features.unsqueeze(0),
            scale_factor=2.0,
            mode="linear",
            align_corners=False,
        ).squeeze(0)  # [1024, T_out]

        return features.cpu()


def create_content_extractor(
    teacher_type: str = "contentvec",
    device: str = "cpu",
    **kwargs,
) -> ContentVecExtractor | WavLMExtractor:
    """Factory for content feature extractors.

    Args:
        teacher_type: ``"contentvec"`` (768d) or ``"wavlm"`` (1024d).
        device: Device for inference.
        **kwargs: Additional arguments (e.g., ``layer`` for WavLM).

    Returns:
        Content extractor instance.
    """
    if teacher_type == "contentvec":
        return ContentVecExtractor(device=device)
    if teacher_type == "wavlm":
        return WavLMExtractor(device=device, **kwargs)
    raise ValueError(f"Unknown content teacher type: {teacher_type!r}")


# ============================================================================
# F0 extraction
# ============================================================================


class F0Extractor(abc.ABC):
    """Abstract base for F0 extractors (pluggable design)."""

    @abc.abstractmethod
    def extract(
        self,
        waveform: torch.Tensor,
        sample_rate: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """Extract F0 contour.

        Args:
            waveform: ``[1, T]`` audio.
            sample_rate: Audio sample rate.

        Returns:
            ``[1, T_frames]`` log-F0 (log(f0+1), 0.0 for unvoiced).
        """
        ...


class TorchCrepeF0Extractor(F0Extractor):
    """F0 extraction using torchcrepe (fallback when RMVPE unavailable)."""

    def __init__(self, device: str = "cpu", model: str = "tiny") -> None:
        self.device = torch.device(device)
        self.model = model

    @torch.inference_mode()
    def extract(
        self,
        waveform: torch.Tensor,
        sample_rate: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """Extract F0 via torchcrepe.

        Returns:
            ``[1, T_frames]`` log-F0 at 10 ms hop.
        """
        import torchcrepe

        # torchcrepe expects 16 kHz
        if sample_rate != 16000:
            import torchaudio.functional as AF

            wav_16k = AF.resample(waveform, sample_rate, 16000)
        else:
            wav_16k = waveform

        wav_16k = wav_16k.to(self.device)

        # hop_length for 10 ms at 16 kHz = 160
        pitch = torchcrepe.predict(
            wav_16k,
            sample_rate=16000,
            hop_length=160,
            fmin=50.0,
            fmax=1100.0,
            model=self.model,
            batch_size=512,
            device=self.device,
            return_periodicity=False,
        )
        # pitch: [B, T_frames]
        if pitch.dim() == 1:
            pitch = pitch.unsqueeze(0)

        # Unvoiced masking: torchcrepe returns 0 for unvoiced already
        # log(f0 + 1), unvoiced → 0.0
        log_f0 = torch.where(
            pitch > 0,
            torch.log(pitch + 1.0),
            torch.zeros_like(pitch),
        )

        return log_f0.cpu()


class RMVPEF0Extractor(F0Extractor):
    """F0 extraction using RMVPE (preferred, higher quality)."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from rmvpe import RMVPE

            self._model = RMVPE(device=self.device)
            logger.info("RMVPE model loaded on %s", self.device)
        except ImportError:
            raise ImportError(
                "RMVPE not available. Install it or use TorchCrepeF0Extractor."
            )

    @torch.inference_mode()
    def extract(
        self,
        waveform: torch.Tensor,
        sample_rate: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """Extract F0 via RMVPE.

        Returns:
            ``[1, T_frames]`` log-F0 at 10 ms hop.
        """
        self._load_model()

        # RMVPE operates on 16 kHz
        if sample_rate != 16000:
            import torchaudio.functional as AF

            wav_16k = AF.resample(waveform, sample_rate, 16000)
        else:
            wav_16k = waveform

        audio_np = wav_16k.squeeze().numpy()
        f0 = self._model.infer_from_audio(audio_np, sample_rate=16000)
        f0 = torch.from_numpy(f0).float()

        # Interpolate to match 10 ms hop at 24 kHz
        expected_frames = waveform.shape[-1] // HOP_LENGTH
        if f0.shape[0] != expected_frames:
            f0 = F.interpolate(
                f0.unsqueeze(0).unsqueeze(0),
                size=expected_frames,
                mode="linear",
                align_corners=False,
            ).squeeze()

        # log(f0+1), unvoiced → 0.0
        log_f0 = torch.where(
            f0 > 0,
            torch.log(f0 + 1.0),
            torch.zeros_like(f0),
        )

        return log_f0.unsqueeze(0).cpu()  # [1, T_frames]


def create_f0_extractor(method: str = "torchcrepe", **kwargs) -> F0Extractor:
    """Factory for F0 extractors.

    Args:
        method: ``"rmvpe"`` or ``"torchcrepe"``.
    """
    if method == "rmvpe":
        return RMVPEF0Extractor(**kwargs)
    if method == "torchcrepe":
        return TorchCrepeF0Extractor(**kwargs)
    raise ValueError(f"Unknown F0 method: {method!r}")
