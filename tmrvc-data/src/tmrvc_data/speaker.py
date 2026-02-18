"""Speaker encoder: ECAPA-TDNN → 192-d L2-normalised embedding."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

from tmrvc_core.constants import D_SPEAKER, SAMPLE_RATE

logger = logging.getLogger(__name__)


class SpeakerEncoder:
    """Extract speaker embeddings using SpeechBrain's ECAPA-TDNN.

    The model outputs 192-dim embeddings, L2-normalised.
    """

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        from speechbrain.inference.speaker import EncoderClassifier

        self._model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": str(self.device)},
        )
        logger.info("ECAPA-TDNN speaker encoder loaded on %s", self.device)

    @torch.inference_mode()
    def extract(
        self,
        waveform: torch.Tensor,
        sample_rate: int = SAMPLE_RATE,
    ) -> torch.Tensor:
        """Extract L2-normalised speaker embedding.

        Args:
            waveform: ``[1, T]`` audio.
            sample_rate: Audio sample rate.

        Returns:
            ``[192]`` speaker embedding (L2-normalised).
        """
        self._load_model()

        # ECAPA-TDNN expects 16 kHz
        if sample_rate != 16000:
            import torchaudio.functional as AF

            wav_16k = AF.resample(waveform, sample_rate, 16000)
        else:
            wav_16k = waveform

        wav_16k = wav_16k.to(self.device)

        # SpeechBrain expects [B, T]
        if wav_16k.dim() == 1:
            wav_16k = wav_16k.unsqueeze(0)

        embedding = self._model.encode_batch(wav_16k)  # [B, 1, D]
        embedding = embedding.squeeze()  # [D]

        # Truncate or pad to D_SPEAKER if necessary
        if embedding.shape[-1] != D_SPEAKER:
            logger.warning(
                "Speaker embedding dim %d != expected %d, projecting",
                embedding.shape[-1],
                D_SPEAKER,
            )
            embedding = embedding[..., :D_SPEAKER]

        # L2 normalise
        embedding = F.normalize(embedding, p=2, dim=-1)

        return embedding.cpu()

    def extract_from_file(self, path: str | object) -> torch.Tensor:
        """Convenience: load audio and extract speaker embedding."""
        from tmrvc_data.preprocessing import load_and_resample

        waveform, sr = load_and_resample(str(path))
        return self.extract(waveform, sr)
