"""TTS generation worker: synthesizes audio from text using UCLM."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tmrvc_gui.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class TTSWorker(BaseWorker):
    """Background worker for UCLM TTS audio generation.

    Config dict keys:
    - text: str
    - language: str
    - uclm_checkpoint: str
    - codec_checkpoint: str
    - speaker_file: str | None
    - speed: float
    - emotion: str
    - breathiness/tension/arousal/valence/roughness/voicing/energy: float
    """

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self.config = config
        self.audio = None
        self.duration_sec = 0.0

    def run(self) -> None:
        self._safe_run(self._generate)

    def _generate(self) -> None:
        import numpy as np
        import torch

        self.log_message.emit("Loading UCLM models...")
        t0 = time.perf_counter()

        from tmrvc_serve.uclm_engine import UCLMEngine
        from tmrvc_core.dialogue_types import StyleParams
        from tmrvc_data.g2p import text_to_phonemes

        engine = UCLMEngine(
            uclm_checkpoint=self.config.get("uclm_checkpoint"),
            codec_checkpoint=self.config.get("codec_checkpoint"),
            device="cpu", # GUI default to CPU for stability
        )
        engine.load_models()

        load_time = time.perf_counter() - t0
        self.log_message.emit(f"Models loaded in {load_time:.1f}s")

        if self.is_cancelled:
            self.finished.emit(False, "Cancelled")
            return

        # Build 8-dim style
        style = StyleParams(
            emotion=self.config.get("emotion", "neutral"),
            breathiness=self.config.get("breathiness", 0.0),
            tension=self.config.get("tension", 0.0),
            arousal=self.config.get("arousal", 0.0),
            valence=self.config.get("valence", 0.0),
            roughness=self.config.get("roughness", 0.0),
            voicing=self.config.get("voicing", 1.0),
            energy=self.config.get("energy", 0.0),
            speech_rate=self.config.get("speed", 1.0),
        )

        # G2P
        self.log_message.emit("Converting text to phonemes...")
        phoneme_ids = text_to_phonemes(
            self.config["text"], 
            language=self.config.get("language", "ja")
        )
        phonemes_t = torch.tensor(phoneme_ids).long().unsqueeze(0)

        # Load speaker embedding
        spk_file = self.config.get("speaker_file")
        if spk_file and Path(spk_file).exists():
            # Support .tmrvc_speaker (contains meta) and .npy
            if spk_file.endswith(".tmrvc_speaker"):
                from tmrvc_export.speaker_file import read_speaker_file
                speaker = read_speaker_file(Path(spk_file))
                spk_t = torch.from_numpy(speaker.spk_embed).float().unsqueeze(0)
            else:
                spk_t = torch.from_numpy(np.load(spk_file)).float().unsqueeze(0)
        else:
            from tmrvc_core.constants import D_SPEAKER
            spk_t = torch.zeros(1, D_SPEAKER)
            self.log_message.emit("Warning: No speaker file, using zero embedding")

        self.log_message.emit("Synthesizing (UCLM)...")
        t1 = time.perf_counter()

        audio_t, metrics = engine.tts(
            phonemes=phonemes_t,
            speaker_embed=spk_t,
            style=style,
        )

        synth_time = time.perf_counter() - t1
        self.audio = audio_t.cpu().numpy()
        self.duration_sec = metrics.output_duration_ms / 1000

        self.log_message.emit(
            f"Done: {self.duration_sec:.2f}s audio in {synth_time:.2f}s "
            f"(RTF={synth_time / max(self.duration_sec, 0.01):.2f}x)"
        )
        self.finished.emit(True, f"Generated {self.duration_sec:.2f}s audio")
