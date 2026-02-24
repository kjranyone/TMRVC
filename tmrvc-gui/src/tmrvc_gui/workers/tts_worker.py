"""TTS generation worker: synthesizes audio from text in a background thread."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tmrvc_gui.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class TTSWorker(BaseWorker):
    """Background worker for TTS audio generation.

    Config dict keys:
    - text: str — Input text
    - language: str — 'ja' or 'en'
    - tts_checkpoint: str — Path to TTS checkpoint
    - vc_checkpoint: str | None — Path to VC checkpoint
    - speaker_file: str | None — Path to .tmrvc_speaker
    - speed: float — Speed factor
    - emotion: str — Emotion category
    - valence/arousal/energy/pitch_range: float — Style sliders
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

        self.log_message.emit("Loading TTS models...")
        t0 = time.perf_counter()

        from tmrvc_serve.tts_engine import TTSEngine
        from tmrvc_core.dialogue_types import StyleParams

        engine = TTSEngine(
            tts_checkpoint=self.config.get("tts_checkpoint"),
            vc_checkpoint=self.config.get("vc_checkpoint"),
            device="cpu",
        )
        engine.load_models()

        load_time = time.perf_counter() - t0
        self.log_message.emit(f"Models loaded in {load_time:.1f}s")

        if self.is_cancelled:
            self.finished.emit(False, "Cancelled")
            return

        # Build style
        style = StyleParams(
            emotion=self.config.get("emotion", "neutral"),
            valence=self.config.get("valence", 0.0),
            arousal=self.config.get("arousal", 0.0),
            energy=self.config.get("energy", 0.0),
            pitch_range=self.config.get("pitch_range", 0.0),
        )

        # Load speaker embedding
        spk_file = self.config.get("speaker_file")
        if spk_file and Path(spk_file).exists():
            spk_embed = torch.from_numpy(np.load(spk_file)).float()
        else:
            spk_embed = torch.zeros(192)
            self.log_message.emit("Warning: No speaker file, using zero embedding")

        self.log_message.emit("Synthesizing...")
        t1 = time.perf_counter()

        audio, duration_sec = engine.synthesize(
            text=self.config["text"],
            language=self.config.get("language", "ja"),
            spk_embed=spk_embed,
            style=style,
            speed=self.config.get("speed", 1.0),
        )

        synth_time = time.perf_counter() - t1
        self.audio = audio
        self.duration_sec = duration_sec

        self.log_message.emit(
            f"Done: {duration_sec:.2f}s audio in {synth_time:.2f}s "
            f"(RTF={synth_time / max(duration_sec, 0.01):.2f}x)"
        )
        self.finished.emit(True, f"Generated {duration_sec:.2f}s audio")
