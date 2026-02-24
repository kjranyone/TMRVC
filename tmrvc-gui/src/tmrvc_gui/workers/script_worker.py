"""Script batch generation worker: generates audio for all dialogue entries."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tmrvc_gui.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class ScriptWorker(BaseWorker):
    """Background worker for batch TTS generation from a YAML script.

    Config dict keys:
    - script_yaml: str — YAML script content
    - output_dir: str — Output directory path
    - tts_checkpoint: str | None — TTS checkpoint path
    - vc_checkpoint: str | None — VC checkpoint path
    """

    def __init__(self, config: dict, parent=None) -> None:
        super().__init__(parent)
        self.config = config

    def run(self) -> None:
        self._safe_run(self._generate)

    def _generate(self) -> None:
        import numpy as np
        import soundfile as sf
        import torch

        from tmrvc_core.constants import SAMPLE_RATE
        from tmrvc_core.dialogue_types import StyleParams
        from tmrvc_data.script_parser import load_script_from_string
        from tmrvc_serve.tts_engine import TTSEngine

        # Parse script
        self.log_message.emit("Parsing script...")
        script = load_script_from_string(self.config["script_yaml"])

        if not script.entries:
            self.finished.emit(False, "No dialogue entries in script.")
            return

        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load engine
        self.log_message.emit("Loading TTS models...")
        t0 = time.perf_counter()

        engine = TTSEngine(
            tts_checkpoint=self.config.get("tts_checkpoint"),
            vc_checkpoint=self.config.get("vc_checkpoint"),
            device="cpu",
        )
        engine.load_models()

        load_time = time.perf_counter() - t0
        self.log_message.emit(f"Models loaded in {load_time:.1f}s")

        # Load speaker embeddings
        speaker_embeds: dict[str, torch.Tensor] = {}
        for char_id, char_profile in script.characters.items():
            if char_profile.speaker_file and char_profile.speaker_file.exists():
                embed = np.load(str(char_profile.speaker_file))
                speaker_embeds[char_id] = torch.from_numpy(embed).float()
            else:
                speaker_embeds[char_id] = torch.zeros(192)

        total = len(script.entries)
        total_duration = 0.0
        t1 = time.perf_counter()

        for i, entry in enumerate(script.entries):
            if self.is_cancelled:
                self.finished.emit(False, f"Cancelled after {i}/{total} entries")
                return

            style = entry.style_override
            if style is None:
                char = script.characters.get(entry.speaker)
                style = char.default_style if char and char.default_style else StyleParams.neutral()

            char = script.characters.get(entry.speaker)
            language = char.language if char else "ja"
            spk_embed = speaker_embeds.get(entry.speaker, torch.zeros(192))

            self.log_message.emit(f"[{i + 1}/{total}] {entry.speaker}: {entry.text[:40]}")

            audio, duration_sec = engine.synthesize(
                text=entry.text,
                language=language,
                spk_embed=spk_embed,
                style=style,
            )
            total_duration += duration_sec

            filename = f"{i + 1:04d}_{entry.speaker}.wav"
            sf.write(str(output_dir / filename), audio, SAMPLE_RATE)

            self.progress.emit(i + 1, total)

        elapsed = time.perf_counter() - t1
        self.finished.emit(
            True,
            f"Generated {total} files ({total_duration:.1f}s audio) in {elapsed:.1f}s",
        )
