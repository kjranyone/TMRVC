"""Script batch generation worker using UCLM."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from tmrvc_gui.workers.base_worker import BaseWorker

logger = logging.getLogger(__name__)


class ScriptWorker(BaseWorker):
    """Background worker for batch TTS generation using UCLM.

    Config dict keys:
    - script_yaml: str
    - output_dir: str
    - uclm_checkpoint: str
    - codec_checkpoint: str
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
        from tmrvc_data.g2p import text_to_phonemes
        from tmrvc_data.script_parser import load_script_from_string
        from tmrvc_serve.uclm_engine import UCLMEngine

        # Parse script
        self.log_message.emit("Parsing script...")
        script = load_script_from_string(self.config["script_yaml"])

        if not script.entries:
            self.finished.emit(False, "No dialogue entries in script.")
            return

        output_dir = Path(self.config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load engine
        self.log_message.emit("Loading UCLM models...")
        t0 = time.perf_counter()

        engine = UCLMEngine(
            uclm_checkpoint=self.config.get("uclm_checkpoint"),
            codec_checkpoint=self.config.get("codec_checkpoint"),
            device="cpu",
        )
        engine.load_models()

        load_time = time.perf_counter() - t0
        self.log_message.emit(f"Models loaded in {load_time:.1f}s")

        # Load speaker embeddings
        speaker_embeds: dict[str, torch.Tensor] = {}
        for char_id, char_profile in script.characters.items():
            if char_profile.speaker_file and char_profile.speaker_file.exists():
                # Support .tmrvc_speaker and .npy
                if str(char_profile.speaker_file).endswith(".tmrvc_speaker"):
                    from tmrvc_export.speaker_file import read_speaker_file
                    speaker = read_speaker_file(char_profile.speaker_file)
                    speaker_embeds[char_id] = torch.from_numpy(speaker.spk_embed).float().unsqueeze(0)
                else:
                    embed = np.load(str(char_profile.speaker_file))
                    speaker_embeds[char_id] = torch.from_numpy(embed).float().unsqueeze(0)
            else:
                speaker_embeds[char_id] = torch.zeros(1, 192)

        total = len(script.entries)
        total_duration_sec = 0.0
        t1 = time.perf_counter()

        for i, entry in enumerate(script.entries):
            if self.is_cancelled:
                self.finished.emit(False, f"Cancelled after {i}/{total} entries")
                return

            char = script.characters.get(entry.speaker)
            language = char.language if char else "ja"
            spk_t = speaker_embeds.get(entry.speaker, torch.zeros(1, 192))

            # Determine style
            style = entry.style_override
            if style is None:
                style = char.default_style if char and char.default_style else StyleParams.neutral()

            self.log_message.emit(f"[{i + 1}/{total}] {entry.speaker}: {entry.text[:40]}")

            # G2P
            phoneme_ids = text_to_phonemes(entry.text, language=language)
            phonemes_t = torch.tensor(phoneme_ids).long().unsqueeze(0)

            # Synthesis
            audio_t, metrics = engine.tts(
                phonemes=phonemes_t,
                speaker_embed=spk_t,
                style=style,
            )
            
            audio = audio_t.cpu().numpy()
            total_duration_sec += metrics.output_duration_ms / 1000

            filename = f"{i + 1:04d}_{entry.speaker}.wav"
            sf.write(str(output_dir / filename), audio, SAMPLE_RATE)

            self.progress.emit(i + 1, total)

        elapsed = time.perf_counter() - t1
        self.finished.emit(
            True,
            f"Generated {total} files ({total_duration_sec:.1f}s audio) in {elapsed:.1f}s",
        )
