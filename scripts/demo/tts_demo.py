#!/usr/bin/env python3
"""UCLM v2 TTS end-to-end demo.

Verifies the unified dual-stream pipeline: 
text → G2P → UCLMEngine (TTS Mode) → Dual Stream Tokens → Codec Decoder → WAV
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="UCLM v2 TTS Demo")
    parser.add_argument("--text", default="こんにちは、世界！テスト音声です。")
    parser.add_argument("--language", default="ja", choices=["ja", "en", "zh", "ko"])
    parser.add_argument("--output", default="tts_demo_uclm.wav", help="Output WAV path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--uclm-checkpoint", type=Path, default="checkpoints/uclm/uclm_latest.pt")
    parser.add_argument("--codec-checkpoint", type=Path, default="checkpoints/codec/codec_latest.pt")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("tts_demo")

    from tmrvc_core.constants import D_SPEAKER, SAMPLE_RATE
    from tmrvc_core.dialogue_types import StyleParams
    from tmrvc_core.text_utils import text_to_phonemes
    from tmrvc_serve.uclm_engine import UCLMEngine

    logger.info("=== UCLM v2 TTS Demo ===")
    logger.info("Text: %s", args.text)
    logger.info("Emotion: %s", args.emotion)

    # 1. Initialize Engine
    t0 = time.perf_counter()
    if not args.uclm_checkpoint.exists() or not args.codec_checkpoint.exists():
        logger.error("Checkpoints not found. Run training or provide valid paths.")
        return

    engine = UCLMEngine(
        uclm_checkpoint=args.uclm_checkpoint,
        codec_checkpoint=args.codec_checkpoint,
        device=args.device,
    )
    engine.load_models()
    logger.info("Engine loaded in %.0fms", (time.perf_counter() - t0) * 1000)

    # 2. Prepare Inputs
    # Phonemes
    phoneme_ids = text_to_phonemes(args.text, language=args.language)
    phonemes_t = torch.tensor(phoneme_ids).long().unsqueeze(0)
    
    # Dummy speaker embedding
    spk_embed = torch.randn(1, D_SPEAKER)
    
    # Style
    style = StyleParams(emotion=args.emotion)

    # 3. Synthesis
    logger.info("--- Generating Audio ---")
    t0 = time.perf_counter()
    audio_t, metrics = engine.tts(
        phonemes=phonemes_t,
        speaker_embed=spk_embed,
        style=style
    )
    total_ms = (time.perf_counter() - t0) * 1000
    
    audio = audio_t.cpu().numpy()
    duration_sec = len(audio) / SAMPLE_RATE
    logger.info("Generated %.2fs audio in %.0fms (RTF=%.2f)", duration_sec, total_ms, metrics.rtf)

    # 4. Save WAV
    import soundfile as sf
    sf.write(args.output, audio, SAMPLE_RATE)
    logger.info("Saved to %s", args.output)

    # Audio stats
    peak = np.max(np.abs(audio))
    logger.info("Audio stats: peak=%.4f", peak)


if __name__ == "__main__":
    main()
