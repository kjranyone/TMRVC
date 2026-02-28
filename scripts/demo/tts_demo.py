#!/usr/bin/env python3
"""Minimal TTS end-to-end demo with random weights.

Verifies the full pipeline: text → tokens → TextEncoder → DurationPredictor
→ F0Predictor → ContentSynthesizer → Converter → Vocoder → iSTFT → WAV

Usage::

    # Basic demo (random weights, tokenizer frontend, CPU)
    python scripts/tts_demo.py

    # With options
    python scripts/tts_demo.py --text "Hello world!" --language en --device xpu

    # Stream mode (sentence-level)
    python scripts/tts_demo.py --stream --text "こんにちは。今日はいい天気ですね。"

    # With emotion
    python scripts/tts_demo.py --emotion happy --text "やったー！嬉しい！"
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

    parser = argparse.ArgumentParser(description="TTS end-to-end demo")
    parser.add_argument("--text", default="こんにちは、世界！テスト音声です。")
    parser.add_argument("--language", default="ja", choices=["ja", "en", "zh", "ko"])
    parser.add_argument("--output", default="tts_demo_output.wav", help="Output WAV path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tts-checkpoint", type=Path, default=None)
    parser.add_argument("--vc-checkpoint", type=Path, default=None)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--stream", action="store_true", help="Use streaming synthesis")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("tts_demo")

    from tmrvc_core.constants import D_SPEAKER, SAMPLE_RATE
    from tmrvc_core.dialogue_types import StyleParams
    from tmrvc_serve.tts_engine import TTSEngine

    logger.info("=== TTS Demo ===")
    logger.info("Text: %s", args.text)
    logger.info("Language: %s", args.language)
    logger.info("Device: %s", args.device)
    logger.info("Emotion: %s", args.emotion)

    # Initialize engine
    t0 = time.perf_counter()
    engine = TTSEngine(
        tts_checkpoint=args.tts_checkpoint,
        vc_checkpoint=args.vc_checkpoint,
        device=args.device,
        text_frontend="tokenizer",
    )
    engine.load_models()

    # If no VC checkpoint, inject random-weight VC backend for demo
    if args.vc_checkpoint is None and engine._converter is None:
        from tmrvc_core.constants import N_STYLE_PARAMS
        from tmrvc_train.models.converter import ConverterStudent
        from tmrvc_train.models.vocoder import VocoderStudent

        logger.info("No VC checkpoint — creating random-weight VC backend for demo")
        engine._converter = ConverterStudent(
            n_acoustic_params=N_STYLE_PARAMS,
        ).to(engine.device).eval()
        engine._vocoder = VocoderStudent().to(engine.device).eval()

    load_ms = (time.perf_counter() - t0) * 1000
    logger.info("Models loaded in %.0fms", load_ms)

    # Dummy speaker embedding
    spk_embed = torch.randn(D_SPEAKER)

    # Style
    style = StyleParams(emotion=args.emotion)

    if args.stream:
        # Streaming mode
        logger.info("--- Streaming synthesis ---")
        t0 = time.perf_counter()
        chunks = list(engine.synthesize_sentences(
            text=args.text,
            language=args.language,
            spk_embed=spk_embed,
            style=style,
            speed=args.speed,
        ))
        total_ms = (time.perf_counter() - t0) * 1000

        if chunks:
            audio = np.concatenate(chunks)
            duration_sec = len(audio) / SAMPLE_RATE
            logger.info(
                "Streamed %d chunks → %.2fs audio in %.0fms (RTF=%.2f)",
                len(chunks), duration_sec, total_ms,
                total_ms / (duration_sec * 1000) if duration_sec > 0 else 0,
            )
            if engine.last_stream_metrics:
                m = engine.last_stream_metrics
                logger.info(
                    "Stream metrics: %d sentences, first_chunk=%.1fms, avg=%.1fms",
                    m.sentence_count, m.first_chunk_ms, m.avg_sentence_ms,
                )
        else:
            logger.warning("No audio chunks produced.")
            return
    else:
        # Single-shot mode
        logger.info("--- Single-shot synthesis ---")
        t0 = time.perf_counter()
        result = engine.synthesize(
            text=args.text,
            language=args.language,
            spk_embed=spk_embed,
            style=style,
            speed=args.speed,
        )
        total_ms = (time.perf_counter() - t0) * 1000

        if result is None:
            logger.error("Synthesis returned None (cancelled?)")
            return
        audio, duration_sec = result
        logger.info("Generated %.2fs audio in %.0fms", duration_sec, total_ms)

        if engine.last_metrics:
            m = engine.last_metrics
            logger.info(
                "Pipeline: g2p=%.1fms enc=%.1fms dur=%.1fms "
                "f0=%.1fms cs=%.1fms conv=%.1fms voc=%.1fms istft=%.1fms "
                "| total=%.1fms RTF=%.2f",
                m.g2p_ms, m.text_encoder_ms, m.duration_predictor_ms,
                m.f0_predictor_ms, m.content_synthesizer_ms,
                m.converter_ms, m.vocoder_ms, m.istft_ms,
                m.total_ms, m.rtf,
            )

    # Save WAV
    import soundfile as sf
    sf.write(args.output, audio, SAMPLE_RATE)
    logger.info("Saved to %s (%.2fs, %d samples)", args.output, duration_sec, len(audio))

    # Audio stats
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    logger.info("Audio stats: peak=%.4f, rms=%.4f", peak, rms)

    if peak < 1e-6:
        logger.warning(
            "Audio is silent! This is expected without trained checkpoints "
            "(VC backend returns silence with random weights)."
        )


if __name__ == "__main__":
    main()
