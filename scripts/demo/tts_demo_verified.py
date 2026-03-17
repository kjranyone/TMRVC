#!/usr/bin/env python3
"""UCLM TTS end-to-end demo with speaker embedding support.

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

    parser = argparse.ArgumentParser(description="UCLM TTS Demo Verified")
    parser.add_argument("--text", default="こんにちは、世界！これは新しく学習したモデルによるテスト音声です。")
    parser.add_argument("--language", default="ja", choices=["ja", "en", "zh", "ko"])
    parser.add_argument("--output", default="tts_demo_verified.wav", help="Output WAV path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--uclm-checkpoint", type=Path, required=True)
    parser.add_argument("--codec-checkpoint", type=Path, default="checkpoints/codec/codec_latest.pt")
    parser.add_argument("--speaker-embed", type=Path, help="Path to speaker embedding .npy")
    parser.add_argument("--emotion", default="neutral")
    parser.add_argument("--hold-bias", type=float, default=2.0, help="Bias to stay on current phoneme (higher = slower/more stable)")
    parser.add_argument("--pace", type=float, default=1.0, help="Speech pace multiplier")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("tts_demo")

    from tmrvc_core.constants import D_SPEAKER, SAMPLE_RATE
    from tmrvc_core.dialogue_types import StyleParams
    from tmrvc_data.g2p import text_to_phonemes
    from tmrvc_serve.uclm_engine import UCLMEngine

    logger.info("=== UCLM TTS Demo Verified ===")
    logger.info("Text: %s", args.text)
    logger.info("Emotion: %s", args.emotion)

    # 1. Initialize Engine
    t0 = time.perf_counter()
    if not args.uclm_checkpoint.exists():
        logger.error("UCLM Checkpoint not found: %s", args.uclm_checkpoint)
        return
    if not args.codec_checkpoint.exists():
        logger.error("Codec Checkpoint not found: %s", args.codec_checkpoint)
        return

    engine = UCLMEngine(
        device=args.device,
    )
    engine.load_models(
        uclm_path=args.uclm_checkpoint,
        codec_path=args.codec_checkpoint,
    )
    logger.info("Engine loaded in %.0fms", (time.perf_counter() - t0) * 1000)

    # 2. Prepare Inputs
    # Phonemes
    g2p_result = text_to_phonemes(args.text, language=args.language)
    phonemes_t = g2p_result.phoneme_ids.unsqueeze(0).to(args.device)
    supra_t = g2p_result.text_suprasegmentals.unsqueeze(0).to(args.device) if g2p_result.text_suprasegmentals is not None else None
    
    # Speaker embedding
    if args.speaker_embed and args.speaker_embed.exists():
        logger.info("Loading speaker embedding from %s", args.speaker_embed)
        spk_embed_np = np.load(args.speaker_embed)
        spk_embed = torch.from_numpy(spk_embed_np).float().to(args.device)
        if spk_embed.dim() == 1:
            spk_embed = spk_embed.unsqueeze(0)
    else:
        logger.warning("Using random speaker embedding!")
        spk_embed = torch.randn(1, D_SPEAKER).to(args.device)
    
    # Style
    style = StyleParams(emotion=args.emotion)

    # 3. Synthesis
    logger.info("--- Generating Audio ---")
    t0 = time.perf_counter()
    audio_t, metrics = engine.tts(
        phonemes=phonemes_t,
        speaker_embed=spk_embed,
        style=style,
        language_id=g2p_result.language_id,
        text_suprasegmentals=supra_t,
        hold_bias=args.hold_bias,
        pace=args.pace,
    )
    total_ms = (time.perf_counter() - t0) * 1000
    
    audio = audio_t.cpu().numpy()
    duration_sec = len(audio) / SAMPLE_RATE
    logger.info(
        "Generated %.2fs audio in %.0fms (RTF=%.2f)",
        duration_sec,
        total_ms,
        float(metrics.get("rtf", 0.0)),
    )

    # 4. Save WAV
    import soundfile as sf
    sf.write(args.output, audio, SAMPLE_RATE)
    logger.info("Saved to %s", args.output)

    # 5. SOTA Validation (GEMINI.md Mandate)
    # Theoretical limits for stability and naturalness
    STD_MIN = 0.04
    STD_MAX = 0.25
    PEAK_MAX = 0.99
    RTF_MAX = 0.8  # Target for high-end serve path

    peak = np.max(np.abs(audio))
    std = np.std(audio)
    rtf = float(metrics.get("rtf", 0.0))

    logger.info("--- Validation Results ---")
    logger.info("Amplitude Stats: peak=%.4f, std=%.4f", peak, std)
    logger.info("Performance: RTF=%.2f", rtf)

    errors = []
    if peak > PEAK_MAX:
        errors.append(f"Peak amplitude too high (%.4f > %.4f) - Possible clipping" % (peak, PEAK_MAX))
    if std < STD_MIN:
        errors.append(f"Standard deviation too low (%.4f < %.4f) - Possible silent or collapsed output" % (std, STD_MIN))
    if std > STD_MAX:
        errors.append(f"Standard deviation too high (%.4f > %.4f) - Possible noisy or unstable output" % (std, STD_MAX))
    if rtf > RTF_MAX:
        errors.append(f"RTF exceeds threshold (%.2f > %.2f) - Serve-path latency regression" % (rtf, RTF_MAX))

    if not errors:
        logger.info("✅ SOTA VALIDATION PASSED: Audio and performance are within theoretical bounds.")
    else:
        logger.error("❌ SOTA VALIDATION FAILED:")
        for err in errors:
            logger.error("  - %s", err)
        sys.exit(1)


if __name__ == "__main__":
    main()
