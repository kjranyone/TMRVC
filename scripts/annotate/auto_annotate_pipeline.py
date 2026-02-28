#!/usr/bin/env python3
"""Complete auto-annotation pipeline for raw audio files.

Takes a directory of audio files (no metadata required) and produces
a fully annotated dataset ready for TMRVC training.

Pipeline stages:
1. Preprocess audio → mel, content, f0, spk_embed
2. Whisper transcription → text
3. Emotion annotation → emotion_id, vad
4. Voice source estimation → breathiness, tension, etc.

Usage:
    uv run python scripts/auto_annotate_pipeline.py \
        --audio-dir data/wav \
        --cache-dir data/cache \
        --dataset custom_speaker \
        --language ja \
        --device cuda

Output structure:
    data/cache/custom_speaker/train/{speaker_id}/{utt_id}/
    ├── mel.npy           # [80, T] log-mel
    ├── content.npy       # [768, T] ContentVec
    ├── f0.npy            # [1, T] Hz
    ├── spk_embed.npy     # [192] speaker embedding
    └── meta.json         # {utterance_id, speaker_id, text, emotion_id, vad, ...}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    audio_dir: Path
    cache_dir: Path
    dataset: str
    split: str = "train"
    language: str = "ja"
    device: str = "cuda"

    whisper_model: str = "large-v3"
    emotion_model: str = ""  # Auto-select based on language

    skip_preprocess: bool = False
    skip_transcribe: bool = False
    skip_emotion: bool = False
    skip_voice_source: bool = True  # Optional, requires external model

    min_duration: float = 0.5
    max_duration: float = 30.0


@dataclass
class UtteranceMeta:
    utterance_id: str
    speaker_id: str
    n_frames: int
    duration_sec: float
    text: str = ""
    language_id: int = 0
    emotion_id: int = 6  # neutral
    emotion_label: str = "neutral"
    emotion_confidence: float = 0.0
    vad: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.5])
    prosody: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.5])


LANGUAGE_IDS = {"ja": 0, "en": 1, "zh": 2}

EMOTION_MAP_JA = {
    "怒り": 0,
    "いかり": 0,
    "anger": 0,
    "嫌悪": 1,
    "けんお": 1,
    "disgust": 1,
    "恐怖": 2,
    "きょうふ": 2,
    "fear": 2,
    "喜び": 3,
    "よろこび": 3,
    "幸福": 3,
    "happy": 3,
    "joy": 3,
    "悲しみ": 4,
    "かなしみ": 4,
    "悲": 4,
    "sad": 4,
    "sadness": 4,
    "驚き": 5,
    "おどろき": 5,
    "surprise": 5,
    "中立": 6,
    "通常": 6,
    "普通": 6,
    "neutral": 6,
    "calm": 6,
    "興奮": 7,
    "excited": 7,
    "不満": 8,
    "frustrated": 8,
    "不安": 9,
    "anxious": 9,
    "謝罪": 10,
    "apologetic": 10,
    "自信": 11,
    "confident": 11,
}


def scan_audio_files(
    root: Path, min_dur: float, max_dur: float
) -> list[tuple[Path, float]]:
    """Scan directory for valid audio files."""
    audio_exts = {".wav", ".flac", ".ogg", ".mp3"}
    files = []

    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in audio_exts:
            continue
        try:
            info = sf.info(str(p))
            if min_dur <= info.duration <= max_dur:
                files.append((p, info.duration))
        except Exception as e:
            logger.debug("Skipping %s: %s", p, e)

    return files


def preprocess_audio(
    audio_path: Path,
    device: str,
    content_encoder: Any = None,
    spk_encoder: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Extract mel, content, f0, spk_embed from audio."""
    from tmrvc_core.audio import compute_mel
    from tmrvc_data.preprocessing import load_and_resample

    try:
        waveform, sr = load_and_resample(str(audio_path))
    except Exception as e:
        logger.warning("Failed to load %s: %s", audio_path, e)
        return None

    mel = compute_mel(waveform.unsqueeze(0)).numpy()[0]
    f0 = np.zeros((1, mel.shape[-1]), dtype=np.float32)

    if content_encoder is not None:
        with torch.no_grad():
            mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
            f0_t = torch.from_numpy(f0).unsqueeze(0).to(device)
            content, _ = content_encoder(mel_t, f0_t)
            content = content.cpu().numpy()[0]
    else:
        content = np.zeros((768, mel.shape[-1]), dtype=np.float32)

    if spk_encoder is not None:
        with torch.no_grad():
            spk_embed = spk_encoder.extract(waveform.to(device))
            spk_embed = spk_embed.cpu().numpy()
    else:
        spk_embed = np.zeros(192, dtype=np.float32)

    return mel, content, f0, spk_embed


def transcribe_audio(audio_path: Path, whisper_model: Any) -> str:
    """Transcribe audio with Whisper."""
    segments, _ = whisper_model.transcribe(str(audio_path), language="ja")
    return "".join(seg.text for seg in segments).strip()


def classify_emotion(
    text: str,
    audio_path: Path,
    emotion_classifier: Any,
    use_audio: bool,
) -> tuple[int, str, float]:
    """Classify emotion from text or audio."""
    if use_audio:
        result = emotion_classifier(str(audio_path))
    else:
        result = emotion_classifier(text)

    label = result[0]["label"].lower()
    confidence = result[0].get("score", 1.0)

    emotion_id = EMOTION_MAP_JA.get(label, 6)

    return emotion_id, label, confidence


def save_utterance(
    cache_dir: Path,
    dataset: str,
    speaker_id: str,
    utt_id: str,
    mel: np.ndarray,
    content: np.ndarray,
    f0: np.ndarray,
    spk_embed: np.ndarray,
    meta: UtteranceMeta,
) -> Path:
    """Save utterance to cache."""
    utt_dir = cache_dir / dataset / "train" / speaker_id / utt_id
    utt_dir.mkdir(parents=True, exist_ok=True)

    np.save(utt_dir / "mel.npy", mel)
    np.save(utt_dir / "content.npy", content)
    np.save(utt_dir / "f0.npy", f0)
    np.save(utt_dir / "spk_embed.npy", spk_embed)

    meta_dict = {
        "utterance_id": meta.utterance_id,
        "speaker_id": meta.speaker_id,
        "n_frames": meta.n_frames,
        "duration_sec": round(meta.duration_sec, 3),
        "text": meta.text,
        "language_id": meta.language_id,
        "emotion_id": meta.emotion_id,
        "emotion_label": meta.emotion_label,
        "emotion_confidence": meta.emotion_confidence,
        "vad": meta.vad,
        "prosody": meta.prosody,
    }

    (utt_dir / "meta.json").write_text(
        json.dumps(meta_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return utt_dir


def main():
    parser = argparse.ArgumentParser(
        description="Complete auto-annotation pipeline for raw audio"
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--language", default="ja", choices=["ja", "en"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--emotion-model", default=None)
    parser.add_argument("--skip-transcribe", action="store_true")
    parser.add_argument("--skip-emotion", action="store_true")
    parser.add_argument("--min-duration", type=float, default=0.5)
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(
        audio_dir=args.audio_dir,
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        language=args.language,
        device=args.device,
        whisper_model=args.whisper_model,
        skip_transcribe=args.skip_transcribe,
        skip_emotion=args.skip_emotion,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    logger.info("=" * 60)
    logger.info("TMRVC Auto-Annotation Pipeline")
    logger.info("=" * 60)
    logger.info("Audio dir: %s", config.audio_dir)
    logger.info("Cache dir: %s", config.cache_dir)
    logger.info("Dataset: %s", config.dataset)
    logger.info("Language: %s", config.language)

    # Scan audio files
    logger.info("\n[Stage 1] Scanning audio files...")
    audio_files = scan_audio_files(
        config.audio_dir,
        config.min_duration,
        config.max_duration,
    )

    if args.max_files > 0:
        audio_files = audio_files[: args.max_files]

    logger.info("Found %d valid audio files", len(audio_files))

    if not audio_files:
        logger.error("No audio files found!")
        return 1

    # Load models
    logger.info("\n[Stage 2] Loading models...")

    from faster_whisper import WhisperModel

    whisper_model = None
    if not config.skip_transcribe:
        logger.info(
            "  Loading Whisper '%s' on %s...", config.whisper_model, config.device
        )
        compute_type = "int8" if config.device == "cpu" else "float16"
        whisper_model = WhisperModel(
            config.whisper_model,
            device=config.device,
            compute_type=compute_type,
        )

    emotion_classifier = None
    use_audio_emotion = True
    if not config.skip_emotion:
        # Use audio-based emotion model (works for any language)
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        use_audio_emotion = True

        if args.emotion_model:
            model_name = args.emotion_model
            use_audio_emotion = "wav2vec2" in model_name or "audio" in model_name

        logger.info("  Loading emotion classifier '%s'...", model_name)
        from transformers import pipeline

        emotion_classifier = pipeline(
            "audio-classification",
            model=model_name,
            device=0 if config.device == "cuda" else -1,
        )

    # Load speaker encoder
    logger.info("  Loading speaker encoder...")
    from tmrvc_data.speaker import SpeakerEncoder

    spk_encoder = SpeakerEncoder(device=config.device)

    # Process files
    logger.info("\n[Stage 3] Processing files...")

    speaker_id = f"{config.dataset}_speaker"
    lang_id = LANGUAGE_IDS.get(config.language, 0)

    stats = {"processed": 0, "errors": 0, "skipped": 0}

    for i, (audio_path, duration) in enumerate(tqdm.tqdm(audio_files, desc="Pipeline")):
        stem = audio_path.stem
        utt_id = f"{config.dataset}_{stem}"

        try:
            # Preprocess
            result = preprocess_audio(
                audio_path, config.device, spk_encoder=spk_encoder
            )
            if result is None:
                stats["errors"] += 1
                continue

            mel, content, f0, spk_embed = result
            n_frames = mel.shape[-1]

            # Transcribe
            text = ""
            if whisper_model is not None:
                text = transcribe_audio(audio_path, whisper_model)

            # Emotion
            emotion_id = 6
            emotion_label = "neutral"
            emotion_confidence = 0.0
            vad = [0.5, 0.3, 0.5]

            if emotion_classifier is not None:
                if use_audio_emotion or text:
                    emotion_id, emotion_label, emotion_confidence = classify_emotion(
                        text, audio_path, emotion_classifier, use_audio_emotion
                    )
                    # Estimate VAD from emotion
                    valence = 0.5 + 0.3 * (emotion_id in [3, 5, 7, 11])
                    arousal = 0.3 + 0.4 * emotion_confidence
                    dominance = 0.5
                    vad = [round(valence, 3), round(arousal, 3), round(dominance, 3)]

            # Create meta
            meta = UtteranceMeta(
                utterance_id=utt_id,
                speaker_id=speaker_id,
                n_frames=n_frames,
                duration_sec=duration,
                text=text,
                language_id=lang_id,
                emotion_id=emotion_id,
                emotion_label=emotion_label,
                emotion_confidence=round(emotion_confidence, 4),
                vad=vad,
            )

            # Save
            save_utterance(
                config.cache_dir,
                config.dataset,
                speaker_id,
                utt_id,
                mel,
                content,
                f0,
                spk_embed,
                meta,
            )

            stats["processed"] += 1

        except Exception as e:
            logger.warning("Error processing %s: %s", audio_path, e)
            stats["errors"] += 1

    # Write manifest
    manifest_dir = config.cache_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{config.dataset}_train.json"

    manifest = {
        "dataset": config.dataset,
        "split": "train",
        "n_utterances": stats["processed"],
        "n_speakers": 1,
        "language": config.language,
        "auto_annotated": True,
        "pipeline_version": "1.0",
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info("Processed: %d", stats["processed"])
    logger.info("Errors:    %d", stats["errors"])
    logger.info("Manifest:  %s", manifest_path)
    logger.info("")
    logger.info(
        "Next step: uv run tmrvc-train-teacher --cache-dir %s --dataset %s --device %s",
        config.cache_dir,
        config.dataset,
        config.device,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
