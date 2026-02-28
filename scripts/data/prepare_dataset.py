#!/usr/bin/env python3
"""Prepare TMRVC dataset from raw audio files.

Complete pipeline:
1. Scan & filter audio files
2. Normalize audio (24kHz, mono, loudness)
3. Extract features (mel, content, f0, spk_embed)
4. Auto-annotate (Whisper transcription + emotion classification)
5. Save to cache directory

Usage:
    # Basic usage
    uv run python scripts/prepare_dataset.py \
        --input data/raw_my_voices \
        --output data/cache \
        --name my_voices \
        --language ja \
        --device cuda

    # Dry run (check file count only)
    uv run python scripts/prepare_dataset.py \
        --input data/raw_my_voices \
        --dry-run

    # With speaker map for flat directory
    uv run python scripts/prepare_dataset.py \
        --input data/voices \
        --output data/cache \
        --name game_voices \
        --speaker-map data/voices/_speaker_map.json

    # Resume from existing cache
    uv run python scripts/prepare_dataset.py \
        --input data/raw_my_voices \
        --output data/cache \
        --name my_voices \
        --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import torch
import tqdm

logger = logging.getLogger(__name__)


# === Configuration ===


@dataclass
class FilterConfig:
    min_duration: float = 0.5
    max_duration: float = 30.0
    min_rms: float = 0.005
    max_rms: float = 0.99
    extensions: tuple[str, ...] = (".wav", ".flac", ".ogg", ".mp3")


@dataclass
class NormalizeConfig:
    target_sr: int = 24000
    target_lufs: float = -23.0
    mono: bool = True


@dataclass
class AnnotateConfig:
    whisper_model: str = "large-v3"
    emotion_model: str = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    language: str = "ja"


@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    dataset_name: str
    speaker_map_path: Path | None = None
    file_list_path: Path | None = None  # NEW: Filter to specific files
    filter: FilterConfig = field(default_factory=FilterConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    annotate: AnnotateConfig = field(default_factory=AnnotateConfig)
    device: str = "cuda"
    resume: bool = False
    dry_run: bool = False


# === Data Classes ===


@dataclass
class AudioFile:
    path: Path
    duration: float
    sample_rate: int
    speaker_id: str
    rms: float = 0.0


@dataclass
class UtteranceMeta:
    utterance_id: str
    speaker_id: str
    n_frames: int
    duration_sec: float
    sample_rate: int = 24000
    text: str = ""
    language_id: int = 0
    emotion_id: int = 6
    emotion_label: str = "neutral"
    emotion_confidence: float = 0.0
    vad: list[float] = field(default_factory=lambda: [0.5, 0.3, 0.5])
    prosody: list[float] = field(default_factory=lambda: [1.0, 0.5, 0.5])
    source_path: str = ""
    # NEW: UCLM specific
    phonemes: str = ""
    phoneme_ids: list[int] = field(default_factory=list)
    durations: list[int] = field(default_factory=list)
    voice_state_mean: list[float] = field(default_factory=lambda: [0.5] * 8)
    has_codec_tokens: bool = False


LANGUAGE_IDS = {"ja": 0, "en": 1, "zh": 2}

EMOTION_MAP = {
    "anger": 0,
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "happiness": 3,
    "joy": 3,
    "sad": 4,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
    "calm": 6,
    "excited": 7,
    "frustrated": 8,
    "anxious": 9,
    "apologetic": 10,
    "confident": 11,
}


# === Stage 1: Scan ===


def scan_audio_files(config: PipelineConfig) -> list[AudioFile]:
    """Scan directory for valid audio files."""
    speaker_map = None
    if config.speaker_map_path and config.speaker_map_path.exists():
        with open(config.speaker_map_path, encoding="utf-8") as f:
            data = json.load(f)
            speaker_map = data.get("mapping", {})

    files = []

    for p in sorted(config.input_dir.rglob("*")):
        if p.suffix.lower() not in config.filter.extensions:
            continue
        if not p.is_file():
            continue

        try:
            info = sf.info(str(p))
            duration = info.duration

            if not (
                config.filter.min_duration <= duration <= config.filter.max_duration
            ):
                continue

            # Determine speaker ID
            if speaker_map:
                speaker_id = speaker_map.get(p.name, "unknown")
                if speaker_id == "spk_noise":
                    continue
            else:
                rel = p.relative_to(config.input_dir)
                speaker_id = rel.parts[0] if len(rel.parts) > 1 else config.dataset_name

            files.append(
                AudioFile(
                    path=p,
                    duration=duration,
                    sample_rate=info.samplerate,
                    speaker_id=speaker_id,
                )
            )

        except Exception as e:
            logger.debug("Skipping %s: %s", p, e)

    return files


# === Stage 2: Normalize ===


def normalize_audio(
    audio_file: AudioFile, config: PipelineConfig
) -> tuple[np.ndarray, int] | None:
    """Load and normalize audio file."""
    from tmrvc_data.preprocessing import load_and_resample
    import pyloudnorm

    try:
        waveform, sr = load_and_resample(str(audio_file.path))

        # Convert to 1D numpy array for pyloudnorm
        audio_np = waveform.numpy().squeeze()

        # Loudness normalization
        meter = pyloudnorm.Meter(sr)
        current_lufs = meter.integrated_loudness(audio_np)

        if not np.isfinite(current_lufs):
            current_lufs = config.normalize.target_lufs

        normalized = pyloudnorm.normalize.loudness(
            audio_np,
            current_lufs,
            config.normalize.target_lufs,
        )

        # Clip prevention
        normalized = np.clip(normalized, -0.99, 0.99)

        return normalized, sr

    except Exception as e:
        logger.warning("Failed to normalize %s: %s", audio_file.path, e)
        return None


# === Stage 3: Extract Features ===


def extract_features(
    waveform: np.ndarray,
    sr: int,
    device: str,
    spk_encoder: Any,
    codec_encoder: Any | None = None,
    voice_state_estimator: Any | None = None,
) -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray | None,
        np.ndarray | None,
    ]
    | None
):
    """Extract mel, content, f0, spk_embed, codec_tokens, voice_state."""
    from tmrvc_core.audio import compute_mel

    try:
        waveform_t = torch.from_numpy(waveform).float()

        # Mel
        mel = compute_mel(waveform_t.unsqueeze(0)).numpy()[0]

        # F0 (placeholder - zero)
        f0 = np.zeros((1, mel.shape[-1]), dtype=np.float32)

        # Content (placeholder - zeros, will be extracted during training)
        content = np.zeros((768, mel.shape[-1]), dtype=np.float32)

        # Speaker embedding
        with torch.no_grad():
            spk_embed = spk_encoder.extract(waveform_t.to(device))
            spk_embed = spk_embed.cpu().numpy()

        # Codec tokens (NEW)
        codec_tokens = None
        if codec_encoder is not None:
            try:
                with torch.no_grad():
                    tokens = codec_encoder.encode_simple(
                        waveform_t.unsqueeze(0).unsqueeze(0).to(device)
                    )
                    codec_tokens = tokens.cpu().numpy()[0]  # [n_codebooks, T]
            except Exception as e:
                logger.debug("Codec encoding failed: %s", e)

        # Voice state (NEW)
        voice_state = None
        if voice_state_estimator is not None and codec_tokens is not None:
            try:
                mel_t = torch.from_numpy(mel).unsqueeze(0).to(device)
                f0_t = torch.from_numpy(f0).to(device)
                with torch.no_grad():
                    vs = voice_state_estimator.estimate(mel_t, f0_t)
                    voice_state = vs.cpu().numpy()[0]  # [T, 8]
            except Exception as e:
                logger.debug("Voice state estimation failed: %s", e)

        return mel, content, f0, spk_embed, codec_tokens, voice_state

    except Exception as e:
        logger.warning("Failed to extract features: %s", e)
        return None


# === Stage 4: Annotate ===


def transcribe_audio(
    audio_path: Path,
    whisper_model: Any,
    language: str,
) -> str:
    """Transcribe with Whisper."""
    try:
        segments, _ = whisper_model.transcribe(str(audio_path), language=language)
        return "".join(seg.text for seg in segments).strip()
    except Exception as e:
        logger.debug("Transcription failed: %s", e)
        return ""


def classify_emotion(
    audio_path: Path,
    emotion_classifier: Any,
) -> tuple[int, str, float]:
    """Classify emotion from audio."""
    try:
        result = emotion_classifier(str(audio_path))
        label = result[0]["label"].lower()
        confidence = result[0].get("score", 1.0)
        emotion_id = EMOTION_MAP.get(label, 6)
        return emotion_id, label, confidence
    except Exception as e:
        logger.debug("Emotion classification failed: %s", e)
        return 6, "neutral", 0.0


# === Stage 5: Save ===


def save_utterance(
    output_dir: Path,
    dataset_name: str,
    meta: UtteranceMeta,
    mel: np.ndarray,
    content: np.ndarray,
    f0: np.ndarray,
    spk_embed: np.ndarray,
    codec_tokens: np.ndarray | None = None,
    voice_state: np.ndarray | None = None,
) -> Path:
    """Save utterance to cache."""
    utt_dir = output_dir / dataset_name / "train" / meta.speaker_id / meta.utterance_id
    utt_dir.mkdir(parents=True, exist_ok=True)

    np.save(utt_dir / "mel.npy", mel)
    np.save(utt_dir / "content.npy", content)
    np.save(utt_dir / "f0.npy", f0)
    np.save(utt_dir / "spk_embed.npy", spk_embed)

    # NEW: Save codec tokens
    if codec_tokens is not None:
        np.save(utt_dir / "codec_tokens.npy", codec_tokens)
        meta.has_codec_tokens = True

    # NEW: Save voice state
    if voice_state is not None:
        np.save(utt_dir / "voice_state.npy", voice_state)
        meta.voice_state_mean = voice_state.mean(axis=0).tolist()

    (utt_dir / "meta.json").write_text(
        json.dumps(asdict(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return utt_dir


def write_manifest(
    output_dir: Path,
    dataset_name: str,
    stats: dict,
) -> Path:
    """Write dataset manifest."""
    manifest_dir = output_dir / "_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset": dataset_name,
        "split": "train",
        "n_utterances": stats["processed"],
        "n_speakers": stats["n_speakers"],
        "total_duration_sec": round(stats["total_duration"], 2),
        "pipeline_version": "1.0",
        "filter_config": asdict(FilterConfig()),
        "normalize_config": asdict(NormalizeConfig()),
    }

    manifest_path = manifest_dir / f"{dataset_name}_train.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return manifest_path


# === Main Pipeline ===


def run_pipeline(config: PipelineConfig) -> int:
    """Run the complete pipeline."""
    logger.info("=" * 60)
    logger.info("TMRVC Dataset Preparation Pipeline")
    logger.info("=" * 60)
    logger.info("Input:  %s", config.input_dir)
    logger.info("Output: %s", config.output_dir)
    logger.info("Dataset: %s", config.dataset_name)
    logger.info("Device:  %s", config.device)

    # Stage 1: Scan
    logger.info("\n[Stage 1/5] Scanning audio files...")
    audio_files = scan_audio_files(config)
    logger.info("Found %d valid files", len(audio_files))

    # Filter by file list if provided
    if config.file_list_path and config.file_list_path.exists():
        with open(config.file_list_path, encoding="utf-8") as f:
            file_set = set(line.strip() for line in f if line.strip())
        audio_files = [af for af in audio_files if af.path.name in file_set]
        logger.info("Filtered to %d files from list", len(audio_files))

    if not audio_files:
        logger.error("No valid audio files found!")
        return 1

    if config.dry_run:
        logger.info("\n[DRY RUN] Would process %d files", len(audio_files))

        # Show speaker distribution
        speakers = {}
        for af in audio_files:
            speakers[af.speaker_id] = speakers.get(af.speaker_id, 0) + 1

        logger.info("\nSpeaker distribution:")
        for spk, count in sorted(speakers.items()):
            logger.info("  %s: %d files", spk, count)

        return 0

    # Load models
    logger.info("\n[Stage 2/5] Loading models...")

    from faster_whisper import WhisperModel
    from transformers import pipeline
    from tmrvc_data.speaker import SpeakerEncoder

    # Whisper
    logger.info("  Loading Whisper '%s'...", config.annotate.whisper_model)
    compute_type = "int8" if config.device == "cpu" else "float16"
    whisper_model = WhisperModel(
        config.annotate.whisper_model,
        device=config.device,
        compute_type=compute_type,
    )

    # Emotion classifier
    logger.info("  Loading emotion classifier...")
    emotion_classifier = pipeline(
        "audio-classification",
        model=config.annotate.emotion_model,
        device=0 if config.device == "cuda" else -1,
    )

    # Speaker encoder
    logger.info("  Loading speaker encoder...")
    spk_encoder = SpeakerEncoder(device=config.device)

    # Codec encoder (for UCLM)
    codec_encoder = None
    voice_state_estimator = None
    try:
        from tmrvc_data.codec import EnCodecWrapper
        from tmrvc_data.voice_state import VoiceStateEstimator

        logger.info("  Loading EnCodec...")
        codec_encoder = EnCodecWrapper(device=config.device)

        logger.info("  Loading voice state estimator...")
        voice_state_estimator = VoiceStateEstimator(device=config.device)
    except ImportError as e:
        logger.warning("UCLM components not available: %s", e)
        logger.warning("Codec tokens and voice state will not be extracted.")

    # Process files
    logger.info("\n[Stage 3-5/5] Processing files...")

    lang_id = LANGUAGE_IDS.get(config.annotate.language, 0)
    stats = {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "n_speakers": 0,
        "total_duration": 0.0,
    }
    speakers = set()

    start_time = time.time()

    for audio_file in tqdm.tqdm(audio_files, desc="Processing"):
        utt_id = f"{config.dataset_name}_{audio_file.path.stem}"

        # Check if already exists (resume)
        if config.resume:
            utt_dir = (
                config.output_dir
                / config.dataset_name
                / "train"
                / audio_file.speaker_id
                / utt_id
            )
            if (utt_dir / "meta.json").exists():
                stats["skipped"] += 1
                continue

        try:
            # Normalize
            result = normalize_audio(audio_file, config)
            if result is None:
                stats["errors"] += 1
                continue

            waveform, sr = result

            # Extract features
            features = extract_features(
                waveform,
                sr,
                config.device,
                spk_encoder,
                codec_encoder,
                voice_state_estimator,
            )
            if features is None:
                stats["errors"] += 1
                continue

            mel, content, f0, spk_embed, codec_tokens, voice_state = features

            # Annotate
            text = transcribe_audio(
                audio_file.path, whisper_model, config.annotate.language
            )
            emotion_id, emotion_label, emotion_conf = classify_emotion(
                audio_file.path, emotion_classifier
            )

            # Estimate VAD
            valence = 0.5 + 0.3 * (emotion_id in [3, 5, 7, 11])
            arousal = 0.3 + 0.4 * emotion_conf
            vad = [round(valence, 3), round(arousal, 3), 0.5]

            # Create meta
            meta = UtteranceMeta(
                utterance_id=utt_id,
                speaker_id=f"{config.dataset_name}_{audio_file.speaker_id}",
                n_frames=mel.shape[-1],
                duration_sec=audio_file.duration,
                text=text,
                language_id=lang_id,
                emotion_id=emotion_id,
                emotion_label=emotion_label,
                emotion_confidence=round(emotion_conf, 4),
                vad=vad,
                source_path=str(audio_file.path),
            )

            # Save
            save_utterance(
                config.output_dir,
                config.dataset_name,
                meta,
                mel,
                content,
                f0,
                spk_embed,
                codec_tokens,
                voice_state,
            )

            stats["processed"] += 1
            stats["total_duration"] += audio_file.duration
            speakers.add(audio_file.speaker_id)

        except Exception as e:
            logger.warning("Error processing %s: %s", audio_file.path, e)
            stats["errors"] += 1

    stats["n_speakers"] = len(speakers)

    # Write manifest
    manifest_path = write_manifest(config.output_dir, config.dataset_name, stats)

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info("Processed:      %d", stats["processed"])
    logger.info("Skipped:        %d", stats["skipped"])
    logger.info("Errors:         %d", stats["errors"])
    logger.info("Speakers:       %d", stats["n_speakers"])
    logger.info(
        "Total duration: %.1f sec (%.1f min)",
        stats["total_duration"],
        stats["total_duration"] / 60,
    )
    logger.info("Elapsed time:   %.1f min", elapsed / 60)
    logger.info("Manifest:       %s", manifest_path)
    logger.info("")
    logger.info("Next step:")
    logger.info(
        "  uv run tmrvc-train-teacher --cache-dir %s --dataset %s --device %s",
        config.output_dir,
        config.dataset_name,
        config.device,
    )

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TMRVC dataset from raw audio files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        required=True,
        help="Input directory with audio files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/cache"),
        help="Output cache directory",
    )
    parser.add_argument("--name", "-n", required=True, help="Dataset name")
    parser.add_argument(
        "--speaker-map",
        type=Path,
        default=None,
        help="JSON file mapping filenames to speaker IDs",
    )
    parser.add_argument(
        "--language",
        "-l",
        default="ja",
        choices=["ja", "en", "zh"],
        help="Language for transcription",
    )
    parser.add_argument("--device", "-d", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--min-duration", type=float, default=0.5, help="Minimum audio duration (sec)"
    )
    parser.add_argument(
        "--max-duration", type=float, default=30.0, help="Maximum audio duration (sec)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Skip already processed files"
    )
    parser.add_argument(
        "--file-list",
        type=Path,
        default=None,
        help="File containing list of audio filenames to process (one per line)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Scan only, don't process"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        dataset_name=args.name,
        speaker_map_path=args.speaker_map,
        file_list_path=args.file_list,
        device=args.device,
        resume=args.resume,
        dry_run=args.dry_run,
        filter=FilterConfig(
            min_duration=args.min_duration,
            max_duration=args.max_duration,
        ),
        annotate=AnnotateConfig(language=args.language),
    )

    return run_pipeline(config)


if __name__ == "__main__":
    sys.exit(main())
