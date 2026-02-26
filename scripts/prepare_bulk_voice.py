#!/usr/bin/env python3
"""Prepare bulk voice data (e.g. eroge/drama CD) for TMRVC training.

Scans a directory of character-based voice folders, filters by quality,
and produces a cleaned dataset ready for tmrvc-preprocess.

Typical input structure::

    raw_root/
    ├── sakura/
    │   ├── ev001_a_01.wav    (2.3s, speech)
    │   ├── ev001_a_02.wav    (0.2s, too short → skip)
    │   └── ...
    └── yuki/
        └── ...

Output structure (ready for tsukuyomi adapter)::

    output_dir/
    ├── sakura/
    │   ├── ev001_a_01.wav    (kept)
    │   └── ...
    └── yuki/
        └── ...

Usage::

    # Scan and report (dry run)
    python scripts/prepare_bulk_voice.py --input data/raw/eroge_all --report

    # Filter and copy
    python scripts/prepare_bulk_voice.py --input data/raw/eroge_all --output data/raw/eroge_clean

    # Filter + compress to FLAC (for server transfer)
    python scripts/prepare_bulk_voice.py --input data/raw/eroge_all --output data/raw/eroge_clean --flac

    # Auto-transcribe with Whisper (requires whisper)
    python scripts/prepare_bulk_voice.py --input data/raw/eroge_all --output data/raw/eroge_clean --transcribe
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# --- Filtering thresholds ---
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0
MIN_RMS = 0.005  # below this = near-silence
MAX_RMS = 0.99  # above this = clipping
AUDIO_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


def scan_audio_files(root: Path) -> dict[str, list[Path]]:
    """Scan directory for audio files grouped by speaker folder."""
    speakers: dict[str, list[Path]] = defaultdict(list)
    for f in sorted(root.rglob("*")):
        if f.suffix.lower() in AUDIO_EXTENSIONS and f.is_file():
            # Speaker = first-level subfolder relative to root
            rel = f.relative_to(root)
            parts = rel.parts
            speaker = parts[0] if len(parts) > 1 else "_flat"
            speakers[speaker].append(f)
    return dict(speakers)


def scan_audio_files_with_map(root: Path, speaker_map_path: Path) -> dict[str, list[Path]]:
    """Group audio files using a speaker_map.json from cluster_speakers.py.

    Files mapped to ``"spk_noise"`` are excluded.
    """
    with open(speaker_map_path, encoding="utf-8") as f:
        data = json.load(f)
    mapping = data["mapping"]

    speakers: dict[str, list[Path]] = defaultdict(list)
    missing = 0
    for filename, speaker_id in mapping.items():
        if speaker_id == "spk_noise":
            continue
        audio_path = root / filename
        if not audio_path.exists():
            missing += 1
            continue
        speakers[speaker_id].append(audio_path)

    if missing > 0:
        logger.warning("Speaker map: %d files not found in %s", missing, root)
    logger.info(
        "Speaker map: %d speakers, %d files (excluded %d noise)",
        len(speakers),
        sum(len(v) for v in speakers.values()),
        sum(1 for v in mapping.values() if v == "spk_noise"),
    )
    return dict(speakers)


def check_audio(path: Path) -> dict:
    """Check audio file quality. Returns info dict with 'ok' flag."""
    info: dict = {"path": str(path), "ok": False, "reason": ""}
    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception as e:
        info["reason"] = f"read_error: {e}"
        return info

    if data.ndim > 1:
        data = data[:, 0]  # mono

    duration = len(data) / sr
    info["duration_sec"] = round(duration, 2)
    info["sample_rate"] = sr
    info["samples"] = len(data)

    if duration < MIN_DURATION_SEC:
        info["reason"] = f"too_short ({duration:.1f}s < {MIN_DURATION_SEC}s)"
        return info
    if duration > MAX_DURATION_SEC:
        info["reason"] = f"too_long ({duration:.1f}s > {MAX_DURATION_SEC}s)"
        return info

    rms = float(np.sqrt(np.mean(data**2)))
    peak = float(np.max(np.abs(data)))
    info["rms"] = round(rms, 4)
    info["peak"] = round(peak, 4)

    if rms < MIN_RMS:
        info["reason"] = f"near_silence (rms={rms:.4f})"
        return info
    if peak > MAX_RMS:
        info["reason"] = f"clipping (peak={peak:.4f})"
        return info

    info["ok"] = True
    return info


def copy_or_convert(
    src: Path, dst: Path, to_flac: bool = False
) -> None:
    """Copy audio file, optionally converting to FLAC."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if to_flac and src.suffix.lower() != ".flac":
        data, sr = sf.read(src)
        dst = dst.with_suffix(".flac")
        sf.write(str(dst), data, sr, format="FLAC")
    else:
        shutil.copy2(src, dst)


def transcribe_whisper(
    audio_paths: list[Path],
    output_path: Path,
    model_name: str = "large-v3",
    language: str = "ja",
    device: str = "cuda",
) -> dict[str, str]:
    """Transcribe audio files using Whisper."""
    try:
        import whisper
    except ImportError:
        logger.error("whisper not installed. Run: pip install openai-whisper")
        return {}

    logger.info("Loading Whisper model '%s'...", model_name)
    model = whisper.load_model(model_name, device=device)

    transcripts: dict[str, str] = {}
    for i, path in enumerate(audio_paths):
        if (i + 1) % 100 == 0:
            logger.info("  Transcribing %d/%d...", i + 1, len(audio_paths))
        try:
            result = model.transcribe(
                str(path), language=language, fp16=(device != "cpu")
            )
            text = result["text"].strip()
            if text:
                transcripts[path.stem] = text
        except Exception as e:
            logger.warning("  Whisper failed for %s: %s", path.name, e)

    # Save transcript file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for stem, text in sorted(transcripts.items()):
            f.write(f"{stem}|{text}\n")
    logger.info("Saved %d transcripts to %s", len(transcripts), output_path)
    return transcripts


def main() -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Prepare bulk voice data for TMRVC training.",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input root directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (clean data).")
    parser.add_argument("--report", action="store_true", help="Scan and report only (no copy).")
    parser.add_argument("--flac", action="store_true", help="Convert output to FLAC.")
    parser.add_argument("--transcribe", action="store_true", help="Auto-transcribe with Whisper.")
    parser.add_argument("--whisper-model", default="large-v3", help="Whisper model name.")
    parser.add_argument("--language", default="ja", help="Language for transcription.")
    parser.add_argument("--device", default="cuda", help="Device for Whisper.")
    parser.add_argument(
        "--speaker-map", type=Path, default=None,
        help="Path to _speaker_map.json from cluster_speakers.py (for flat directories).",
    )
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_SEC)
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION_SEC)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    global MIN_DURATION_SEC, MAX_DURATION_SEC
    MIN_DURATION_SEC = args.min_duration
    MAX_DURATION_SEC = args.max_duration

    if not args.report and args.output is None:
        parser.error("--output is required unless --report is used")

    # Scan
    logger.info("Scanning %s ...", args.input)
    if args.speaker_map:
        speakers = scan_audio_files_with_map(args.input, args.speaker_map)
    else:
        speakers = scan_audio_files(args.input)
    total_files = sum(len(v) for v in speakers.values())
    logger.info("Found %d speakers, %d files", len(speakers), total_files)

    # Check quality
    stats: dict[str, dict] = {}
    kept_paths: dict[str, list[Path]] = defaultdict(list)
    skip_reasons: dict[str, int] = defaultdict(int)

    for spk, files in sorted(speakers.items()):
        ok_count = 0
        total_dur = 0.0
        for f in files:
            info = check_audio(f)
            if info["ok"]:
                ok_count += 1
                total_dur += info.get("duration_sec", 0)
                kept_paths[spk].append(f)
            else:
                reason = info["reason"].split("(")[0].strip()
                skip_reasons[reason] += 1
                if args.verbose:
                    logger.debug("  SKIP %s: %s", f.name, info["reason"])

        stats[spk] = {
            "total": len(files),
            "kept": ok_count,
            "skipped": len(files) - ok_count,
            "duration_h": round(total_dur / 3600, 2),
        }

    # Report
    total_kept = sum(s["kept"] for s in stats.values())
    total_dur_h = sum(s["duration_h"] for s in stats.values())

    logger.info("")
    logger.info("=== Report ===")
    logger.info("%-20s %8s %8s %8s %8s", "Speaker", "Total", "Kept", "Skip", "Hours")
    logger.info("-" * 60)
    for spk, s in sorted(stats.items()):
        logger.info(
            "%-20s %8d %8d %8d %8.1f",
            spk[:20], s["total"], s["kept"], s["skipped"], s["duration_h"],
        )
    logger.info("-" * 60)
    logger.info("%-20s %8d %8d %8d %8.1f", "TOTAL", total_files, total_kept,
                total_files - total_kept, total_dur_h)
    logger.info("")
    logger.info("Skip reasons:")
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        logger.info("  %-30s %d", reason, count)

    if args.report:
        # Save report JSON
        report_path = args.input / "_bulk_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"speakers": stats, "skip_reasons": dict(skip_reasons)}, f, indent=2)
        logger.info("Report saved to %s", report_path)
        return

    # Copy/convert
    logger.info("")
    logger.info("Copying %d files to %s ...", total_kept, args.output)
    copied = 0
    for spk, files in sorted(kept_paths.items()):
        for src in files:
            rel = src.relative_to(args.input)
            dst = args.output / rel
            copy_or_convert(src, dst, to_flac=args.flac)
            copied += 1
            if copied % 1000 == 0:
                logger.info("  Copied %d/%d ...", copied, total_kept)

    logger.info("Copied %d files (%.1fh audio)", copied, total_dur_h)

    # Transcribe
    if args.transcribe:
        logger.info("")
        logger.info("=== Whisper Transcription ===")
        for spk, files in sorted(kept_paths.items()):
            output_files = []
            for src in files:
                rel = src.relative_to(args.input)
                dst = args.output / rel
                if args.flac:
                    dst = dst.with_suffix(".flac")
                output_files.append(dst)

            transcript_path = args.output / spk / "transcripts.txt"
            logger.info("Transcribing %s (%d files)...", spk, len(output_files))
            transcribe_whisper(
                output_files,
                transcript_path,
                model_name=args.whisper_model,
                language=args.language,
                device=args.device,
            )

    logger.info("Done.")


if __name__ == "__main__":
    main()
