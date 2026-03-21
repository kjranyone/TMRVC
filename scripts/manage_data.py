#!/usr/bin/env python3
"""Dataset management for UCLMv4 training.

Provides incremental corpus registration and cache building.

Commands:
    add <name> <path>    Register a corpus (scan wavs, append to manifest)
    build                Build cache for uncached entries
    status               Show registration and cache status
    remove <name>        Remove a corpus from manifest (does not delete files)

Examples:
    # Register existing corpora
    .venv/bin/python scripts/manage_data.py add jvs data/raw/jvs --speaker-from-dir
    .venv/bin/python scripts/manage_data.py add moe data/raw/moe_voices --speaker-from-dir
    .venv/bin/python scripts/manage_data.py add tsukuyomi data/raw/tsukuyomi --speaker single
    .venv/bin/python scripts/manage_data.py add vctk data/raw/vctk --speaker-from-dir --ext flac

    # Add new corpus later
    .venv/bin/python scripts/manage_data.py add my_voices /path/to/wavs --speaker single
    .venv/bin/python scripts/manage_data.py add drama /path/to/drama --speaker-from-dir

    # Build cache (only uncached entries)
    .venv/bin/python scripts/manage_data.py build
    .venv/bin/python scripts/manage_data.py build --max 5000 --workers 1

    # Check status
    .venv/bin/python scripts/manage_data.py status
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import re
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tmrvc-core" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-data" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-train" / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("manage_data")

DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifest.jsonl"
CACHE_DIR = DATA_DIR / "cache" / "v4"

# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    entries = []
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            entries.append(json.loads(line))
    return entries


def save_manifest(entries: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def append_manifest(new_entries: list[dict]):
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        for e in new_entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def stable_utt_id(corpus: str, wav_path: str) -> str:
    """Generate a stable utterance ID from corpus name and wav path."""
    h = hashlib.sha256(f"{corpus}:{wav_path}".encode()).hexdigest()[:12]
    return f"{corpus}_{h}"


# ---------------------------------------------------------------------------
# Command: add
# ---------------------------------------------------------------------------

def cmd_add(args):
    name = args.name
    path = Path(args.path).resolve()

    if not path.exists():
        logger.error("Path does not exist: %s", path)
        sys.exit(1)

    # Scan for audio files
    exts = args.ext.split(",")
    wavs = []
    for ext in exts:
        wavs.extend(sorted(path.rglob(f"*.{ext.strip()}")))

    if not wavs:
        logger.error("No audio files found in %s (extensions: %s)", path, args.ext)
        sys.exit(1)

    # Load existing manifest to check for duplicates
    existing = load_manifest()
    existing_paths = {e["wav_path"] for e in existing}
    existing_corpora = {e["corpus"] for e in existing}

    if name in existing_corpora and not args.force:
        count = sum(1 for e in existing if e["corpus"] == name)
        logger.error("Corpus '%s' already registered (%d entries). Use --force to re-add.", name, count)
        sys.exit(1)

    if args.force and name in existing_corpora:
        existing = [e for e in existing if e["corpus"] != name]
        save_manifest(existing)
        existing_paths = {e["wav_path"] for e in existing}
        logger.info("Removed existing entries for '%s'", name)

    # Determine speaker assignment strategy
    new_entries = []
    for wav in wavs:
        wav_str = str(wav)
        if wav_str in existing_paths:
            continue

        # Speaker ID
        if args.speaker == "single":
            speaker = f"{name}_default"
        elif args.speaker_from_dir:
            # Use immediate parent directory as speaker
            speaker = f"{name}_{wav.parent.name}"
        else:
            speaker = f"{name}_default"

        utt_id = stable_utt_id(name, wav_str)

        new_entries.append({
            "utt_id": utt_id,
            "corpus": name,
            "speaker": speaker,
            "wav_path": wav_str,
            "cached": False,
        })

    if not new_entries:
        logger.info("No new files to add for '%s'", name)
        return

    append_manifest(new_entries)

    # Summary
    speakers = Counter(e["speaker"] for e in new_entries)
    logger.info("Added %d entries for corpus '%s'", len(new_entries), name)
    logger.info("  Speakers: %d", len(speakers))
    for spk, cnt in speakers.most_common(10):
        logger.info("    %s: %d", spk, cnt)
    if len(speakers) > 10:
        logger.info("    ... and %d more", len(speakers) - 10)


# ---------------------------------------------------------------------------
# Command: build
# ---------------------------------------------------------------------------

def cmd_build(args):
    from tmrvc_core.constants import (
        D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
        HOP_LENGTH, SAMPLE_RATE,
    )
    from tmrvc_core.audio import compute_mel

    entries = load_manifest()
    uncached = [e for e in entries if not e.get("cached", False)]

    if not uncached:
        logger.info("All %d entries are cached. Nothing to build.", len(entries))
        return

    if args.max and args.max < len(uncached):
        uncached = uncached[:args.max]

    logger.info("Building cache: %d uncached / %d total", len(uncached), len(entries))

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load models
    logger.info("Loading ASR (Whisper large-v3)...")
    from faster_whisper import WhisperModel
    compute = "float16" if device == "cuda" else "int8"
    whisper = WhisperModel("large-v3", device=device, compute_type=compute)

    logger.info("Loading speaker encoder (ECAPA-TDNN)...")
    from tmrvc_data.speaker import SpeakerEncoder
    spk_encoder = SpeakerEncoder(device=device)

    logger.info("Loading voice state estimator...")
    from tmrvc_data.voice_state import VoiceStateEstimator
    vs_estimator = VoiceStateEstimator(device=device)

    logger.info("Loading codec (EnCodec 24kHz)...")
    from tmrvc_data.encodec_codec import EnCodecWrapper
    codec = EnCodecWrapper(device=device)

    logger.info("All models loaded. Starting cache build...")

    CODEC_HOP = 320  # EnCodec 75 Hz
    n_ok = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()

    # Index for fast manifest update
    entry_idx = {e["utt_id"]: i for i, e in enumerate(entries)}

    for i, entry in enumerate(uncached):
        utt_id = entry["utt_id"]
        speaker = entry["speaker"]
        wav_path = entry["wav_path"]

        utt_dir = CACHE_DIR / "train" / speaker / utt_id

        # Skip if already cached on disk
        if (utt_dir / "meta.json").exists():
            idx = entry_idx[utt_id]
            entries[idx]["cached"] = True
            n_ok += 1
            continue

        try:
            # Load and resample
            from tmrvc_data.preprocessing import load_and_resample
            waveform, _ = load_and_resample(wav_path, target_sr=SAMPLE_RATE)
            if waveform is None:
                n_skip += 1
                continue

            waveform_np = waveform.squeeze().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform).squeeze()
            n_samples = len(waveform_np)
            n_frames = n_samples // CODEC_HOP

            if n_frames < 20:
                n_skip += 1
                continue

            # ASR
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, waveform_np, SAMPLE_RATE)
                segments, info = whisper.transcribe(tmp.name, beam_size=5, vad_filter=True)
                transcript = "".join(seg.text for seg in segments).strip()
                lang = info.language if hasattr(info, "language") else "ja"
                asr_conf = info.language_probability if hasattr(info, "language_probability") else 0.5

            if not transcript:
                n_skip += 1
                continue

            # G2P
            from tmrvc_data.g2p import text_to_phonemes
            g2p_result = text_to_phonemes(transcript, language=lang)
            phoneme_ids = g2p_result.phoneme_ids
            if isinstance(phoneme_ids, torch.Tensor):
                phoneme_ids = phoneme_ids.detach().cpu().numpy()
            phoneme_ids = np.asarray(phoneme_ids, dtype=np.int64)
            if len(phoneme_ids) < 3:
                n_skip += 1
                continue

            # Voice state (12-D)
            waveform_t = torch.from_numpy(waveform_np).float().unsqueeze(0)
            mel = compute_mel(waveform_t).to(device)
            f0 = torch.zeros(1, 1, mel.shape[-1], device=torch.device(device))
            vs_raw = vs_estimator.estimate(mel, f0)
            if isinstance(vs_raw, torch.Tensor):
                vs = vs_raw.detach().squeeze(0).cpu().numpy()
            else:
                vs = np.zeros((n_frames, D_VOICE_STATE), dtype=np.float32)
            vs = np.clip(vs[:n_frames], 0, 1).astype(np.float32)
            if vs.shape[0] < n_frames:
                vs = np.pad(vs, ((0, n_frames - vs.shape[0]), (0, 0)), mode="edge")

            # Speaker embedding
            spk = spk_encoder.extract(waveform_t, sample_rate=SAMPLE_RATE)
            spk = spk.detach().cpu().numpy().flatten().astype(np.float32) if isinstance(spk, torch.Tensor) else np.zeros(D_SPEAKER, np.float32)

            # Codec tokens
            waveform_t_codec = torch.from_numpy(waveform_np).float().unsqueeze(0).unsqueeze(0)
            codec_tokens = codec.encode(waveform_t_codec).squeeze(0).numpy().astype(np.int64)

            # Supervision metadata
            observed_mask = np.ones((n_frames, D_VOICE_STATE), dtype=bool)
            observed_mask[:, 8:] = False
            confidence = np.ones((n_frames, D_VOICE_STATE), dtype=np.float32) * 0.8
            confidence[:, 8:] = 0.1

            # Write cache
            utt_dir.mkdir(parents=True, exist_ok=True)
            np.save(utt_dir / "codec_tokens.npy", codec_tokens)
            np.save(utt_dir / "voice_state.npy", vs)
            np.save(utt_dir / "spk_embed.npy", spk)
            np.save(utt_dir / "phoneme_ids.npy", phoneme_ids)
            np.save(utt_dir / "text_suprasegmentals.npy",
                    np.zeros((len(phoneme_ids), 4), dtype=np.float32))
            np.save(utt_dir / "voice_state_targets.npy", vs)
            np.save(utt_dir / "voice_state_observed_mask.npy", observed_mask)
            np.save(utt_dir / "voice_state_confidence.npy", confidence)

            meta = {
                "utterance_id": utt_id,
                "speaker_id": speaker,
                "corpus": entry["corpus"],
                "n_frames": int(n_frames),
                "n_codec_frames": int(n_frames),
                "n_control_frames": int(n_frames),
                "text": transcript,
                "enriched_transcript": transcript,
                "language_id": 0,
                "language": lang,
                "duration_sec": n_samples / SAMPLE_RATE,
                "quality_score": 0.85,
                "supervision_tier": "tier_b",
                "asr_confidence": float(asr_conf),
                "acting_annotations": {},
            }
            with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            # Mark cached in manifest
            idx = entry_idx[utt_id]
            entries[idx]["cached"] = True
            n_ok += 1

        except Exception as e:
            logger.warning("Failed %s: %s", utt_id, e)
            n_fail += 1

        # Progress log
        done = n_ok + n_skip + n_fail
        if done % 100 == 0 and done > 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = (len(uncached) - done) / max(rate, 0.001)
            logger.info("Progress: %d/%d (ok=%d, skip=%d, fail=%d) | %.1f/s | ETA %.0fm",
                         done, len(uncached), n_ok, n_skip, n_fail, rate, remaining / 60)

        # Periodic manifest save (every 500)
        if done % 500 == 0 and done > 0:
            save_manifest(entries)

    # Final save
    save_manifest(entries)
    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Cache build complete: %d ok, %d skipped, %d failed (%.0fs)",
                 n_ok, n_skip, n_fail, elapsed)
    logger.info("Cache dir: %s", CACHE_DIR)


# ---------------------------------------------------------------------------
# Command: status
# ---------------------------------------------------------------------------

def cmd_status(args):
    entries = load_manifest()
    if not entries:
        logger.info("No entries in manifest. Use 'add' to register corpora.")
        return

    # Per-corpus stats
    corpora = {}
    for e in entries:
        c = e["corpus"]
        if c not in corpora:
            corpora[c] = {"total": 0, "cached": 0, "speakers": set()}
        corpora[c]["total"] += 1
        corpora[c]["speakers"].add(e["speaker"])
        if e.get("cached", False):
            corpora[c]["cached"] += 1

    total = len(entries)
    cached = sum(1 for e in entries if e.get("cached", False))

    logger.info("=" * 60)
    logger.info("DATASET STATUS")
    logger.info("=" * 60)
    logger.info("  %-20s %8s %8s %8s %8s", "Corpus", "Total", "Cached", "Pending", "Speakers")
    logger.info("  " + "-" * 56)
    for name, stats in sorted(corpora.items()):
        pending = stats["total"] - stats["cached"]
        logger.info("  %-20s %8d %8d %8d %8d",
                     name, stats["total"], stats["cached"], pending, len(stats["speakers"]))
    logger.info("  " + "-" * 56)
    logger.info("  %-20s %8d %8d %8d", "TOTAL", total, cached, total - cached)
    logger.info("=" * 60)
    logger.info("Manifest: %s", MANIFEST_PATH)
    logger.info("Cache:    %s", CACHE_DIR)


# ---------------------------------------------------------------------------
# Command: remove
# ---------------------------------------------------------------------------

def cmd_remove(args):
    entries = load_manifest()
    before = len(entries)
    entries = [e for e in entries if e["corpus"] != args.name]
    after = len(entries)
    removed = before - after

    if removed == 0:
        logger.info("No entries found for corpus '%s'", args.name)
        return

    save_manifest(entries)
    logger.info("Removed %d entries for corpus '%s' from manifest", removed, args.name)
    logger.info("Cache files NOT deleted. Remove manually if needed: %s/train/%s_*", CACHE_DIR, args.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dataset management for UCLMv4")
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Register a corpus")
    p_add.add_argument("name", help="Corpus name (e.g. jvs, moe, my_voices)")
    p_add.add_argument("path", help="Path to audio directory")
    p_add.add_argument("--speaker-from-dir", action="store_true",
                       help="Use parent directory name as speaker ID")
    p_add.add_argument("--speaker", default=None,
                       help="Speaker assignment: 'single' = one speaker for all")
    p_add.add_argument("--ext", default="wav",
                       help="Audio file extensions, comma-separated (default: wav)")
    p_add.add_argument("--force", action="store_true",
                       help="Replace existing corpus entries")

    # build
    p_build = sub.add_parser("build", help="Build cache for uncached entries")
    p_build.add_argument("--device", default="auto")
    p_build.add_argument("--max", type=int, default=None,
                         help="Max entries to process")

    # status
    sub.add_parser("status", help="Show dataset status")

    # remove
    p_rm = sub.add_parser("remove", help="Remove corpus from manifest")
    p_rm.add_argument("name", help="Corpus name to remove")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args)
    elif args.command == "build":
        cmd_build(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "remove":
        cmd_remove(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
