#!/usr/bin/env python3
"""v4 Idempotent Bootstrap Pipeline.

Phase-based, restartable at any point. Each utterance tracks completion
state in meta.json. Re-running skips already-completed phases.

Phases:
  A: Audio ingest + ASR + G2P + Speaker embed + Voice state
  B: Codec tokenization (EnCodec)
  C: Vocal event detection (DSP, CPU-only)
  D: LLM annotation + enriched transcript (separate VRAM budget)

Usage:
    .venv/bin/python scripts/bootstrap_v4.py --phase all
    .venv/bin/python scripts/bootstrap_v4.py --phase ab      # audio + codec only
    .venv/bin/python scripts/bootstrap_v4.py --phase c       # vocal events only (CPU)
    .venv/bin/python scripts/bootstrap_v4.py --phase d       # LLM annotation only
    .venv/bin/python scripts/bootstrap_v4.py --status        # show progress
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tmrvc-core" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-data" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-train" / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("bootstrap")

from tmrvc_core.constants import D_VOICE_STATE, D_SPEAKER, SAMPLE_RATE
from tmrvc_core.acting_tags import ALL_ACTING_TAGS


CACHE_DIR = ROOT / "data" / "cache" / "v4full"
RAW_DIR = ROOT / "data" / "raw"
CODEC_HOP = 320  # EnCodec 75 Hz


def parse_args():
    p = argparse.ArgumentParser(description="v4 idempotent bootstrap")
    p.add_argument("--phase", default="all", help="Phases to run: all, ab, c, d, or any combination")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample-pct", type=float, default=100, help="Percent of raw audio to process")
    p.add_argument("--annotation-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--status", action="store_true", help="Show progress only")
    p.add_argument("--max-workers", type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def _read_meta(utt_dir: Path) -> dict:
    meta_path = utt_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return {}


def _write_meta(utt_dir: Path, meta: dict):
    utt_dir.mkdir(parents=True, exist_ok=True)
    (utt_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _phase_done(meta: dict, phase: str) -> bool:
    return meta.get("phases", {}).get(phase, {}).get("done", False)


def _mark_phase(meta: dict, phase: str) -> dict:
    if "phases" not in meta:
        meta["phases"] = {}
    meta["phases"][phase] = {"done": True, "timestamp": datetime.now().isoformat()}
    return meta


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_raw_audio(sample_pct: float) -> list[Path]:
    all_wavs = sorted(RAW_DIR.rglob("*.wav")) if RAW_DIR.exists() else []
    if not all_wavs:
        logger.error("No raw audio in %s", RAW_DIR)
        return []
    n = max(1, int(len(all_wavs) * sample_pct / 100))
    random.seed(42)
    sampled = random.sample(all_wavs, min(n, len(all_wavs)))
    logger.info("Discovered %d / %d audio files (%.0f%%)", len(sampled), len(all_wavs), sample_pct)
    return sampled


def wav_to_utt_dir(wav_path: Path, index: int) -> Path:
    speaker_id = f"spk_{wav_path.parent.name.replace(' ', '_')[:20]}"
    utt_id = f"v4full_{index:06d}"
    return CACHE_DIR / "train" / speaker_id / utt_id


# ---------------------------------------------------------------------------
# Phase A: Audio ingest + ASR + G2P + Speaker + VoiceState
# ---------------------------------------------------------------------------

def run_phase_a(wav_paths: list[Path], device: str):
    from tmrvc_data.preprocessing import load_and_resample
    from tmrvc_core.audio import compute_mel

    logger.info("=== Phase A: Audio + ASR + G2P + Speaker + VoiceState ===")

    # Load models
    from faster_whisper import WhisperModel
    compute_type = "float16" if device == "cuda" else "int8"
    whisper = WhisperModel("large-v3", device=device, compute_type=compute_type)
    logger.info("Whisper loaded on %s", device)

    from tmrvc_data.speaker import SpeakerEncoder
    spk_enc = SpeakerEncoder(device=device)

    from tmrvc_data.voice_state import VoiceStateEstimator
    vs_est = VoiceStateEstimator(device=device)

    n_done, n_skip = 0, 0
    for i, wav_path in enumerate(wav_paths):
        utt_dir = wav_to_utt_dir(wav_path, i)
        meta = _read_meta(utt_dir)

        if _phase_done(meta, "a_ingest"):
            n_skip += 1
            continue

        try:
            # Load
            waveform, _ = load_and_resample(str(wav_path), target_sr=SAMPLE_RATE)
            if waveform is None:
                continue
            w_np = waveform.squeeze().detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform).squeeze()
            n_samples = len(w_np)
            n_frames = n_samples // CODEC_HOP
            if n_frames < 20:
                continue

            # ASR
            import tempfile, soundfile as sf
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                sf.write(tmp.name, w_np, SAMPLE_RATE)
                segments, info = whisper.transcribe(tmp.name, beam_size=5, vad_filter=True)
                transcript = "".join(s.text for s in segments).strip()
                lang = info.language if hasattr(info, 'language') else "ja"
                asr_conf = info.language_probability if hasattr(info, 'language_probability') else 0.5

            if not transcript:
                continue

            # G2P
            from tmrvc_data.g2p import text_to_phonemes
            g2p_result = text_to_phonemes(transcript, language=lang)
            phoneme_ids = g2p_result.phoneme_ids
            if isinstance(phoneme_ids, torch.Tensor):
                phoneme_ids = phoneme_ids.detach().cpu().numpy()
            phoneme_ids = np.asarray(phoneme_ids, dtype=np.int64)
            if len(phoneme_ids) < 3:
                continue

            # Voice state (12-D DSP)
            w_t = torch.from_numpy(w_np).float().unsqueeze(0)
            mel = compute_mel(w_t).to(device)
            f0 = torch.zeros(1, 1, mel.shape[-1], device=torch.device(device))
            vs = vs_est.estimate(mel, f0)
            vs = vs.detach().squeeze(0).cpu().numpy() if isinstance(vs, torch.Tensor) else np.zeros((n_frames, D_VOICE_STATE), dtype=np.float32)
            vs = np.clip(vs[:n_frames], 0, 1).astype(np.float32)
            if vs.shape[0] < n_frames:
                vs = np.pad(vs, ((0, n_frames - vs.shape[0]), (0, 0)), mode='edge')

            # Speaker embed
            spk = spk_enc.extract(w_t, sample_rate=SAMPLE_RATE)
            spk = spk.detach().cpu().numpy().flatten().astype(np.float32) if isinstance(spk, torch.Tensor) else np.zeros(D_SPEAKER, np.float32)

            # Observed mask & confidence
            observed_mask = np.ones((n_frames, D_VOICE_STATE), dtype=bool)
            observed_mask[:, 8:] = False
            confidence = np.ones((n_frames, D_VOICE_STATE), dtype=np.float32) * 0.8
            confidence[:, 8:] = 0.1

            # Save
            utt_dir.mkdir(parents=True, exist_ok=True)
            np.save(utt_dir / "voice_state.npy", vs)
            np.save(utt_dir / "spk_embed.npy", spk)
            np.save(utt_dir / "phoneme_ids.npy", phoneme_ids)
            np.save(utt_dir / "text_suprasegmentals.npy", np.zeros((len(phoneme_ids), 4), dtype=np.float32))
            np.save(utt_dir / "voice_state_targets.npy", vs)
            np.save(utt_dir / "voice_state_observed_mask.npy", observed_mask)
            np.save(utt_dir / "voice_state_confidence.npy", confidence)

            # Save raw waveform path for Phase B/C
            speaker_id = utt_dir.parent.name
            meta.update({
                "utterance_id": utt_dir.name,
                "speaker_id": speaker_id,
                "source_wav": str(wav_path),
                "n_frames": int(n_frames),
                "n_samples": int(n_samples),
                "text": transcript,
                "language": lang,
                "language_id": 0,
                "asr_confidence": float(asr_conf),
                "duration_sec": n_samples / SAMPLE_RATE,
                "quality_score": 0.85,
                "supervision_tier": "tier_b",
            })
            _mark_phase(meta, "a_ingest")
            _write_meta(utt_dir, meta)
            n_done += 1

            if (n_done + n_skip) % 100 == 0:
                logger.info("Phase A: %d done, %d skipped / %d total", n_done, n_skip, len(wav_paths))

        except Exception as e:
            logger.debug("Phase A skip %s: %s", wav_path.name, e)

    logger.info("Phase A complete: %d new, %d skipped", n_done, n_skip)


# ---------------------------------------------------------------------------
# Phase B: EnCodec tokenization
# ---------------------------------------------------------------------------

def run_phase_b(device: str):
    logger.info("=== Phase B: EnCodec tokenization ===")
    from tmrvc_data.encodec_codec import EnCodecWrapper
    from tmrvc_data.preprocessing import load_and_resample

    codec = EnCodecWrapper(device=device)

    n_done, n_skip = 0, 0
    for meta_path in sorted(CACHE_DIR.rglob("meta.json")):
        utt_dir = meta_path.parent
        meta = _read_meta(utt_dir)

        if not _phase_done(meta, "a_ingest"):
            continue
        if _phase_done(meta, "b_codec"):
            n_skip += 1
            continue

        try:
            source_wav = meta.get("source_wav", "")
            if not source_wav or not Path(source_wav).exists():
                continue

            waveform, _ = load_and_resample(source_wav, target_sr=SAMPLE_RATE)
            w_np = waveform.squeeze().detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform).squeeze()

            w_t = torch.from_numpy(w_np).float().unsqueeze(0).unsqueeze(0)
            tokens = codec.encode(w_t)  # [1, 8, T]
            tokens_np = tokens.squeeze(0).numpy().astype(np.int64)

            np.save(utt_dir / "codec_tokens.npy", tokens_np)
            meta["n_codec_frames"] = int(tokens_np.shape[1])
            _mark_phase(meta, "b_codec")
            _write_meta(utt_dir, meta)
            n_done += 1

            if (n_done + n_skip) % 200 == 0:
                logger.info("Phase B: %d done, %d skipped", n_done, n_skip)

        except Exception as e:
            logger.debug("Phase B skip %s: %s", utt_dir.name, e)

    logger.info("Phase B complete: %d new, %d skipped", n_done, n_skip)


# ---------------------------------------------------------------------------
# Phase C: Vocal event detection (CPU only, no GPU needed)
# ---------------------------------------------------------------------------

def run_phase_c():
    logger.info("=== Phase C: Vocal event detection (DSP, CPU) ===")
    from tmrvc_data.vocal_event_detector import VocalEventDetector
    from tmrvc_data.preprocessing import load_and_resample

    detector = VocalEventDetector()

    n_done, n_skip = 0, 0
    for meta_path in sorted(CACHE_DIR.rglob("meta.json")):
        utt_dir = meta_path.parent
        meta = _read_meta(utt_dir)

        if not _phase_done(meta, "a_ingest"):
            continue
        if _phase_done(meta, "c_vocal_events"):
            n_skip += 1
            continue

        try:
            source_wav = meta.get("source_wav", "")
            if not source_wav or not Path(source_wav).exists():
                continue

            waveform, _ = load_and_resample(source_wav, target_sr=SAMPLE_RATE)
            w_np = waveform.squeeze().detach().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform).squeeze()

            report = detector.detect(w_np)
            events_data = [
                {"type": e.event_type, "tag": e.tag, "start": float(e.start_sec),
                 "end": float(e.end_sec), "confidence": float(e.confidence)}
                for e in report.events if e.confidence > 0.5
            ]

            meta["vocal_events"] = events_data
            meta["n_vocal_events"] = len(events_data)
            _mark_phase(meta, "c_vocal_events")
            _write_meta(utt_dir, meta)
            n_done += 1

            if (n_done + n_skip) % 500 == 0:
                logger.info("Phase C: %d done, %d skipped", n_done, n_skip)

        except Exception as e:
            logger.warning("Phase C skip %s: %s", utt_dir.name, e)

    logger.info("Phase C complete: %d new, %d skipped", n_done, n_skip)


# ---------------------------------------------------------------------------
# Phase D: LLM annotation + enriched transcript
# ---------------------------------------------------------------------------

def run_phase_d(device: str, annotation_model: str):
    import re

    logger.info("=== Phase D: LLM annotation (4-bit) ===")
    logger.info("Loading %s...", annotation_model)

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(annotation_model, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        annotation_model, trust_remote_code=True,
        quantization_config=bnb_config, device_map="auto",
    )
    model.eval()
    logger.info("LLM loaded")

    prompt_template = (
        "You are a speech acting annotator. Given the transcript and detected vocal events, "
        "produce an enriched transcript with inline tags and acting annotations.\n\n"
        "Transcript: {text}\n"
        "Detected events: {events}\n"
        "Language: {lang}\n\n"
        "Respond in JSON:\n"
        '{{"scene_summary":"...","dialogue_intent":"...","emotion_description":"...",'
        '"acting_hint":"...","enriched_transcript":"...(transcript with inline event tags)"}}'
    )

    n_done, n_skip = 0, 0
    for meta_path in sorted(CACHE_DIR.rglob("meta.json")):
        utt_dir = meta_path.parent
        meta = _read_meta(utt_dir)

        if not _phase_done(meta, "a_ingest"):
            continue
        if _phase_done(meta, "d_llm_annotation"):
            n_skip += 1
            continue

        try:
            transcript = meta.get("text", "")
            if not transcript:
                continue

            lang = meta.get("language", "ja")
            events = meta.get("vocal_events", [])
            events_str = ", ".join(e["tag"] for e in events) if events else "none"

            prompt = prompt_template.format(text=transcript, events=events_str, lang=lang)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True, top_p=0.9)

            generated = out[0][inputs["input_ids"].shape[-1]:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Parse JSON
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}

            meta["acting_annotations"] = {
                "scene_summary": data.get("scene_summary", ""),
                "dialogue_intent": data.get("dialogue_intent", ""),
                "emotion_description": data.get("emotion_description", ""),
                "acting_hint": data.get("acting_hint", ""),
            }
            enriched_text = data.get("enriched_transcript", transcript)

            # Validate enriched transcript tags against ALL_ACTING_TAGS
            import re as _re
            found_tags = _re.findall(r'\[([^\]]+)\]', enriched_text)
            if found_tags:
                valid_tag_set = set(f"[{t}]" for t in ALL_ACTING_TAGS)
                n_valid = sum(1 for t in found_tags if f"[{t}]" in valid_tag_set)
                if len(found_tags) > 0 and n_valid / len(found_tags) < 0.5:
                    logger.warning(
                        "Phase D %s: >50%% invalid tags (%d/%d), falling back to plain transcript",
                        utt_dir.name, len(found_tags) - n_valid, len(found_tags),
                    )
                    enriched_text = transcript

            meta["enriched_transcript"] = enriched_text

            # Multi-factor supervision tier assignment (A/B/C/D)
            asr_conf = meta.get("asr_confidence", 0.5)
            has_semantic = bool(data.get("emotion_description") or data.get("acting_hint"))
            has_enriched = enriched_text != transcript
            has_speaker = bool(meta.get("speaker_id"))
            # Physical coverage: 8/12 dims observed by default
            physical_coverage = 8 / D_VOICE_STATE if D_VOICE_STATE > 0 else 0.667

            if asr_conf > 0.85 and has_semantic and has_enriched and has_speaker:
                tier = "tier_a"
            elif asr_conf > 0.7 and (has_semantic or has_enriched):
                tier = "tier_b"
            elif asr_conf > 0.5:
                tier = "tier_c"
            else:
                tier = "tier_d"
            meta["supervision_tier"] = tier

            _mark_phase(meta, "d_llm_annotation")
            _write_meta(utt_dir, meta)
            n_done += 1

            if (n_done + n_skip) % 50 == 0:
                logger.info("Phase D: %d done, %d skipped", n_done, n_skip)

        except Exception as e:
            logger.warning("Phase D skip %s: %s", utt_dir.name, e)

    logger.info("Phase D complete: %d new, %d skipped", n_done, n_skip)


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def show_status():
    total = 0
    phase_counts = {"a_ingest": 0, "b_codec": 0, "c_vocal_events": 0, "d_llm_annotation": 0}
    tiers = {"tier_a": 0, "tier_b": 0, "tier_c": 0, "tier_d": 0}
    event_counts = {}

    for meta_path in CACHE_DIR.rglob("meta.json"):
        total += 1
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for phase in phase_counts:
            if _phase_done(meta, phase):
                phase_counts[phase] += 1
        tier = meta.get("supervision_tier", "tier_d")
        tiers[tier] = tiers.get(tier, 0) + 1
        for ev in meta.get("vocal_events", []):
            t = ev.get("type", "?")
            event_counts[t] = event_counts.get(t, 0) + 1

    print(f"\n=== v4 Bootstrap Status ({CACHE_DIR}) ===")
    print(f"Total utterances: {total}")
    print(f"\nPhase completion:")
    for phase, count in phase_counts.items():
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {phase:<20} {bar} {count:>6}/{total} ({pct:.0f}%)")

    print(f"\nSupervision tiers:")
    for tier in ("tier_a", "tier_b", "tier_c", "tier_d"):
        print(f"  {tier}: {tiers.get(tier, 0)}")

    if event_counts:
        print(f"\nVocal events detected:")
        for ev, count in sorted(event_counts.items(), key=lambda x: -x[1]):
            print(f"  [{ev}]: {count}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()

    if args.status:
        show_status()
        return

    phases = args.phase.lower()
    wav_paths = discover_raw_audio(args.sample_pct)
    if not wav_paths:
        return

    t0 = time.time()

    if "a" in phases or phases == "all":
        run_phase_a(wav_paths, args.device)
        _free_gpu()

    if "b" in phases or phases == "all":
        run_phase_b(args.device)
        _free_gpu()

    if "c" in phases or phases == "all":
        # Free GPU for CPU-only phase
        _free_gpu()
        run_phase_c()

    if "d" in phases or phases == "all":
        # Free everything before LLM
        _free_gpu()
        run_phase_d(args.device, args.annotation_model)
        _free_gpu()

    elapsed = time.time() - t0
    logger.info("Bootstrap finished in %.0fs", elapsed)
    show_status()


if __name__ == "__main__":
    main()
