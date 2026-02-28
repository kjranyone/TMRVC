#!/usr/bin/env python3
"""Evaluate TTS frontends from a YAML script.

Generates per-frontend audio and objective runtime metrics from the same script
entries, then writes:
- audio files per frontend
- per-entry CSV
- summary JSON
- optional A/B blind-listening package
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_core.text_utils import analyze_inline_stage_directions
from tmrvc_data.script_parser import load_script, load_script_from_string
from tmrvc_serve.tts_engine import TTSEngine

logger = logging.getLogger(__name__)


@dataclass
class EvalRow:
    entry_index: int
    speaker: str
    language: str
    frontend: str
    source_text: str
    spoken_text: str
    stage_directions: str
    audio_path: str
    speed: float
    sentence_pause_ms: int
    duration_sec: float
    samples: int
    wall_ms: float
    first_chunk_ms: float
    stream_total_ms: float
    avg_sentence_ms: float
    rtf: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate tokenizer/phoneme TTS frontends from a script.",
    )
    parser.add_argument("script", type=Path, help="YAML script file path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_tts_frontends"),
        help="Output root directory.",
    )
    parser.add_argument("--tts-checkpoint", type=Path, default=None, help="TTS checkpoint.")
    parser.add_argument("--vc-checkpoint", type=Path, default=None, help="VC checkpoint.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument(
        "--frontends",
        nargs="+",
        default=["tokenizer", "phoneme"],
        choices=["tokenizer", "phoneme"],
        help="Frontend list to evaluate.",
    )
    parser.add_argument(
        "--sentence-pause-ms",
        type=int,
        default=120,
        help="Base sentence pause for synthesize_sentences.",
    )
    parser.add_argument(
        "--chunk-duration-ms",
        type=int,
        default=100,
        help="Chunk duration for sentence streaming synthesis.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Global speed multiplier.",
    )
    parser.add_argument(
        "--auto-style",
        action="store_true",
        help="Enable per-sentence auto style inside synthesize_sentences.",
    )
    parser.add_argument(
        "--stage-blend-weight",
        type=float,
        default=0.60,
        help="Blend weight for inline stage style overlay.",
    )
    parser.add_argument(
        "--create-ab",
        action="store_true",
        help="Create A/B blind-listening package (requires exactly 2 frontends).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for A/B pairing.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _clamp_style(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _clamp_speed(v: float) -> float:
    return max(0.5, min(2.0, v))


def _blend_style(
    base: StyleParams | None,
    overlay: StyleParams | None,
    weight: float,
) -> StyleParams | None:
    if overlay is None:
        return base
    if base is None:
        return overlay

    emotion = base.emotion
    if base.emotion == "neutral" and overlay.emotion != "neutral":
        emotion = overlay.emotion

    return StyleParams(
        emotion=emotion,
        valence=_clamp_style(base.valence * (1.0 - weight) + overlay.valence * weight),
        arousal=_clamp_style(base.arousal * (1.0 - weight) + overlay.arousal * weight),
        dominance=_clamp_style(base.dominance * (1.0 - weight) + overlay.dominance * weight),
        speech_rate=_clamp_style(
            base.speech_rate * (1.0 - weight) + overlay.speech_rate * weight
        ),
        energy=_clamp_style(base.energy * (1.0 - weight) + overlay.energy * weight),
        pitch_range=_clamp_style(
            base.pitch_range * (1.0 - weight) + overlay.pitch_range * weight
        ),
        reasoning="; ".join(
            p for p in [base.reasoning, overlay.reasoning, "eval_inline_stage"] if p
        ),
    )


def _append_silence(audio: np.ndarray, leading_ms: int, trailing_ms: int) -> np.ndarray:
    lead_samples = max(0, int(SAMPLE_RATE * leading_ms / 1000))
    trail_samples = max(0, int(SAMPLE_RATE * trailing_ms / 1000))
    if lead_samples == 0 and trail_samples == 0:
        return audio.astype(np.float32, copy=False)
    parts: list[np.ndarray] = []
    if lead_samples > 0:
        parts.append(np.zeros(lead_samples, dtype=np.float32))
    parts.append(audio.astype(np.float32, copy=False))
    if trail_samples > 0:
        parts.append(np.zeros(trail_samples, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32, copy=False)


def _summary(rows: list[EvalRow]) -> dict:
    by_frontend: dict[str, list[EvalRow]] = {}
    for row in rows:
        by_frontend.setdefault(row.frontend, []).append(row)

    summary: dict[str, dict] = {}
    for frontend, items in by_frontend.items():
        count = len(items)
        total_audio = sum(i.duration_sec for i in items)
        total_wall = sum(i.wall_ms for i in items)
        mean_rtf = sum(i.rtf for i in items) / max(count, 1)
        mean_first = sum(i.first_chunk_ms for i in items) / max(count, 1)
        mean_stream_total = sum(i.stream_total_ms for i in items) / max(count, 1)
        summary[frontend] = {
            "entries": count,
            "total_audio_sec": total_audio,
            "total_wall_ms": total_wall,
            "aggregate_rtf": total_wall / max(total_audio * 1000.0, 1e-6),
            "mean_entry_rtf": mean_rtf,
            "mean_first_chunk_ms": mean_first,
            "mean_stream_total_ms": mean_stream_total,
        }
    return summary


def _load_speaker_embeddings(script_obj) -> dict[str, torch.Tensor]:
    from tmrvc_export.speaker_file import read_speaker_file

    speaker_embeds: dict[str, torch.Tensor] = {}
    for char_id, profile in script_obj.characters.items():
        if profile.speaker_file and Path(profile.speaker_file).exists():
            spk_embed, _lora, _meta, _thumb = read_speaker_file(profile.speaker_file)
            speaker_embeds[char_id] = torch.from_numpy(spk_embed).float()
        else:
            speaker_embeds[char_id] = torch.zeros(192)
    return speaker_embeds


def _load_script_obj(script_path: Path):
    """Load script with compatibility for ``entries`` key used in examples."""
    script_obj = load_script(script_path)
    if script_obj.entries:
        return script_obj

    import yaml

    with open(script_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if isinstance(data, dict) and "entries" in data and "dialogue" not in data:
        data["dialogue"] = data.pop("entries")
        text = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        script_obj = load_script_from_string(text, base_dir=script_path.parent)
    return script_obj


def _resolve_entry_style(script_obj, entry) -> StyleParams:
    if entry.style_override is not None:
        return entry.style_override
    char = script_obj.characters.get(entry.speaker)
    if char and char.default_style is not None:
        return char.default_style
    return StyleParams.neutral()


def _write_rows_csv(rows: list[EvalRow], out_csv: Path) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _create_ab_package(
    rows: list[EvalRow],
    output_dir: Path,
    frontend_a: str,
    frontend_b: str,
    seed: int,
) -> None:
    by_entry: dict[int, dict[str, EvalRow]] = {}
    for row in rows:
        by_entry.setdefault(row.entry_index, {})[row.frontend] = row

    rng = random.Random(seed)
    ab_dir = output_dir / "ab_blind"
    ab_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = ab_dir / "manifest.csv"
    answer_path = ab_dir / "answer_key.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as manifest_f, open(
        answer_path, "w", newline="", encoding="utf-8"
    ) as answer_f:
        manifest_writer = csv.writer(manifest_f)
        answer_writer = csv.writer(answer_f)
        manifest_writer.writerow(["entry_index", "speaker", "source_text", "a_path", "b_path"])
        answer_writer.writerow(
            ["entry_index", "a_frontend", "b_frontend", "a_path", "b_path", "source_text"]
        )

        for idx in sorted(by_entry):
            pair = by_entry[idx]
            if frontend_a not in pair or frontend_b not in pair:
                continue
            row_a = pair[frontend_a]
            row_b = pair[frontend_b]

            swap = rng.random() < 0.5
            left = row_b if swap else row_a
            right = row_a if swap else row_b

            a_path = ab_dir / f"{idx:04d}_A.wav"
            b_path = ab_dir / f"{idx:04d}_B.wav"
            shutil.copy2(left.audio_path, a_path)
            shutil.copy2(right.audio_path, b_path)

            manifest_writer.writerow(
                [idx, row_a.speaker, row_a.source_text, str(a_path), str(b_path)]
            )
            answer_writer.writerow(
                [
                    idx,
                    left.frontend,
                    right.frontend,
                    str(a_path),
                    str(b_path),
                    row_a.source_text,
                ]
            )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.script.exists():
        raise FileNotFoundError(f"Script file not found: {args.script}")
    if args.create_ab and len(args.frontends) != 2:
        raise ValueError("--create-ab requires exactly 2 frontends.")

    import soundfile as sf

    script_obj = _load_script_obj(args.script)
    if not script_obj.entries:
        raise ValueError("Script has no dialogue entries.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded script: %s (%d entries)", args.script, len(script_obj.entries))

    speaker_embeds = _load_speaker_embeddings(script_obj)

    engines: dict[str, TTSEngine] = {}
    for frontend in args.frontends:
        logger.info("Loading engine for frontend=%s", frontend)
        engine = TTSEngine(
            tts_checkpoint=args.tts_checkpoint,
            vc_checkpoint=args.vc_checkpoint,
            device=args.device,
            text_frontend=frontend,
        )
        engine.load_models()
        engine.warmup()
        engines[frontend] = engine

    rows: list[EvalRow] = []
    for idx, entry in enumerate(script_obj.entries, start=1):
        char = script_obj.characters.get(entry.speaker)
        language = char.language if char else "ja"
        spk_embed = speaker_embeds.get(entry.speaker, torch.zeros(192))
        base_style = _resolve_entry_style(script_obj, entry)
        stage = analyze_inline_stage_directions(entry.text, language=language)
        spoken_text = stage.spoken_text

        for frontend in args.frontends:
            engine = engines[frontend]
            style = _blend_style(base_style, stage.style_overlay, args.stage_blend_weight)
            speed = _clamp_speed(args.speed * entry.speed * stage.speed_scale)
            sentence_pause_ms = max(0, min(1600, args.sentence_pause_ms + stage.sentence_pause_ms_delta))

            t0 = time.perf_counter()
            chunks = list(engine.synthesize_sentences(
                text=spoken_text,
                language=language,
                spk_embed=spk_embed,
                style=style,
                speed=speed,
                chunk_duration_ms=args.chunk_duration_ms,
                sentence_pause_ms=sentence_pause_ms,
                auto_style=args.auto_style,
            ))
            wall_ms = (time.perf_counter() - t0) * 1000.0

            if chunks:
                audio = np.concatenate(chunks).astype(np.float32)
            else:
                audio = np.zeros(0, dtype=np.float32)
            audio = _append_silence(audio, stage.leading_silence_ms, stage.trailing_silence_ms)
            duration_sec = len(audio) / SAMPLE_RATE

            out_wav = output_dir / frontend / f"{idx:04d}_{entry.speaker}.wav"
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out_wav, audio, SAMPLE_RATE)

            stream_metrics = engine.last_stream_metrics
            first_chunk_ms = stream_metrics.first_chunk_ms if stream_metrics else 0.0
            stream_total_ms = stream_metrics.total_ms if stream_metrics else 0.0
            avg_sentence_ms = stream_metrics.avg_sentence_ms if stream_metrics else 0.0
            rtf = stream_total_ms / max(duration_sec * 1000.0, 1e-6)

            row = EvalRow(
                entry_index=idx,
                speaker=entry.speaker,
                language=language,
                frontend=frontend,
                source_text=entry.text,
                spoken_text=spoken_text,
                stage_directions=" | ".join(stage.stage_directions),
                audio_path=str(out_wav),
                speed=float(speed),
                sentence_pause_ms=int(sentence_pause_ms),
                duration_sec=float(duration_sec),
                samples=int(len(audio)),
                wall_ms=float(wall_ms),
                first_chunk_ms=float(first_chunk_ms),
                stream_total_ms=float(stream_total_ms),
                avg_sentence_ms=float(avg_sentence_ms),
                rtf=float(rtf),
            )
            rows.append(row)
            logger.info(
                "[%04d][%s] %.2fs audio, wall=%.1fms, rtf=%.3f",
                idx, frontend, duration_sec, wall_ms, rtf,
            )

    rows_csv = output_dir / "rows.csv"
    _write_rows_csv(rows, rows_csv)

    summary = {
        "script": str(args.script),
        "frontends": args.frontends,
        "sample_rate": SAMPLE_RATE,
        "entries": len(script_obj.entries),
        "summary": _summary(rows),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.create_ab:
        _create_ab_package(
            rows=rows,
            output_dir=output_dir,
            frontend_a=args.frontends[0],
            frontend_b=args.frontends[1],
            seed=args.seed,
        )

    logger.info("Wrote evaluation artifacts to %s", output_dir)


if __name__ == "__main__":
    main()
