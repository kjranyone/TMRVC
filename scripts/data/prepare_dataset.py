#!/usr/bin/env python3
"""Prepare TMRVC UCLM v2 dataset from raw audio files.

Comprehensive UCLM v2 Extraction Pipeline:
1. Scan & Filter
2. Normalize (24kHz LUFS)
3. Extract Dual-Stream Tokens (A_t, B_t)
4. Extract Voice State (8-dim Physical + 128-dim SSL)
5. Extract TTS Alignment (Phonemes + Durations via Forced Alignment)
6. Save structured cache (.npy + meta.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_data.codec import UCLMCodecWrapper
from tmrvc_data.voice_state import SSLVoiceStateEstimator
from tmrvc_data.speaker import SpeakerEncoder
from tmrvc_data.alignment import extract_alignment # NEW: For TTS data

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    dataset_name: str
    language: str = "ja"
    codec_checkpoint: Path | None = None
    device: str = "cuda"
    resume: bool = False
    dry_run: bool = False

@dataclass
class UtteranceMeta:
    utterance_id: str
    speaker_id: str
    duration_sec: float
    text: str = ""
    language_id: int = 0
    voice_state_mean: list[float] = field(default_factory=list)
    has_alignment: bool = False

LANGUAGE_IDS = {"ja": 0, "en": 1, "zh": 2, "ko": 3}

def run_pipeline(config: PipelineConfig) -> int:
    logger.info("Starting UCLM v2 Dataset Preparation Pipeline")
    
    # Load Models
    logger.info("Loading UCLM extraction models...")
    codec = UCLMCodecWrapper(config.codec_checkpoint, device=config.device)
    vs_estimator = SSLVoiceStateEstimator(device=config.device)
    spk_encoder = SpeakerEncoder(device=config.device)
    
    # Faster-Whisper for transcription (needed for alignment)
    from faster_whisper import WhisperModel
    compute_type = "float16" if config.device == "cuda" else "int8"
    whisper = WhisperModel("large-v3", device=config.device, compute_type=compute_type)

    # Scan Files
    audio_files = sorted(list(config.input_dir.rglob("*.wav")) + list(config.input_dir.rglob("*.flac")))
    logger.info("Found %d files to process", len(audio_files))

    for audio_path in tqdm.tqdm(audio_files, desc="Processing"):
        utt_id = audio_path.stem
        # Improved speaker logic: Use the top-level directory name relative to input_dir
        rel_path = audio_path.relative_to(config.input_dir)
        if len(rel_path.parts) > 1:
            spk_id = rel_path.parts[0]
        else:
            spk_id = config.dataset_name # Fallback to dataset name for flat files
        
        utt_dir = config.output_dir / config.dataset_name / "train" / spk_id / utt_id
        if config.resume and (utt_dir / "meta.json").exists():
            continue
            
        try:
            from tmrvc_data.preprocessing import load_and_resample
            waveform, sr = load_and_resample(str(audio_path), target_sr=SAMPLE_RATE)
            waveform_t = waveform.unsqueeze(0).to(config.device) # [1, 1, T]
            
            # 1. Extract Tokens (A_t, B_t)
            a_tokens, b_logits = codec.encode(waveform_t)
            b_tokens = b_logits.argmax(dim=-1)
            
            # 2. Extract Voice State (8-dim + SSL)
            # Need mel and f0 for estimator
            from tmrvc_core.audio import compute_mel
            mel = compute_mel(waveform_t.squeeze(1)).to(config.device)
            f0 = torch.zeros(1, 1, mel.shape[-1], device=config.device) # Placeholder
            
            # WavLM resample
            import torchaudio.transforms as T
            waveform_16k = T.Resample(SAMPLE_RATE, 16000).to(config.device)(waveform_t.squeeze(1))
            
            vs_dict = vs_estimator(waveform_16k, waveform_t.squeeze(1), mel, f0)
            explicit_state = vs_dict["explicit_state"].cpu().numpy()[0]
            ssl_state = vs_dict["ssl_state"].cpu().numpy()[0]
            
            # 3. Transcribe & Align (for TTS)
            segments, _ = whisper.transcribe(str(audio_path), language=config.language)
            text = "".join(seg.text for seg in segments).strip()
            
            phoneme_ids = None
            durations = None
            if text:
                try:
                    # Forced Alignment logic
                    align_res = extract_alignment(str(audio_path), text, language=config.language)
                    phoneme_ids = align_res["phoneme_ids"]
                    durations = align_res["durations"]
                except Exception as e:
                    logger.debug("Alignment failed for %s: %s", utt_id, e)

            # 4. Extract Speaker Embed
            with torch.no_grad():
                spk_embed = spk_encoder.extract(waveform_t.squeeze(1))
                spk_embed = spk_embed.cpu().numpy()

            # 5. Save
            utt_dir.mkdir(parents=True, exist_ok=True)
            np.save(utt_dir / "codec_tokens.npy", a_tokens.cpu().numpy()[0])
            np.save(utt_dir / "control_tokens.npy", b_tokens.cpu().numpy()[0])
            np.save(utt_dir / "explicit_state.npy", explicit_state)
            np.save(utt_dir / "ssl_state.npy", ssl_state)
            np.save(utt_dir / "spk_embed.npy", spk_embed)
            if phoneme_ids is not None:
                np.save(utt_dir / "phoneme_ids.npy", np.array(phoneme_ids))
                np.save(utt_dir / "durations.npy", np.array(durations))
            
            meta = UtteranceMeta(
                utterance_id=utt_id,
                speaker_id=spk_id,
                duration_sec=waveform.shape[-1] / SAMPLE_RATE,
                text=text,
                language_id=LANGUAGE_IDS.get(config.language, 0),
                voice_state_mean=explicit_state.mean(axis=0).tolist(),
                has_alignment=phoneme_ids is not None
            )
            (utt_dir / "meta.json").write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2))

        except Exception as e:
            logger.warning("Failed to process %s: %s", audio_path, e)

    return 0

def main():
    parser = argparse.ArgumentParser(description="UCLM v2 Data Prep")
    parser.add_argument("--input", "-i", type=Path, required=True)
    parser.add_argument("--output", "-o", type=Path, default=Path("data/cache"))
    parser.add_argument("--name", "-n", required=True)
    parser.add_argument("--language", "-l", default="ja")
    parser.add_argument("--codec-checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    config = PipelineConfig(
        input_dir=args.input,
        output_dir=args.output,
        dataset_name=args.name,
        language=args.language,
        codec_checkpoint=args.codec_checkpoint,
        device=args.device,
        resume=args.resume
    )
    return run_pipeline(config)

if __name__ == "__main__":
    sys.exit(main())
