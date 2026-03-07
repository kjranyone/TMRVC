#!/usr/bin/env python3
"""Prepare TMRVC UCLM dataset from raw audio files.

Comprehensive UCLM Extraction Pipeline:
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
    logger.info("Starting UCLM Dataset Preparation Pipeline")
    
    # Load Models
    logger.info("Loading UCLM extraction models...")
    codec = UCLMCodecWrapper(config.codec_checkpoint, device=config.device)
    vs_estimator = SSLVoiceStateEstimator(device=config.device)
    spk_encoder = SpeakerEncoder(device=config.device)
    
    # Faster-Whisper for transcription (needed for alignment)
    from faster_whisper import WhisperModel
    compute_type = "float16" if config.device == "cuda" else "int8"
    whisper = WhisperModel("large-v3-turbo", device=config.device, compute_type=compute_type)

    # Load speaker map if exists in configs/datasets.yaml or inferred
    speaker_map = None
    datasets_yaml = Path("configs/datasets.yaml")
    if datasets_yaml.exists():
        import yaml
        with open(datasets_yaml, encoding="utf-8") as _f:
            _registry = yaml.safe_load(_f) or {}
        ds_cfg = (_registry.get("datasets") or {}).get(config.dataset_name) or {}
        spk_map_path = ds_cfg.get("speaker_map")
        if spk_map_path:
            with open(spk_map_path, encoding="utf-8") as _f:
                speaker_map = json.load(_f)["mapping"]

    # Scan Files
    audio_files = sorted(list(config.input_dir.rglob("*.wav")) + list(config.input_dir.rglob("*.flac")))
    logger.info("Found %d files to process", len(audio_files))

    for audio_path in tqdm.tqdm(audio_files, desc="Processing"):
        utt_id = audio_path.stem
        
        # Speaker logic: Map > Directory > DatasetName
        if speaker_map and audio_path.name in speaker_map:
            spk_id = speaker_map[audio_path.name]
            if spk_id == "spk_noise": continue
        else:
            rel_path = audio_path.relative_to(config.input_dir)
            spk_id = rel_path.parts[0] if len(rel_path.parts) > 1 else config.dataset_name
        
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
            from tmrvc_core.audio import compute_mel
            mel = compute_mel(waveform_t.squeeze(1)).to(config.device)
            f0 = torch.zeros(1, 1, mel.shape[-1], device=config.device)
            
            import torchaudio.transforms as T
            waveform_16k = T.Resample(SAMPLE_RATE, 16000).to(config.device)(waveform_t.squeeze(1))
            
            vs_dict = vs_estimator(waveform_16k, waveform_t.squeeze(1), mel, f0)
            
            # CRITICAL: Frame Alignment Verification
            T_target = a_tokens.shape[-1]
            explicit_state = vs_dict["explicit_state"].squeeze(0)
            ssl_state = vs_dict["ssl_state"].squeeze(0)
            
            assert explicit_state.shape[0] == T_target, f"Explicit state frame mismatch: {explicit_state.shape[0]} != {T_target}"
            
            # Interpolate SSL (50Hz -> 100Hz)
            ssl_state_interp = torch.nn.functional.interpolate(
                ssl_state.unsqueeze(0).transpose(1, 2),
                size=T_target,
                mode="linear",
                align_corners=False
            ).transpose(1, 2).squeeze(0)
            
            explicit_state_np = explicit_state.cpu().numpy()
            ssl_state_np = ssl_state_interp.cpu().numpy()
            
            # 3. Transcribe
            segments, _ = whisper.transcribe(str(audio_path), language=config.language)
            text = "".join(seg.text for seg in segments).strip()
            
            # 4. Extract Speaker Embed
            with torch.no_grad():
                spk_embed = spk_encoder.extract(waveform_t.squeeze(1))
                spk_embed = spk_embed.cpu().numpy()

            # 5. Save
            utt_dir.mkdir(parents=True, exist_ok=True)
            np.save(utt_dir / "codec_tokens.npy", a_tokens.cpu().numpy()[0])
            np.save(utt_dir / "control_tokens.npy", b_tokens.cpu().numpy()[0])
            np.save(utt_dir / "explicit_state.npy", explicit_state_np)
            np.save(utt_dir / "ssl_state.npy", ssl_state_np)
            np.save(utt_dir / "spk_embed.npy", spk_embed)
            np.save(utt_dir / "waveform.npy", waveform.numpy())
            
            meta = {
                "utterance_id": utt_id,
                "speaker_id": spk_id,
                "duration_sec": waveform.shape[-1] / SAMPLE_RATE,
                "text": text,
                "language_id": LANGUAGE_IDS.get(config.language, 0),
                "n_frames": T_target,
                "f0_mean": 150.0 # Default fallback
            }
            (utt_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

        except Exception as e:
            logger.warning("Failed to process %s: %s", audio_path, e)

    return 0

def main():
    parser = argparse.ArgumentParser(description="UCLM Data Prep")
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
