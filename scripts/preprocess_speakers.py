#!/usr/bin/env python3
"""Preprocess specific speakers from VCTK dataset."""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import tqdm
import soundfile as sf
import torchaudio.transforms as T
from faster_whisper import WhisperModel

sys.path.insert(0, "tmrvc-core/src")
sys.path.insert(0, "tmrvc-data/src")

from tmrvc_data.dataset_adapters import VCTKAdapter
from tmrvc_data.preprocessing import preprocess_audio
from tmrvc_data.codec import UCLMCodecWrapper
from tmrvc_data.voice_state import SSLVoiceStateEstimator
from tmrvc_data.speaker import SpeakerEncoder
from tmrvc_core.audio import compute_mel
from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.types import UCLMFeatureSet
from tmrvc_data.cache import FeatureCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker-list", required=True, type=Path)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Load speaker list
    speakers = set(s.strip() for s in args.speaker_list.read_text().strip().split("\n"))
    print(f"Worker {args.worker_id}: Processing {len(speakers)} speakers")

    # Load models
    print(f"Worker {args.worker_id}: Loading models...")
    codec = UCLMCodecWrapper(None, device=args.device)
    vs_estimator = SSLVoiceStateEstimator(device=args.device)
    spk_encoder = SpeakerEncoder(device=args.device)
    whisper = WhisperModel("large-v3-turbo", device=args.device, compute_type="float16")
    cache = FeatureCache(Path("data/cache"))

    # Get utterances
    adapter = VCTKAdapter()
    all_utterances = list(adapter.iter_utterances(Path("data/raw"), "train"))
    utterances = [
        u for u in all_utterances if u.speaker_id.replace("vctk_", "") in speakers
    ]
    print(f"Worker {args.worker_id}: {len(utterances)} utterances to process")

    processed = 0
    errors = 0

    for utt in tqdm.tqdm(utterances, desc=f"Worker {args.worker_id}"):
        try:
            # Duration check
            info = sf.info(str(utt.audio_path))
            if info.duration < 0.1 or info.duration > 30.0:
                continue

            # Load audio
            waveform, sr = preprocess_audio(str(utt.audio_path), target_sr=SAMPLE_RATE)
            waveform_t = waveform.unsqueeze(0).to(args.device)

            # Extract codec tokens
            a_tokens, b_logits = codec.encode(waveform_t)
            b_tokens = b_logits.argmax(dim=-1)

            # Extract voice state
            mel = compute_mel(waveform_t.squeeze(1)).to(args.device)
            f0 = torch.zeros(1, 1, mel.shape[-1], device=args.device)
            waveform_16k = T.Resample(SAMPLE_RATE, 16000).to(args.device)(
                waveform_t.squeeze(1)
            )
            vs_dict = vs_estimator(waveform_16k, waveform_t.squeeze(1), mel, f0)

            # Transcribe
            segments, _ = whisper.transcribe(str(utt.audio_path), language="en")
            text = "".join(seg.text for seg in segments).strip()

            # Speaker embedding
            spk_embed = spk_encoder.extract(waveform_t.squeeze(1))

            # Frame alignment with assertions
            T_target = a_tokens.shape[-1]
            T_mel = mel.shape[-1]
            assert T_mel == T_target, f"Frame mismatch: mel={T_mel}, codec={T_target}"

            # Explicit state
            explicit_state = vs_dict["explicit_state"].detach().cpu()
            if explicit_state.dim() == 3:
                explicit_state = explicit_state.squeeze(0)
            T_explicit = explicit_state.shape[0]
            assert T_explicit == T_target, (
                f"Frame mismatch: explicit={T_explicit}, codec={T_target}"
            )
            explicit_state = explicit_state.transpose(0, 1)

            # SSL state with interpolation
            ssl_state = vs_dict["ssl_state"].detach().cpu()
            if ssl_state.dim() == 3:
                ssl_state = ssl_state.squeeze(0)
            ssl_state = (
                torch.nn.functional.interpolate(
                    ssl_state.unsqueeze(0).transpose(1, 2),
                    size=T_target,
                    mode="linear",
                    align_corners=False,
                )
                .transpose(1, 2)
                .squeeze(0)
                .transpose(0, 1)
            )

            # B tokens alignment
            b_tokens_aligned = b_tokens.detach().cpu().squeeze(0)
            T_b = b_tokens_aligned.shape[-1]
            assert T_b == T_target, f"Frame mismatch: b={T_b}, codec={T_target}"

            # Save
            features = UCLMFeatureSet(
                codec_tokens_a=a_tokens.detach().cpu().squeeze(0),
                codec_tokens_b=b_tokens_aligned,
                voice_state_explicit=explicit_state,
                voice_state_ssl=ssl_state,
                spk_embed=spk_embed.detach().cpu().squeeze(0),
                text=text,
                utterance_id=utt.utterance_id,
                speaker_id=utt.speaker_id,
                n_frames=T_target,
                waveform=waveform.detach(),
            )
            cache.save(features, "vctk", "train")
            processed += 1

        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"Error {utt.utterance_id}: {e}")
            continue

    print(f"Worker {args.worker_id}: Done. Processed={processed}, Errors={errors}")


if __name__ == "__main__":
    main()
