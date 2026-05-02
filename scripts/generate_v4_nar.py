#!/usr/bin/env python3
"""NAR (parallel) generation for v4 textless TTS.

Single forward pass — all frames generated in parallel from text + speaker +
voice_state, with a_ctx=zeros (matches NAR training regime). No AR feedback,
no exposure bias.

Usage:
    .venv/bin/python scripts/generate_v4_nar.py --checkpoint <path> --output <dir>
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tmrvc-core" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-data" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-train" / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("nar")

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    N_CODEBOOKS, RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE,
    PHONEME_VOCAB_SIZE, N_ACTING_TAGS, SAMPLE_RATE,
)


def load_model(ckpt_path: str, device: str, codec_condition: str = "D"):
    from tmrvc_train.models.uclm_model import DisentangledUCLM
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt["model"]
    d_model = sd["uclm_core.layers.0.norm1.weight"].shape[0]
    n_layers = max(int(k.split(".")[2]) for k in sd if k.startswith("uclm_core.layers.")) + 1
    q_dim = sd["uclm_core.layers.0.attn.q_proj.weight"].shape[0]
    k_dim = sd["uclm_core.layers.0.attn.k_proj.weight"].shape[0]
    n_heads = 8
    for nh in [8, 12, 16, 4]:
        hd = d_model // nh
        if hd * nh == d_model and k_dim % hd == 0:
            n_heads = nh
            break
    n_codebooks_eff = 1 if codec_condition == "D" else N_CODEBOOKS
    rvq_vocab_eff = 4096 if codec_condition == "D" else RVQ_VOCAB_SIZE
    num_speakers = 256
    for k, v in sd.items():
        if "speaker_classifier" in k or "speaker_adv_classifier" in k:
            if hasattr(v, "shape") and len(v.shape) == 2:
                num_speakers = v.shape[0]
                break
    model = DisentangledUCLM(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        d_voice_state_explicit=D_VOICE_STATE, d_voice_state_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER, n_codebooks=n_codebooks_eff,
        rvq_vocab_size=rvq_vocab_eff, control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE, num_speakers=num_speakers,
        acting_tag_vocab_size=N_ACTING_TAGS, codec_condition=codec_condition,
    ).to(device)
    filtered = {k: v for k, v in sd.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    logger.info("Model loaded (step %d, %d/%d keys, d_model=%d, n_layers=%d)",
                ckpt.get("step", 0), len(filtered), len(sd), d_model, n_layers)
    return model


def build_window_mask(pos: torch.Tensor, L: int, w_left: int = 2, w_right: int = 5) -> torch.Tensor:
    """Window cross-attention mask matching training-time window."""
    device = pos.device
    B, T = pos.shape
    phon_pos = torch.arange(L, device=device).view(1, 1, 1, L)
    p = pos.view(B, 1, T, 1)
    within = (phon_pos >= (p - w_left)) & (phon_pos <= (p + w_right))
    mask = torch.zeros(B, 1, T, L, device=device, dtype=torch.float32)
    return mask.masked_fill(~within, float("-inf"))


@torch.inference_mode()
def generate_nar(
    model, phoneme_ids, speaker_embed, voice_state, device,
    duration_frames: int = None,
    pace: float = 1.0,
    temperature: float = 0.0,  # 0 = argmax
    codec_condition: str = "D",
):
    """Single-pass NAR generation. All frames predicted in parallel."""
    B = 1
    phon = phoneme_ids.unsqueeze(0).to(device) if phoneme_ids.dim() == 1 else phoneme_ids.to(device)
    L = phon.size(1)

    # Estimate duration: ~10 frames per phoneme at 75 Hz (≈ 130 ms each), adjusted by pace
    if duration_frames is None:
        duration_frames = max(int(L * 10 / pace), 32)
    T = duration_frames

    n_cb = model.n_codebooks if hasattr(model, "n_codebooks") else 1
    lang_ids = torch.zeros(B, L, dtype=torch.long, device=device)
    ssl_zeros = torch.zeros(B, T, D_VOICE_STATE_SSL, device=device)
    vs_t = voice_state.view(1, 1, -1).expand(1, T, -1).to(device)
    target_a = torch.zeros(B, n_cb, T, dtype=torch.long, device=device)
    target_b = torch.zeros(B, 4, T, dtype=torch.long, device=device)
    pos = (torch.arange(T, device=device).float() * L / T).long().clamp(max=L - 1).unsqueeze(0)
    frame_off = torch.arange(T, device=device).long().unsqueeze(0)
    ca_mask = build_window_mask(pos, L)

    t0 = time.time()
    out = model.forward_tts_pointer(
        phoneme_ids=phon, language_ids=lang_ids, pointer_state=None,
        speaker_embed=speaker_embed, explicit_state=vs_t, ssl_state=ssl_zeros,
        target_a=target_a, target_b=target_b, target_length=T,
        position_indices=pos, frame_offsets=frame_off, cross_attn_mask=ca_mask,
        acoustic_history=(target_a, target_b),  # explicit zeros
    )
    logits_a = out["logits_a"]  # [B, n_cb, T, V]
    if temperature <= 0:
        sampled = logits_a.argmax(dim=-1)
    else:
        probs = F.softmax(logits_a / temperature, dim=-1)
        flat = probs.view(-1, probs.shape[-1])
        idx = torch.multinomial(flat, 1).view(B, n_cb, T)
        sampled = idx
    gen_time = time.time() - t0

    # Decode
    if codec_condition == "D":
        from tmrvc_data.wavtokenizer_codec import WavTokenizerWrapper
        codec = WavTokenizerWrapper(device=device)
    else:
        from tmrvc_data.encodec_codec import EnCodecWrapper
        codec = EnCodecWrapper(device=device)
    audio = codec.decode(sampled).squeeze().cpu().numpy()
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.9
    duration = len(audio) / SAMPLE_RATE
    return audio, {
        "n_frames": T,
        "duration_sec": duration,
        "gen_time_sec": gen_time,
        "rtf": gen_time / max(duration, 0.001),
        "phonemes_total": L,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--codec-condition", default="D", choices=["A", "B", "C", "D"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--pace", type=float, default=1.0,
                        help="Speech pace; >1 = faster (fewer frames per phoneme)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = argmax (deterministic), >0 = stochastic sampling")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output) if args.output else (ROOT / "output" / "v4_nar")
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, device, codec_condition=args.codec_condition)

    # Speaker embed from cache
    cache_dir = ROOT / "data" / "cache" / ("v4d" if args.codec_condition == "D" else "v4") / "train"
    speaker_embed = None
    if cache_dir.exists():
        for spk_dir in sorted(cache_dir.iterdir()):
            for utt_dir in sorted(spk_dir.iterdir()):
                p = utt_dir / "spk_embed.npy"
                if p.exists():
                    speaker_embed = torch.from_numpy(np.load(p)).float().unsqueeze(0).to(device)
                    break
            if speaker_embed is not None:
                break
    if speaker_embed is None:
        speaker_embed = torch.randn(1, D_SPEAKER, device=device)
        speaker_embed = speaker_embed / speaker_embed.norm(dim=-1, keepdim=True)

    def g2p(text, lang="ja"):
        from tmrvc_data.g2p import text_to_phonemes
        r = text_to_phonemes(text, language=lang)
        ids = r.phoneme_ids
        return ids.detach() if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)

    styles = {
        "neutral":          ("本当にありがとうございます。",
                             [0.5, 0.3, 0.5, 0.35, 0.5, 0.2, 0.15, 0.5, 0.2, 0.5, 0.4, 0.1]),
        "angry":            ("いい加減にしてよ！",
                             [0.7, 0.6, 0.85, 0.7, 0.7, 0.1, 0.3, 0.6, 0.3, 0.5, 0.8, 0.2]),
        "whisper":          ("ねえ、ちょっと聞いて。",
                             [0.4, 0.15, 0.15, 0.1, 0.3, 0.8, 0.1, 0.3, 0.4, 0.5, 0.15, 0.05]),
        "sad":              ("もう会えないのかな。",
                             [0.35, 0.15, 0.25, 0.2, 0.35, 0.3, 0.25, 0.4, 0.15, 0.5, 0.25, 0.15]),
        "excited":          ("すごい！信じられない！",
                             [0.7, 0.7, 0.8, 0.5, 0.6, 0.15, 0.2, 0.7, 0.2, 0.5, 0.7, 0.1]),
        "tender":           ("大丈夫だよ、心配しないで。",
                             [0.45, 0.2, 0.3, 0.15, 0.4, 0.35, 0.1, 0.45, 0.1, 0.5, 0.25, 0.05]),
        "professional":     ("本日の会議を始めさせていただきます。",
                             [0.5, 0.25, 0.5, 0.4, 0.5, 0.1, 0.05, 0.5, 0.1, 0.5, 0.5, 0.05]),
        "creaky_dramatic":  ("それは...嘘でしょう。",
                             [0.3, 0.4, 0.35, 0.6, 0.4, 0.2, 0.4, 0.35, 0.3, 0.5, 0.3, 0.6]),
        "mesugaki":         ("ざぁ〜こ♡ ざぁ〜こ♡ お兄さん弱すぎじゃない？",
                             [0.75, 0.6, 0.65, 0.25, 0.7, 0.15, 0.1, 0.6, 0.1, 0.75, 0.55, 0.05]),
    }

    logger.info("=" * 60)
    logger.info("NAR generation: %d samples (pace=%.1f, temp=%.2f)",
                len(styles), args.pace, args.temperature)
    logger.info("=" * 60)

    results = []
    for name, (text, vs_list) in styles.items():
        phonemes = g2p(text)
        vs = torch.tensor(vs_list, dtype=torch.float32)
        logger.info("--- %s: %s → %d phonemes ---", name, text, len(phonemes))
        audio, meta = generate_nar(
            model, phonemes, speaker_embed, vs, device,
            pace=args.pace, temperature=args.temperature,
            codec_condition=args.codec_condition,
        )
        wav_path = output_dir / f"{name}.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)
        logger.info("  → %.2fs | %d frames | RTF=%.2f",
                    meta["duration_sec"], meta["n_frames"], meta["rtf"])
        results.append({"style": name, **meta})

    with open(output_dir / "generation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Done. %s", output_dir)


if __name__ == "__main__":
    main()
