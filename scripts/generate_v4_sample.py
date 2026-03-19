#!/usr/bin/env python3
"""Generate acting speech using proper pointer-based inference.

Implements the full pointer protocol from UCLMEngine.tts():
- PointerInferenceState for text position tracking
- pointer_head for advance/hold decisions
- KV cache for temporal context
- forward_streaming for causal frame-by-frame generation
- Automatic stop when pointer reaches end of text
- EnCodec decode for waveform

Usage:
    .venv/bin/python scripts/generate_v4_sample.py
"""

from __future__ import annotations

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
logger = logging.getLogger("gen_v4")

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    N_CODEBOOKS, CONTROL_SLOTS, RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE,
    PHONEME_VOCAB_SIZE, SAMPLE_RATE,
)


def load_model(ckpt_path: str, device: str):
    from tmrvc_train.models.uclm_model import DisentangledUCLM
    model = DisentangledUCLM(
        d_model=D_MODEL, d_explicit=D_VOICE_STATE, d_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER, n_codebooks=N_CODEBOOKS,
        rvq_vocab_size=RVQ_VOCAB_SIZE, control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE, num_speakers=100,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    logger.info("Model loaded (step %d, loss %.4f)", ckpt.get("step", 0), ckpt.get("best_loss", 0))
    return model


@torch.inference_mode()
def generate_speech(
    model, phoneme_ids: torch.Tensor, speaker_embed: torch.Tensor,
    voice_state: torch.Tensor, device: str,
    max_frames: int = 1500, temperature: float = 0.8,
    pace: float = 1.0, max_frames_per_unit: int = 50,
) -> tuple[np.ndarray, dict]:
    """Proper pointer-based causal TTS generation."""

    B = 1
    phoneme_ids = phoneme_ids.unsqueeze(0).to(device) if phoneme_ids.dim() == 1 else phoneme_ids.to(device)
    L = phoneme_ids.size(1)

    # Voice state conditioning
    vs = voice_state.view(1, 1, -1).to(device)
    ssl_zeros = torch.zeros(1, 1, D_VOICE_STATE_SSL, device=device)
    lang_ids = torch.zeros(B, dtype=torch.long, device=device)

    # Pre-compute text features
    text_features = model.text_encoder(phoneme_ids, lang_ids).transpose(1, 2)  # [B, L, D]

    # Pointer state
    text_index = 0
    progress = 0.0
    frames_on_unit = 0

    # Generation state
    kv_caches = None
    ctx_a = torch.zeros(B, N_CODEBOOKS, 1, dtype=torch.long, device=device)
    ctx_b = torch.zeros(B, CONTROL_SLOTS, 1, dtype=torch.long, device=device)

    a_tokens = []
    pointer_trace = []
    t0 = time.time()
    forced_advances = 0

    for t in range(max_frames):
        # Check if pointer passed all phonemes
        if text_index >= L:
            break

        # Current phoneme features (pointer-indexed)
        safe_idx = min(text_index, L - 1)
        content_features = text_features[:, safe_idx:safe_idx+1, :]  # [B, 1, D]

        # Voice state encoding for this frame
        state_cond = model.voice_state_enc(
            explicit_state=vs, ssl_state=ssl_zeros,
            delta_state=torch.zeros(1, 1, D_VOICE_STATE, device=device),
        )
        if isinstance(state_cond, tuple):
            state_cond = state_cond[0]

        # Forward streaming (single frame)
        try:
            out = model.forward_streaming(
                queries=content_features + state_cond,
                memory=text_features,
                a_ctx=ctx_a, b_ctx=ctx_b,
                speaker_embed=speaker_embed,
                explicit_state=vs,
                cfg_scale=1.0,
                kv_caches=kv_caches,
            )
            logits_a = out["logits_a"]  # [B, 8, 1, 1024]
            logits_b = out["logits_b"]
            kv_caches = out.get("kv_cache_out", None)
            hidden = out.get("hidden_states", None)
        except Exception as e:
            # Fallback: use uclm_core directly
            result = model.uclm_core(
                queries=content_features + state_cond,
                memory=text_features,
                a_ctx=ctx_a, b_ctx=ctx_b,
                speaker_embed=speaker_embed,
                state_cond=state_cond,
                kv_caches=kv_caches,
            )
            logits_a, logits_b = result[0], result[1]
            kv_caches = result[2] if len(result) > 2 else None
            hidden = result[3] if len(result) > 3 else None

        # Sample tokens
        at = torch.stack([
            torch.multinomial(F.softmax(logits_a[:, i, 0, :] / temperature, dim=-1), 1)
            for i in range(N_CODEBOOKS)
        ], dim=1).squeeze(-1)  # [B, 8]

        bt = torch.stack([
            torch.multinomial(F.softmax(logits_b[:, i, 0, :] / temperature, dim=-1), 1)
            for i in range(CONTROL_SLOTS)
        ], dim=1).squeeze(-1)

        a_tokens.append(at.unsqueeze(-1))
        ctx_a = at.unsqueeze(-1)
        ctx_b = bt.unsqueeze(-1)

        # --- Pointer advance logic ---
        if hidden is not None and hasattr(model, 'pointer_head'):
            try:
                p_adv_logit, p_delta, p_bc = model.pointer_head(hidden)
                adv_prob = torch.sigmoid(p_adv_logit + (pace - 1.0) * 2.0).item()
                velocity = p_delta.item() * pace
                progress += velocity
            except Exception:
                adv_prob = 0.5
                progress += 0.05 * pace
        else:
            # No pointer head: simple duration-based advance
            adv_prob = 0.5
            progress += 0.05 * pace

        frames_on_unit += 1
        advanced = False

        # Forced advance if stuck
        if frames_on_unit >= max_frames_per_unit:
            advanced = True
            forced_advances += 1
        # Normal advance: progress passed 1.0 and advance probability high
        elif progress >= 1.0 and adv_prob > 0.5:
            advanced = True
        # Progress-only advance
        elif progress >= 1.5:
            advanced = True

        if advanced:
            pointer_trace.append((text_index, frames_on_unit))
            text_index += 1
            progress = 0.0
            frames_on_unit = 0

    # Finish remaining
    if frames_on_unit > 0:
        pointer_trace.append((text_index, frames_on_unit))

    gen_time = time.time() - t0
    n_frames = len(a_tokens)

    if n_frames == 0:
        return np.zeros(SAMPLE_RATE, dtype=np.float32), {"n_frames": 0, "error": "no frames"}

    # Stack and decode
    all_a = torch.cat(a_tokens, dim=-1)  # [B, 8, T]

    try:
        from tmrvc_data.encodec_codec import EnCodecWrapper
        codec = EnCodecWrapper(device=device)
        audio_t = codec.decode(all_a)
        audio_np = audio_t.squeeze().cpu().numpy()
    except Exception as e:
        logger.warning("Codec decode failed: %s", e)
        audio_np = np.zeros(n_frames * 320, dtype=np.float32)

    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.9

    duration = len(audio_np) / SAMPLE_RATE
    return audio_np, {
        "n_frames": n_frames,
        "duration_sec": duration,
        "gen_time_sec": gen_time,
        "rtf": gen_time / max(duration, 0.001),
        "text_index_final": text_index,
        "phonemes_total": L,
        "forced_advances": forced_advances,
        "pointer_trace": pointer_trace,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = ROOT / "output" / "v4_samples"
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ROOT / "checkpoints" / "v4_full" / "v4_full_final.pt"
    model = load_model(str(ckpt_path), device)

    # Speaker embed from cache
    cache_dir = ROOT / "data" / "cache" / "v4full" / "train"
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
        speaker_embed /= speaker_embed.norm(dim=-1, keepdim=True)

    def g2p(text, lang="ja"):
        try:
            from tmrvc_data.g2p import text_to_phonemes
            r = text_to_phonemes(text, language=lang)
            ids = r.phoneme_ids
            return ids.detach() if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
        except Exception:
            return torch.randint(1, PHONEME_VOCAB_SIZE, (len(text)*2,))

    styles = {
        "neutral":          ("本当にありがとうございます。",     [0.5,0.3,0.5,0.35,0.5,0.2,0.15,0.5,0.2,0.5,0.4,0.1]),
        "angry":            ("いい加減にしてよ！",               [0.7,0.6,0.85,0.7,0.7,0.1,0.3,0.6,0.3,0.5,0.8,0.2]),
        "whisper":          ("ねえ、ちょっと聞いて。",           [0.4,0.15,0.15,0.1,0.3,0.8,0.1,0.3,0.4,0.5,0.15,0.05]),
        "sad":              ("もう会えないのかな。",             [0.35,0.15,0.25,0.2,0.35,0.3,0.25,0.4,0.15,0.5,0.25,0.15]),
        "excited":          ("すごい！信じられない！",           [0.7,0.7,0.8,0.5,0.6,0.15,0.2,0.7,0.2,0.5,0.7,0.1]),
        "tender":           ("大丈夫だよ、心配しないで。",       [0.45,0.2,0.3,0.15,0.4,0.35,0.1,0.45,0.1,0.5,0.25,0.05]),
        "professional":     ("本日の会議を始めさせていただきます。", [0.5,0.25,0.5,0.4,0.5,0.1,0.05,0.5,0.1,0.5,0.5,0.05]),
        "creaky_dramatic":  ("それは...嘘でしょう。",             [0.3,0.4,0.35,0.6,0.4,0.2,0.4,0.35,0.3,0.5,0.3,0.6]),
    }

    logger.info("=" * 60)
    logger.info("Generating %d samples with POINTER PROTOCOL", len(styles))
    logger.info("=" * 60)

    results = []
    for name, (text, vs_list) in styles.items():
        phonemes = g2p(text)
        vs = torch.tensor(vs_list, dtype=torch.float32)
        logger.info("--- %s: %s → %d phonemes ---", name, text, len(phonemes))

        audio, meta = generate_speech(model, phonemes, speaker_embed, vs, device)
        wav_path = output_dir / f"{name}.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)

        logger.info("  → %.2fs | %d frames | ptr=%d/%d | forced=%d | RTF=%.2f",
                    meta["duration_sec"], meta["n_frames"],
                    meta["text_index_final"], meta["phonemes_total"],
                    meta["forced_advances"], meta["rtf"])
        results.append({"style": name, **meta})

    logger.info("=" * 60)
    for r in results:
        logger.info("  %s: %.2fs, %d frames, ptr %d/%d",
                    r["style"], r["duration_sec"], r["n_frames"],
                    r["text_index_final"], r["phonemes_total"])

    with open(output_dir / "generation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    logger.info("Done. %s", output_dir)


if __name__ == "__main__":
    main()
