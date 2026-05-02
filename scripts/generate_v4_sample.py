#!/usr/bin/env python3
"""Generate acting speech using proper pointer-based inference.

Implements the full pointer protocol from UCLMEngine.tts():
- PointerInferenceState for text position tracking
- pointer_head for advance/hold decisions
- KV cache for temporal context
- forward_streaming for causal frame-by-frame generation
- Automatic stop when pointer reaches end of text
- EnCodec decode for waveform
- v4: acting_texture_latent conditioning via ActingMacroProjector

Usage:
    .venv/bin/python scripts/generate_v4_sample.py
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
logger = logging.getLogger("gen_v4")

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    D_ACTING_LATENT, D_ACTING_MACRO,
    N_CODEBOOKS, CONTROL_SLOTS, RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE,
    PHONEME_VOCAB_SIZE, SAMPLE_RATE,
)


def _infer_hparams(state_dict: dict) -> dict:
    """Infer model hyperparameters from checkpoint weight shapes."""
    d_model = state_dict["uclm_core.layers.0.norm1.weight"].shape[0]
    n_layers = max(int(k.split(".")[2]) for k in state_dict if k.startswith("uclm_core.layers.")) + 1
    q_dim = state_dict["uclm_core.layers.0.attn.q_proj.weight"].shape[0]
    k_dim = state_dict["uclm_core.layers.0.attn.k_proj.weight"].shape[0]
    for n_h in [8, 12, 16, 4]:
        hd = d_model // n_h
        if hd * n_h == d_model and k_dim % hd == 0:
            n_heads = n_h
            break
    else:
        n_heads = 8
    return {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads}


def load_model(ckpt_path: str, device: str, codec_condition: str = "A"):
    from tmrvc_train.models.uclm_model import DisentangledUCLM
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = _infer_hparams(ckpt["model"])
    logger.info("Checkpoint hparams: d_model=%d, n_layers=%d, n_heads=%d",
                hp["d_model"], hp["n_layers"], hp["n_heads"])
    # Condition D uses single codebook with larger vocab
    n_codebooks_eff = 1 if codec_condition == "D" else N_CODEBOOKS
    rvq_vocab_eff = 4096 if codec_condition == "D" else RVQ_VOCAB_SIZE
    # Infer num_speakers from checkpoint
    num_speakers = 256
    for k, v in ckpt["model"].items():
        if k.endswith("speaker_adv_classifier.weight") or "speaker_classifier" in k:
            if hasattr(v, "shape") and len(v.shape) == 2:
                num_speakers = v.shape[0]
                break
    model = DisentangledUCLM(
        d_model=hp["d_model"], n_heads=hp["n_heads"], n_layers=hp["n_layers"],
        d_voice_state_explicit=D_VOICE_STATE,
        d_voice_state_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER, n_codebooks=n_codebooks_eff,
        rvq_vocab_size=rvq_vocab_eff, control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE, num_speakers=num_speakers,
        codec_condition=codec_condition,
    ).to(device)
    model_sd = model.state_dict()
    ckpt_sd = ckpt["model"]
    filtered = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    skipped = [k for k in ckpt_sd if k in model_sd and model_sd[k].shape != ckpt_sd[k].shape]
    model.load_state_dict(filtered, strict=False)
    if skipped:
        logger.warning("Skipped %d keys with shape mismatch: %s", len(skipped), skipped[:5])
    model.eval()
    logger.info("Model loaded (step %d, loss %.4f, %d/%d keys)",
                ckpt.get("step", 0), ckpt.get("best_loss", 0), len(filtered), len(ckpt_sd))
    return model


@torch.inference_mode()
def generate_speech(
    model, phoneme_ids: torch.Tensor, speaker_embed: torch.Tensor,
    voice_state: torch.Tensor, device: str,
    acting_texture_latent: torch.Tensor | None = None,
    max_frames: int = 1500, temperature: float = 0.8,
    pace: float = 1.0, max_frames_per_unit: int = 50,
    codec_condition: str = "A",
) -> tuple[np.ndarray, dict]:
    """Proper pointer-based causal TTS generation with v4 acting latent."""

    B = 1
    phoneme_ids = phoneme_ids.unsqueeze(0).to(device) if phoneme_ids.dim() == 1 else phoneme_ids.to(device)
    L = phoneme_ids.size(1)

    # Voice state conditioning
    vs = voice_state.view(1, 1, -1).to(device)
    ssl_zeros = torch.zeros(1, 1, D_VOICE_STATE_SSL, device=device)
    lang_ids = torch.zeros(B, dtype=torch.long, device=device)

    # Acting texture latent [B, D_ACTING_LATENT]
    act_latent = None
    if acting_texture_latent is not None:
        act_latent = acting_texture_latent.view(1, D_ACTING_LATENT).to(device)

    # Pre-compute text features
    text_features = model.text_encoder(phoneme_ids, lang_ids).transpose(1, 2)  # [B, L, D]

    # Pointer state
    text_index = 0
    progress = 0.0
    frames_on_unit = 0

    # Generation state — use model's actual n_codebooks (may be 1 for Condition D)
    n_cb = model.n_codebooks if hasattr(model, 'n_codebooks') else N_CODEBOOKS
    kv_caches = None
    ctx_a = torch.zeros(B, n_cb, 1, dtype=torch.long, device=device)
    ctx_b = torch.zeros(B, CONTROL_SLOTS, 1, dtype=torch.long, device=device)

    # Estimate total frames (still used for logging / progress bounds).
    estimated_total_frames = int(L * 10)

    # Window for pointer-centered cross-attention (matches training-time mask).
    W_LEFT, W_RIGHT = 2, 5

    a_tokens = []
    pointer_trace = []
    t0 = time.time()
    forced_advances = 0

    for t in range(max_frames):
        # Check if pointer passed all phonemes
        if text_index >= L:
            break

        # TEXTLESS queries: built from a learned slot + pointer scalars, NEVER
        # from phoneme features. Text reaches acoustics only via cross-attention.
        idx_t = torch.tensor([[min(text_index, L - 1)]], dtype=torch.long, device=device)
        content_features = model._build_frame_queries(
            batch_size=1,
            target_length=1,
            phoneme_count=L,
            idx=idx_t,
            device=device,
            dtype=text_features.dtype,
            pointer_state=None,
        )

        # Per-frame cross-attention window mask: only phonemes in
        # [text_index - W_LEFT, text_index + W_RIGHT] are visible.
        phon_positions = torch.arange(L, device=device)
        w_start = max(0, text_index - W_LEFT)
        w_end = min(L, text_index + W_RIGHT + 1)
        within = (phon_positions >= w_start) & (phon_positions < w_end)
        ca_mask = torch.zeros(1, 1, 1, L, device=device, dtype=torch.float32)
        ca_mask = ca_mask.masked_fill(~within.view(1, 1, 1, L), float("-inf"))

        # Forward streaming (single frame)
        try:
            out = model.forward_streaming(
                queries=content_features,
                memory=text_features,
                a_ctx=ctx_a, b_ctx=ctx_b,
                speaker_embed=speaker_embed,
                explicit_state=vs,
                ssl_state=ssl_zeros,
                acting_texture_latent=act_latent,
                cfg_scale=1.0,
                kv_caches=kv_caches,
                frame_index=t,
                frame_offsets=torch.tensor([[t]], dtype=torch.long, device=device),
                cross_attn_mask=ca_mask,
            )
            logits_a = out["logits_a"]
            logits_b = out["logits_b"]
            kv_caches = out.get("kv_cache_out", None)
            hidden = out.get("hidden_states", None)
        except Exception as e:
            if t == 0:
                logger.warning("forward_streaming failed: %s — falling back to uclm_core", e)
            # Fallback: manual voice state encoding + uclm_core
            state_cond = model.voice_state_enc(
                explicit_state=vs, ssl_state=ssl_zeros,
            )
            if isinstance(state_cond, tuple):
                state_cond = state_cond[0]
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
            for i in range(n_cb)
        ], dim=1).squeeze(-1)  # [B, n_cb]

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
    all_a = torch.cat(a_tokens, dim=-1)  # [B, n_cb, T]

    try:
        if codec_condition == "D":
            from tmrvc_data.wavtokenizer_codec import WavTokenizerWrapper
            codec = WavTokenizerWrapper(device=device)
        else:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (default: v4_full_final.pt)")
    parser.add_argument("--codec-condition", default="A", choices=["A", "B", "C", "D"])
    parser.add_argument("--output", default=None, help="Output dir")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output) if args.output else (ROOT / "output" / "v4_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (ROOT / "checkpoints" / "v4_full" / "v4_full_final.pt")
    model = load_model(str(ckpt_path), device, codec_condition=args.codec_condition)

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
        speaker_embed /= speaker_embed.norm(dim=-1, keepdim=True)

    def g2p(text, lang="ja"):
        try:
            from tmrvc_data.g2p import text_to_phonemes
            r = text_to_phonemes(text, language=lang)
            ids = r.phoneme_ids
            return ids.detach() if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
        except Exception:
            return torch.randint(1, PHONEME_VOCAB_SIZE, (len(text)*2,))

    # v4: Load acting macro projector from checkpoint for latent computation
    from tmrvc_train.models.acting_latent import ActingMacroProjector
    macro_proj = ActingMacroProjector()
    if hasattr(model, "acting_macro_proj"):
        macro_proj.load_state_dict(model.acting_macro_proj.state_dict())
    macro_proj.to(device).eval()

    def macro_to_latent(intensity=0.5, instability=0.2, tenderness=0.3,
                        tension=0.3, spontaneity=0.5, reference_mix=0.0):
        """Convert 6-D acting macro to 24-D acting texture latent."""
        macro = torch.tensor([[intensity, instability, tenderness,
                               tension, spontaneity, reference_mix]],
                             dtype=torch.float32, device=device)
        return macro_proj(macro).squeeze(0)  # [24]

    # Style dict: (text, 12-D voice_state, 6-D acting_macro or None)
    #   voice_state order: pitch_level, pitch_range, energy_level, pressedness,
    #     spectral_tilt, breathiness, voice_irregularity, openness,
    #     aperiodicity, formant_shift, vocal_effort, creak
    styles = {
        "neutral":          ("本当にありがとうございます。",
                             [0.5,0.3,0.5,0.35,0.5,0.2,0.15,0.5,0.2,0.5,0.4,0.1], None),
        "angry":            ("いい加減にしてよ！",
                             [0.7,0.6,0.85,0.7,0.7,0.1,0.3,0.6,0.3,0.5,0.8,0.2], None),
        "whisper":          ("ねえ、ちょっと聞いて。",
                             [0.4,0.15,0.15,0.1,0.3,0.8,0.1,0.3,0.4,0.5,0.15,0.05], None),
        "sad":              ("もう会えないのかな。",
                             [0.35,0.15,0.25,0.2,0.35,0.3,0.25,0.4,0.15,0.5,0.25,0.15], None),
        "excited":          ("すごい！信じられない！",
                             [0.7,0.7,0.8,0.5,0.6,0.15,0.2,0.7,0.2,0.5,0.7,0.1], None),
        "tender":           ("大丈夫だよ、心配しないで。",
                             [0.45,0.2,0.3,0.15,0.4,0.35,0.1,0.45,0.1,0.5,0.25,0.05], None),
        "professional":     ("本日の会議を始めさせていただきます。",
                             [0.5,0.25,0.5,0.4,0.5,0.1,0.05,0.5,0.1,0.5,0.5,0.05], None),
        "creaky_dramatic":  ("それは...嘘でしょう。",
                             [0.3,0.4,0.35,0.6,0.4,0.2,0.4,0.35,0.3,0.5,0.3,0.6], None),
        # v4: character archetypes with acting texture latent
        "mesugaki":         ("ざぁ〜こ♡ ざぁ〜こ♡ お兄さん弱すぎじゃない？",
                             [0.75,0.6,0.65,0.25,0.7,0.15,0.1,0.6,0.1,0.75,0.55,0.05],
                             {"intensity": 0.8, "instability": 0.35, "tenderness": 0.1,
                              "tension": 0.15, "spontaneity": 0.85, "reference_mix": 0.0}),
    }

    logger.info("=" * 60)
    logger.info("Generating %d samples with POINTER PROTOCOL + ACTING LATENT", len(styles))
    logger.info("=" * 60)

    results = []
    for name, (text, vs_list, macro_params) in styles.items():
        phonemes = g2p(text)
        vs = torch.tensor(vs_list, dtype=torch.float32)

        act_latent = None
        if macro_params is not None:
            act_latent = macro_to_latent(**macro_params)

        logger.info("--- %s: %s → %d phonemes (latent=%s) ---",
                     name, text, len(phonemes), "yes" if act_latent is not None else "no")

        audio, meta = generate_speech(
            model, phonemes, speaker_embed, vs, device,
            acting_texture_latent=act_latent,
            temperature=args.temperature,
            codec_condition=args.codec_condition,
        )
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
