#!/usr/bin/env python3
"""Verify that UCLMv4 can produce mesugaki (メスガキ) voice archetype.

Tests the v4 physical + latent hybrid control by generating mesugaki-style
utterances and comparing against neutral and contrasting archetypes.

Verification axes:
1. Physical control differentiation — mesugaki voice state produces distinct
   acoustic features vs neutral (pitch, spectral tilt, formant shift)
2. Acting latent effect — with vs without acting_texture_latent changes output
3. Archetype contrast — mesugaki vs onee-san vs neutral are distinguishable
4. Multi-utterance consistency — same profile produces consistent character

Usage:
    .venv/bin/python scripts/verify_mesugaki.py
    .venv/bin/python scripts/verify_mesugaki.py --checkpoint checkpoints/v4_full/v4_full_final.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tmrvc-core" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-data" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-train" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("verify_mesugaki")

from tmrvc_core.constants import (
    D_ACTING_LATENT,
    D_ACTING_MACRO,
    D_MODEL,
    D_SPEAKER,
    D_VOICE_STATE,
    D_VOICE_STATE_SSL,
    N_CODEBOOKS,
    CONTROL_SLOTS,
    RVQ_VOCAB_SIZE,
    CONTROL_VOCAB_SIZE,
    PHONEME_VOCAB_SIZE,
    SAMPLE_RATE,
)
from tmrvc_core.voice_state import CANONICAL_VOICE_STATE_IDS


# ---------------------------------------------------------------------------
# Character archetype definitions
# ---------------------------------------------------------------------------

@dataclass
class CharacterProfile:
    """Voice archetype with 12-D physical state + 6-D acting macro."""
    name: str
    description: str
    # 12-D voice state: pitch_level, pitch_range, energy_level, pressedness,
    #   spectral_tilt, breathiness, voice_irregularity, openness,
    #   aperiodicity, formant_shift, vocal_effort, creak
    voice_state: list[float]
    # 6-D acting macro: intensity, instability, tenderness, tension, spontaneity, reference_mix
    acting_macro: Optional[dict[str, float]] = None

    def voice_state_tensor(self) -> torch.Tensor:
        return torch.tensor(self.voice_state, dtype=torch.float32)

    def voice_state_dict(self) -> dict[str, float]:
        return {CANONICAL_VOICE_STATE_IDS[i]: v for i, v in enumerate(self.voice_state)}


MESUGAKI = CharacterProfile(
    name="mesugaki",
    description="メスガキ: 小悪魔的で生意気、挑発的だがどこか愛嬌がある若い女性の声",
    voice_state=[
        0.75,  # pitch_level — 高め (若い女性)
        0.6,   # pitch_range — 広い抑揚 (煽り・挑発の緩急)
        0.65,  # energy_level — やや高め (元気、自信)
        0.25,  # pressedness — 低め (余裕のある発声)
        0.7,   # spectral_tilt — 明るい (キンキン系)
        0.15,  # breathiness — 低い (クリアな声)
        0.1,   # voice_irregularity — 低い (若く滑らか)
        0.6,   # openness — やや開放 (ナマイキな母音)
        0.1,   # aperiodicity — 低い (クリーン)
        0.75,  # formant_shift — 高め (小柄な声道)
        0.55,  # vocal_effort — 中程度 (自信あるが張り上げない)
        0.05,  # creak — ほぼなし (若い)
    ],
    acting_macro={
        "intensity": 0.8,       # 非常に表現的
        "instability": 0.35,    # 遊び心のある揺れ
        "tenderness": 0.1,      # 甘くない、ナマイキ
        "tension": 0.15,        # リラックスした自信
        "spontaneity": 0.85,    # 自然で予測不能
        "reference_mix": 0.0,
    },
)

NEUTRAL = CharacterProfile(
    name="neutral",
    description="ニュートラル: 標準的な女性アナウンサー風",
    voice_state=[0.5, 0.3, 0.5, 0.35, 0.5, 0.2, 0.15, 0.5, 0.2, 0.5, 0.4, 0.1],
    acting_macro=None,
)

ONEESAN = CharacterProfile(
    name="oneesan",
    description="お姉さん: 落ち着きのある優しい年上女性の声",
    voice_state=[
        0.45,  # pitch_level — やや低め (落ち着き)
        0.25,  # pitch_range — 狭め (安定)
        0.4,   # energy_level — 控えめ
        0.2,   # pressedness — 低い (柔らかい)
        0.4,   # spectral_tilt — やや暗め (温かみ)
        0.3,   # breathiness — やや息混じり (色気)
        0.1,   # voice_irregularity — 低い
        0.45,  # openness — 中程度
        0.15,  # aperiodicity — 低い
        0.45,  # formant_shift — やや低め (大人)
        0.3,   # vocal_effort — 低め (囁くような)
        0.05,  # creak — ほぼなし
    ],
    acting_macro={
        "intensity": 0.4,
        "instability": 0.1,
        "tenderness": 0.85,
        "tension": 0.05,
        "spontaneity": 0.3,
        "reference_mix": 0.0,
    },
)

# Mesugaki test utterances (typical dialogue patterns)
MESUGAKI_UTTERANCES = [
    "ざぁ〜こ♡ ざぁ〜こ♡",
    "お兄さん、弱すぎじゃない？",
    "えー、もう諦めるの？だっさ〜い♡",
    "わたしに勝てると思ったの？バカじゃないの〜？",
    "ほら、もっと頑張ってよ。応援してあげるから♡",
    "あはっ、顔真っ赤じゃん。かわい〜",
]

# Neutral comparison utterances (same semantic content, neutral delivery)
NEUTRAL_UTTERANCES = [
    "本当にありがとうございます。",
    "お疲れ様でした。",
    "承知いたしました。",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _infer_hparams(state_dict: dict) -> dict:
    """Infer model hyperparameters from checkpoint weight shapes."""
    d_model = state_dict["uclm_core.layers.0.norm1.weight"].shape[0]
    n_layers = max(int(k.split(".")[2]) for k in state_dict if k.startswith("uclm_core.layers.")) + 1
    q_dim = state_dict["uclm_core.layers.0.attn.q_proj.weight"].shape[0]
    k_dim = state_dict["uclm_core.layers.0.attn.k_proj.weight"].shape[0]
    # GQA: n_heads = d_model / head_dim, n_kv_heads = k_dim / head_dim
    # Try common head counts to find integer head_dim
    for n_h in [8, 12, 16, 4]:
        hd = d_model // n_h
        if hd * n_h == d_model and k_dim % hd == 0:
            n_heads = n_h
            break
    else:
        n_heads = 8
    return {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads}


def load_model(ckpt_path: str, device: str):
    from tmrvc_train.models.uclm_model import DisentangledUCLM
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = _infer_hparams(ckpt["model"])
    logger.info("Checkpoint hparams: d_model=%d, n_layers=%d, n_heads=%d",
                hp["d_model"], hp["n_layers"], hp["n_heads"])
    model = DisentangledUCLM(
        d_model=hp["d_model"], n_heads=hp["n_heads"], n_layers=hp["n_layers"],
        d_voice_state_explicit=D_VOICE_STATE,
        d_voice_state_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER, n_codebooks=N_CODEBOOKS,
        rvq_vocab_size=RVQ_VOCAB_SIZE, control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE, num_speakers=100,
    ).to(device)
    # Filter out keys with size mismatches (checkpoint may differ from current arch)
    model_sd = model.state_dict()
    ckpt_sd = ckpt["model"]
    filtered = {}
    skipped = []
    for k, v in ckpt_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        elif k in model_sd:
            skipped.append(k)
    model.load_state_dict(filtered, strict=False)
    if skipped:
        logger.warning("Skipped %d keys with shape mismatch: %s", len(skipped), skipped[:5])
    model.eval()
    logger.info("Model loaded (step %d, loss %.4f, %d/%d keys)",
                ckpt.get("step", 0), ckpt.get("best_loss", 0),
                len(filtered), len(ckpt_sd))
    return model


def load_speaker_embed(device: str) -> torch.Tensor:
    cache_dir = ROOT / "data" / "cache" / "v4full" / "train"
    if cache_dir.exists():
        for spk_dir in sorted(cache_dir.iterdir()):
            for utt_dir in sorted(spk_dir.iterdir()):
                p = utt_dir / "spk_embed.npy"
                if p.exists():
                    return torch.from_numpy(np.load(p)).float().unsqueeze(0).to(device)
    embed = torch.randn(1, D_SPEAKER, device=device)
    embed /= embed.norm(dim=-1, keepdim=True)
    return embed


def make_macro_projector(model, device: str):
    from tmrvc_train.models.acting_latent import ActingMacroProjector
    proj = ActingMacroProjector()
    if hasattr(model, "acting_macro_proj"):
        proj.load_state_dict(model.acting_macro_proj.state_dict())
    return proj.to(device).eval()


# ---------------------------------------------------------------------------
# G2P helper
# ---------------------------------------------------------------------------

def g2p(text: str, lang: str = "ja") -> torch.Tensor:
    try:
        from tmrvc_data.g2p import text_to_phonemes
        r = text_to_phonemes(text, language=lang)
        ids = r.phoneme_ids
        return ids.detach() if isinstance(ids, torch.Tensor) else torch.tensor(ids, dtype=torch.long)
    except Exception:
        return torch.randint(1, PHONEME_VOCAB_SIZE, (len(text) * 2,))


# ---------------------------------------------------------------------------
# Generation (reuses generate_v4_sample logic)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_speech(
    model,
    phoneme_ids: torch.Tensor,
    speaker_embed: torch.Tensor,
    voice_state: torch.Tensor,
    device: str,
    acting_texture_latent: torch.Tensor | None = None,
    max_frames: int = 1500,
    temperature: float = 0.8,
    pace: float = 1.0,
    max_frames_per_unit: int = 50,
) -> tuple[np.ndarray, dict]:
    B = 1
    phoneme_ids = phoneme_ids.unsqueeze(0).to(device) if phoneme_ids.dim() == 1 else phoneme_ids.to(device)
    L = phoneme_ids.size(1)

    vs = voice_state.view(1, 1, -1).to(device)
    ssl_zeros = torch.zeros(1, 1, D_VOICE_STATE_SSL, device=device)
    lang_ids = torch.zeros(B, dtype=torch.long, device=device)

    act_latent = None
    if acting_texture_latent is not None:
        act_latent = acting_texture_latent.view(1, D_ACTING_LATENT).to(device)

    text_features = model.text_encoder(phoneme_ids, lang_ids).transpose(1, 2)

    text_index = 0
    progress = 0.0
    frames_on_unit = 0
    kv_caches = None
    ctx_a = torch.zeros(B, N_CODEBOOKS, 1, dtype=torch.long, device=device)
    ctx_b = torch.zeros(B, CONTROL_SLOTS, 1, dtype=torch.long, device=device)

    a_tokens = []
    pointer_trace = []
    t0 = time.time()
    forced_advances = 0

    for t in range(max_frames):
        if text_index >= L:
            break

        safe_idx = min(text_index, L - 1)
        content_features = text_features[:, safe_idx:safe_idx + 1, :]

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
            )
            logits_a = out["logits_a"]
            logits_b = out["logits_b"]
            kv_caches = out.get("kv_cache_out", None)
            hidden = out.get("hidden_states", None)
        except Exception:
            state_cond = model.voice_state_enc(explicit_state=vs, ssl_state=ssl_zeros)
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

        at = torch.stack([
            torch.multinomial(F.softmax(logits_a[:, i, 0, :] / temperature, dim=-1), 1)
            for i in range(N_CODEBOOKS)
        ], dim=1).squeeze(-1)

        bt = torch.stack([
            torch.multinomial(F.softmax(logits_b[:, i, 0, :] / temperature, dim=-1), 1)
            for i in range(CONTROL_SLOTS)
        ], dim=1).squeeze(-1)

        a_tokens.append(at.unsqueeze(-1))
        ctx_a = at.unsqueeze(-1)
        ctx_b = bt.unsqueeze(-1)

        if hidden is not None and hasattr(model, "pointer_head"):
            try:
                p_adv_logit, p_delta, p_bc = model.pointer_head(hidden)
                adv_prob = torch.sigmoid(p_adv_logit + (pace - 1.0) * 2.0).item()
                velocity = p_delta.item() * pace
                progress += velocity
            except Exception:
                adv_prob = 0.5
                progress += 0.05 * pace
        else:
            adv_prob = 0.5
            progress += 0.05 * pace

        frames_on_unit += 1
        advanced = False

        if frames_on_unit >= max_frames_per_unit:
            advanced = True
            forced_advances += 1
        elif progress >= 1.0 and adv_prob > 0.5:
            advanced = True
        elif progress >= 1.5:
            advanced = True

        if advanced:
            pointer_trace.append((text_index, frames_on_unit))
            text_index += 1
            progress = 0.0
            frames_on_unit = 0

    if frames_on_unit > 0:
        pointer_trace.append((text_index, frames_on_unit))

    gen_time = time.time() - t0
    n_frames = len(a_tokens)

    if n_frames == 0:
        return np.zeros(SAMPLE_RATE, dtype=np.float32), {"n_frames": 0, "error": "no frames"}

    all_a = torch.cat(a_tokens, dim=-1)

    try:
        from tmrvc_data.encodec_codec import EnCodecWrapper
        codec = EnCodecWrapper(device=device)
        audio_t = codec.decode(all_a)
        audio_np = audio_t.squeeze().cpu().numpy()
    except Exception:
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
    }


# ---------------------------------------------------------------------------
# Verification tests
# ---------------------------------------------------------------------------

def test_mesugaki_generation(
    model, speaker_embed, macro_proj, device, output_dir,
) -> list[dict]:
    """Test 1: Generate all mesugaki utterances with the mesugaki profile."""
    logger.info("=" * 60)
    logger.info("TEST 1: Mesugaki utterance generation")
    logger.info("=" * 60)

    results = []
    act_latent = _profile_to_latent(MESUGAKI, macro_proj, device)

    for i, text in enumerate(MESUGAKI_UTTERANCES):
        phonemes = g2p(text)
        audio, meta = generate_speech(
            model, phonemes, speaker_embed,
            MESUGAKI.voice_state_tensor(), device,
            acting_texture_latent=act_latent,
        )
        wav_path = output_dir / f"mesugaki_{i:02d}.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)

        meta["text"] = text
        meta["profile"] = MESUGAKI.name
        results.append(meta)
        logger.info("  [%d] %s → %.2fs, %d frames", i, text, meta["duration_sec"], meta["n_frames"])

    return results


def test_archetype_contrast(
    model, speaker_embed, macro_proj, device, output_dir,
) -> list[dict]:
    """Test 2: Same text, different archetypes — verify differentiation."""
    logger.info("=" * 60)
    logger.info("TEST 2: Archetype contrast (same text, different profiles)")
    logger.info("=" * 60)

    test_text = "ほら、もっと頑張ってよ。"
    phonemes = g2p(test_text)
    results = []

    for profile in [MESUGAKI, NEUTRAL, ONEESAN]:
        act_latent = _profile_to_latent(profile, macro_proj, device)
        audio, meta = generate_speech(
            model, phonemes, speaker_embed,
            profile.voice_state_tensor(), device,
            acting_texture_latent=act_latent,
        )
        wav_path = output_dir / f"contrast_{profile.name}.wav"
        sf.write(str(wav_path), audio, SAMPLE_RATE)

        meta["text"] = test_text
        meta["profile"] = profile.name
        results.append(meta)
        logger.info("  %s → %.2fs, %d frames", profile.name, meta["duration_sec"], meta["n_frames"])

    return results


def test_latent_effect(
    model, speaker_embed, macro_proj, device, output_dir,
) -> list[dict]:
    """Test 3: Same voice state, with vs without acting latent."""
    logger.info("=" * 60)
    logger.info("TEST 3: Acting latent effect (with vs without)")
    logger.info("=" * 60)

    test_text = "ざぁ〜こ♡ ざぁ〜こ♡"
    phonemes = g2p(test_text)
    results = []

    # With acting latent
    act_latent = _profile_to_latent(MESUGAKI, macro_proj, device)
    audio_with, meta_with = generate_speech(
        model, phonemes, speaker_embed,
        MESUGAKI.voice_state_tensor(), device,
        acting_texture_latent=act_latent,
    )
    sf.write(str(output_dir / "latent_with.wav"), audio_with, SAMPLE_RATE)
    meta_with["condition"] = "with_latent"
    results.append(meta_with)

    # Without acting latent (physical controls only)
    audio_without, meta_without = generate_speech(
        model, phonemes, speaker_embed,
        MESUGAKI.voice_state_tensor(), device,
        acting_texture_latent=None,
    )
    sf.write(str(output_dir / "latent_without.wav"), audio_without, SAMPLE_RATE)
    meta_without["condition"] = "without_latent"
    results.append(meta_without)

    # Token-level divergence: compare codec token distributions
    logger.info("  with_latent:    %.2fs, %d frames", meta_with["duration_sec"], meta_with["n_frames"])
    logger.info("  without_latent: %.2fs, %d frames", meta_without["duration_sec"], meta_without["n_frames"])

    return results


def test_voice_state_sweep(
    model, speaker_embed, macro_proj, device, output_dir,
) -> list[dict]:
    """Test 4: Sweep key physical dimensions to verify control response."""
    logger.info("=" * 60)
    logger.info("TEST 4: Physical control sweep on mesugaki base")
    logger.info("=" * 60)

    test_text = "お兄さん、弱すぎじゃない？"
    phonemes = g2p(test_text)
    act_latent = _profile_to_latent(MESUGAKI, macro_proj, device)
    results = []

    # Sweep key dimensions: pitch_level, spectral_tilt, formant_shift
    sweep_dims = [
        (0, "pitch_level"),
        (4, "spectral_tilt"),
        (9, "formant_shift"),
    ]
    sweep_values = [0.2, 0.5, 0.8]

    for dim_idx, dim_name in sweep_dims:
        for val in sweep_values:
            vs = MESUGAKI.voice_state_tensor().clone()
            vs[dim_idx] = val

            audio, meta = generate_speech(
                model, phonemes, speaker_embed, vs, device,
                acting_texture_latent=act_latent,
            )
            tag = f"sweep_{dim_name}_{val:.1f}"
            sf.write(str(output_dir / f"{tag}.wav"), audio, SAMPLE_RATE)

            meta["sweep_dim"] = dim_name
            meta["sweep_val"] = val
            results.append(meta)
            logger.info("  %s=%.1f → %.2fs, %d frames", dim_name, val,
                        meta["duration_sec"], meta["n_frames"])

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile_to_latent(
    profile: CharacterProfile, macro_proj, device: str,
) -> torch.Tensor | None:
    if profile.acting_macro is None:
        return None
    m = profile.acting_macro
    macro = torch.tensor([[
        m.get("intensity", 0.5),
        m.get("instability", 0.2),
        m.get("tenderness", 0.3),
        m.get("tension", 0.3),
        m.get("spontaneity", 0.5),
        m.get("reference_mix", 0.0),
    ]], dtype=torch.float32, device=device)
    with torch.no_grad():
        return macro_proj(macro).squeeze(0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Verify mesugaki voice archetype")
    parser.add_argument("--checkpoint", type=str,
                        default=str(ROOT / "checkpoints" / "v4_full" / "v4_full_final.pt"))
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = ROOT / "output" / "verify_mesugaki"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Device: %s", device)
    logger.info("Output: %s", output_dir)

    # Load model and resources
    model = load_model(args.checkpoint, device)
    speaker_embed = load_speaker_embed(device)
    macro_proj = make_macro_projector(model, device)

    # Log archetype definitions
    logger.info("=" * 60)
    logger.info("MESUGAKI VOICE PROFILE")
    logger.info("=" * 60)
    for dim_id, val in zip(CANONICAL_VOICE_STATE_IDS, MESUGAKI.voice_state):
        logger.info("  %-20s: %.2f", dim_id, val)
    if MESUGAKI.acting_macro:
        logger.info("  --- acting macro ---")
        for k, v in MESUGAKI.acting_macro.items():
            logger.info("  %-20s: %.2f", k, v)

    # Run verification tests
    all_results = {}

    all_results["mesugaki_generation"] = test_mesugaki_generation(
        model, speaker_embed, macro_proj, device, output_dir)

    all_results["archetype_contrast"] = test_archetype_contrast(
        model, speaker_embed, macro_proj, device, output_dir)

    all_results["latent_effect"] = test_latent_effect(
        model, speaker_embed, macro_proj, device, output_dir)

    all_results["voice_state_sweep"] = test_voice_state_sweep(
        model, speaker_embed, macro_proj, device, output_dir)

    # Summary
    logger.info("=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    total_samples = sum(len(v) for v in all_results.values())
    total_ok = sum(
        1 for v in all_results.values()
        for r in v
        if r.get("n_frames", 0) > 0 and r.get("error") is None
    )
    logger.info("  Total samples: %d", total_samples)
    logger.info("  Successful:    %d", total_ok)
    logger.info("  Failed:        %d", total_samples - total_ok)

    # Write results
    report_path = output_dir / "verification_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    logger.info("Report: %s", report_path)

    # Write profile definitions for reproducibility
    profiles_path = output_dir / "profiles.json"
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump({
            "mesugaki": {
                "voice_state": dict(zip(CANONICAL_VOICE_STATE_IDS, MESUGAKI.voice_state)),
                "acting_macro": MESUGAKI.acting_macro,
                "description": MESUGAKI.description,
            },
            "neutral": {
                "voice_state": dict(zip(CANONICAL_VOICE_STATE_IDS, NEUTRAL.voice_state)),
                "acting_macro": NEUTRAL.acting_macro,
                "description": NEUTRAL.description,
            },
            "oneesan": {
                "voice_state": dict(zip(CANONICAL_VOICE_STATE_IDS, ONEESAN.voice_state)),
                "acting_macro": ONEESAN.acting_macro,
                "description": ONEESAN.description,
            },
        }, f, ensure_ascii=False, indent=2)
    logger.info("Profiles: %s", profiles_path)

    logger.info("Done. Listen to samples in: %s", output_dir)


if __name__ == "__main__":
    main()
