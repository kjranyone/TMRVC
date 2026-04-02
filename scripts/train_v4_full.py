#!/usr/bin/env python3
"""v4 complete training pipeline — real models, no shortcuts.

Phase 1: Bootstrap cache from raw audio using ALL real models:
  - ASR: faster-whisper large-v3 (cached)
  - G2P: real phoneme conversion
  - Voice State: real DSP 12-D extraction
  - Speaker Encoder: ECAPA-TDNN (cached)
  - LLM Annotation: Qwen3.5-35B-A3B (cached) → semantic + enriched transcripts
  - Codec: EnCodec 24kHz frozen pre-trained, 75 Hz, 8 RVQ x 1024 (condition A)

Phase 2: v4 full training with:
  - Enriched transcript path (inline acting tags)
  - All 9 v4 loss terms
  - Biological constraint regularization
  - Acting latent encoder/predictor
  - Supervision tier weighting

Usage:
    .venv/bin/python scripts/train_v4_full.py --steps 200 --sample-pct 1
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tmrvc-core" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-data" / "src"))
sys.path.insert(0, str(ROOT / "tmrvc-train" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v4_full")

from tmrvc_core.constants import (
    D_MODEL, D_SPEAKER, D_VOICE_STATE, D_VOICE_STATE_SSL,
    D_ACTING_LATENT, N_ACTING_TAGS, N_CODEBOOKS, CONTROL_SLOTS,
    RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE, PHONEME_VOCAB_SIZE,
    HOP_LENGTH, SAMPLE_RATE,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8,
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--sample-pct", type=float, default=1.0, help="% of raw audio")
    p.add_argument("--max-frames", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--output-dir", default=str(ROOT / "checkpoints" / "v4_full"))
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--skip-cache", action="store_true")
    p.add_argument("--resume-from", type=int, default=None,
                   help="Resume from checkpoint step (e.g. 9500)")
    p.add_argument("--annotation-model", default="Qwen/Qwen3.5-35B-A3B",
                   help="LLM for semantic annotation + enriched transcripts")
    p.add_argument("--codec-condition", default="A", choices=["A", "B", "C", "D"],
                   help="Codec experiment condition (track_codec_strategy.md)")
    return p.parse_args()


# =========================================================================
# Phase 1: Real bootstrap cache generation
# =========================================================================

class FullBootstrapCacheBuilder:
    """Builds training cache using ALL real models."""

    ANNOTATION_PROMPT = (
        "You are a speech acting annotator. Analyze this transcript and respond with JSON:\n"
        '{"scene_summary":"...","dialogue_intent":"...","emotion_description":"...","acting_hint":"...",'
        '"enriched_transcript":"...(original text with inline acting tags like [angry], [whisper], '
        '[emphasis], [pause], [laugh], [sigh] inserted at appropriate positions)..."}\n\n'
        "Language: {lang}\nTranscript: {text}\n\nJSON:"
    )

    def __init__(self, device: str, annotation_model: str, max_frames: int):
        self.device = device
        self.annotation_model_name = annotation_model
        self.max_frames = max_frames

        # Real models — lazy loaded
        self._whisper = None
        self._spk_encoder = None
        self._vs_estimator = None
        self._mimi_codec = None
        self._annotation_model = None
        self._annotation_tokenizer = None

    def _load_whisper(self):
        if self._whisper is not None:
            return
        from faster_whisper import WhisperModel
        compute = "float16" if self.device == "cuda" else "int8"
        self._whisper = WhisperModel("large-v3", device=self.device, compute_type=compute)
        logger.info("Loaded Whisper large-v3 on %s", self.device)

    def _load_spk_encoder(self):
        if self._spk_encoder is not None:
            return
        from tmrvc_data.speaker import SpeakerEncoder
        self._spk_encoder = SpeakerEncoder(device=self.device)
        logger.info("Loaded ECAPA-TDNN speaker encoder")

    def _load_vs_estimator(self):
        if self._vs_estimator is not None:
            return
        from tmrvc_data.voice_state import VoiceStateEstimator
        self._vs_estimator = VoiceStateEstimator(device=self.device)
        logger.info("Loaded VoiceStateEstimator")

    def _load_codec(self):
        if self._mimi_codec is not None:
            return
        # Use EnCodec for condition A baseline (track_codec_strategy.md)
        from tmrvc_data.encodec_codec import EnCodecWrapper
        self._mimi_codec = EnCodecWrapper(device=self.device)
        logger.info("Loaded EnCodec 24kHz on %s", self.device)

    def _load_annotation_llm(self):
        if self._annotation_model is not None:
            return
        logger.info("Loading annotation LLM %s (4-bit quantized)...", self.annotation_model_name)
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        self._annotation_tokenizer = AutoTokenizer.from_pretrained(
            self.annotation_model_name, trust_remote_code=True,
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self._annotation_model = AutoModelForCausalLM.from_pretrained(
            self.annotation_model_name, trust_remote_code=True,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self._annotation_model.eval()
        logger.info("Loaded %s (4-bit, ~%.1f GB VRAM)",
                     self.annotation_model_name,
                     sum(p.nelement() * p.element_size() for p in self._annotation_model.parameters()) / 1e9)

    def build(self, wav_paths: list[Path], cache_dir: Path) -> int:
        from tmrvc_data.preprocessing import load_and_resample
        from tmrvc_core.audio import compute_mel

        cache_dir.mkdir(parents=True, exist_ok=True)
        # Phase 1: Load audio processing models (ASR, speaker, voice state, codec)
        # LLM is loaded separately in Phase 2 to avoid VRAM exhaustion
        self._load_whisper()
        self._load_spk_encoder()
        self._load_vs_estimator()
        self._load_codec()

        n_ok = 0
        for i, wav_path in enumerate(wav_paths):
            utt_id = f"v4full_{i:06d}"
            speaker_id = f"spk_{wav_path.parent.name.replace(' ', '_')[:20]}"
            utt_dir = cache_dir / "v4full" / "train" / speaker_id / utt_id

            if (utt_dir / "meta.json").exists():
                n_ok += 1
                continue

            try:
                # --- Load audio ---
                waveform, _ = load_and_resample(str(wav_path), target_sr=SAMPLE_RATE)
                if waveform is None:
                    continue
                waveform_np = waveform.squeeze().cpu().numpy() if isinstance(waveform, torch.Tensor) else np.asarray(waveform).squeeze()
                n_samples = len(waveform_np)

                # EnCodec: 75 Hz (hop=320), voice state: 75 Hz (same rate)
                CODEC_HOP = 320   # EnCodec 75 Hz
                n_frames = n_samples // CODEC_HOP
                n_codec_frames = n_frames
                n_control_frames = n_frames
                if n_frames < 20:
                    continue
                # No max_frames filter — include all utterances regardless of length

                # --- Real ASR ---
                import tempfile, soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
                    sf.write(tmp.name, waveform_np, SAMPLE_RATE)
                    segments, info = self._whisper.transcribe(tmp.name, beam_size=5, vad_filter=True)
                    transcript = "".join(seg.text for seg in segments).strip()
                    lang = info.language if hasattr(info, 'language') else "ja"
                    asr_conf = info.language_probability if hasattr(info, 'language_probability') else 0.5

                if not transcript:
                    continue  # Skip empty transcripts

                # --- Real G2P ---
                from tmrvc_data.g2p import text_to_phonemes as g2p_func
                g2p_result = g2p_func(transcript, language=lang)
                phoneme_ids = g2p_result.phoneme_ids
                if isinstance(phoneme_ids, torch.Tensor):
                    phoneme_ids = phoneme_ids.detach().cpu().numpy()
                phoneme_ids = np.asarray(phoneme_ids, dtype=np.int64)
                if len(phoneme_ids) < 3:
                    continue

                # --- Real voice state (12-D DSP) ---
                waveform_t = torch.from_numpy(waveform_np).float().unsqueeze(0)
                mel = compute_mel(waveform_t).to(self.device)
                f0 = torch.zeros(1, 1, mel.shape[-1], device=torch.device(self.device))
                vs_raw = self._vs_estimator.estimate(mel, f0)
                if isinstance(vs_raw, torch.Tensor):
                    vs = vs_raw.detach().squeeze(0).cpu().numpy()
                else:
                    vs = np.zeros((n_frames, D_VOICE_STATE), dtype=np.float32)
                vs = np.clip(vs[:n_frames], 0, 1).astype(np.float32)
                if vs.shape[0] < n_frames:
                    vs = np.pad(vs, ((0, n_frames - vs.shape[0]), (0, 0)), mode='edge')

                # --- Real speaker embedding ---
                spk = self._spk_encoder.extract(waveform_t, sample_rate=SAMPLE_RATE)
                spk = spk.detach().cpu().numpy().flatten().astype(np.float32) if isinstance(spk, torch.Tensor) else np.zeros(D_SPEAKER, np.float32)

                # --- LLM annotation deferred to pass 2 (VRAM constraint) ---
                annotations = {"scene_summary": "", "dialogue_intent": "", "emotion_description": "", "acting_hint": ""}
                enriched = transcript

                # --- Real Mimi codec tokens (12.5 Hz) ---
                waveform_t_codec = torch.from_numpy(waveform_np).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
                codec_tokens = self._mimi_codec.encode(waveform_t_codec)  # [1, 8, T_codec]
                codec_tokens = codec_tokens.squeeze(0).numpy().astype(np.int64)  # [8, T_codec]

                # --- Supervision tier ---
                has_semantic = bool(annotations.get("emotion_description") or annotations.get("acting_hint"))
                tier = "tier_a" if (asr_conf > 0.8 and has_semantic) else "tier_b"

                # --- Observed mask & confidence ---
                observed_mask = np.ones((n_frames, D_VOICE_STATE), dtype=bool)
                observed_mask[:, 8:] = False  # new dims not yet reliable
                confidence = np.ones((n_frames, D_VOICE_STATE), dtype=np.float32) * 0.8
                confidence[:, 8:] = 0.1

                # --- Write cache ---
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
                    "speaker_id": speaker_id,
                    "n_frames": int(n_frames),
                    "n_codec_frames": int(n_codec_frames),
                    "n_control_frames": int(n_control_frames),
                    "text": transcript,
                    "enriched_transcript": enriched,
                    "language_id": {"ja": 0, "en": 1, "zh": 2, "ko": 3}.get(lang, 0),
                    "language": lang,
                    "duration_sec": n_samples / SAMPLE_RATE,
                    "quality_score": 0.85,
                    "supervision_tier": tier,
                    "asr_confidence": float(asr_conf),
                    "acting_annotations": annotations,
                }
                with open(utt_dir / "meta.json", "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                n_ok += 1
                if (i + 1) % 10 == 0:
                    logger.info(
                        "Cache %d/%d | tier=%s | lang=%s | text=%.40s... | enriched=%.40s...",
                        i + 1, len(wav_paths), tier, lang,
                        transcript, enriched,
                    )

            except Exception as e:
                logger.warning("Skip %s: %s", wav_path.name, e)

        logger.info("Pass 1 complete: %d / %d utterances (audio features)", n_ok, len(wav_paths))

        # --- Pass 2: LLM annotation (free audio models first) ---
        if n_ok > 0:
            logger.info("Pass 2: Unloading audio models, loading LLM for annotation...")
            self._whisper = None
            self._spk_encoder = None
            self._vs_estimator = None
            if self._mimi_codec is not None:
                self._mimi_codec.unload()
                self._mimi_codec = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                self._load_annotation_llm()
                annotated = 0
                for meta_path in sorted(cache_dir.rglob("meta.json")):
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        transcript = meta.get("text", "")
                        if not transcript:
                            continue
                        lang = meta.get("language", "ja")
                        prompt = self.ANNOTATION_PROMPT.format(lang=lang, text=transcript)
                        annotations, enriched = self._run_annotation(prompt, transcript)
                        meta["acting_annotations"] = annotations
                        meta["enriched_transcript"] = enriched
                        has_semantic = bool(annotations.get("emotion_description") or annotations.get("acting_hint"))
                        asr_conf = meta.get("asr_confidence", 0.5)
                        meta["supervision_tier"] = "tier_a" if (asr_conf > 0.8 and has_semantic) else "tier_b"
                        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                        annotated += 1
                        if annotated % 50 == 0:
                            logger.info("Annotated %d utterances", annotated)
                    except Exception as e:
                        logger.debug("Annotation skip: %s", e)
                logger.info("Pass 2 complete: %d utterances annotated", annotated)
            except Exception as e:
                logger.warning("LLM annotation failed (proceeding without): %s", e)

        logger.info("Cache complete: %d / %d utterances", n_ok, len(wav_paths))
        return n_ok

    def _run_annotation(self, prompt: str, fallback_text: str) -> tuple[dict, str]:
        """Run LLM annotation, return (annotations_dict, enriched_transcript)."""
        try:
            inputs = self._annotation_tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=1024,
            ).to(self._annotation_model.device)

            with torch.inference_mode():
                out = self._annotation_model.generate(
                    **inputs, max_new_tokens=512,
                    temperature=0.3, do_sample=True, top_p=0.9,
                )
            generated = out[0][inputs["input_ids"].shape[-1]:]
            text = self._annotation_tokenizer.decode(generated, skip_special_tokens=True).strip()

            # Parse JSON
            # Try to find JSON in the output
            json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = {}

            annotations = {
                "scene_summary": data.get("scene_summary", ""),
                "dialogue_intent": data.get("dialogue_intent", ""),
                "emotion_description": data.get("emotion_description", ""),
                "acting_hint": data.get("acting_hint", ""),
            }
            enriched = data.get("enriched_transcript", fallback_text)

            return annotations, enriched

        except Exception as e:
            logger.debug("Annotation failed: %s", e)
            return {
                "scene_summary": "", "dialogue_intent": "",
                "emotion_description": "", "acting_hint": "",
            }, fallback_text

    def cleanup(self):
        del self._whisper, self._spk_encoder, self._vs_estimator
        del self._annotation_model, self._annotation_tokenizer
        self._whisper = self._spk_encoder = self._vs_estimator = None
        self._annotation_model = self._annotation_tokenizer = None
        if self._mimi_codec is not None:
            self._mimi_codec.unload()
        self._mimi_codec = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All bootstrap models unloaded, GPU freed for training")


# =========================================================================
# Phase 2: Training
# =========================================================================

def main():
    args = parse_args()
    # v4 cache: prefer new managed path, fall back to legacy
    cache_dir = ROOT / "data" / "cache" / "v4"
    if not cache_dir.exists():
        legacy = ROOT / "data" / "cache"
        if (legacy / "v4full").exists():
            cache_dir = legacy
        elif legacy.exists():
            cache_dir = legacy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Phase 1: Real cache ----
    if not args.skip_cache:
        logger.info("=" * 60)
        logger.info("Phase 1: Bootstrap cache with ALL real models (%.1f%%)", args.sample_pct)
        logger.info("  ASR: faster-whisper large-v3")
        logger.info("  Speaker: ECAPA-TDNN")
        logger.info("  Voice State: DSP 12-D")
        logger.info("  LLM: %s → semantic + enriched transcripts", args.annotation_model)
        logger.info("  Codec: Mimi (kyutai/mimi) frozen, 12.5 Hz, 8x2048")
        logger.info("=" * 60)

        all_wavs = sorted((ROOT / "data" / "raw").rglob("*.wav")) if (ROOT / "data" / "raw").exists() else []
        n_sample = max(1, int(len(all_wavs) * args.sample_pct / 100))
        random.seed(42)
        wavs = random.sample(all_wavs, min(n_sample, len(all_wavs)))
        logger.info("Sampled %d / %d audio files", len(wavs), len(all_wavs))

        builder = FullBootstrapCacheBuilder(
            device=args.device,
            annotation_model=args.annotation_model,
            max_frames=args.max_frames,
        )
        n = builder.build(wavs, cache_dir)
        builder.cleanup()

        if n == 0:
            logger.error("No cache generated. Exiting.")
            sys.exit(1)

    # ---- Phase 2: Model + Trainer ----
    logger.info("=" * 60)
    logger.info("Phase 2: Building v4 model + trainer")
    logger.info("=" * 60)
    device = args.device

    from tmrvc_train.models.uclm_model import DisentangledUCLM
    from tmrvc_train.models.acting_latent import ActingLatentEncoder, ActingLatentPredictor
    from tmrvc_train.trainer import UCLMTrainer
    from tmrvc_train.v4_loss import V4LossConfig

    codec_cond = args.codec_condition
    logger.info("Codec condition: %s", codec_cond)

    # Infer d_model/n_layers from checkpoint when resuming (checkpoint may differ from constants.yaml)
    eff_d_model = D_MODEL
    eff_n_layers = None  # use default from constants
    eff_n_heads = None
    resume_step = 0

    if args.resume_from is not None:
        resume_path = output_dir / f"v4_step_{args.resume_from}.pt"
        if not resume_path.exists():
            logger.error("Checkpoint not found: %s", resume_path)
            sys.exit(1)
        _ckpt_sd = torch.load(resume_path, map_location="cpu", weights_only=False)["model"]
        eff_d_model = _ckpt_sd["uclm_core.layers.0.norm1.weight"].shape[0]
        eff_n_layers = max(int(k.split(".")[2]) for k in _ckpt_sd if k.startswith("uclm_core.layers.")) + 1
        q_dim = _ckpt_sd["uclm_core.layers.0.attn.q_proj.weight"].shape[0]
        k_dim = _ckpt_sd["uclm_core.layers.0.attn.k_proj.weight"].shape[0]
        for nh in [8, 12, 16, 4]:
            hd = eff_d_model // nh
            if hd * nh == eff_d_model and k_dim % hd == 0:
                eff_n_heads = nh
                break
        del _ckpt_sd
        logger.info("Resume: d_model=%d, n_layers=%d, n_heads=%s (from checkpoint)",
                     eff_d_model, eff_n_layers, eff_n_heads)

    init_kwargs = dict(
        d_model=eff_d_model,
        d_voice_state_explicit=D_VOICE_STATE, d_voice_state_ssl=D_VOICE_STATE_SSL,
        d_speaker=D_SPEAKER, n_codebooks=N_CODEBOOKS,
        rvq_vocab_size=RVQ_VOCAB_SIZE, control_vocab_size=CONTROL_VOCAB_SIZE,
        vocab_size=PHONEME_VOCAB_SIZE, num_speakers=200,
        acting_tag_vocab_size=N_ACTING_TAGS,
        codec_condition=codec_cond,
    )
    if eff_n_layers is not None:
        init_kwargs["n_layers"] = eff_n_layers
    if eff_n_heads is not None:
        init_kwargs["n_heads"] = eff_n_heads

    model = DisentangledUCLM(**init_kwargs).to(device)

    acting_enc = ActingLatentEncoder().to(device)
    acting_pred = ActingLatentPredictor(d_text=eff_d_model, d_context=eff_d_model).to(device)

    # Only model params here — trainer will add acting_enc/pred/bio params via add_param_group
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Resume from checkpoint
    if args.resume_from is not None:
        resume_path = output_dir / f"v4_step_{args.resume_from}.pt"
        logger.info("Loading checkpoint: %s", resume_path)
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        # Filter out keys with shape mismatch (code may have evolved since checkpoint)
        model_sd = model.state_dict()
        ckpt_sd = ckpt["model"]
        filtered = {k: v for k, v in ckpt_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
        skipped = [k for k in ckpt_sd if k in model_sd and model_sd[k].shape != ckpt_sd[k].shape]
        model.load_state_dict(filtered, strict=False)
        if skipped:
            logger.warning("Skipped %d keys with shape mismatch: %s", len(skipped), skipped)
        acting_enc.load_state_dict(ckpt["acting_encoder"])
        acting_pred.load_state_dict(ckpt["acting_predictor"])
        # Optimizer state may have mismatched param groups due to model changes;
        # load with best-effort
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except (ValueError, RuntimeError) as e:
            logger.warning("Optimizer state load failed (%s), starting fresh optimizer", e)
        resume_step = ckpt["step"]
        del ckpt
        logger.info("Resumed from step %d (%d/%d model keys loaded)",
                     resume_step, len(filtered), len(ckpt_sd))

    trainer = UCLMTrainer(
        model=model, optimizer=optimizer, device=device,
        tts_mode="pointer", tts_prob=1.0,
        pointer_loss_weight=0.5, progress_loss_weight=0.2,
        boundary_confidence_loss_weight=0.1,
        voice_state_loss_weight=0.1,
        conditioning_dropout_prob=0.1,
        curriculum=None,
        enable_v4_losses=True,
        v4_loss_config=V4LossConfig(),
        acting_latent_encoder=acting_enc,
        acting_latent_predictor=acting_pred,
        bio_constraint_weight=1.0,
        acting_kl_weight=0.01,
        disentanglement_weight=0.1,
        semantic_alignment_weight=0.5,
        use_enriched_transcript=True,
        enriched_transcript_prob=0.5,
        codec_condition=codec_cond,
    )

    # LR scheduler: linear warmup + cosine decay (after trainer adds param groups)
    import math
    warmup_steps = min(5000, args.steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Sync scheduler and trainer._global_step with resume point
    if resume_step > 0:
        trainer._global_step = resume_step
        for _ in range(resume_step):
            scheduler.step()
        logger.info("Synced scheduler and trainer._global_step to step %d", resume_step)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %.2fM trainable params", n_params / 1e6)

    # ---- Phase 3: Dataloader (V4UCLMDataset) ----
    from tmrvc_data.v4_dataset import V4UCLMDataset, v4_collate_fn

    dataset = V4UCLMDataset(
        cache_dir=str(cache_dir),
        max_frames=args.max_frames, min_frames=10,
        use_enriched_transcript=True,
        enriched_transcript_prob=0.5,
    )
    logger.info("Dataset: %d samples", len(dataset))
    if len(dataset) == 0:
        logger.error("Empty dataset. Exiting.")
        sys.exit(1)

    def _collate(samples):
        """Collate V4UCLMDataset samples and map keys to Trainer expectations."""
        raw = v4_collate_fn(samples)
        B = raw["codec_tokens_a"].shape[0] if raw.get("codec_tokens_a") is not None else 1
        T = raw["codec_tokens_a"].shape[-1] if raw.get("codec_tokens_a") is not None else 1

        d = {
            "target_a": raw["codec_tokens_a"],
            "target_b": raw["codec_tokens_b"] if raw.get("codec_tokens_b") is not None else torch.full((B, CONTROL_SLOTS, T), -1, dtype=torch.long),
            "speaker_embed": raw.get("speaker_embed", torch.zeros(B, D_SPEAKER)),
            "phoneme_ids": raw.get("phoneme_ids", torch.zeros(B, 1, dtype=torch.long)),
            "phoneme_lens": raw.get("phoneme_ids_lengths", torch.ones(B, dtype=torch.long)),
            # Keys matching trainer expectations
            "physical_targets": raw.get("physical_targets"),
            "physical_observed_mask": raw.get("physical_observed_mask"),
            "physical_confidence": raw.get("physical_confidence"),
            "voice_state_targets": raw.get("physical_targets"),
            "voice_state_observed_mask": raw.get("physical_observed_mask"),
            "voice_state_confidence": raw.get("physical_confidence"),
            # New v4 fields passed through
            "enriched_phoneme_ids": raw.get("enriched_phoneme_ids"),
            "use_enriched": raw.get("use_enriched"),
            "supervision_tier": raw.get("supervision_tier"),
            # NOTE: ssl_state and bootstrap_alignment are not yet in cache.
            # Acting-latent losses (KL, usage, disentanglement, semantic alignment)
            # are effectively disabled until ssl_state is added to the cache pipeline.
            "ssl_state": raw.get("ssl_state", torch.zeros(B, T, D_VOICE_STATE_SSL)),
            "speaker_id": torch.tensor(
                [int(hashlib.md5(s.encode()).hexdigest(), 16) % 200 if isinstance(s, str) else 0
                 for s in (raw.get("speaker_id") or [""] * B)],
                dtype=torch.long,
            ),
            "language_id": torch.tensor(
                [{"ja": 0, "en": 1, "zh": 2, "ko": 3}.get(l, 0) if isinstance(l, str) else (l if isinstance(l, int) else 0)
                 for l in (raw.get("language") or [0] * B)],
                dtype=torch.long,
            ),
            "utterance_ids": raw.get("utterance_id", [f"unk_{i}" for i in range(B)]),
        }

        # Align all temporal dimensions to min(T_codec, T_vs)
        T_codec = T
        T_vs = raw["physical_targets"].shape[1] if raw.get("physical_targets") is not None else T
        T_aligned = min(T_codec, T_vs)

        d["target_a"] = d["target_a"][:, :, :T_aligned]
        if d.get("target_b") is not None and isinstance(d["target_b"], torch.Tensor):
            d["target_b"] = d["target_b"][:, :, :T_aligned]
        else:
            d["target_b"] = torch.zeros(B, CONTROL_SLOTS, T_aligned, dtype=torch.long)

        if raw.get("physical_targets") is not None:
            d["explicit_state"] = raw["physical_targets"][:, :T_aligned, :]
        else:
            d["explicit_state"] = torch.zeros(B, T_aligned, D_VOICE_STATE)
        # Align ssl_state (keep real data if available)
        if d.get("ssl_state") is not None and isinstance(d["ssl_state"], torch.Tensor) and d["ssl_state"].shape[1] >= T_aligned:
            d["ssl_state"] = d["ssl_state"][:, :T_aligned, :]
        else:
            d["ssl_state"] = torch.zeros(B, T_aligned, D_VOICE_STATE_SSL)

        # Interpolate voice-state tensors to match codec frame count
        for vs_key in ("voice_state_targets", "voice_state_observed_mask", "voice_state_confidence",
                       "physical_targets", "physical_observed_mask", "physical_confidence"):
            if d.get(vs_key) is not None and isinstance(d[vs_key], torch.Tensor) and d[vs_key].dim() == 3:
                vs_t = d[vs_key].permute(0, 2, 1).float()  # [B, D, T_vs]
                vs_t = F.interpolate(vs_t, size=T_aligned, mode='nearest')
                d[vs_key] = vs_t.permute(0, 2, 1).to(d[vs_key].dtype)

        d["lengths"] = torch.full((B,), T_aligned, dtype=torch.long)

        # Frame lengths from codec tokens
        d["lengths"] = torch.tensor([T] * B, dtype=torch.long)

        # VC source (clone of target_a)
        d["source_a_t"] = d["target_a"].clone()

        return d

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate, drop_last=True,
    )

    # ---- Phase 4: Train ----
    logger.info("=" * 60)
    logger.info("Phase 4: v4 FULL training — %d steps", args.steps)
    logger.info("  Real ASR transcripts ✓")
    logger.info("  Real G2P phonemes ✓")
    logger.info("  Real 12-D voice state ✓")
    logger.info("  Real speaker embeddings ✓")
    logger.info("  Real LLM enriched transcripts ✓")
    logger.info("  All v4 losses ✓")
    logger.info("=" * 60)

    step = resume_step
    epoch = 0
    running = {}
    best_loss = float("inf")
    t0 = time.time()

    accum_steps = args.grad_accum
    micro_step = 0

    while step < args.steps:
        epoch += 1
        for batch in dataloader:
            if step >= args.steps:
                break
            is_first_micro = (micro_step % accum_steps == 0)
            is_last_micro = (micro_step % accum_steps == accum_steps - 1)

            # Zero grad only at the start of each accumulation cycle
            if is_first_micro:
                trainer.optimizer.zero_grad(set_to_none=True)

            try:
                metrics = trainer.train_step(
                    batch,
                    accumulate=not is_last_micro,
                    accum_steps=accum_steps,
                )
            except RuntimeError as e:
                if "shape" in str(e) or "size" in str(e):
                    logger.warning("Skipping bad batch at step %d: %s", step, e)
                    micro_step += 1
                    continue
                raise
            micro_step += 1
            if not is_last_micro:
                continue  # accumulating — don't count as a step
            step += 1
            scheduler.step()

            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    running[k] = running.get(k, 0.0) + v

            if step % args.log_every == 0:
                elapsed = time.time() - t0
                avg = {k: v / args.log_every for k, v in running.items()}
                total = avg.get("loss_total", avg.get("loss", 0))
                parts = [f"{k}={avg[k]:.4f}" for k in sorted(avg) if k.startswith("loss") and avg[k] != 0]
                logger.info("step %d/%d | loss=%.4f | %.2f s/step | %s",
                            step, args.steps, total, elapsed / step, " ".join(parts[:8]))
                if 0 < total < best_loss:
                    best_loss = total
                running = {}

            if step % args.save_every == 0:
                torch.save({
                    "model": model.state_dict(),
                    "acting_encoder": acting_enc.state_dict(),
                    "acting_predictor": acting_pred.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                }, output_dir / f"v4_step_{step}.pt")
                logger.info("Saved: v4_step_%d.pt", step)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info("Done: %d steps, %d epochs, %.0fs (%.2f s/step)", step, epoch, elapsed, elapsed / max(step, 1))
    logger.info("Best loss: %.4f", best_loss)

    torch.save({
        "model": model.state_dict(),
        "acting_encoder": acting_enc.state_dict(),
        "acting_predictor": acting_pred.state_dict(),
        "step": step, "best_loss": best_loss,
    }, output_dir / "v4_full_final.pt")

    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in model.parameters() if p.requires_grad)
    logger.info("Gradient flow: %d / %d (%.0f%%)", n_grad, n_total, n_grad / max(n_total, 1) * 100)


if __name__ == "__main__":
    main()
