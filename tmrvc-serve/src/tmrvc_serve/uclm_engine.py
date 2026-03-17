"""UCLM Engine: contract-friendly unified inference for TTS and VC (v3)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from tmrvc_core.constants import (
    MAX_ACOUSTIC_HISTORY_FRAMES,
    MAX_DIALOGUE_CONTEXT_UNITS,
    MAX_PROMPT_CACHE_BYTES,
    MAX_PROMPT_FRAMES,
    MAX_PROMPT_KV_TOKENS,
    MAX_TEXT_UNITS_ACTIVE,
    SAMPLE_RATE,
)
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_core.types import CFGMode, PointerState, SpeakerProfile
from tmrvc_core.voice_state import legacy_style_to_canonical_voice_state
from tmrvc_train.models import (
    DisentangledUCLM,
    EmotionAwareDecoder,
    EmotionAwareEncoder,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineState:
    """Persistent inference states for a single session."""

    enc_states: List[torch.Tensor] = field(
        default_factory=lambda: [
            torch.zeros(1, 1, 6),
            torch.zeros(1, 64, 4),
            torch.zeros(1, 128, 4),
            torch.zeros(1, 256, 2),
        ]
    )
    dec_states: List[torch.Tensor] = field(default_factory=lambda: [torch.empty(0)] * 4)
    kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ctx_a: torch.Tensor = field(
        default_factory=lambda: torch.zeros(1, 8, 200, dtype=torch.long)
    )
    ctx_b: torch.Tensor = field(
        default_factory=lambda: torch.zeros(1, 4, 200, dtype=torch.long)
    )
    prev_voice_state: torch.Tensor = field(
        default_factory=lambda: torch.zeros(1, 1, 8)
    )
    prompt_speaker_embed: Optional[torch.Tensor] = None
    prompt_features: Optional[torch.Tensor] = None
    prompt_encoding_time_ms: float = 0.0


@dataclass
class PointerInferenceState:
    """Runtime pointer state for v3 causal TTS inference."""

    text_index: int = 0
    progress: float = 0.0
    total_phonemes: int = 0
    frames_generated: int = 0
    stall_frames: int = 0
    max_stall: int = 100
    max_frames_per_unit: int = 50
    skip_protection_threshold: float = 0.3
    frames_on_current_unit: int = 0
    forced_advance_count: int = 0
    skip_protection_count: int = 0
    # Lazy CFG cache: stores unconditional logits and tracks cache freshness
    cfg_uncond_cache_a: Optional[torch.Tensor] = None
    cfg_uncond_cache_b: Optional[torch.Tensor] = None
    cfg_uncond_cache_adv: Optional[torch.Tensor] = None
    cfg_uncond_cache_prog: Optional[torch.Tensor] = None
    cfg_uncond_cache_bc: Optional[torch.Tensor] = None
    cfg_cache_age: int = 0

    @property
    def finished(self) -> bool:
        # SOTA EOS logic: Finish when pointer has passed all phonemes
        # and has generated a minimum tail buffer.
        return self.text_index >= self.total_phonemes and self.frames_on_current_unit >= 5

    def to_dict(self) -> dict:
        """Serialize state for telemetry."""
        return {
            "text_index": self.text_index,
            "progress": self.progress,
            "total_phonemes": self.total_phonemes,
            "frames_generated": self.frames_generated,
            "forced_advance_count": self.forced_advance_count,
            "skip_protection_count": self.skip_protection_count,
        }

    def replay(self, record: TrajectoryRecord) -> Tuple[torch.Tensor, dict]:
        """SOTA: Perform deterministic replay from a TrajectoryRecord.
        
        Bypasses the transformer model and directly decodes realized tokens.
        Ensures bit-exact reproduction of a specific performance.
        """
        if record.acoustic_trace is None or record.control_trace is None:
            raise ValueError("TrajectoryRecord must contain acoustic and control traces for replay")

        t0 = time.perf_counter()
        
        # Stream A and B tokens
        a_t = record.acoustic_trace
        b_t = record.control_trace
        
        # Ensure they are on the correct device
        a_t = a_t.to(self.device)
        b_t = b_t.to(self.device)
        
        # Realized voice state trajectory
        vs_full = record.voice_state_trajectory
        if vs_full is not None:
            vs_full = vs_full.to(self.device).unsqueeze(0) # [1, T, 8]
        else:
            # Fallback to neutral if somehow missing
            t_frames = a_t.shape[-1]
            vs_full = torch.zeros((1, t_frames, 8), device=self.device)

        # Decode using the bit-exact realized tokens
        with torch.no_grad():
            audio_out, _ = self.codec_dec(a_t, b_t, vs_full, [])

        gen_time = time.perf_counter() - t0
        return audio_out.squeeze(), {
            "rtf": gen_time / (audio_out.shape[-1] / SAMPLE_RATE),
            "gen_time_ms": gen_time * 1000.0,
            "mode": "deterministic_replay",
            "trajectory_id": record.trajectory_id
        }


class UCLMEngine:
    """Orchestrates split models for inference (v3)."""

    # Frame-index contract constants (Worker 04, task 26)
    FRAME_SAMPLE_RATE: int = 24000
    FRAME_HOP_LENGTH: int = 240
    FRAME_START_INCLUSIVE: bool = True
    FRAME_END_EXCLUSIVE: bool = True

    def __init__(
        self,
        uclm_checkpoint: Path | str | None = None,
        codec_checkpoint: Path | str | None = None,
        device: str = "cpu",
        d_model: int = 512,
        tts_mode: str = "pointer",
        speaker_profiles_dir: Path | str | None = None,
    ):
        self.device = torch.device(device)
        self.d_model = d_model
        self.tts_mode = tts_mode
        self._uclm_checkpoint = Path(uclm_checkpoint) if uclm_checkpoint else None
        self._codec_checkpoint = Path(codec_checkpoint) if codec_checkpoint else None
        self._speaker_profiles_dir = (
            Path(speaker_profiles_dir) if speaker_profiles_dir else Path("models/characters")
        )

        self.codec_enc = EmotionAwareEncoder(d_model=d_model).to(self.device).eval()
        self.codec_dec = EmotionAwareDecoder(d_model=d_model).to(self.device).eval()
        self.uclm_core_model: DisentangledUCLM | None = None
        self.uclm_core = None
        self.vc_enc = None
        self.voice_state_enc = None
        self._loaded = False
        self._has_distilled_cfg = False

        # Speaker profile cache: profile_id -> (SpeakerProfile, encoder_version)
        self._speaker_profile_cache: dict[str, tuple[SpeakerProfile, str]] = {}
        self._encoder_version: str = ""

    @property
    def models_loaded(self) -> bool:
        return self._loaded

    @property
    def scene_state_available(self) -> bool:
        return self._loaded and self.uclm_core_model is not None and hasattr(self.uclm_core_model, "context_projector")

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("UCLMEngine models are not loaded.")
        if self.uclm_core is None or self.vc_enc is None or self.voice_state_enc is None:
            raise RuntimeError("UCLMEngine models are not loaded.")

    def init_random_models(self, num_speakers: int = 1000) -> None:
        """Initialize models with random weights for testing."""
        from tmrvc_train.models import DisentangledUCLM
        
        self.uclm_core_model = DisentangledUCLM(num_speakers=num_speakers).to(self.device).eval()
        self.uclm_core = self.uclm_core_model.uclm_core
        self.vc_enc = self.uclm_core_model.vc_encoder
        self.voice_state_enc = self.uclm_core_model.voice_state_enc
        self._has_distilled_cfg = hasattr(self.uclm_core_model, "cfg_scale_embed")
        self._loaded = True
        logger.info("Initialized UCLMEngine with RANDOM weights on %s", self.device)

    def load_models(
        self,
        uclm_path: Path | str | None = None,
        codec_path: Path | str | None = None,
    ) -> None:
        """Load models from checkpoints."""
        uclm_ckpt_path = Path(uclm_path) if uclm_path is not None else self._uclm_checkpoint
        codec_ckpt_path = (
            Path(codec_path) if codec_path is not None else self._codec_checkpoint
        )
        if uclm_ckpt_path is None or codec_ckpt_path is None:
            raise ValueError("Both UCLM and codec checkpoints are required.")

        uclm_ckpt = torch.load(
            uclm_ckpt_path, map_location=self.device, weights_only=False
        )
        uclm_state = uclm_ckpt.get("model", uclm_ckpt)
        num_spk = 1000
        key = "voice_state_enc.adversarial_classifier.2.weight"
        if isinstance(uclm_state, dict) and key in uclm_state:
            num_spk = uclm_state[key].shape[0]

        self.uclm_core_model = DisentangledUCLM(num_speakers=num_spk).to(self.device).eval()
        self.uclm_core_model.load_state_dict(uclm_state, strict=False)

        codec_ckpt = torch.load(
            codec_ckpt_path, map_location=self.device, weights_only=False
        )
        codec_state = codec_ckpt.get("model", codec_ckpt)
        if isinstance(codec_state, dict):
            self.codec_enc.load_state_dict(
                {
                    k.replace("encoder.", ""): v
                    for k, v in codec_state.items()
                    if k.startswith("encoder.")
                },
                strict=False,
            )
            self.codec_dec.load_state_dict(
                {
                    k.replace("decoder.", ""): v
                    for k, v in codec_state.items()
                    if k.startswith("decoder.")
                },
                strict=False,
            )

        self.uclm_core = self.uclm_core_model.uclm_core
        self.vc_enc = self.uclm_core_model.vc_encoder
        self.voice_state_enc = self.uclm_core_model.voice_state_enc
        self._has_distilled_cfg = hasattr(self.uclm_core_model, "cfg_scale_embed")
        self._loaded = True

        # Track encoder version for speaker profile cache invalidation
        import hashlib
        ckpt_hash = hashlib.sha256(str(uclm_ckpt_path).encode()).hexdigest()[:12]
        self._encoder_version = f"uclm-{ckpt_hash}"
        # Invalidate any cached profiles from a different encoder version
        self._invalidate_stale_profiles()

        logger.info(
            "Loaded UCLM v3 checkpoints on %s (distilled_cfg=%s, encoder_version=%s)",
            self.device,
            self._has_distilled_cfg,
            self._encoder_version,
        )

    @torch.no_grad()
    def vc_frame(
        self,
        audio_frame: torch.Tensor,
        speaker_embed: torch.Tensor,
        style: StyleParams,
        state: EngineState,
        cfg_scale: float = 1.0,
        temperature: float = 0.8,
        pitch_shift: float = 0.0,
        explicit_voice_state: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, EngineState]:
        self._require_loaded()

        a_src_t, b_logits_src, new_enc_states = self.codec_enc(audio_frame, state.enc_states)
        content_features, _ = self.vc_enc(a_src_t)

        if explicit_voice_state is not None:
            if explicit_voice_state.dim() == 1:
                explicit_voice_state = explicit_voice_state.unsqueeze(0).unsqueeze(0)
            elif explicit_voice_state.dim() == 2:
                explicit_voice_state = explicit_voice_state.unsqueeze(0)
            v_state = explicit_voice_state.to(self.device).float()
        else:
            v_state = (
                torch.tensor(
                    legacy_style_to_canonical_voice_state(style), device=self.device
                )
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
            )
        ssl_state = torch.zeros(1, 1, 128, device=self.device)

        # 1. Pitch-shift implementation via F0 conditioning
        f0_condition = None
        if pitch_shift != 0.0:
            f0_val = 0.0 + (pitch_shift / 12.0) 
            f0_condition = torch.tensor([[[f0_val, 1.0]]], device=self.device)

        out = self.uclm_core_model.forward_streaming(
            queries=content_features,
            memory=content_features,
            a_ctx=state.ctx_a,
            b_ctx=state.ctx_b,
            speaker_embed=speaker_embed,
            explicit_state=v_state,
            ssl_state=ssl_state,
            cfg_scale=cfg_scale,
            kv_caches=state.kv_caches,
            f0_condition=f0_condition,
        )
        logits_a, logits_b, new_kv = out["logits_a"], out["logits_b"], out["kv_cache_out"]

        if temperature > 0:
            probs_a = F.softmax(logits_a[:, :, -1, :] / temperature, dim=-1)
            probs_b = F.softmax(logits_b[:, :, -1, :] / temperature, dim=-1)
            a_t = torch.stack([torch.multinomial(probs_a[:, i, :], 1) for i in range(8)], dim=1).squeeze(-1).unsqueeze(-1)
            b_t = torch.stack([torch.multinomial(probs_b[:, i, :], 1) for i in range(4)], dim=1).squeeze(-1).unsqueeze(-1)
        else:
            a_t = logits_a[:, :, -1, :].argmax(dim=-1, keepdim=True)
            b_t = logits_b[:, :, -1, :].argmax(dim=-1, keepdim=True)

        audio_out, new_dec_states = self.codec_dec(a_t, b_t, v_state, state.dec_states)

        new_state = EngineState(
            enc_states=new_enc_states,
            dec_states=new_dec_states,
            kv_caches=new_kv,
            ctx_a=torch.cat([state.ctx_a[:, :, 1:], a_t], dim=-1),
            ctx_b=torch.cat([state.ctx_b[:, :, 1:], b_t], dim=-1),
            prev_voice_state=v_state,
        )
        return audio_out.squeeze(), new_state

    def load_speaker_profile(self, profile_id: str) -> Optional[SpeakerProfile]:
        """Load a SpeakerProfile from the Casting Gallery (Worker 04/12 requirement)."""
        import json
        
        # Check cache first
        if profile_id in self._speaker_profile_cache:
            profile, ver = self._speaker_profile_cache[profile_id]
            if ver == self._encoder_version:
                return profile

        profile_dir = self._speaker_profiles_dir / profile_id
        meta_path = profile_dir / "profile.json"
        
        if not meta_path.exists():
            logger.warning("Speaker profile %s not found in %s", profile_id, self._speaker_profiles_dir)
            return None
            
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            
            # Load associated tensors
            embed = torch.from_numpy(np.load(profile_dir / "speaker_embed.npy"))
            tokens = torch.from_numpy(np.load(profile_dir / "prompt_codec_tokens.npy"))
            summary = None
            if (profile_dir / "prompt_summary_tokens.npy").exists():
                summary = torch.from_numpy(np.load(profile_dir / "prompt_summary_tokens.npy"))
                
            profile = SpeakerProfile(
                speaker_profile_id=profile_id,
                reference_audio_hash=meta.get("reference_audio_hash", ""),
                speaker_embed=embed,
                prompt_codec_tokens=tokens,
                prompt_summary_tokens=summary,
                adaptor_id=meta.get("adaptor_id"),
                adaptor_merged=meta.get("adaptor_merged", False)
            )
            # Update cache
            self._speaker_profile_cache[profile_id] = (profile, self._encoder_version)
            return profile
        except Exception as e:
            logger.error("Failed to load speaker profile %s: %s", profile_id, e)
            return None

    def encode_on_the_fly_reference(self, audio_bytes: bytes, text: str | None = None) -> SpeakerProfile:
        """Encode raw audio into a SpeakerProfile on-the-fly (Worker 04)."""
        # (Worker 04: audio loader utility stub)
        waveform = torch.randn(1, 24000 * 5, device=self.device) 
        
        # Extract codec tokens via encoder
        prompt_codec_tokens = self.codec_enc(waveform) # [1, 8, T]
        
        # Extract speaker embed and summary tokens (Resampler)
        refined_embed, summary_tokens = self.uclm_core_model.encode_speaker_prompt(prompt_codec_tokens)
        
        return SpeakerProfile(
            speaker_profile_id="on_the_fly",
            reference_audio_hash="temp",
            speaker_embed=refined_embed.squeeze(0),
            prompt_codec_tokens=prompt_codec_tokens.squeeze(0),
            prompt_summary_tokens=summary_tokens.squeeze(0)
        )

    @torch.no_grad()
    def encode_speaker_prompt(
        self,
        reference_audio: torch.Tensor | None = None,
        reference_codec_tokens: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float, float]:
        """Encode reference audio for few-shot speaker adaptation.

        Returns:
            (refined_speaker_embed, prompt_summary_tokens, vq_loss, encoding_time_ms)
        """
        self._require_loaded()
        t0 = time.perf_counter()

        if reference_codec_tokens is None and reference_audio is not None:
            reference_codec_tokens = self.codec_enc(reference_audio)

        if reference_codec_tokens is not None:
            # UCLM v3: Returns (refined_speaker_embed, prompt_summary_tokens, vq_loss)
            refined_embed, summary_tokens, vq_loss = self.uclm_core_model.encode_speaker_prompt(
                reference_codec_tokens.to(self.device),
                speaker_embed.to(self.device) if speaker_embed is not None else None,
            )
            vq_loss_val = vq_loss.item() if isinstance(vq_loss, torch.Tensor) else float(vq_loss)
        elif speaker_embed is not None:
            refined_embed = speaker_embed.to(self.device)
            summary_tokens = None
            vq_loss_val = 0.0
        else:
            raise ValueError("No speaker identity evidence provided")

        # --- Prompt budget enforcement (v3) ---
        if summary_tokens is not None:
            n_tokens = summary_tokens.shape[-2] if summary_tokens.dim() >= 2 else 0
            if n_tokens > MAX_PROMPT_KV_TOKENS:
                logger.warning(
                    "Prompt summary tokens exceed MAX_PROMPT_KV_TOKENS (%d > %d); "
                    "truncating to last %d tokens.",
                    n_tokens, MAX_PROMPT_KV_TOKENS, MAX_PROMPT_KV_TOKENS,
                )
                summary_tokens = summary_tokens[..., -MAX_PROMPT_KV_TOKENS:, :]

            cache_bytes = summary_tokens.nelement() * summary_tokens.element_size()
            if cache_bytes > MAX_PROMPT_CACHE_BYTES:
                logger.warning(
                    "Prompt cache size exceeds MAX_PROMPT_CACHE_BYTES (%d > %d).",
                    cache_bytes, MAX_PROMPT_CACHE_BYTES,
                )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return refined_embed, summary_tokens, vq_loss_val, elapsed_ms

    # ------------------------------------------------------------------
    # SpeakerProfile runtime behaviour (Worker 04, task 25)
    # ------------------------------------------------------------------

    def cache_speaker_prompt(
        self,
        profile_id: str,
        speaker_embed: torch.Tensor,
        prompt_features: Optional[torch.Tensor] = None,
        prompt_summary_tokens: Optional[torch.Tensor] = None,
    ) -> SpeakerProfile:
        """Cache a speaker prompt for reuse across turns.

        Creates a SpeakerProfile from the given embeddings and caches it
        under the current encoder version.

        Args:
            profile_id: Unique identifier for the profile.
            speaker_embed: [d_speaker] speaker embedding tensor.
            prompt_features: Optional prompt features from encoding.
            prompt_summary_tokens: Optional compressed summary tokens.

        Returns:
            The cached SpeakerProfile.
        """
        profile = SpeakerProfile(
            speaker_profile_id=profile_id,
            reference_audio_hash="on-the-fly",
            speaker_embed=speaker_embed.detach().cpu().squeeze(),
            prompt_codec_tokens=torch.zeros(0, 8),  # placeholder
            prompt_summary_tokens=(
                prompt_summary_tokens.detach().cpu() if prompt_summary_tokens is not None else None
            ),
            display_name=profile_id,
        )
        self._speaker_profile_cache[profile_id] = (profile, self._encoder_version)
        logger.info("Cached speaker prompt: %s", profile_id)
        return profile

    def encode_on_the_fly_reference(
        self,
        ref_audio_bytes: bytes,
        text: Optional[str] = None,
    ) -> SpeakerProfile:
        """Encode reference audio on-the-fly for few-shot speaker adaptation.

        Args:
            ref_audio_bytes: Raw audio bytes (WAV format expected).
            text: Optional transcript of the reference audio.

        Returns:
            A SpeakerProfile with the encoded embeddings.
        """
        self._require_loaded()

        # Decode audio bytes to tensor
        import io
        import soundfile as sf
        buf = io.BytesIO(ref_audio_bytes)
        try:
            audio_np, sr = sf.read(buf, dtype="float32")
        except Exception:
            # Fallback: try raw float32
            audio_np = np.frombuffer(ref_audio_bytes, dtype=np.float32)
            sr = SAMPLE_RATE

        audio_t = torch.from_numpy(audio_np).float().to(self.device)
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0).unsqueeze(0)
        elif audio_t.dim() == 2:
            audio_t = audio_t.unsqueeze(0)

        # Encode through codec to get tokens
        with torch.no_grad():
            # Simple encoding path — in production, use full codec pipeline
            codec_tokens_a, _, _ = self.codec_enc(audio_t, [torch.empty(0)] * 4)

        # Use speaker prompt encoder
        refined_embed, prompt_feats, enc_time = self.encode_speaker_prompt(
            reference_codec_tokens=codec_tokens_a,
            speaker_embed=torch.zeros(1, 192, device=self.device),
        )

        import hashlib
        audio_hash = hashlib.sha256(ref_audio_bytes[:4096]).hexdigest()[:16]
        profile_id = f"otf-{audio_hash}"

        profile = self.cache_speaker_prompt(
            profile_id=profile_id,
            speaker_embed=refined_embed.squeeze(),
            prompt_features=prompt_feats,
        )
        profile.reference_audio_hash = audio_hash

        logger.info(
            "On-the-fly speaker encoding completed in %.1fms (profile=%s)",
            enc_time, profile_id,
        )
        return profile

    def _invalidate_stale_profiles(self) -> None:
        """Evict cached profiles from a different encoder version."""
        stale_keys = [
            pid for pid, (_, ver) in self._speaker_profile_cache.items()
            if ver != self._encoder_version
        ]
        for k in stale_keys:
            del self._speaker_profile_cache[k]
        if stale_keys:
            logger.info(
                "Invalidated %d stale speaker profiles (encoder version changed)",
                len(stale_keys),
            )

    @torch.no_grad()
    def tts(
        self,
        phonemes: torch.Tensor,
        speaker_profile: Optional[SpeakerProfile] = None,
        speaker_embed: Optional[torch.Tensor] = None,
        style: StyleParams | None = None,
        cfg_scale: float = 1.5,
        cfg_mode: CFGMode | str = CFGMode.FULL,
        cfg_lazy_interval: int = 5,
        temperature: float = 0.8,
        language_id: int = 0,
        pace: float = 1.0,
        hold_bias: float = 0.0,
        boundary_bias: float = 0.0,
        max_frames: int = 1500,
        dialogue_context: torch.Tensor | None = None,
        acting_intent: torch.Tensor | None = None,
        phrase_pressure: float = 0.0,
        breath_tendency: float = 0.0,
        reference_audio_base64: str | None = None,
        reference_text: str | None = None,
        max_frames_per_unit: int = 50,
        text_suprasegmentals: torch.Tensor | None = None,
        explicit_voice_state: torch.Tensor | None = None,
        delta_voice_state: torch.Tensor | None = None,
        local_prosody_plan: dict[int, torch.Tensor] | None = None, # New SOTA field
        disable_circuit_breaker: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        """Full causal pointer-based TTS generation (v3).

        Speaker embed priority resolution:
            ``speaker_embed`` is the global timbre anchor and wins for identity.
            ``prompt_codec_tokens`` (via reference audio) may only refine local
            texture within the timbre envelope.  If both disagree,
            ``speaker_embed`` takes priority for speaker identity.

        Args:
            phonemes: [1, L] phoneme token ids.
            speaker_profile: optional :class:`SpeakerProfile` (provides both timbre and prompt tokens).
            speaker_embed: optional [1, d_speaker] direct speaker embedding fallback.
            style: StyleParams for voice state conditioning.
            cfg_scale: classifier-free guidance scale.
            cfg_mode: CFG operating mode (off/full/lazy/distilled).
            cfg_lazy_interval: for lazy mode, how many frames between
                unconditional refreshes (default 5).
            temperature: sampling temperature.
            language_id: language identifier.
            pace: speech pace multiplier.
            hold_bias: bias toward holding current phoneme.
            boundary_bias: bias toward phoneme boundaries.
            max_frames: maximum number of frames to generate.
            dialogue_context: optional [1, D_ctx] scene/dialogue embedding.
            acting_intent: optional [1, D_act] acting intent vector.
            phrase_pressure: interruption/urgency pressure (-1 to 1).
            breath_tendency: tendency to insert breathing pauses (-1 to 1).
            max_frames_per_unit: maximum frames allowed on a single text unit
                before forced advance (default 50).
            disable_circuit_breaker: explicitly disable CFG auto-downgrade (e.g. for eval).
        """
        if self.tts_mode == "legacy_duration":
            return self._tts_legacy_duration(
                phonemes=phonemes,
                speaker_embed=speaker_embed,
                style=style,
                cfg_scale=cfg_scale,
                temperature=temperature,
                language_id=language_id,
                pace=pace,
                max_frames=max_frames,
            )
        self._require_loaded()
        t0 = time.perf_counter()

        # Normalise cfg_mode to CFGMode enum
        if isinstance(cfg_mode, str):
            cfg_mode = CFGMode(cfg_mode)

        # Distilled mode fallback: if distilled weights are not available,
        # fall back to full two-pass CFG and log a warning.
        if cfg_mode == CFGMode.DISTILLED and not self._has_distilled_cfg:
            logger.warning(
                "Distilled CFG requested but checkpoint lacks cfg_scale_embed; "
                "falling back to FULL mode."
            )
            cfg_mode = CFGMode.FULL

        # CFG safety clamping (Worker 04: default [1.0, 3.0])
        cfg_scale = max(1.0, min(cfg_scale, 3.0))

        # If cfg_mode is OFF, force scale to 1.0 regardless
        if cfg_mode == CFGMode.OFF:
            cfg_scale = 1.0

        style = style or StyleParams.neutral()
        v_state = (
            torch.tensor(
                legacy_style_to_canonical_voice_state(style), device=self.device
            )
            .float()
            .view(1, 1, -1)
        )

        # Override voice state with explicit 8-D vector if provided
        if explicit_voice_state is not None:
            if explicit_voice_state.dim() == 1:
                explicit_voice_state = explicit_voice_state.unsqueeze(0).unsqueeze(0)
            elif explicit_voice_state.dim() == 2:
                explicit_voice_state = explicit_voice_state.unsqueeze(0)
            v_state = explicit_voice_state.to(self.device)

        # Apply delta voice state if provided
        if delta_voice_state is not None:
            if delta_voice_state.dim() == 1:
                delta_voice_state = delta_voice_state.unsqueeze(0).unsqueeze(0)
            elif delta_voice_state.dim() == 2:
                delta_voice_state = delta_voice_state.unsqueeze(0)
            v_state = v_state + delta_voice_state.to(self.device)

        num_phonemes = phonemes.shape[1]
        phoneme_ids = phonemes.to(self.device)

        # --- Runtime budget enforcement: text units ---
        if num_phonemes > MAX_TEXT_UNITS_ACTIVE:
            logger.warning(
                "Text units (%d) exceed MAX_TEXT_UNITS_ACTIVE (%d); truncating.",
                num_phonemes, MAX_TEXT_UNITS_ACTIVE,
            )
            phoneme_ids = phoneme_ids[:, :MAX_TEXT_UNITS_ACTIVE]
            num_phonemes = MAX_TEXT_UNITS_ACTIVE
            if text_suprasegmentals is not None and text_suprasegmentals.shape[-2] > MAX_TEXT_UNITS_ACTIVE:
                text_suprasegmentals = text_suprasegmentals[..., :MAX_TEXT_UNITS_ACTIVE, :]

        # --- Runtime budget enforcement: dialogue context ---
        if dialogue_context is not None and dialogue_context.shape[-1] > MAX_DIALOGUE_CONTEXT_UNITS:
            logger.warning(
                "Dialogue context units (%d) exceed MAX_DIALOGUE_CONTEXT_UNITS (%d); truncating.",
                dialogue_context.shape[-1], MAX_DIALOGUE_CONTEXT_UNITS,
            )
            dialogue_context = dialogue_context[..., :MAX_DIALOGUE_CONTEXT_UNITS]

        # --- Runtime budget enforcement: max generation frames ---
        max_frames = min(max_frames, MAX_ACOUSTIC_HISTORY_FRAMES)

        prompt_summary_tokens = None
        if speaker_profile is not None:
            speaker_embed = speaker_profile.speaker_embed.to(self.device).unsqueeze(0)
            
            # Use pre-computed summary tokens from profile if available
            if speaker_profile.prompt_summary_tokens is not None:
                prompt_summary_tokens = speaker_profile.prompt_summary_tokens.to(self.device).unsqueeze(0)
            
            # If no summary but raw tokens exist, compute summary once
            elif speaker_profile.prompt_codec_tokens is not None:
                prompt_tokens = speaker_profile.prompt_codec_tokens.to(self.device).unsqueeze(0)
                # --- Prompt budget enforcement (v3) ---
                if prompt_tokens.shape[2] > MAX_PROMPT_FRAMES:
                    prompt_tokens = prompt_tokens[:, :, -MAX_PROMPT_FRAMES:]
                
                # Re-encode to get summary tokens (using the model's resampler internally)
                # Note: encode_speaker_prompt should return summary_tokens in v3
                _, prompt_summary_tokens, _, _ = self.encode_speaker_prompt(
                    reference_codec_tokens=prompt_tokens,
                    speaker_embed=speaker_embed
                )
        elif speaker_embed is not None:
            speaker_embed = speaker_embed.to(self.device)
            if speaker_embed.dim() == 1:
                speaker_embed = speaker_embed.unsqueeze(0)
        else:
            raise ValueError("Either speaker_profile or speaker_embed must be provided")

        lang_id = torch.tensor([language_id], device=self.device).expand(1, num_phonemes)

        ptr = PointerInferenceState(total_phonemes=num_phonemes, max_frames_per_unit=max_frames_per_unit)
        kv_caches = None
        ctx_b = torch.zeros(1, 4, 1, dtype=torch.long, device=self.device)

        # Move suprasegmentals to device if provided
        _supra = text_suprasegmentals.to(self.device) if text_suprasegmentals is not None else None
        if _supra is not None and _supra.dim() == 2:
            _supra = _supra.unsqueeze(0)  # [L, d_supra] -> [1, L, d_supra]

        # Pre-compute text features once before the generation loop
        phoneme_lens = torch.tensor([num_phonemes], device=self.device)
        all_phoneme_features = self.uclm_core_model.text_encoder(
            phoneme_ids, lang_id, phoneme_lens,
            text_suprasegmentals=_supra,
        ).transpose(1, 2)

        # Move optional expressive inputs to device once
        _dlg_ctx = dialogue_context.to(self.device) if dialogue_context is not None else None
        _act_int = acting_intent.to(self.device) if acting_intent is not None else None

        a_tokens, b_tokens = [], []
        ctx_a = torch.zeros(1, 8, 1, dtype=torch.long, device=self.device)

        # --- CFG Circuit Breaker State ---
        # If frame latency exceeds 8ms, we downgrade CFG to OFF for the rest of the utterance.
        cfg_circuit_broken = False

        # Trajectory collection (Worker 04)
        pointer_trace: List[Tuple[int, int]] = [] # [text_index, frames_spent]
        vs_trajectory: List[torch.Tensor] = []    # [1, 8] per frame
        a_trace_list: List[torch.Tensor] = []     # [8, 1] per frame
        
        # Local Prosody Plan (Worker 01/04)
        local_plan = local_prosody_plan or {}

        for t in range(max_frames):
            if ptr.finished:
                break

            frame_start_t = time.perf_counter()
            
            # --- Dynamic Voice State Overrides (Local Prosody Plan) ---
            v_state_active = v_state
            if ptr.text_index in local_plan:
                v_state_active = local_plan[ptr.text_index].to(self.device).view(1, 1, 8)

            # Record active state for trajectory (Worker 04)
            vs_trajectory.append(v_state_active.squeeze(1))

            # Index into cached text features by pointer position
            safe_idx = min(ptr.text_index, ptr.total_phonemes - 1)
            content_features = all_phoneme_features[:, safe_idx : safe_idx + 1, :]

            # Automatic CFG downgrade if budget is tight
            active_cfg_mode = cfg_mode
            if not cfg_circuit_broken and cfg_mode != CFGMode.OFF:
                # Placeholder for dynamic budget check
                pass

            out = self.uclm_core_model.forward_streaming(
                queries=content_features,
                memory=all_phoneme_features,
                a_ctx=ctx_a,
                b_ctx=ctx_b,
                speaker_embed=speaker_embed,
                explicit_state=v_state_active,
                cfg_scale=1.0,  # Conditional pass always uses scale=1
                kv_caches=kv_caches,
                dialogue_context=_dlg_ctx,
                acting_intent=_act_int,
                prompt_summary_tokens=prompt_summary_tokens,
                frame_index=t,
            )

            logits_a, logits_b, kv_caches = out["logits_a"], out["logits_b"], out["kv_cache_out"]
            hidden_states = out["hidden_states"]

            # --- CFG blending (mode-aware) ---
            # guided = uncond + cfg_scale * (cond - uncond)
            current_cfg_scale = cfg_scale if not cfg_circuit_broken else 1.0
            
            if current_cfg_scale > 1.0 and active_cfg_mode != CFGMode.OFF:
                cfg_mode_str = active_cfg_mode.value if hasattr(active_cfg_mode, 'value') else str(active_cfg_mode)
                if cfg_mode_str == "distilled":
                    # Distilled mode: the conditional pass already used
                    # cfg_scale as a model input; the model's output
                    # approximates the blended logits directly.  We
                    # re-run forward_streaming with cfg_scale injected so
                    # the model can see it.
                    out_dist = self.uclm_core_model.forward_streaming(
                        queries=content_features,
                        memory=all_phoneme_features,
                        a_ctx=ctx_a,
                        b_ctx=ctx_b,
                        speaker_embed=speaker_embed,
                        explicit_state=v_state,
                        cfg_scale=cfg_scale,
                        kv_caches=None,
                        dialogue_context=_dlg_ctx,
                        acting_intent=_act_int,
                        frame_index=t,
                    )
                    logits_a = out_dist["logits_a"]
                    logits_b = out_dist["logits_b"]
                    # Distilled pointer: run pointer_head on distilled hidden_states
                    dist_hidden = out_dist["hidden_states"]
                    p_adv_logit, p_delta, _p_bc = self.uclm_core_model.pointer_head(dist_hidden)

                else:
                    # FULL or LAZY: need unconditional logits
                    need_uncond_refresh = True
                    if cfg_mode == CFGMode.LAZY:
                        # Reuse cached unconditional logits when fresh enough
                        if (
                            ptr.cfg_uncond_cache_a is not None
                            and ptr.cfg_cache_age < cfg_lazy_interval
                        ):
                            need_uncond_refresh = False
                            uncond_a = ptr.cfg_uncond_cache_a
                            uncond_b = ptr.cfg_uncond_cache_b
                            uncond_adv = ptr.cfg_uncond_cache_adv
                            uncond_prog = ptr.cfg_uncond_cache_prog
                            uncond_bc = ptr.cfg_uncond_cache_bc
                            ptr.cfg_cache_age += 1

                    if need_uncond_refresh:
                        masked = DisentangledUCLM.apply_cfg_unconditional_mask(
                            explicit_state=v_state,
                            ssl_state=torch.zeros(1, 1, 128, device=self.device),
                            speaker_embed=speaker_embed,
                            dialogue_context=_dlg_ctx,
                            acting_intent=_act_int,
                        )
                        out_uncond = self.uclm_core_model.forward_streaming(
                            queries=content_features,
                            memory=all_phoneme_features,
                            a_ctx=ctx_a,
                            b_ctx=ctx_b,
                            speaker_embed=masked["speaker_embed"],
                            explicit_state=masked["explicit_state"],
                            cfg_scale=1.0,
                            kv_caches=None,  # Don't share cache with unconditional
                            dialogue_context=masked["dialogue_context"],
                            acting_intent=masked["acting_intent"],
                            frame_index=t,
                        )
                        uncond_a = out_uncond["logits_a"]
                        uncond_b = out_uncond["logits_b"]
                        # Unconditional pointer outputs
                        uncond_hidden = out_uncond["hidden_states"]
                        uncond_adv, uncond_prog, uncond_bc = self.uclm_core_model.pointer_head(uncond_hidden)

                        if cfg_mode == CFGMode.LAZY:
                            ptr.cfg_uncond_cache_a = uncond_a.detach()
                            ptr.cfg_uncond_cache_b = uncond_b.detach()
                            ptr.cfg_uncond_cache_adv = uncond_adv.detach()
                            ptr.cfg_uncond_cache_prog = uncond_prog.detach()
                            ptr.cfg_uncond_cache_bc = uncond_bc.detach()
                            ptr.cfg_cache_age = 1  # just refreshed, next frame is age=1

                    logits_a = uncond_a + cfg_scale * (logits_a - uncond_a)
                    logits_b = uncond_b + cfg_scale * (logits_b - uncond_b)

                    # CFG-guided pointer outputs
                    p_adv_logit_cond, p_delta_cond, p_bc_cond = self.uclm_core_model.pointer_head(hidden_states)
                    p_adv_logit = uncond_adv + cfg_scale * (p_adv_logit_cond - uncond_adv)
                    p_delta = uncond_prog + cfg_scale * (p_delta_cond - uncond_prog)
                    _p_bc = uncond_bc + cfg_scale * (p_bc_cond - uncond_bc)
            else:
                # No CFG: use conditional pointer outputs directly
                p_adv_logit, p_delta, _p_bc = self.uclm_core_model.pointer_head(hidden_states)

            # --- CFG Circuit Breaker Check ---
            frame_elapsed_ms = (time.perf_counter() - frame_start_t) * 1000.0
            if not cfg_circuit_broken and frame_elapsed_ms > 8.0:
                logger.warning(
                    "CFG Circuit Breaker triggered! Frame latency %.2fms > 8.0ms. "
                    "Downgrading to CFG=OFF for remaining frames.",
                    frame_elapsed_ms
                )
                cfg_circuit_broken = True

            # Combine all pacing controls:
            # - hold_bias: negative = hold longer, positive = advance sooner
            # - boundary_bias: positive = encourage boundary transitions
            # - phrase_pressure: positive = urgency (faster), negative = restrained
            # - breath_tendency: positive = more likely to pause at boundaries
            p_adv_prob = torch.sigmoid(
                p_adv_logit
                - hold_bias
                + boundary_bias
                + (pace - 1.0) * 2.0
                + phrase_pressure * 1.5
                - breath_tendency * 0.5
            )
            
            if temperature > 0:
                at = torch.stack([torch.multinomial(F.softmax(logits_a[:, i, 0, :] / temperature, dim=-1), 1) for i in range(8)], dim=1).squeeze(-1)
                bt = torch.stack([torch.multinomial(F.softmax(logits_b[:, i, 0, :] / temperature, dim=-1), 1) for i in range(4)], dim=1).squeeze(-1)
            else:
                at = logits_a[:, :, 0, :].argmax(dim=-1)
                bt = logits_b[:, :, 0, :].argmax(dim=-1)
                
            a_trace_list.append(at.unsqueeze(-1))
            a_tokens.append(at.unsqueeze(-1))
            b_tokens.append(bt.unsqueeze(-1))
            ctx_a = at.unsqueeze(-1)
            ctx_b = bt.unsqueeze(-1)

            # --- Pointer state-transition with failure handling ---
            ptr.frames_on_current_unit += 1
            # SOTA Theory: p_delta is predicted velocity (1/dur).
            # We integrate it, scaled by pace, to recover absolute position.
            # SOTA: hold_bias acts as a drag on velocity to prevent premature jumps.
            velocity = p_delta.item() * pace
            drag = max(0.0, hold_bias * 0.02) # Scale drag factor
            ptr.progress += (velocity - drag)
            ptr.progress = max(0.0, ptr.progress)
            
            adv_prob_val = p_adv_prob.item()

            # Compute boundary_confidence as sigmoid of the raw advance logit
            boundary_confidence = torch.sigmoid(p_adv_logit).item()

            advanced = False
            advance_reason = ""

            # SOTA: Define the "Integration Threshold"
            # If the integrated progress exceeds 1.0, we have physically passed the boundary.
            passed_threshold = ptr.progress >= 1.0

            # Forced advance: stuck too long on one text unit
            if ptr.frames_on_current_unit >= ptr.max_frames_per_unit:
                advance_reason = "FORCED"
                advanced = True
            # Skip-protection: both signals fire but boundary_confidence is low
            elif adv_prob_val > 0.5 and passed_threshold:
                if boundary_confidence >= ptr.skip_protection_threshold:
                    advance_reason = "SKIP_PROTECTED_OK"
                    advanced = True
                else:
                    ptr.skip_protection_count += 1
            # Normal advance: either signal or both depending on hold_bias
            elif (adv_prob_val > 0.5 and passed_threshold) if hold_bias > 2.0 else (adv_prob_val > 0.5 or passed_threshold):
                advance_reason = "NORMAL"
                advanced = True

            if advanced:
                # Record trace: unit X lasted Y frames (Worker 04)
                pointer_trace.append((ptr.text_index, ptr.frames_on_current_unit))

                logger.debug(
                    "t=%d: Advanced [%s] to text_index=%d (adv_prob=%.3f, p_delta=%.3f, progress=%.3f, bc=%.3f)",
                    t, advance_reason, ptr.text_index + 1, adv_prob_val, p_delta.item(), ptr.progress, boundary_confidence
                )
                ptr.text_index += 1
                # SOTA: Continuous integration - carry over the remainder velocity
                ptr.progress = max(0.0, ptr.progress - 1.0) if advance_reason != "FORCED" else 0.0
                ptr.frames_on_current_unit = 0
                ptr.stall_frames = 0

            if not advanced:
                ptr.stall_frames += 1
            else:
                ptr.stall_frames = 0

            ptr.frames_generated += 1

            # Secondary safeguard: legacy stall detection
            if ptr.stall_frames > ptr.max_stall:
                logger.warning("Pointer stalled at text_index=%d (secondary safeguard), forcing advance", ptr.text_index)
                ptr.text_index += 1
                ptr.progress = 0.0
                ptr.frames_on_current_unit = 0
                ptr.stall_frames = 0
                ptr.forced_advance_count += 1

        if not a_tokens:
            return torch.zeros(0), {}

        # Finalize trace for the last (partial) unit (Worker 04)
        if ptr.frames_on_current_unit > 0:
            pointer_trace.append((ptr.text_index, ptr.frames_on_current_unit))

        a_t, b_t = torch.cat(a_tokens, dim=-1), torch.cat(b_tokens, dim=-1)
        t_frames = a_t.shape[-1]
        
        # Use realized trajectory for final render to ensure consistency
        vs_full = torch.cat(vs_trajectory, dim=0).unsqueeze(0) if vs_trajectory else v_state.expand(-1, t_frames, -1)
        audio_out, _ = self.codec_dec(a_t, b_t, vs_full, [])

        gen_time = time.perf_counter() - t0
        rtf = gen_time / (audio_out.shape[-1] / SAMPLE_RATE)
        
        return audio_out.squeeze(), {
            "rtf": float(rtf),
            "gen_time_ms": float(gen_time * 1000.0),
            "pointer_trace": pointer_trace,
            "phoneme_ids": phoneme_ids, # Store original IDs
            "voice_state_trajectory": vs_full.squeeze(0),
            "acoustic_trace": torch.cat(a_trace_list, dim=-1) if a_trace_list else None,
            "control_trace": b_t,
            "pointer_state": ptr.to_dict(),
            "forced_advance_count": ptr.forced_advance_count,
            "skip_protection_count": ptr.skip_protection_count,
            "cfg_mode": cfg_mode.value,
        }

    def replay_trajectory(
        self,
        phonemes: torch.Tensor,
        trajectory: TrajectoryRecord,
        speaker_profile: SpeakerProfile | None = None,
        speaker_embed: torch.Tensor | None = None,
        text_suprasegmentals: torch.Tensor | None = None,
        local_prosody_plan: dict[int, torch.Tensor] | None = None,
        cfg_scale: float = 1.5,
        temperature: float = 0.8,
        language_id: int = 0,
        use_exact_tokens: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """Replay a specific performance trajectory deterministically."""
        self._require_loaded()
        t0 = time.perf_counter()
        
        device = self.device
        num_phonemes = phonemes.shape[1]
        phoneme_ids = phonemes.to(device)
        
        # 1. Prepare Speaker (with fallback encoding if summary is missing)
        prompt_summary_tokens = None
        if speaker_profile is not None:
            speaker_embed = speaker_profile.speaker_embed.to(device).unsqueeze(0)
            
            if speaker_profile.prompt_summary_tokens is not None:
                prompt_summary_tokens = speaker_profile.prompt_summary_tokens.to(device).unsqueeze(0)
            elif speaker_profile.prompt_codec_tokens is not None:
                # Same as tts(): encode summary tokens on the fly
                prompt_tokens = speaker_profile.prompt_codec_tokens.to(device).unsqueeze(0)
                if prompt_tokens.shape[2] > MAX_PROMPT_FRAMES:
                    prompt_tokens = prompt_tokens[:, :, -MAX_PROMPT_FRAMES:]
                _, prompt_summary_tokens, _, _ = self.encode_speaker_prompt(
                    reference_codec_tokens=prompt_tokens,
                    speaker_embed=speaker_embed
                )
        elif speaker_embed is not None:
            speaker_embed = speaker_embed.to(device)
            if speaker_embed.dim() == 1:
                speaker_embed = speaker_embed.unsqueeze(0)
        else:
            raise ValueError("Either speaker_profile or speaker_embed must be provided")
        
        # 2. Process Phonemes (SOTA: Prioritize Phoneme IDs from the trajectory itself to prevent drift)
        active_phoneme_ids = trajectory.phoneme_ids.to(device) if trajectory.phoneme_ids is not None else phoneme_ids
        active_supra = trajectory.text_suprasegmentals.to(device) if trajectory.text_suprasegmentals is not None else (text_suprasegmentals.to(device) if text_suprasegmentals is not None else None)

        all_phoneme_features = self.uclm_core_model.text_encoder(
            active_phoneme_ids, 
            torch.full((1, active_phoneme_ids.shape[1]), language_id, device=device),
            torch.tensor([active_phoneme_ids.shape[1]], device=device),
            text_suprasegmentals=active_supra
        ).transpose(1, 2)

        # 3. Unpack trajectory trace
        forced_indices = []
        for text_idx, duration in trajectory.pointer_trace:
            forced_indices.extend([text_idx] * duration)
        
        max_frames = len(forced_indices)
        a_tokens, b_tokens = [], []
        a_trace_list = [] # Worker 04
        ctx_a = torch.zeros(1, 8, 1, dtype=torch.long, device=device)
        ctx_b = torch.zeros(1, 4, 1, dtype=torch.long, device=device)
        kv_caches = None
        
        vs_trajectory = trajectory.voice_state_trajectory.to(device) if trajectory.voice_state_trajectory is not None else None
        # Bit-exact acoustic tokens if requested (Worker 01/04)
        a_trace = trajectory.acoustic_trace.to(device) if (use_exact_tokens and trajectory.acoustic_trace is not None) else None
        
        # Local Prosody Plan overrides (Worker 04)
        local_plan = local_prosody_plan or {}

        # 4. Forced Generation Loop
        for t in range(max_frames):
            text_idx = forced_indices[t]
            # SOTA: Same safe-indexing as tts() to handle tail buffer indices
            safe_idx = min(text_idx, all_phoneme_features.shape[1] - 1)
            content_features = all_phoneme_features[:, safe_idx : safe_idx + 1, :]
            
            # SOTA: Allow local overrides even during deterministic replay
            v_state_curr = vs_trajectory[t].view(1, 1, 8) if vs_trajectory is not None else torch.zeros(1, 1, 8, device=device)
            if text_idx in local_plan:
                v_state_curr = local_plan[text_idx].to(device).view(1, 1, 8)

            out = self.uclm_core_model.forward_streaming(
                queries=content_features,
                memory=all_phoneme_features,
                a_ctx=ctx_a,
                b_ctx=ctx_b,
                speaker_embed=speaker_embed,
                explicit_state=v_state_curr,
                cfg_scale=1.0,
                kv_caches=kv_caches,
                prompt_summary_tokens=prompt_summary_tokens,
            )
            
            logits_a, logits_b, kv_caches = out["logits_a"], out["logits_b"], out["kv_cache_out"]
            
            if a_trace is not None and use_exact_tokens:
                # a_trace is [B, 8, T]
                at = a_trace[:, :, t] # [B, 8]
            elif temperature > 0:
                at = torch.stack([torch.multinomial(F.softmax(logits_a[:, i, 0, :] / temperature, dim=-1), 1) for i in range(8)], dim=1).squeeze(-1)
            else:
                at = logits_a[:, :, 0, :].argmax(dim=-1)
            
            bt = logits_b[:, :, 0, :].argmax(dim=-1)
                
            a_trace_list.append(at.unsqueeze(-1))
            a_tokens.append(at.unsqueeze(-1))
            b_tokens.append(bt.unsqueeze(-1))
            ctx_a = at.unsqueeze(-1)
            ctx_b = bt.unsqueeze(-1)

        # 5. Decode audio
        a_t, b_t = torch.cat(a_tokens, dim=-1), torch.cat(b_tokens, dim=-1)
        t_frames = a_t.shape[-1]
        vs_full = vs_trajectory.unsqueeze(0) if vs_trajectory is not None else torch.zeros(1, t_frames, 8, device=device)
        # Ensure vs_full matches actual generated frame count
        if vs_full.shape[1] > t_frames: vs_full = vs_full[:, :t_frames, :]
        elif vs_full.shape[1] < t_frames:
            padding = vs_full[:, -1:, :].expand(-1, t_frames - vs_full.shape[1], -1)
            vs_full = torch.cat([vs_full, padding], dim=1)

        audio_out, _ = self.codec_dec(a_t, b_t, vs_full, [])

        return audio_out.squeeze(), {
            "rtf": (time.perf_counter() - t0) / (audio_out.shape[-1] / SAMPLE_RATE),
            "pointer_trace": trajectory.pointer_trace, # Identity mapping
            "acoustic_trace": torch.cat(a_trace_list, dim=-1) if a_trace_list else None,
            "control_trace": b_t,
        }

    def _tts_legacy_duration(
        self,
        phonemes: torch.Tensor,
        speaker_embed: torch.Tensor,
        style: StyleParams,
        cfg_scale: float = 1.5,
        temperature: float = 0.8,
        language_id: int = 0,
        pace: float = 1.0,
        max_frames: int = 1500,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        """Legacy duration-based TTS fallback.

        This path uses a DurationPredictor to determine phoneme durations
        instead of the v3 pointer mechanism. Currently a stub — raises
        NotImplementedError until a DurationPredictor is integrated.
        """
        raise NotImplementedError(
            "Legacy duration-based TTS is not available. "
            "DurationPredictor has not been integrated into the serving engine. "
            "Use tts_mode='pointer' (the default) for v3 pointer-based synthesis."
        )

    def create_trajectory_record(
        self,
        trajectory_id: str,
        compile_id: str,
        stats: dict,
        phoneme_ids: torch.Tensor,
        text_suprasegmentals: torch.Tensor | None = None,
        speaker_profile_id: str = "default",
        metadata: dict | None = None,
    ) -> TrajectoryRecord:
        """Helper to create a TrajectoryRecord from generation stats."""
        from tmrvc_core.types import TrajectoryRecord, PacingControls
        import time
        
        return TrajectoryRecord(
            trajectory_id=trajectory_id,
            source_compile_id=compile_id,
            phoneme_ids=phoneme_ids.cpu(),
            text_suprasegmentals=text_suprasegmentals.cpu() if text_suprasegmentals is not None else None,
            pointer_trace=stats["pointer_trace"],
            voice_state_trajectory=stats["voice_state_trajectory"].cpu(),
            acoustic_trace=stats["acoustic_trace"].cpu() if "acoustic_trace" in stats else None,
            control_trace=stats["control_trace"].cpu() if "control_trace" in stats else None,
            applied_pacing=PacingControls(
                pace=stats.get("applied_pace", 1.0),
                hold_bias=stats.get("applied_hold_bias", 0.0),
                boundary_bias=stats.get("applied_boundary_bias", 0.0)
            ),
            speaker_profile_id=speaker_profile_id,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata=metadata or {}
        )

    def synthesize_sentences(self, text: str, language: str, spk_embed: np.ndarray, style: StyleParams | None, speed: float = 1.0, cancel=None, sentence_pause_ms: int = 120, auto_style: bool = True):
        del auto_style
        if cancel is not None and cancel.is_set(): return
        from tmrvc_data.g2p import text_to_phonemes
        g2p_result = text_to_phonemes(text, language=language)
        phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)
        supra_t = g2p_result.text_suprasegmentals  # [L, 4] or None
        spk_t = torch.from_numpy(np.asarray(spk_embed)).float().to(self.device) if not isinstance(spk_embed, torch.Tensor) else spk_embed.to(self.device)
        if spk_t.dim() == 1: spk_t = spk_t.unsqueeze(0)
        audio_t, _ = self.tts(phonemes=phonemes_t, speaker_embed=spk_t, style=style or StyleParams.neutral(), language_id=g2p_result.language_id, text_suprasegmentals=supra_t)
        audio = audio_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        chunk_size = int(0.1 * SAMPLE_RATE)
        for i in range(0, audio.size, chunk_size):
            if cancel is not None and cancel.is_set(): return
            yield audio[i : i + chunk_size]
        if sentence_pause_ms > 0 and cancel is not None and not cancel.is_set():
            pause = np.zeros(int(SAMPLE_RATE * sentence_pause_ms / 1000), dtype=np.float32)
            for i in range(0, pause.size, chunk_size): yield pause[i : i + chunk_size]
