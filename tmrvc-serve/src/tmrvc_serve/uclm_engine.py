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

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_core.types import PointerState, SpeakerProfile
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

    @property
    def finished(self) -> bool:
        return self.text_index >= self.total_phonemes

    def to_dict(self) -> dict:
        return {
            "text_index": self.text_index,
            "progress": self.progress,
            "total_phonemes": self.total_phonemes,
            "frames_generated": self.frames_generated,
            "stall_frames": self.stall_frames,
            "max_frames_per_unit": self.max_frames_per_unit,
            "skip_protection_threshold": self.skip_protection_threshold,
            "frames_on_current_unit": self.frames_on_current_unit,
            "forced_advance_count": self.forced_advance_count,
            "skip_protection_count": self.skip_protection_count,
        }


class UCLMEngine:
    """Orchestrates split models for inference (v3)."""

    def __init__(
        self,
        uclm_checkpoint: Path | str | None = None,
        codec_checkpoint: Path | str | None = None,
        device: str = "cpu",
        d_model: int = 512,
        tts_mode: str = "pointer",
    ):
        self.device = torch.device(device)
        self.d_model = d_model
        self.tts_mode = tts_mode
        self._uclm_checkpoint = Path(uclm_checkpoint) if uclm_checkpoint else None
        self._codec_checkpoint = Path(codec_checkpoint) if codec_checkpoint else None

        self.codec_enc = EmotionAwareEncoder(d_model=d_model).to(self.device).eval()
        self.codec_dec = EmotionAwareDecoder(d_model=d_model).to(self.device).eval()
        self.uclm_core_model: DisentangledUCLM | None = None
        self.uclm_core = None
        self.vc_enc = None
        self.voice_state_enc = None
        self._loaded = False

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
        self._loaded = True
        logger.info("Loaded UCLM v3 checkpoints on %s", self.device)

    @torch.no_grad()
    def vc_frame(
        self,
        audio_frame: torch.Tensor,
        speaker_embed: torch.Tensor,
        style: StyleParams,
        state: EngineState,
        cfg_scale: float = 1.0,
        temperature: float = 0.8,
    ) -> Tuple[torch.Tensor, EngineState]:
        self._require_loaded()

        a_src_t, b_logits_src, new_enc_states = self.codec_enc(audio_frame, state.enc_states)
        content_features, _ = self.vc_enc(a_src_t)

        v_state = (
            torch.tensor(style.to_vector(), device=self.device)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        ssl_state = torch.zeros(1, 1, 128, device=self.device)

        v_out = self.voice_state_enc(v_state, ssl_state)
        state_cond = v_out[0] if isinstance(v_out, tuple) else v_out

        logits_a, logits_b, new_kv, _ = self.uclm_core(
            content_features, state.ctx_b, speaker_embed, state_cond, cfg_scale, state.kv_caches
        )

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
            ctx_b=torch.cat([state.ctx_b[:, :, 1:], b_t], dim=-1),
            prev_voice_state=v_state,
        )
        return audio_out.squeeze(), new_state

    @torch.no_grad()
    def encode_speaker_prompt(
        self,
        reference_audio: torch.Tensor | None = None,
        reference_codec_tokens: torch.Tensor | None = None,
        speaker_embed: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Encode reference audio for few-shot speaker adaptation.

        Returns:
            (refined_speaker_embed, prompt_features, encoding_time_ms)
        """
        self._require_loaded()
        t0 = time.perf_counter()

        if reference_codec_tokens is not None and hasattr(self.uclm_core_model, 'speaker_prompt_encoder'):
            refined_embed, prompt_feats = self.uclm_core_model.encode_speaker_prompt(
                reference_codec_tokens.to(self.device),
                speaker_embed.to(self.device) if speaker_embed is not None else None,
            )
        elif speaker_embed is not None:
            refined_embed = speaker_embed.to(self.device)
            prompt_feats = None
        else:
            raise ValueError("Either reference_codec_tokens or speaker_embed must be provided")

        encoding_time = (time.perf_counter() - t0) * 1000.0
        return refined_embed, prompt_feats, encoding_time

    @torch.no_grad()
    def tts(
        self,
        phonemes: torch.Tensor,
        speaker_profile: Optional[SpeakerProfile] = None,
        speaker_embed: Optional[torch.Tensor] = None,
        style: StyleParams | None = None,
        cfg_scale: float = 1.5,
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

        style = style or StyleParams.neutral()
        v_state = torch.tensor(style.to_vector(), device=self.device).float().view(1, 1, -1)
        num_phonemes = phonemes.shape[1]
        phoneme_ids = phonemes.to(self.device)
        
        prompt_kv_cache = None
        if speaker_profile is not None:
            speaker_embed = speaker_profile.speaker_embed.to(self.device).unsqueeze(0)
            if speaker_profile.prompt_codec_tokens is not None:
                # Precompute prompt KV cache using the encoder
                # Note: Assuming encode_speaker_prompt or a helper can recreate the cache from tokens
                # For now, we'll just encode the prompt codec tokens directly if model supports it
                pass # TODO: implement prompt_kv_cache restoration from prompt_codec_tokens
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

        # Pre-compute text features once before the generation loop
        phoneme_lens = torch.tensor([num_phonemes], device=self.device)
        all_phoneme_features = self.uclm_core_model.text_encoder(
            phoneme_ids, lang_id, phoneme_lens
        ).transpose(1, 2)

        # Move optional expressive inputs to device once
        _dlg_ctx = dialogue_context.to(self.device) if dialogue_context is not None else None
        _act_int = acting_intent.to(self.device) if acting_intent is not None else None

        a_tokens, b_tokens = [], []
        ctx_a = torch.zeros(1, 8, 1, dtype=torch.long, device=self.device)

        for t in range(max_frames):
            if ptr.finished:
                break

            # Index into cached text features by pointer position
            content_features = all_phoneme_features[:, ptr.text_index : ptr.text_index + 1, :]

            out = self.uclm_core_model.forward_streaming(
                content_features=content_features,
                a_ctx=ctx_a,
                b_ctx=ctx_b,
                speaker_embed=speaker_embed,
                state_cond=v_state,
                cfg_scale=cfg_scale,
                kv_caches=kv_caches,
                dialogue_context=_dlg_ctx,
                acting_intent=_act_int,
            )
            
            logits_a, logits_b, kv_caches = out["logits_a"], out["logits_b"], out["kv_cache_out"]
            hidden_states = out["hidden_states"]
            
            p_adv_logit, p_delta, _p_bc = self.uclm_core_model.pointer_head(hidden_states)
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
                
            a_tokens.append(at.unsqueeze(-1))
            b_tokens.append(bt.unsqueeze(-1))
            ctx_a = at.unsqueeze(-1)
            ctx_b = bt.unsqueeze(-1)

            # --- Pointer state-transition with failure handling ---
            ptr.frames_on_current_unit += 1
            progress_delta = p_delta.item() / max(0.1, 1.0 / (pace + 1e-6))
            ptr.progress += progress_delta
            adv_prob_val = p_adv_prob.item()

            # Compute boundary_confidence as sigmoid of the raw advance logit
            boundary_confidence = torch.sigmoid(p_adv_logit).item()

            advanced = False

            # Forced advance: stuck too long on one text unit
            if ptr.frames_on_current_unit >= ptr.max_frames_per_unit:
                logger.warning(
                    "Forced advance at text_index=%d after %d frames on unit",
                    ptr.text_index, ptr.frames_on_current_unit,
                )
                ptr.text_index += 1
                ptr.progress = 0.0
                ptr.frames_on_current_unit = 0
                ptr.stall_frames = 0
                ptr.forced_advance_count += 1
                advanced = True
            # Skip-protection: both signals fire but boundary_confidence is low
            elif adv_prob_val > 0.5 and ptr.progress >= 1.0:
                if boundary_confidence >= ptr.skip_protection_threshold:
                    ptr.text_index += 1
                    ptr.progress = 0.0
                    ptr.frames_on_current_unit = 0
                    ptr.stall_frames = 0
                    advanced = True
                else:
                    ptr.skip_protection_count += 1
            # Normal advance on either signal
            elif adv_prob_val > 0.5 or ptr.progress >= 1.0:
                ptr.text_index += 1
                ptr.progress = 0.0
                ptr.frames_on_current_unit = 0
                ptr.stall_frames = 0
                advanced = True

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

        a_t, b_t = torch.cat(a_tokens, dim=-1), torch.cat(b_tokens, dim=-1)
        t_frames = a_t.shape[-1]
        audio_out, _ = self.codec_dec(a_t, b_t, v_state.expand(-1, t_frames, -1), [])

        gen_time = time.perf_counter() - t0
        rtf = gen_time / (audio_out.shape[-1] / SAMPLE_RATE)
        return audio_out.squeeze(), {
            "rtf": float(rtf),
            "gen_time_ms": float(gen_time * 1000.0),
            "pointer_state": ptr.to_dict(),
            "prompt_encoding_time_ms": 0.0,
            "stall_events": 0,
            "forced_advance_count": ptr.forced_advance_count,
            "skip_protection_count": ptr.skip_protection_count,
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

    def synthesize_sentences(self, text: str, language: str, spk_embed: np.ndarray, style: StyleParams | None, speed: float = 1.0, cancel=None, sentence_pause_ms: int = 120, auto_style: bool = True):
        del auto_style
        if cancel is not None and cancel.is_set(): return
        from tmrvc_data.g2p import text_to_phonemes
        g2p_result = text_to_phonemes(text, language=language)
        phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)
        spk_t = torch.from_numpy(np.asarray(spk_embed)).float().to(self.device) if not isinstance(spk_embed, torch.Tensor) else spk_embed.to(self.device)
        if spk_t.dim() == 1: spk_t = spk_t.unsqueeze(0)
        audio_t, _ = self.tts(phonemes=phonemes_t, speaker_embed=spk_t, style=style or StyleParams.neutral(), language_id=g2p_result.language_id)
        audio = audio_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        chunk_size = int(0.1 * SAMPLE_RATE)
        for i in range(0, audio.size, chunk_size):
            if cancel is not None and cancel.is_set(): return
            yield audio[i : i + chunk_size]
        if sentence_pause_ms > 0 and cancel is not None and not cancel.is_set():
            pause = np.zeros(int(SAMPLE_RATE * sentence_pause_ms / 1000), dtype=np.float32)
            for i in range(0, pause.size, chunk_size): yield pause[i : i + chunk_size]
