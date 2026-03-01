"""UCLM Engine: Unified inference for TTS and VC using dual-stream tokens."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Generator

import numpy as np
import torch
import torch.nn.functional as F

from tmrvc_core.constants import (
    HOP_LENGTH,
    SAMPLE_RATE,
    PHONEME_VOCAB_SIZE,
    D_MODEL,
)
from tmrvc_core.dialogue_types import CharacterProfile, StyleParams
from tmrvc_train.models import DisentangledUCLM, EmotionAwareCodec

logger = logging.getLogger(__name__)


@dataclass
class UCLMMetrics:
    """Timing breakdown for UCLM pipeline."""
    g2p_ms: float = 0.0
    encoder_ms: float = 0.0
    uclm_ms: float = 0.0
    decoder_ms: float = 0.0
    total_ms: float = 0.0
    output_duration_ms: float = 0.0

    @property
    def rtf(self) -> float:
        if self.output_duration_ms <= 0:
            return 0.0
        return self.total_ms / self.output_duration_ms


class UCLMEngine:
    """Unified engine for real-time TTS and VC.
    
    Uses DisentangledUCLM (dual-stream) and EmotionAwareCodec.
    """

    def __init__(
        self,
        uclm_checkpoint: Path | str | None = None,
        codec_checkpoint: Path | str | None = None,
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.uclm_checkpoint = Path(uclm_checkpoint) if uclm_checkpoint else None
        self.codec_checkpoint = Path(codec_checkpoint) if codec_checkpoint else None
        
        self.uclm = None
        self.codec = None
        self._loaded = False

    def load_from_state_dicts(self, uclm_state: dict, codec_state: dict):
        """Initialize from state dicts (for ModelPool)."""
        self.uclm = DisentangledUCLM(d_model=D_MODEL).to(self.device)
        self.uclm.load_state_dict(uclm_state)
        self.uclm.eval()
        
        self.codec = EmotionAwareCodec().to(self.device)
        self.codec.load_state_dict(codec_state)
        self.codec.eval()
        self._loaded = True

    def load_models(self):
        """Load UCLM and Codec models from files."""
        if not self.uclm_checkpoint or not self.codec_checkpoint:
            raise ValueError("Checkpoints must be provided to load_models()")
            
        uclm_ckpt = torch.load(self.uclm_checkpoint, map_location=self.device)
        codec_ckpt = torch.load(self.codec_checkpoint, map_location=self.device)
        
        self.load_from_state_dicts(uclm_ckpt["model"], codec_ckpt["model"])

    @torch.no_grad()
    def tts(
        self,
        phonemes: torch.Tensor,
        speaker_embed: torch.Tensor,
        style: StyleParams,
        cfg_scale: float = 1.5,
    ) -> tuple[torch.Tensor, UCLMMetrics]:
        """Synchronous TTS for a single utterance."""
        t_start = time.perf_counter()
        metrics = UCLMMetrics()
        
        # Prepare inputs
        phoneme_lens = torch.tensor([phonemes.shape[1]], device=self.device)
        lang_ids = torch.tensor([0], device=self.device) # Default JA
        
        # VoiceState from StyleParams
        voice_state = torch.tensor(style.to_vector(), device=self.device).float()
        voice_state = voice_state.unsqueeze(0).unsqueeze(0) # [1, 1, 8]
        
        # We need a temporal sequence for VoiceState. 
        # In TTS, we usually don't know the length yet. 
        # UCLM forward_tts handles expansion, but needs a target length or duration.
        # Here we mock a long enough state and let forward_tts trim/pad.
        voice_state = voice_state.expand(1, 1000, 8) 
        ssl_state = torch.zeros(1, 1000, 128, device=self.device)
        
        # Forward UCLM
        t0 = time.perf_counter()
        out = self.uclm.forward_tts(
            phonemes=phonemes.to(self.device),
            phoneme_lens=phoneme_lens,
            language_ids=lang_ids,
            explicit_state=voice_state,
            ssl_state=ssl_state,
            speaker_embed=speaker_embed.to(self.device),
            cfg_scale=cfg_scale
        )
        metrics.uclm_ms = (time.perf_counter() - t0) * 1000
        
        # Sample tokens (Greedy for now)
        a_tokens = out["logits_a"].argmax(dim=-1) # [1, 8, T]
        b_tokens = out["logits_b"].argmax(dim=-1) # [1, 4, T]
        
        # Decode
        t0 = time.perf_counter()
        # Trim voice state to match tokens
        T = a_tokens.shape[-1]
        audio = self.codec.decode(
            a_tokens, 
            b_tokens, 
            voice_state[:, :T, :],
        )
        metrics.decoder_ms = (time.perf_counter() - t0) * 1000
        
        metrics.total_ms = (time.perf_counter() - t_start) * 1000
        metrics.output_duration_ms = T * 10.0
        
        return audio.squeeze(), metrics

    @torch.no_grad()
    def vc_frame(
        self,
        audio_frame: torch.Tensor,
        speaker_embed: torch.Tensor,
        style: StyleParams,
        kv_cache: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Real-time VC for a single 10ms frame."""
        # 1. Encode source
        a_tokens, b_tokens = self.codec.encode(audio_frame) # [1, 8, 1], [1, 4, 1]
        
        # 2. Extract content (VQ Bottleneck)
        content, _ = self.uclm.vc_encoder(a_tokens)
        
        # 3. Voice State
        v_state = torch.tensor(style.to_vector(), device=self.device).float()
        v_state = v_state.unsqueeze(0).unsqueeze(0) # [1, 1, 8]
        ssl_state = torch.zeros(1, 1, 128, device=self.device) # Placeholder
        
        state_cond = self.uclm.voice_state_enc(v_state, ssl_state)
        
        # 4. UCLM Streaming
        out = self.uclm.forward_streaming(
            content,
            state_cond,
            speaker_embed,
            kv_cache_in=kv_cache
        )
        
        # 5. Decode
        target_a = out["logits_a"].argmax(dim=-1)
        target_b = out["logits_b"].argmax(dim=-1)
        
        audio_out = self.codec.decode(target_a, target_b, v_state)
        
        return audio_out.squeeze(), out["kv_cache_out"]
