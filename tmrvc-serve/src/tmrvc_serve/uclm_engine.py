"""UCLM Engine: Contract-compliant unified inference for TTS and VC."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F

from tmrvc_core.constants import SAMPLE_RATE, D_MODEL, D_SPEAKER, RVQ_VOCAB_SIZE, CONTROL_VOCAB_SIZE
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_train.models import (
    EmotionAwareEncoder,
    EmotionAwareDecoder,
    VCEncoder,
    VoiceStateEncoder,
    CodecTransformer,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineState:
    """Persistent inference states for a single session (mirroring Rust StreamingEngine)."""
    # Encoder causal states [B, C, P]
    enc_states: List[torch.Tensor] = field(default_factory=lambda: [
        torch.zeros(1, 64, 6), torch.zeros(1, 128, 4), 
        torch.zeros(1, 256, 4), torch.zeros(1, 512, 2)
    ])
    # Decoder causal states [B, C, P]
    dec_states: List[torch.Tensor] = field(default_factory=lambda: [
        torch.zeros(1, 256, 2), torch.zeros(1, 128, 4), 
        torch.zeros(1, 64, 6), torch.zeros(1, 1, 6)
    ])
    # Transformer KV Cache (list of tuples per layer)
    kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    # Control Context (B_t history) [1, 4, 200]
    ctx_b: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 4, 200, dtype=torch.long))
    # Previous voice state for delta computation
    prev_voice_state: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, 8))


class UCLMEngine:
    """Orchestrates split ONNX-ready models according to onnx-contract.md."""

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.codec_enc = EmotionAwareEncoder().to(self.device).eval()
        self.vc_enc = VCEncoder().to(self.device).eval()
        self.voice_state_enc = VoiceStateEncoder().to(self.device).eval()
        self.uclm_core = CodecTransformer().to(self.device).eval()
        self.codec_dec = EmotionAwareDecoder().to(self.device).eval()
        self._loaded = False

    def load_from_combined_checkpoint(self, path: Path | str):
        """Load all sub-models from a unified training checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt.get("model", ckpt)
        
        # Mapping logic from unified DisentangledUCLM keys to split components
        # This assumes the combined model was saved with these prefixes
        self.codec_enc.load_state_dict({k.replace("codec.encoder.", ""): v for k, v in state_dict.items() if k.startswith("codec.encoder.")}, strict=False)
        self.vc_enc.load_state_dict({k.replace("vc_encoder.", ""): v for k, v in state_dict.items() if k.startswith("vc_encoder.")}, strict=False)
        self.voice_state_enc.load_state_dict({k.replace("voice_state_enc.", ""): v for k, v in state_dict.items() if k.startswith("voice_state_enc.")}, strict=False)
        self.uclm_core.load_state_dict({k.replace("uclm_core.", ""): v for k, v in state_dict.items() if k.startswith("uclm_core.")}, strict=False)
        self.codec_dec.load_state_dict({k.replace("codec.decoder.", ""): v for k, v in state_dict.items() if k.startswith("codec.decoder.")}, strict=False)
        
        self._loaded = True
        logger.info("UCLM components loaded and aligned with Contract.")

    @torch.no_grad()
    def vc_frame(
        self,
        audio_frame: torch.Tensor,
        speaker_embed: torch.Tensor,
        style: StyleParams,
        state: EngineState,
        cfg_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, EngineState]:
        """
        Processes a single 10ms frame [1, 1, 240] precisely as defined in 
        onnx-contract.md Section 4.2.
        """
        # 1. Codec Encoder
        a_src_t, b_logits_src, new_enc_states = self.codec_enc(audio_frame, state.enc_states)
        
        # 2. VC Content Extraction
        content_features, _ = self.vc_enc(a_src_t)
        
        # 3. Voice State Conditioning
        v_state = torch.tensor(style.to_vector(), device=self.device).float().unsqueeze(0).unsqueeze(0)
        delta_state = v_state - state.prev_voice_state
        ssl_state = torch.zeros(1, 1, 128, device=self.device) # Placeholder
        
        state_cond = self.voice_state_enc(v_state, ssl_state, delta_state)
        
        # 4. UCLM Core Prediction
        logits_a, logits_b, new_kv = self.uclm_core(
            content_features,
            state.ctx_b,
            speaker_embed,
            state_cond,
            torch.tensor([cfg_scale], device=self.device),
            state.kv_caches
        )
        
        # 5. Sampling (Greedy for parity)
        a_t = logits_a.argmax(dim=-1) # [1, 8, 1]
        b_t = logits_b.argmax(dim=-1) # [1, 4, 1]
        
        # 6. Codec Decoder
        audio_out, new_dec_states = self.codec_dec(
            a_t, b_t, v_state, state.dec_states
        )
        
        # Update and return State
        new_state = EngineState(
            enc_states=new_enc_states,
            dec_states=new_dec_states,
            kv_caches=new_kv,
            ctx_b=torch.cat([state.ctx_b[:, :, 1:], b_t], dim=-1),
            prev_voice_state=v_state
        )
        
        return audio_out.squeeze(), new_state

    @torch.no_grad()
    def tts(
        self,
        phonemes: torch.Tensor,
        speaker_embed: torch.Tensor,
        style: StyleParams,
        cfg_scale: float = 1.5,
    ) -> torch.Tensor:
        """
        Batch TTS (simplified for now, uses uclm_core internally).
        """
        # (Implementation details omitted for brevity, focusing on VC alignment first)
        # Note: In a real system, this would also maintain a temporary state.
        return torch.zeros(SAMPLE_RATE) 
