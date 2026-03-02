"""UCLM Engine: Contract-compliant unified inference for TTS and VC."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List, Union

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
    DisentangledUCLM,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineState:
    """Persistent inference states for a single session (mirroring Rust StreamingEngine)."""
    enc_states: List[torch.Tensor] = field(default_factory=lambda: [
        torch.zeros(1, 1, 6), torch.zeros(1, 64, 4), 
        torch.zeros(1, 128, 4), torch.zeros(1, 256, 2)
    ])
    dec_states: List[torch.Tensor] = field(default_factory=lambda: [torch.empty(0)]*4)
    kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ctx_b: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 4, 200, dtype=torch.long))
    prev_voice_state: torch.Tensor = field(default_factory=lambda: torch.zeros(1, 1, 8))


class UCLMEngine:
    """Orchestrates split models for inference."""

    def __init__(self, device: str = "cpu", d_model: int = 512):
        self.device = torch.device(device)
        self.d_model = d_model
        
        self.codec_enc = EmotionAwareEncoder(d_model=d_model).to(self.device).eval()
        self.codec_dec = EmotionAwareDecoder(d_model=d_model).to(self.device).eval()
        self.uclm_core_model = None
        self.uclm_core = None
        self.vc_enc = None
        self.voice_state_enc = None
        self._loaded = False

    def load_models(self, uclm_path: Path | str, codec_path: Path | str):
        """Load models from checkpoints."""
        # Load UCLM
        uclm_ckpt = torch.load(uclm_path, map_location=self.device, weights_only=False)
        num_spk = 1000
        if "voice_state_enc.adversarial_classifier.2.weight" in uclm_ckpt.get("model", {}):
            num_spk = uclm_ckpt["model"]["voice_state_enc.adversarial_classifier.2.weight"].shape[0]
            
        self.uclm_core_model = DisentangledUCLM(num_speakers=num_spk).to(self.device).eval()
        self.uclm_core_model.load_state_dict(uclm_ckpt.get("model", uclm_ckpt), strict=False)
        
        # Load Codec
        codec_ckpt = torch.load(codec_path, map_location=self.device, weights_only=False)
        self.codec_enc.load_state_dict({k.replace("encoder.", ""): v for k, v in codec_ckpt.get("model", codec_ckpt).items() if k.startswith("encoder.")}, strict=False)
        self.codec_dec.load_state_dict({k.replace("decoder.", ""): v for k, v in codec_ckpt.get("model", codec_ckpt).items() if k.startswith("decoder.")}, strict=False)
        
        self.uclm_core = self.uclm_core_model.uclm_core
        self.vc_enc = self.uclm_core_model.vc_encoder
        self.voice_state_enc = self.uclm_core_model.voice_state_enc
        self._loaded = True

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
        # 1. Codec Encoder
        a_src_t, b_logits_src, new_enc_states = self.codec_enc(audio_frame, state.enc_states)
        
        # 2. VC Content Extraction
        content_features, _ = self.vc_enc(a_src_t)
        
        # 3. Voice State Conditioning
        v_state = torch.tensor(style.to_vector(), device=self.device).float().unsqueeze(0).unsqueeze(0)
        delta_state = v_state - state.prev_voice_state
        ssl_state = torch.zeros(1, 1, 128, device=self.device)
        
        v_out = self.voice_state_enc(v_state, ssl_state, delta_state)
        state_cond = v_out[0] if isinstance(v_out, tuple) else v_out
        
        # 4. UCLM Core Prediction
        logits_a, logits_b, new_kv = self.uclm_core(
            content_features, state.ctx_b, speaker_embed, state_cond,
            cfg_scale, state.kv_caches
        )
        
        # 5. Sampling
        if temperature > 0:
            probs_a = F.softmax(logits_a[:, :, -1, :] / temperature, dim=-1)
            probs_b = F.softmax(logits_b[:, :, -1, :] / temperature, dim=-1)
            a_t = torch.stack([torch.multinomial(probs_a[:, i, :], 1) for i in range(8)], dim=1).squeeze(-1).unsqueeze(-1)
            b_t = torch.stack([torch.multinomial(probs_b[:, i, :], 1) for i in range(4)], dim=1).squeeze(-1).unsqueeze(-1)
        else:
            a_t = logits_a[:, :, -1, :].argmax(dim=-1, keepdim=True)
            b_t = logits_b[:, :, -1, :].argmax(dim=-1, keepdim=True)
        
        # 6. Codec Decoder
        audio_out, new_dec_states = self.codec_dec(a_t, b_t, v_state, state.dec_states)
        
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
        temperature: float = 0.8,
    ) -> Tuple[torch.Tensor, dict]:
        """Full end-to-end TTS generation."""
        t0 = time.perf_counter()
        
        v_state = torch.tensor(style.to_vector(), device=self.device).float().unsqueeze(0).unsqueeze(0)
        ssl_state = torch.zeros(1, 1, 128, device=self.device)
        phoneme_lens = torch.tensor([phonemes.shape[1]], device=self.device)
        lang_id = torch.tensor([0], device=self.device)
        
        out = self.uclm_core_model.forward_tts(
            phonemes=phonemes.to(self.device),
            phoneme_lens=phoneme_lens,
            language_ids=lang_id,
            target_b=torch.zeros(1, 4, 1, dtype=torch.long, device=self.device),
            explicit_state=v_state.expand(-1, 50, -1), # Dummy expansion
            ssl_state=ssl_state.expand(-1, 50, -1),
            speaker_embed=speaker_embed.to(self.device),
            cfg_scale=cfg_scale
        )
        
        logits_a, logits_b = out["logits_a"], out["logits_b"]
        T = logits_a.shape[2]
        
        if temperature > 0:
            probs_a = F.softmax(logits_a / temperature, dim=-1)
            # Sample for all frames
            a_list = []
            for t in range(T):
                at = torch.stack([torch.multinomial(probs_a[:, i, t, :], 1) for i in range(8)], dim=1)
                a_list.append(at)
            a_t = torch.cat(a_list, dim=-1)
            
            probs_b = F.softmax(logits_b / temperature, dim=-1)
            b_list = []
            for t in range(T):
                bt = torch.stack([torch.multinomial(probs_b[:, i, t, :], 1) for i in range(4)], dim=1)
                b_list.append(bt)
            b_t = torch.cat(b_list, dim=-1)
        else:
            a_t = logits_a.argmax(dim=-1)
            b_t = logits_b.argmax(dim=-1)
            
        audio_out, _ = self.codec_dec(a_t, b_t, v_state.expand(-1, T, -1), [])
        
        gen_time = (time.perf_counter() - t0)
        rtf = gen_time / (audio_out.shape[-1] / SAMPLE_RATE)
        
        return audio_out.squeeze(), {"rtf": rtf, "gen_time_ms": gen_time * 1000}
