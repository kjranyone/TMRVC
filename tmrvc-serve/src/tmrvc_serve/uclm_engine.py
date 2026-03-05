"""UCLM Engine: contract-friendly unified inference for TTS and VC."""

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


class UCLMEngine:
    """Orchestrates split models for inference."""

    def __init__(
        self,
        uclm_checkpoint: Path | str | None = None,
        codec_checkpoint: Path | str | None = None,
        device: str = "cpu",
        d_model: int = 512,
    ):
        self.device = torch.device(device)
        self.d_model = d_model
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
        return False

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
        """Load models from checkpoints.

        Paths are optional to keep compatibility with `app.py` init flow.
        """
        uclm_ckpt_path = Path(uclm_path) if uclm_path is not None else self._uclm_checkpoint
        codec_ckpt_path = (
            Path(codec_path) if codec_path is not None else self._codec_checkpoint
        )
        if uclm_ckpt_path is None or codec_ckpt_path is None:
            raise ValueError("Both UCLM and codec checkpoints are required.")

        self._uclm_checkpoint = uclm_ckpt_path
        self._codec_checkpoint = codec_ckpt_path

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
        logger.info("Loaded UCLM/codec checkpoints on %s", self.device)

    def load_from_combined_checkpoint(self, checkpoint: Path | str) -> None:
        """Backward-compatible fallback for legacy callers."""
        ckpt_path = Path(checkpoint)
        self.load_models(uclm_path=ckpt_path, codec_path=ckpt_path)

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
        delta_state = v_state - state.prev_voice_state
        ssl_state = torch.zeros(1, 1, 128, device=self.device)

        v_out = self.voice_state_enc(v_state, ssl_state, delta_state)
        state_cond = v_out[0] if isinstance(v_out, tuple) else v_out

        logits_a, logits_b, new_kv = self.uclm_core(
            content_features, state.ctx_b, speaker_embed, state_cond, cfg_scale, state.kv_caches
        )

        if temperature > 0:
            probs_a = F.softmax(logits_a[:, :, -1, :] / temperature, dim=-1)
            probs_b = F.softmax(logits_b[:, :, -1, :] / temperature, dim=-1)
            a_t = (
                torch.stack([torch.multinomial(probs_a[:, i, :], 1) for i in range(8)], dim=1)
                .squeeze(-1)
                .unsqueeze(-1)
            )
            b_t = (
                torch.stack([torch.multinomial(probs_b[:, i, :], 1) for i in range(4)], dim=1)
                .squeeze(-1)
                .unsqueeze(-1)
            )
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
    def tts(
        self,
        phonemes: torch.Tensor,
        speaker_embed: torch.Tensor,
        style: StyleParams,
        cfg_scale: float = 1.5,
        temperature: float = 0.8,
    ) -> Tuple[torch.Tensor, dict]:
        """Full end-to-end TTS generation."""
        self._require_loaded()
        t0 = time.perf_counter()

        v_state = (
            torch.tensor(style.to_vector(), device=self.device)
            .float()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        ssl_state = torch.zeros(1, 1, 128, device=self.device)
        phoneme_lens = torch.tensor([phonemes.shape[1]], device=self.device)
        lang_id = torch.tensor([0], device=self.device)

        out = self.uclm_core_model.forward_tts(
            phonemes=phonemes.to(self.device),
            phoneme_lens=phoneme_lens,
            language_ids=lang_id,
            target_b=torch.zeros(1, 4, 1, dtype=torch.long, device=self.device),
            explicit_state=v_state.expand(-1, 50, -1),
            ssl_state=ssl_state.expand(-1, 50, -1),
            speaker_embed=speaker_embed.to(self.device),
            cfg_scale=cfg_scale,
        )

        logits_a, logits_b = out["logits_a"], out["logits_b"]
        t_frames = logits_a.shape[2]

        if temperature > 0:
            probs_a = F.softmax(logits_a / temperature, dim=-1)
            a_list = []
            for t in range(t_frames):
                at = torch.stack(
                    [torch.multinomial(probs_a[:, i, t, :], 1) for i in range(8)],
                    dim=1,
                )
                a_list.append(at)
            a_t = torch.cat(a_list, dim=-1)

            probs_b = F.softmax(logits_b / temperature, dim=-1)
            b_list = []
            for t in range(t_frames):
                bt = torch.stack(
                    [torch.multinomial(probs_b[:, i, t, :], 1) for i in range(4)],
                    dim=1,
                )
                b_list.append(bt)
            b_t = torch.cat(b_list, dim=-1)
        else:
            a_t = logits_a.argmax(dim=-1)
            b_t = logits_b.argmax(dim=-1)

        audio_out, _ = self.codec_dec(a_t, b_t, v_state.expand(-1, t_frames, -1), [])

        gen_time = time.perf_counter() - t0
        rtf = gen_time / (audio_out.shape[-1] / SAMPLE_RATE)
        return audio_out.squeeze(), {"rtf": float(rtf), "gen_time_ms": float(gen_time * 1000.0)}

    @staticmethod
    def _apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
        if speed <= 0:
            return audio
        if abs(speed - 1.0) < 1e-3 or audio.size < 2:
            return audio
        x_old = np.linspace(0.0, 1.0, num=audio.size, endpoint=True, dtype=np.float64)
        new_len = max(1, int(round(audio.size / speed)))
        x_new = np.linspace(0.0, 1.0, num=new_len, endpoint=True, dtype=np.float64)
        return np.interp(x_new, x_old, audio).astype(np.float32)

    def prefetch_g2p(self, text: str, language: str) -> None:
        from tmrvc_core.text_utils import text_to_phonemes

        _ = text_to_phonemes(text, language=language)

    def initial_scene_state(self) -> np.ndarray:
        return np.zeros((1,), dtype=np.float32)

    def update_scene_state(
        self,
        text: str,
        language: str,
        spk_embed: np.ndarray,
        z_prev: np.ndarray,
    ) -> np.ndarray:
        del text, language, spk_embed
        return z_prev

    def synthesize_sentences(
        self,
        text: str,
        language: str,
        spk_embed: np.ndarray,
        style: StyleParams | None,
        speed: float = 1.0,
        cancel=None,
        sentence_pause_ms: int = 120,
        auto_style: bool = True,
    ):
        """Compatibility generator used by `/ws/chat` route."""
        del auto_style
        if cancel is not None and cancel.is_set():
            return

        from tmrvc_core.text_utils import text_to_phonemes

        phoneme_ids = text_to_phonemes(text, language=language)
        phonemes_t = torch.tensor(phoneme_ids).long().unsqueeze(0)
        spk_t = torch.from_numpy(spk_embed).float().unsqueeze(0)

        audio_t, _ = self.tts(
            phonemes=phonemes_t,
            speaker_embed=spk_t,
            style=style or StyleParams.neutral(),
        )
        audio = audio_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
        audio = self._apply_speed(audio, speed)

        chunk_size = int(0.1 * SAMPLE_RATE)  # 100 ms
        for i in range(0, audio.size, chunk_size):
            if cancel is not None and cancel.is_set():
                return
            yield audio[i : i + chunk_size]

        if sentence_pause_ms > 0 and cancel is not None and not cancel.is_set():
            pause = np.zeros(int(SAMPLE_RATE * sentence_pause_ms / 1000), dtype=np.float32)
            for i in range(0, pause.size, chunk_size):
                yield pause[i : i + chunk_size]
