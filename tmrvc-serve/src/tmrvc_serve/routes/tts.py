"""POST /tts and POST /tts/stream endpoints (UCLM v2)."""

from __future__ import annotations

import asyncio
import logging
import torch

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_serve._helpers import (
    _append_silence,
    _audio_to_wav_base64,
    _load_speaker_embed,
    _to_dialogue_turns,
)
from tmrvc_serve.schemas import TTSRequest, TTSResponse, TTSStreamRequest
from tmrvc_serve.style_resolver import (
    _apply_inline_stage_overlay,
    _predict_style_from_inputs,
    _resolve_effective_speed,
    _resolve_sentence_pause,
    _resolve_stage_speed,
    _resolve_style_preset,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/tts", response_model=TTSResponse)
async def generate_tts(req: TTSRequest) -> TTSResponse:
    from tmrvc_serve.app import get_engine, _characters, _context_predictor
    from tmrvc_core.text_utils import analyze_inline_stage_directions
    from tmrvc_data.g2p import text_to_phonemes

    engine = get_engine()

    character = _characters.get(req.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail=f"Character '{req.character_id}' not found.")

    # 1. Analyze text and directions
    inline_stage = analyze_inline_stage_directions(req.text, language=character.language)
    spoken_text = inline_stage.spoken_text

    # 2. Predict style
    history = _to_dialogue_turns(req.context)
    style = await _predict_style_from_inputs(
        character=character,
        text=spoken_text,
        emotion=req.emotion,
        history=history,
        situation=req.situation,
        hint=req.hint,
        speaker=req.character_id,
        context_predictor=_context_predictor,
    )

    style, preset_cfg = _resolve_style_preset(style, req.style_preset)
    style = _apply_inline_stage_overlay(style, inline_stage.style_overlay)
    
    # 3. G2P: text -> phoneme IDs
    g2p_result = text_to_phonemes(spoken_text, language=character.language)
    phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)

    # 4. Load speaker embedding
    spk_embed = _load_speaker_embed(character)
    spk_t = spk_embed.to(dtype=torch.float32).unsqueeze(0)

    # 5. Unified Synthesis (UCLM)
    audio_t, metrics = engine.tts(
        phonemes=phonemes_t,
        speaker_embed=spk_t,
        style=style
    )
    
    audio = audio_t.cpu().numpy()

    # 6. Post-processing
    audio = _append_silence(
        audio,
        leading_ms=inline_stage.leading_silence_ms,
        trailing_ms=inline_stage.trailing_silence_ms,
    )
    duration_sec = len(audio) / SAMPLE_RATE

    audio_b64 = _audio_to_wav_base64(audio)

    # Prepare metadata for response
    style_used = vars(style).copy() if style else {}
    style_used["style_preset"] = req.style_preset
    style_used["spoken_text"] = spoken_text
    style_used["metrics"] = metrics

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        style_used=style_used,
    )


@router.post("/tts/stream")
async def stream_tts(req: TTSStreamRequest) -> StreamingResponse:
    """Streaming TTS endpoint (Batch fallback for UCLM v2 initial implementation)."""
    # For now, UCLM v2 synthesis is fast enough to return as a single chunk 
    # while we implement full causal streaming for TTS.
    res = await generate_tts(req)
    
    import base64
    audio_bytes = base64.b64decode(res.audio_base64)
    
    async def _yield_once():
        yield audio_bytes

    return StreamingResponse(
        _yield_once(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Sample-Format": "float32",
        },
    )
