"""POST /tts, POST /tts/stream, and POST /tts/stream/sse endpoints (UCLM v3 pointer mode)."""

from __future__ import annotations

import asyncio
import base64
import json
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

    # 4. Load speaker embedding or profile
    spk_embed = _load_speaker_embed(character)
    spk_t = spk_embed.to(dtype=torch.float32).unsqueeze(0)

    # Future: _load_speaker_profile logic to fetch prompt_codec_tokens
    speaker_profile = None

    # 5. Convert optional embedding lists to tensors
    dlg_ctx = torch.tensor(req.dialogue_context, dtype=torch.float32).unsqueeze(0) if req.dialogue_context is not None else None
    act_int = torch.tensor(req.acting_intent, dtype=torch.float32).unsqueeze(0) if req.acting_intent is not None else None

    # Few-shot speaker adaptation: reference audio encoding
    # (Placeholder: actual codec encoding from raw audio will be added when the
    # full Speaker Prompt Encoder pipeline is integrated)

    # 6. Unified Synthesis (UCLM)
    audio_t, metrics = engine.tts(
        phonemes=phonemes_t,
        speaker_profile=speaker_profile,
        speaker_embed=spk_t,
        style=style,
        language_id=g2p_result.language_id,
        pace=req.pace,
        hold_bias=req.hold_bias,
        boundary_bias=req.boundary_bias,
        phrase_pressure=req.phrase_pressure,
        breath_tendency=req.breath_tendency,
        dialogue_context=dlg_ctx,
        acting_intent=act_int,
    )
    
    audio = audio_t.cpu().numpy()

    # 7. Post-processing
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
    """Streaming TTS endpoint (batch fallback while full causal streaming is implemented)."""
    # Batch fallback: returns the full audio as a single chunk while
    # full causal streaming is being implemented.
    res = await generate_tts(req)

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


@router.post("/tts/stream/sse")
async def stream_tts_sse(req: TTSStreamRequest) -> StreamingResponse:
    """Server-Sent Events streaming TTS endpoint.

    Uses batch fallback (generate full audio, then chunk into SSE events).
    The inner loop is structured so that real causal streaming can replace
    the batch generation later without changing the SSE event contract.

    Events emitted:
        ``event: audio``  — base64-encoded PCM float32 chunk.
        ``event: pointer`` — JSON pointer telemetry snapshot.
        ``event: done``   — JSON final metrics.
    """
    from tmrvc_serve.app import get_engine, _characters, _context_predictor
    from tmrvc_core.text_utils import analyze_inline_stage_directions
    from tmrvc_data.g2p import text_to_phonemes

    engine = get_engine()

    character = _characters.get(req.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail=f"Character '{req.character_id}' not found.")

    # --- Resolve style and phonemes (mirrors /tts) ---
    inline_stage = analyze_inline_stage_directions(req.text, language=character.language)
    spoken_text = inline_stage.spoken_text

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

    g2p_result = text_to_phonemes(spoken_text, language=character.language)
    phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)

    # 4. Load speaker embedding or profile
    spk_embed = _load_speaker_embed(character)
    spk_t = spk_embed.to(dtype=torch.float32).unsqueeze(0)

    # Future: _load_speaker_profile logic to fetch prompt_codec_tokens
    speaker_profile = None

    dlg_ctx = torch.tensor(req.dialogue_context, dtype=torch.float32).unsqueeze(0) if req.dialogue_context is not None else None
    act_int = torch.tensor(req.acting_intent, dtype=torch.float32).unsqueeze(0) if req.acting_intent is not None else None

    # --- Batch synthesis (will be replaced by causal streaming) ---
    audio_t, metrics = engine.tts(
        phonemes=phonemes_t,
        speaker_profile=speaker_profile,
        speaker_embed=spk_t,
        style=style,
        language_id=g2p_result.language_id,
        pace=req.pace,
        hold_bias=req.hold_bias,
        boundary_bias=req.boundary_bias,
        phrase_pressure=req.phrase_pressure,
        breath_tendency=req.breath_tendency,
        dialogue_context=dlg_ctx,
        acting_intent=act_int,
    )

    audio = audio_t.cpu().numpy().astype(np.float32)
    audio = _append_silence(
        audio,
        leading_ms=inline_stage.leading_silence_ms,
        trailing_ms=inline_stage.trailing_silence_ms,
    )

    # --- Chunk audio into SSE events ---
    chunk_samples = int(SAMPLE_RATE * req.chunk_duration_ms / 1000)
    pointer_state = metrics.get("pointer_state", {})

    async def _sse_generator():
        total_chunks = max(1, (len(audio) + chunk_samples - 1) // chunk_samples)

        for i in range(0, len(audio), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            chunk_b64 = base64.b64encode(chunk.tobytes()).decode("ascii")
            yield f"event: audio\ndata: {chunk_b64}\n\n"

            # Interleave pointer telemetry (interpolated position)
            chunk_idx = i // chunk_samples
            progress_frac = (chunk_idx + 1) / total_chunks
            tel = dict(pointer_state)
            tel["stream_progress"] = round(progress_frac, 4)
            yield f"event: pointer\ndata: {json.dumps(tel)}\n\n"

            # Yield control to the event loop between chunks
            await asyncio.sleep(0)

        # Final done event
        yield f"event: done\ndata: {json.dumps(metrics)}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
