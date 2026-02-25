"""POST /tts and POST /tts/stream endpoints."""

from __future__ import annotations

import asyncio
import logging
import queue as stdlib_queue
import threading

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_serve._helpers import (
    _append_silence,
    _audio_to_wav_base64,
    _iter_silence_chunks,
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

    engine = get_engine()

    character = _characters.get(req.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail=f"Character '{req.character_id}' not found.")

    from tmrvc_core.text_utils import analyze_inline_stage_directions
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
    effective_speed = _resolve_stage_speed(
        _resolve_effective_speed(req.speed, preset_cfg),
        inline_stage.speed_scale,
    )
    effective_sentence_pause_ms = _resolve_sentence_pause(
        preset_cfg.sentence_pause_ms,
        inline_stage.sentence_pause_ms_delta,
    )

    # Load speaker embedding
    spk_embed = _load_speaker_embed(character)
    if req.style_preset == "default":
        audio, _duration_sec = engine.synthesize(
            text=spoken_text,
            language=character.language,
            spk_embed=spk_embed,
            style=style,
            speed=effective_speed,
        )
    else:
        chunks = list(engine.synthesize_sentences(
            text=spoken_text,
            language=character.language,
            spk_embed=spk_embed,
            style=style,
            speed=effective_speed,
            sentence_pause_ms=effective_sentence_pause_ms,
            auto_style=preset_cfg.auto_style,
        ))
        if chunks:
            audio = np.concatenate(chunks).astype(np.float32)
        else:
            audio = np.zeros(0, dtype=np.float32)

    audio = _append_silence(
        audio,
        leading_ms=inline_stage.leading_silence_ms,
        trailing_ms=inline_stage.trailing_silence_ms,
    )
    duration_sec = len(audio) / SAMPLE_RATE

    audio_b64 = _audio_to_wav_base64(audio)

    style_used = vars(style) if style else {}
    style_used["style_preset"] = req.style_preset
    style_used["effective_speed"] = effective_speed
    style_used["effective_sentence_pause_ms"] = effective_sentence_pause_ms
    style_used["spoken_text"] = spoken_text
    style_used["inline_stage_directions"] = inline_stage.stage_directions
    style_used["inline_stage_speed_scale"] = inline_stage.speed_scale
    style_used["inline_stage_silence_ms"] = {
        "leading": inline_stage.leading_silence_ms,
        "trailing": inline_stage.trailing_silence_ms,
    }

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        style_used=style_used,
    )


@router.post("/tts/stream")
async def stream_tts(req: TTSStreamRequest) -> StreamingResponse:
    """Streaming TTS endpoint.

    Returns chunked raw PCM float32 audio (24kHz mono) using
    sentence-level incremental synthesis.  Each chunk is emitted
    as soon as a sentence (or sub-sentence chunk) is ready.

    Content-Type: application/octet-stream
    X-Sample-Rate: 24000
    X-Sample-Format: float32
    """
    from tmrvc_serve.app import get_engine, _characters, _context_predictor

    engine = get_engine()

    character = _characters.get(req.character_id)
    if character is None:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{req.character_id}' not found.",
        )

    from tmrvc_core.text_utils import analyze_inline_stage_directions
    inline_stage = analyze_inline_stage_directions(req.text, language=character.language)
    spoken_text = inline_stage.spoken_text

    spk_embed = _load_speaker_embed(character)

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
    effective_speed = _resolve_stage_speed(
        _resolve_effective_speed(req.speed, preset_cfg),
        inline_stage.speed_scale,
    )
    effective_sentence_pause_ms = _resolve_sentence_pause(
        preset_cfg.sentence_pause_ms,
        inline_stage.sentence_pause_ms_delta,
    )

    loop = asyncio.get_running_loop()

    async def _generate():
        sync_q: stdlib_queue.Queue[np.ndarray | None] = stdlib_queue.Queue(maxsize=4)

        _stream_done = threading.Event()

        def _produce():
            try:
                for chunk in _iter_silence_chunks(
                    inline_stage.leading_silence_ms,
                    req.chunk_duration_ms,
                ):
                    while not _stream_done.is_set():
                        try:
                            sync_q.put(chunk, timeout=0.05)
                            break
                        except stdlib_queue.Full:
                            continue
                    if _stream_done.is_set():
                        return

                for chunk in engine.synthesize_sentences(
                    text=spoken_text,
                    language=character.language,
                    spk_embed=spk_embed,
                    style=style,
                    speed=effective_speed,
                    chunk_duration_ms=req.chunk_duration_ms,
                    sentence_pause_ms=effective_sentence_pause_ms,
                    auto_style=preset_cfg.auto_style,
                ):
                    # Use timeout to avoid deadlock when consumer
                    # disconnects and stops reading.
                    while not _stream_done.is_set():
                        try:
                            sync_q.put(chunk, timeout=0.05)
                            break
                        except stdlib_queue.Full:
                            continue
                    if _stream_done.is_set():
                        return

                for chunk in _iter_silence_chunks(
                    inline_stage.trailing_silence_ms,
                    req.chunk_duration_ms,
                ):
                    while not _stream_done.is_set():
                        try:
                            sync_q.put(chunk, timeout=0.05)
                            break
                        except stdlib_queue.Full:
                            continue
                    if _stream_done.is_set():
                        return

                sync_q.put(None)
            except Exception:
                logger.exception("Streaming TTS synthesis failed")
                sync_q.put(None)

        producer = loop.run_in_executor(None, _produce)
        try:
            while True:
                try:
                    chunk = await loop.run_in_executor(
                        None,
                        lambda: sync_q.get(timeout=0.1),
                    )
                except stdlib_queue.Empty:
                    continue
                if chunk is None:
                    break
                yield chunk.astype(np.float32).tobytes()
        finally:
            _stream_done.set()
            await producer

    return StreamingResponse(
        _generate(),
        media_type="application/octet-stream",
        headers={
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Sample-Format": "float32",
        },
    )
