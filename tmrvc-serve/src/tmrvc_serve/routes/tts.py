"""POST /tts, POST /tts/stream, POST /tts/stream/sse, POST /tts/stream/v4, POST /tts/simple, POST /tts/prompt endpoints (UCLM pointer mode)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import torch

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_serve._helpers import (
    _append_silence,
    _audio_to_wav_base64,
    _load_speaker_embed,
    _to_dialogue_turns,
)
from tmrvc_serve.schemas import TTSRequestAdvanced, TTSRequestSimple, TTSRequestPrompt, TTSResponse
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
async def generate_tts(req: TTSRequestAdvanced) -> TTSResponse:
    from tmrvc_serve.app import get_engine, _characters, _context_predictor
    from tmrvc_core.text_utils import analyze_inline_stage_directions
    from tmrvc_data.g2p import text_to_phonemes

    engine = get_engine()

    # 1. G2P: text -> phoneme IDs
    g2p_result = text_to_phonemes(req.text, language=req.language or "ja")
    phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)

    # 2. Load speaker profile
    speaker_profile = None
    spk_t = None
    if req.speaker_profile_id:
        speaker_profile = engine.load_speaker_profile(req.speaker_profile_id)
        if speaker_profile is None:
            raise HTTPException(status_code=404, detail=f"Speaker profile '{req.speaker_profile_id}' not found.")
        spk_t = speaker_profile.speaker_embed.unsqueeze(0).to(engine.device)

    # 3. Handle on-the-fly few-shot adaptation
    if req.reference_audio_base64:
        ref_audio_bytes = base64.b64decode(req.reference_audio_base64)
        logger.info("Encoding on-the-fly reference audio...")
        ref_profile = engine.encode_on_the_fly_reference(ref_audio_bytes, text=None)
        speaker_profile = ref_profile
        spk_t = ref_profile.speaker_embed.unsqueeze(0).to(engine.device)

    # 4. Build 12-D physical controls tensor
    physical_controls_list = req.physical_controls.to_list()
    explicit_vs = torch.tensor(physical_controls_list, dtype=torch.float32).unsqueeze(0).to(engine.device)

    if req.delta_physical_controls is not None:
        delta_list = req.delta_physical_controls.to_list()
        delta_vs = torch.tensor(delta_list, dtype=torch.float32).unsqueeze(0).to(engine.device)
    else:
        delta_vs = None

    # 5. Extract pacing controls
    pacing = req.pacing

    # 6a. v4: Acting texture latent
    acting_tex_t = None
    if req.acting_texture_latent is not None:
        acting_tex_t = torch.tensor([req.acting_texture_latent], dtype=torch.float32).to(engine.device)

    # 6. Unified Synthesis
    audio_t, metrics = engine.tts(
        phonemes=phonemes_t,
        speaker_profile=speaker_profile,
        speaker_embed=spk_t,
        style=None,
        language_id=g2p_result.language_id,
        pace=pacing.pace,
        hold_bias=pacing.hold_bias,
        boundary_bias=pacing.boundary_bias,
        phrase_pressure=pacing.phrase_pressure,
        breath_tendency=pacing.breath_tendency,
        explicit_voice_state=explicit_vs,
        delta_voice_state=delta_vs,
        acting_texture_latent=acting_tex_t,
        cfg_scale=req.cfg_scale,
        cfg_mode=req.cfg_mode,
    )

    audio = audio_t.cpu().numpy()
    duration_sec = len(audio) / SAMPLE_RATE
    audio_b64 = _audio_to_wav_base64(audio)

    trajectory_id = metrics.get("trajectory_id")

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        trajectory_id=trajectory_id,
        provenance="fresh_compile",
        rtf=metrics.get("rtf", 0.0),
        gen_time_ms=metrics.get("gen_time_ms", 0.0),
        cfg_mode=metrics.get("cfg_mode", req.cfg_mode),
        forced_advance_count=metrics.get("forced_advance_count", 0),
        skip_protection_count=metrics.get("skip_protection_count", 0),
    )


@router.post("/tts/stream")
async def stream_tts(req: TTSRequestAdvanced) -> StreamingResponse:
    """Streaming TTS endpoint (batch fallback while full causal streaming is implemented)."""
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
async def stream_tts_sse(req: TTSRequestAdvanced) -> StreamingResponse:
    """Server-Sent Events streaming TTS endpoint.

    Uses batch fallback (generate full audio, then chunk into SSE events).
    The inner loop is structured so that real causal streaming can replace
    the batch generation later without changing the SSE event contract.

    Events emitted:
        ``event: audio``  -- base64-encoded PCM float32 chunk.
        ``event: pointer`` -- JSON pointer telemetry snapshot.
        ``event: done``   -- JSON final metrics.
    """
    res = await generate_tts(req)

    audio_bytes = base64.b64decode(res.audio_base64)
    audio = np.frombuffer(audio_bytes, dtype=np.float32) if len(audio_bytes) > 0 else np.array([], dtype=np.float32)

    chunk_samples = int(SAMPLE_RATE * 100 / 1000)  # 100ms chunks

    async def _sse_generator():
        total_chunks = max(1, (len(audio) + chunk_samples - 1) // chunk_samples)

        for i in range(0, max(1, len(audio)), chunk_samples):
            chunk = audio[i : i + chunk_samples]
            chunk_b64 = base64.b64encode(chunk.tobytes()).decode("ascii")
            yield f"event: audio\ndata: {chunk_b64}\n\n"

            chunk_idx = i // chunk_samples
            progress_frac = (chunk_idx + 1) / total_chunks
            tel = {"stream_progress": round(progress_frac, 4)}
            yield f"event: pointer\ndata: {json.dumps(tel)}\n\n"

            await asyncio.sleep(0)

        done_data = {
            "rtf": res.rtf,
            "gen_time_ms": res.gen_time_ms,
            "cfg_mode": res.cfg_mode,
        }
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


_V4_SCHEMA_VERSION = "1.0"


@router.post("/tts/stream/v4")
async def tts_stream_v4(req: TTSRequestAdvanced) -> StreamingResponse:
    """Real causal streaming TTS with v4 physical + acting controls.

    Returns Server-Sent Events with incremental audio chunks and telemetry.
    v4 does not permit batch fallback to count as claim-valid streaming.

    Events emitted:
        ``event: audio``      -- base64-encoded PCM float32 chunk (one per frame group).
        ``event: telemetry``  -- pointer state, physical trajectory slice, RTF snapshot.
        ``event: done``       -- final trajectory_id, provenance, and aggregate metrics.
    """
    from tmrvc_serve.app import get_engine
    from tmrvc_core.text_utils import analyze_inline_stage_directions
    from tmrvc_data.g2p import text_to_phonemes

    # --- Schema version gate ---
    if req.schema_version != _V4_SCHEMA_VERSION:
        raise HTTPException(
            status_code=422,
            detail=(
                f"v4 streaming requires schema_version='{_V4_SCHEMA_VERSION}', "
                f"got '{req.schema_version}'."
            ),
        )

    engine = get_engine()

    # 1. G2P: text -> phoneme IDs
    g2p_result = text_to_phonemes(req.text, language=req.language or "ja")
    phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)

    # 2. Load speaker profile
    speaker_profile = None
    spk_t = None
    if req.speaker_profile_id:
        speaker_profile = engine.load_speaker_profile(req.speaker_profile_id)
        if speaker_profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Speaker profile '{req.speaker_profile_id}' not found.",
            )
        spk_t = speaker_profile.speaker_embed.unsqueeze(0).to(engine.device)

    # 3. Handle on-the-fly few-shot adaptation
    if req.reference_audio_base64:
        ref_audio_bytes = base64.b64decode(req.reference_audio_base64)
        logger.info("v4-stream: encoding on-the-fly reference audio...")
        ref_profile = engine.encode_on_the_fly_reference(ref_audio_bytes, text=None)
        speaker_profile = ref_profile
        spk_t = ref_profile.speaker_embed.unsqueeze(0).to(engine.device)

    # 4. Build 12-D physical controls tensor
    physical_controls_list = req.physical_controls.to_list()
    explicit_vs = (
        torch.tensor(physical_controls_list, dtype=torch.float32)
        .unsqueeze(0)
        .to(engine.device)
    )

    if req.delta_physical_controls is not None:
        delta_list = req.delta_physical_controls.to_list()
        delta_vs = (
            torch.tensor(delta_list, dtype=torch.float32)
            .unsqueeze(0)
            .to(engine.device)
        )
    else:
        delta_vs = None

    # 5. Pacing controls
    pacing = req.pacing

    # 5b. v4: Acting texture latent
    acting_tex_t = None
    if req.acting_texture_latent is not None:
        acting_tex_t = torch.tensor([req.acting_texture_latent], dtype=torch.float32).to(engine.device)

    # 6. Causal streaming generation
    #    Use engine.tts_stream() if available, otherwise fall back to
    #    frame-by-frame slicing of a full generation.  The SSE contract
    #    remains identical either way, but the v4 endpoint marks its
    #    provenance accordingly so callers can distinguish true causal
    #    streaming from structured-batch streaming.
    import time as _time

    gen_start = _time.monotonic()

    has_causal = hasattr(engine, "tts_stream")

    if has_causal:
        # Real causal path: engine yields (chunk_audio, pointer_telemetry) tuples
        stream_iter = engine.tts_stream(
            phonemes=phonemes_t,
            speaker_profile=speaker_profile,
            speaker_embed=spk_t,
            style=None,
            language_id=g2p_result.language_id,
            pace=pacing.pace,
            hold_bias=pacing.hold_bias,
            boundary_bias=pacing.boundary_bias,
            phrase_pressure=pacing.phrase_pressure,
            breath_tendency=pacing.breath_tendency,
            explicit_voice_state=explicit_vs,
            delta_voice_state=delta_vs,
            acting_texture_latent=acting_tex_t,
            cfg_scale=req.cfg_scale,
            cfg_mode=req.cfg_mode,
        )
        provenance = "causal_stream_v4"
    else:
        # Structured-batch path: generate fully, then chunk.
        # Marked as structured_batch so claim auditing can distinguish it.
        audio_t, metrics = engine.tts(
            phonemes=phonemes_t,
            speaker_profile=speaker_profile,
            speaker_embed=spk_t,
            style=None,
            language_id=g2p_result.language_id,
            pace=pacing.pace,
            hold_bias=pacing.hold_bias,
            boundary_bias=pacing.boundary_bias,
            phrase_pressure=pacing.phrase_pressure,
            breath_tendency=pacing.breath_tendency,
            explicit_voice_state=explicit_vs,
            delta_voice_state=delta_vs,
            acting_texture_latent=acting_tex_t,
            cfg_scale=req.cfg_scale,
            cfg_mode=req.cfg_mode,
        )
        provenance = "structured_batch_v4"
        stream_iter = None

    chunk_samples = int(SAMPLE_RATE * 100 / 1000)  # 100 ms

    async def _v4_sse_generator():
        trajectory_id = None
        total_frames = 0
        total_samples = 0
        physical_trajectory: list[list[float]] = []

        if has_causal and stream_iter is not None:
            # --- True causal streaming ---
            for chunk_audio_t, pointer_state in stream_iter:
                chunk_np = chunk_audio_t.cpu().numpy().ravel()
                total_samples += len(chunk_np)
                total_frames += 1

                # Audio event
                chunk_b64 = base64.b64encode(chunk_np.astype(np.float32).tobytes()).decode("ascii")
                yield f"event: audio\ndata: {chunk_b64}\n\n"

                # Telemetry event
                elapsed = _time.monotonic() - gen_start
                audio_sec = total_samples / SAMPLE_RATE
                rtf = elapsed / audio_sec if audio_sec > 0 else 0.0

                tel = {
                    "pointer": pointer_state if isinstance(pointer_state, dict) else {},
                    "physical_trajectory_slice": physical_controls_list,
                    "rtf": round(rtf, 4),
                    "frames_emitted": total_frames,
                    "samples_emitted": total_samples,
                }
                yield f"event: telemetry\ndata: {json.dumps(tel)}\n\n"

                await asyncio.sleep(0)

            trajectory_id = getattr(stream_iter, "trajectory_id", None)
        else:
            # --- Structured-batch streaming ---
            audio_np = audio_t.cpu().numpy().ravel()
            trajectory_id = metrics.get("trajectory_id")

            num_chunks = max(1, (len(audio_np) + chunk_samples - 1) // chunk_samples)

            for i in range(0, max(1, len(audio_np)), chunk_samples):
                chunk = audio_np[i : i + chunk_samples]
                total_samples += len(chunk)
                total_frames += 1

                chunk_b64 = base64.b64encode(chunk.astype(np.float32).tobytes()).decode("ascii")
                yield f"event: audio\ndata: {chunk_b64}\n\n"

                elapsed = _time.monotonic() - gen_start
                audio_sec = total_samples / SAMPLE_RATE
                rtf = elapsed / audio_sec if audio_sec > 0 else 0.0
                progress_frac = total_frames / num_chunks

                tel = {
                    "pointer": {
                        "progress": round(progress_frac, 4),
                        "frames_generated": total_frames,
                    },
                    "physical_trajectory_slice": physical_controls_list,
                    "rtf": round(rtf, 4),
                    "frames_emitted": total_frames,
                    "samples_emitted": total_samples,
                }
                yield f"event: telemetry\ndata: {json.dumps(tel)}\n\n"

                await asyncio.sleep(0)

        # Done event
        gen_elapsed = _time.monotonic() - gen_start
        done_data = {
            "trajectory_id": trajectory_id,
            "provenance": provenance,
            "schema_version": _V4_SCHEMA_VERSION,
            "total_frames": total_frames,
            "total_samples": total_samples,
            "duration_sec": round(total_samples / SAMPLE_RATE, 4) if total_samples > 0 else 0.0,
            "gen_time_ms": round(gen_elapsed * 1000, 2),
            "rtf": round(gen_elapsed / (total_samples / SAMPLE_RATE), 4) if total_samples > 0 else 0.0,
        }
        yield f"event: done\ndata: {json.dumps(done_data)}\n\n"

    return StreamingResponse(
        _v4_sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Schema-Version": _V4_SCHEMA_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# Helpers for Simple / Prompt mode endpoints
# ---------------------------------------------------------------------------


def _resolve_speaker(engine, pseudo_speaker_id=None, speaker_profile_id=None):
    """Resolve speaker embed from available identifiers."""
    if speaker_profile_id:
        profile = engine.load_speaker_profile(speaker_profile_id)
        if profile is not None:
            return profile.speaker_embed
    if pseudo_speaker_id:
        profile = engine.load_speaker_profile(pseudo_speaker_id)
        if profile is not None:
            return profile.speaker_embed
    return np.zeros(192, dtype=np.float32)


# ---------------------------------------------------------------------------
# Simple mode TTS
# ---------------------------------------------------------------------------


@router.post("/tts/simple")
async def tts_simple(req: TTSRequestSimple, request: Request):
    """Simple mode TTS: emotion + speed only."""
    from tmrvc_serve.app import get_engine
    from tmrvc_data.g2p import text_to_phonemes

    engine = get_engine()

    # Resolve speaker
    spk_embed = _resolve_speaker(engine, req.pseudo_speaker_id, req.speaker_profile_id)

    # Convert speed to pace (simple linear mapping)
    pace = req.speed if req.speed else 1.0

    # Resolve emotion to style
    style = None
    if req.emotion:
        style = _resolve_style_preset(req.emotion)

    # G2P
    g2p_result = text_to_phonemes(req.text, language=req.language or "ja")
    phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)
    spk_t = torch.from_numpy(np.asarray(spk_embed)).float().to(engine.device)
    if spk_t.dim() == 1:
        spk_t = spk_t.unsqueeze(0)

    audio_t, stats = engine.tts(
        phonemes=phonemes_t,
        speaker_embed=spk_t,
        style=style,
        language_id=g2p_result.language_id,
        pace=pace,
    )

    audio = audio_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
    audio_b64 = base64.b64encode(audio.tobytes()).decode()

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=len(audio) / SAMPLE_RATE,
        provenance="simple_mode",
    )


# ---------------------------------------------------------------------------
# Prompt mode TTS
# ---------------------------------------------------------------------------


@router.post("/tts/prompt")
async def tts_prompt(req: TTSRequestPrompt, request: Request):
    """Prompt mode TTS: natural language acting intent -> synthesis."""
    from tmrvc_serve.app import get_engine
    from tmrvc_serve.intent_compiler import IntentCompiler
    from tmrvc_data.g2p import text_to_phonemes

    engine = get_engine()

    # Compile intent
    compiler = getattr(request.app.state, "intent_compiler", None)
    if compiler is None:
        compiler = IntentCompiler(device=engine.device)
        request.app.state.intent_compiler = compiler

    compiled = compiler.compile(
        prompt=req.acting_prompt or "",
        context={
            "text": req.text,
            "scene_context": req.scene_context,
        },
    )

    # Resolve speaker
    spk_embed = _resolve_speaker(engine, None, req.speaker_profile_id)

    # G2P
    g2p_result = text_to_phonemes(req.text, language=req.language or "ja")
    phonemes_t = g2p_result.phoneme_ids.to(dtype=torch.long).unsqueeze(0)
    spk_t = torch.from_numpy(np.asarray(spk_embed)).float().to(engine.device)
    if spk_t.dim() == 1:
        spk_t = spk_t.unsqueeze(0)

    # Build voice state from compiled physical targets
    voice_state = compiled.physical_targets.float().to(engine.device)
    if voice_state.dim() == 1:
        voice_state = voice_state.unsqueeze(0)

    # Build acting texture latent
    acting_latent = None
    if compiled.acting_latent_prior is not None:
        acting_latent = compiled.acting_latent_prior.float().to(engine.device)
        if acting_latent.dim() == 1:
            acting_latent = acting_latent.unsqueeze(0)

    # Extract pacing from compiled output
    pacing = compiled.pacing

    audio_t, stats = engine.tts(
        phonemes=phonemes_t,
        speaker_embed=spk_t,
        style=None,
        language_id=g2p_result.language_id,
        explicit_voice_state=voice_state,
        acting_texture_latent=acting_latent,
        pace=pacing.pace,
        hold_bias=pacing.hold_bias,
        boundary_bias=pacing.boundary_bias,
        phrase_pressure=pacing.phrase_pressure,
        breath_tendency=pacing.breath_tendency,
    )

    audio = audio_t.detach().cpu().numpy().astype(np.float32).reshape(-1)
    audio_b64 = base64.b64encode(audio.tobytes()).decode()

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=len(audio) / SAMPLE_RATE,
        provenance="prompt_compile",
        trajectory_id=compiled.compile_id,
    )
