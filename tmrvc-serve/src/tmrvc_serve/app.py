"""FastAPI application for TMRVC TTS server.

Endpoints:
- POST /tts            Generate audio from text (batch)
- POST /tts/stream     Streaming audio generation (chunked PCM)
- WS   /ws/chat        WebSocket for live chat TTS (real-time priority queue)
- GET  /characters     List available characters
- POST /characters     Register a new character
- GET  /health         Health check
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import io
import json
import logging
import queue as stdlib_queue
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import (
    CharacterProfile,
    DialogueTurn,
    StyleParams,
)
from tmrvc_serve.schemas import (
    CharacterCreateRequest,
    CharacterInfo,
    HealthResponse,
    Priority,
    StylePreset,
    TTSRequest,
    TTSResponse,
    TTSStreamRequest,
    WSAudioMessage,
    WSError,
    WSQueueStatus,
    WSSkipped,
    WSStyleMessage,
)
from tmrvc_serve.tts_engine import FADEOUT_SAMPLES, TTSEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="TMRVC TTS Server",
    description="Text-to-Speech API with emotion/style control and WebSocket chat support.",
    version="0.1.0",
)

# Global state (initialized in lifespan or startup)
_engine: TTSEngine | None = None
_characters: dict[str, CharacterProfile] = {}
_context_predictor = None

_MAX_QUEUE_SIZE = 20
_DIALOGUE_HISTORY_MAX = 24
_HINT_BLEND_WEIGHT = 0.35
_SITUATION_BLEND_WEIGHT = 0.20
_BACKCHANNEL_TOKENS = frozenset({
    "uh-huh", "yeah", "yes", "no", "okay", "ok",
    "うん", "はい", "ええ", "そう", "そうだね",
    "嗯", "好", "是", "对",
    "응", "네", "그래",
})


@dataclasses.dataclass(frozen=True)
class StylePresetConfig:
    """High-level preset for ASMR-style speaking characteristics."""

    emotion: str | None = None
    delta_valence: float = 0.0
    delta_arousal: float = 0.0
    delta_dominance: float = 0.0
    delta_speech_rate: float = 0.0
    delta_energy: float = 0.0
    delta_pitch_range: float = 0.0
    speed_multiplier: float = 1.0
    sentence_pause_ms: int = 120
    auto_style: bool = True


_STYLE_PRESET_TABLE: dict[str, StylePresetConfig] = {
    "default": StylePresetConfig(),
    "asmr_soft": StylePresetConfig(
        emotion="whisper",
        delta_valence=0.10,
        delta_arousal=-0.35,
        delta_speech_rate=-0.25,
        delta_energy=-0.55,
        delta_pitch_range=-0.10,
        speed_multiplier=0.90,
        sentence_pause_ms=220,
        auto_style=False,
    ),
    "asmr_intimate": StylePresetConfig(
        emotion="whisper",
        delta_valence=0.15,
        delta_arousal=-0.45,
        delta_speech_rate=-0.35,
        delta_energy=-0.65,
        delta_pitch_range=-0.20,
        speed_multiplier=0.82,
        sentence_pause_ms=280,
        auto_style=False,
    ),
}


def _clamp_style(v: float) -> float:
    return max(-1.0, min(1.0, v))


def _clamp_speed(v: float) -> float:
    return max(0.5, min(2.0, v))


def _resolve_style_preset(
    base_style: StyleParams | None,
    preset: StylePreset,
) -> tuple[StyleParams | None, StylePresetConfig]:
    """Apply high-level style preset to a base style."""
    cfg = _STYLE_PRESET_TABLE[str(preset)]
    if preset == "default" and base_style is None:
        return None, cfg

    src = base_style or StyleParams.neutral()
    result = StyleParams(
        emotion=cfg.emotion or src.emotion,
        valence=_clamp_style(src.valence + cfg.delta_valence),
        arousal=_clamp_style(src.arousal + cfg.delta_arousal),
        dominance=_clamp_style(src.dominance + cfg.delta_dominance),
        speech_rate=_clamp_style(src.speech_rate + cfg.delta_speech_rate),
        energy=_clamp_style(src.energy + cfg.delta_energy),
        pitch_range=_clamp_style(src.pitch_range + cfg.delta_pitch_range),
        reasoning=(src.reasoning + "; " if src.reasoning else "") + f"preset={preset}",
    )
    return result, cfg


def _resolve_effective_speed(base_speed: float, cfg: StylePresetConfig) -> float:
    return _clamp_speed(base_speed * cfg.speed_multiplier)


def _merge_reasoning(*parts: str | None) -> str:
    chunks = [p.strip() for p in parts if p and p.strip()]
    return "; ".join(chunks)


def _blend_style_values(base: float, overlay: float, overlay_weight: float) -> float:
    return _clamp_style((base * (1.0 - overlay_weight)) + (overlay * overlay_weight))


def _blend_styles(
    base: StyleParams | None,
    overlay: StyleParams | None,
    overlay_weight: float,
    reason_tag: str,
) -> StyleParams | None:
    if base is None and overlay is None:
        return None
    if base is None:
        assert overlay is not None
        return StyleParams(
            emotion=overlay.emotion,
            valence=overlay.valence,
            arousal=overlay.arousal,
            dominance=overlay.dominance,
            speech_rate=overlay.speech_rate,
            energy=overlay.energy,
            pitch_range=overlay.pitch_range,
            reasoning=_merge_reasoning(overlay.reasoning, reason_tag),
        )
    if overlay is None:
        return base

    emotion = base.emotion
    if base.emotion == "neutral" and overlay.emotion != "neutral":
        emotion = overlay.emotion

    return StyleParams(
        emotion=emotion,
        valence=_blend_style_values(base.valence, overlay.valence, overlay_weight),
        arousal=_blend_style_values(base.arousal, overlay.arousal, overlay_weight),
        dominance=_blend_style_values(base.dominance, overlay.dominance, overlay_weight),
        speech_rate=_blend_style_values(base.speech_rate, overlay.speech_rate, overlay_weight),
        energy=_blend_style_values(base.energy, overlay.energy, overlay_weight),
        pitch_range=_blend_style_values(base.pitch_range, overlay.pitch_range, overlay_weight),
        reasoning=_merge_reasoning(base.reasoning, overlay.reasoning, reason_tag),
    )


def _predict_style_rule_based(text: str, character: CharacterProfile) -> StyleParams:
    if _context_predictor is not None:
        return _context_predictor.predict_rule_based(text, character)

    from tmrvc_core.text_utils import infer_sentence_style

    fallback = infer_sentence_style(text, character.language, StyleParams.neutral())
    if isinstance(fallback, StyleParams):
        return fallback
    return StyleParams.neutral()


def _is_backchannel(text: str) -> bool:
    normalized = text.strip().lower().strip(" \t\r\n.。!?！？…")
    return normalized in _BACKCHANNEL_TOKENS


def _apply_dialogue_dynamics(
    style: StyleParams | None,
    history: list[DialogueTurn],
    speaker: str | None,
    text: str,
) -> StyleParams | None:
    if style is None or not history:
        return style

    prev_turn = history[-1]
    valence = style.valence
    arousal = style.arousal
    dominance = style.dominance
    speech_rate = style.speech_rate
    energy = style.energy
    pitch_range = style.pitch_range
    tags: list[str] = []

    turn_changed = bool(speaker) and prev_turn.speaker != speaker
    if turn_changed and ("?" in prev_turn.text or "？" in prev_turn.text):
        arousal += 0.12
        energy += 0.08
        pitch_range += 0.10
        tags.append("reply_to_question")

    if turn_changed and prev_turn.emotion in {"angry", "excited"}:
        arousal += 0.08
        dominance += 0.08
        tags.append("carry_tension")

    if turn_changed and prev_turn.emotion in {"sad", "fearful", "whisper"}:
        valence -= 0.08
        speech_rate -= 0.08
        energy -= 0.08
        tags.append("carry_softness")

    if _is_backchannel(text):
        speech_rate -= 0.08
        energy -= 0.10
        pitch_range -= 0.05
        tags.append("backchannel")

    return StyleParams(
        emotion=style.emotion,
        valence=_clamp_style(valence),
        arousal=_clamp_style(arousal),
        dominance=_clamp_style(dominance),
        speech_rate=_clamp_style(speech_rate),
        energy=_clamp_style(energy),
        pitch_range=_clamp_style(pitch_range),
        reasoning=_merge_reasoning(style.reasoning, ",".join(tags) if tags else None),
    )


def _to_dialogue_turns(
    context: list[object] | None,
) -> list[DialogueTurn]:
    if not context:
        return []
    return [
        DialogueTurn(
            speaker=str(getattr(turn, "speaker")),
            text=str(getattr(turn, "text")),
            emotion=getattr(turn, "emotion", None),
        )
        for turn in context
    ]


async def _predict_style_from_inputs(
    *,
    character: CharacterProfile,
    text: str,
    emotion: str | None,
    history: list[DialogueTurn],
    situation: str | None,
    hint: str | None,
    speaker: str | None,
) -> StyleParams | None:
    style: StyleParams | None = None

    if emotion:
        style = StyleParams(emotion=emotion, reasoning="emotion_override")
    else:
        if _context_predictor is not None:
            try:
                if history or situation:
                    style = await _context_predictor.predict(
                        character,
                        history,
                        text,
                        situation,
                    )
                else:
                    style = _context_predictor.predict_rule_based(text, character)
            except Exception as e:
                logger.warning("Context prediction failed, using rule-based fallback: %s", e)
                style = None
        if style is None:
            style = _predict_style_rule_based(text, character)

    style = _apply_dialogue_dynamics(style, history, speaker, text)

    if situation:
        situation_style = _predict_style_rule_based(situation, character)
        style = _blend_styles(
            style,
            situation_style,
            overlay_weight=_SITUATION_BLEND_WEIGHT,
            reason_tag="situation_soft",
        )

    if hint:
        hint_style = _predict_style_rule_based(hint, character)
        style = _blend_styles(
            style,
            hint_style,
            overlay_weight=_HINT_BLEND_WEIGHT,
            reason_tag="hint_soft",
        )

    return style


def get_engine() -> TTSEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="TTS engine not initialized.")
    return _engine


def init_app(
    tts_checkpoint: str | Path | None = None,
    vc_checkpoint: str | Path | None = None,
    device: str = "cpu",
    api_key: str | None = None,
) -> None:
    """Initialize the TTS engine and context predictor.

    Called by the CLI or manually before serving.
    """
    global _engine, _context_predictor

    _engine = TTSEngine(
        tts_checkpoint=tts_checkpoint,
        vc_checkpoint=vc_checkpoint,
        device=device,
    )
    _engine.load_models()
    _engine.warmup()

    try:
        from tmrvc_train.context_predictor import ContextStylePredictor
        _context_predictor = ContextStylePredictor(api_key=api_key)
    except Exception as e:
        logger.warning("Context predictor unavailable; using local fallback only: %s", e)
        _context_predictor = None


def _audio_to_wav_base64(audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    """Encode float32 audio to base64 WAV string."""
    import struct

    buf = io.BytesIO()
    n_samples = len(audio)
    data_size = n_samples * 4  # float32 = 4 bytes

    # WAV header (float32 format, IEEE)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))  # chunk size
    buf.write(struct.pack("<H", 3))   # format = IEEE float
    buf.write(struct.pack("<H", 1))   # channels
    buf.write(struct.pack("<I", sr))  # sample rate
    buf.write(struct.pack("<I", sr * 4))  # byte rate
    buf.write(struct.pack("<H", 4))   # block align
    buf.write(struct.pack("<H", 32))  # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(audio.astype(np.float32).tobytes())

    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engine = _engine
    return HealthResponse(
        status="ok",
        models_loaded=engine.models_loaded if engine else False,
        characters_count=len(_characters),
    )


@app.get("/characters", response_model=list[CharacterInfo])
async def list_characters() -> list[CharacterInfo]:
    return [
        CharacterInfo(
            id=cid,
            name=c.name,
            personality=c.personality,
            voice_description=c.voice_description,
            language=c.language,
        )
        for cid, c in _characters.items()
    ]


@app.post("/characters", response_model=CharacterInfo)
async def create_character(req: CharacterCreateRequest) -> CharacterInfo:
    if req.id in _characters:
        raise HTTPException(status_code=409, detail=f"Character '{req.id}' already exists.")

    speaker_file = None
    if req.speaker_file:
        sp = Path(req.speaker_file).resolve()
        # Validate: must end with .tmrvc_speaker and not traverse outside models/
        if not sp.name.endswith(".tmrvc_speaker"):
            raise HTTPException(status_code=400, detail="Speaker file must be .tmrvc_speaker")
        if not sp.exists():
            raise HTTPException(status_code=400, detail=f"Speaker file not found: {sp.name}")
        speaker_file = sp

    profile = CharacterProfile(
        name=req.name,
        personality=req.personality,
        voice_description=req.voice_description,
        language=req.language,
        speaker_file=speaker_file,
    )
    _characters[req.id] = profile

    return CharacterInfo(
        id=req.id,
        name=req.name,
        personality=req.personality,
        voice_description=req.voice_description,
        language=req.language,
    )


@app.post("/tts", response_model=TTSResponse)
async def generate_tts(req: TTSRequest) -> TTSResponse:
    engine = get_engine()

    character = _characters.get(req.character_id)
    if character is None:
        raise HTTPException(status_code=404, detail=f"Character '{req.character_id}' not found.")

    history = _to_dialogue_turns(req.context)
    style = await _predict_style_from_inputs(
        character=character,
        text=req.text,
        emotion=req.emotion,
        history=history,
        situation=req.situation,
        hint=req.hint,
        speaker=req.character_id,
    )

    style, preset_cfg = _resolve_style_preset(style, req.style_preset)
    effective_speed = _resolve_effective_speed(req.speed, preset_cfg)

    # Load speaker embedding
    spk_embed = _load_speaker_embed(character)
    if req.style_preset == "default":
        audio, duration_sec = engine.synthesize(
            text=req.text,
            language=character.language,
            spk_embed=spk_embed,
            style=style,
            speed=effective_speed,
        )
    else:
        chunks = list(engine.synthesize_sentences(
            text=req.text,
            language=character.language,
            spk_embed=spk_embed,
            style=style,
            speed=effective_speed,
            sentence_pause_ms=preset_cfg.sentence_pause_ms,
            auto_style=preset_cfg.auto_style,
        ))
        if chunks:
            audio = np.concatenate(chunks).astype(np.float32)
        else:
            audio = np.zeros(0, dtype=np.float32)
        duration_sec = len(audio) / SAMPLE_RATE

    audio_b64 = _audio_to_wav_base64(audio)

    style_used = vars(style) if style else {}
    style_used["style_preset"] = req.style_preset
    style_used["effective_speed"] = effective_speed

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        style_used=style_used,
    )


@app.post("/tts/stream")
async def stream_tts(req: TTSStreamRequest) -> StreamingResponse:
    """Streaming TTS endpoint.

    Returns chunked raw PCM float32 audio (24kHz mono) using
    sentence-level incremental synthesis.  Each chunk is emitted
    as soon as a sentence (or sub-sentence chunk) is ready.

    Content-Type: application/octet-stream
    X-Sample-Rate: 24000
    X-Sample-Format: float32
    """
    engine = get_engine()

    character = _characters.get(req.character_id)
    if character is None:
        raise HTTPException(
            status_code=404,
            detail=f"Character '{req.character_id}' not found.",
        )

    spk_embed = _load_speaker_embed(character)

    history = _to_dialogue_turns(req.context)
    style = await _predict_style_from_inputs(
        character=character,
        text=req.text,
        emotion=req.emotion,
        history=history,
        situation=req.situation,
        hint=req.hint,
        speaker=req.character_id,
    )
    style, preset_cfg = _resolve_style_preset(style, req.style_preset)
    effective_speed = _resolve_effective_speed(req.speed, preset_cfg)

    loop = asyncio.get_running_loop()

    async def _generate():
        sync_q: stdlib_queue.Queue[np.ndarray | None] = stdlib_queue.Queue(maxsize=4)

        _stream_done = threading.Event()

        def _produce():
            try:
                for chunk in engine.synthesize_sentences(
                    text=req.text,
                    language=character.language,
                    spk_embed=spk_embed,
                    style=style,
                    speed=effective_speed,
                    chunk_duration_ms=req.chunk_duration_ms,
                    sentence_pause_ms=preset_cfg.sentence_pause_ms,
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


# ---------------------------------------------------------------------------
# WebSocket: real-time TTS with priority queue
# ---------------------------------------------------------------------------


@dataclasses.dataclass(order=False)
class SpeakItem:
    """An item in the priority queue for the consumer task."""

    priority: int
    timestamp: float
    text: str
    character_id: str
    emotion: str | None
    style_preset: StylePreset
    seq: int
    hint: str | None = None
    situation: str | None = None
    speed: float | None = None

    def __lt__(self, other: SpeakItem) -> bool:
        """Lower priority value = higher priority; break ties by timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp


@app.websocket("/ws/chat")
async def chat_websocket(ws: WebSocket) -> None:
    """WebSocket endpoint for live chat TTS.

    Protocol (new):
    - ``speak``      遯ｶ繝ｻenqueue TTS with priority + interrupt
    - ``cancel``     遯ｶ繝ｻdrain queue and interrupt current speech
    - ``configure``  遯ｶ繝ｻupdate session character_id / speed
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    queue: asyncio.PriorityQueue[SpeakItem] = asyncio.PriorityQueue(maxsize=_MAX_QUEUE_SIZE)
    interrupt_event = asyncio.Event()
    cancel_event = threading.Event()  # thread-safe cancel for synthesize_sentences
    is_speaking = False
    seq_counter = 0

    # Session-level config (mutable by ``configure`` messages)
    session_character_id: str = ""
    session_speed: float = 1.0
    session_style_preset: StylePreset = "default"
    session_situation: str | None = None
    dialogue_history: list[DialogueTurn] = []

    async def _send(msg: dict) -> None:
        """Helper to send JSON, silently ignoring closed connections."""
        try:
            await ws.send_json(msg)
        except Exception:
            pass

    async def _send_queue_status() -> None:
        await _send(WSQueueStatus(
            pending=queue.qsize(),
            speaking=is_speaking,
        ).model_dump())

    # ------------------------------------------------------------------
    # Receiver task 遯ｶ繝ｻreads client messages and dispatches
    # ------------------------------------------------------------------
    async def receiver() -> None:
        nonlocal seq_counter, session_character_id, session_speed
        nonlocal session_style_preset, session_situation
        try:
            while True:
                data = await ws.receive_text()
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    await _send(WSError(detail="Invalid JSON").model_dump())
                    continue

                msg_type = msg.get("type")

                if msg_type == "speak":
                    text = msg.get("text", "")
                    if not text:
                        await _send(WSError(detail="Empty text in speak request").model_dump())
                        continue

                    char_id = msg.get("character_id") or session_character_id
                    priority = msg.get("priority", Priority.NORMAL)
                    do_interrupt = msg.get("interrupt", False)

                    if do_interrupt:
                        interrupt_event.set()
                        cancel_event.set()
                        # Drain the queue
                        while not queue.empty():
                            try:
                                skipped = queue.get_nowait()
                                await _send(WSSkipped(
                                    text=skipped.text,
                                    reason="interrupted",
                                ).model_dump())
                            except asyncio.QueueEmpty:
                                break

                    raw_speed = msg.get("speed")
                    speak_speed = float(raw_speed) if raw_speed is not None else None
                    speak_situation = (
                        msg.get("situation")
                        if "situation" in msg
                        else session_situation
                    )
                    style_preset_raw = msg.get("style_preset")
                    style_preset = style_preset_raw or session_style_preset
                    if style_preset not in _STYLE_PRESET_TABLE:
                        await _send(
                            WSError(detail=f"Unknown style_preset: {style_preset}").model_dump(),
                        )
                        continue

                    seq_counter += 1
                    item = SpeakItem(
                        priority=int(priority),
                        timestamp=time.monotonic(),
                        text=text,
                        character_id=char_id,
                        emotion=msg.get("emotion"),
                        style_preset=style_preset,
                        seq=seq_counter,
                        hint=msg.get("hint"),
                        situation=speak_situation,
                        speed=speak_speed,
                    )
                    if queue.full():
                        await _send(WSSkipped(
                            text=text,
                            reason="queue_full",
                        ).model_dump())
                    else:
                        await queue.put(item)
                        await _send_queue_status()

                elif msg_type == "cancel":
                    interrupt_event.set()
                    cancel_event.set()
                    while not queue.empty():
                        try:
                            skipped = queue.get_nowait()
                            await _send(WSSkipped(
                                text=skipped.text,
                                reason="cancelled",
                            ).model_dump())
                        except asyncio.QueueEmpty:
                            break
                    await _send_queue_status()

                elif msg_type == "configure":
                    if "character_id" in msg and msg["character_id"] is not None:
                        session_character_id = msg["character_id"]
                    if "speed" in msg and msg["speed"] is not None:
                        session_speed = max(0.5, min(2.0, float(msg["speed"])))
                    if "situation" in msg:
                        session_situation = msg["situation"]
                    if "style_preset" in msg and msg["style_preset"] is not None:
                        style_preset = msg["style_preset"]
                        if style_preset not in _STYLE_PRESET_TABLE:
                            await _send(
                                WSError(detail=f"Unknown style_preset: {style_preset}").model_dump(),
                            )
                            continue
                        session_style_preset = style_preset

                else:
                    await _send(WSError(detail=f"Unknown message type: {msg_type}").model_dump())

        except WebSocketDisconnect:
            pass

    # ------------------------------------------------------------------
    # Consumer task 窶・pulls from queue, synthesizes sentences, streams
    # ------------------------------------------------------------------
    async def consumer() -> None:
        nonlocal is_speaking
        loop = asyncio.get_running_loop()
        try:
            while True:
                item = await queue.get()
                interrupt_event.clear()
                cancel_event.clear()
                is_speaking = True
                await _send_queue_status()

                char_id = item.character_id
                character = _characters.get(char_id)
                if not character:
                    await _send(WSError(detail=f"Unknown character: {char_id}").model_dump())
                    is_speaking = False
                    continue

                try:
                    engine = get_engine()
                except HTTPException:
                    await _send(WSError(detail="TTS engine not initialized").model_dump())
                    is_speaking = False
                    continue

                spk_embed = _load_speaker_embed(character)

                history_snapshot = dialogue_history[-_DIALOGUE_HISTORY_MAX:]
                style = await _predict_style_from_inputs(
                    character=character,
                    text=item.text,
                    emotion=item.emotion,
                    history=history_snapshot,
                    situation=item.situation,
                    hint=item.hint,
                    speaker=item.character_id,
                )
                style, preset_cfg = _resolve_style_preset(style, item.style_preset)
                style_msg = style or StyleParams.neutral()

                # Send style message (before audio, for avatar sync)
                await _send(WSStyleMessage(
                    emotion=style_msg.emotion,
                    vad=[style_msg.valence, style_msg.arousal, style_msg.dominance],
                    reasoning=style_msg.reasoning,
                    seq=item.seq,
                ).model_dump())
                dialogue_history.append(DialogueTurn(
                    speaker=item.character_id or character.name,
                    text=item.text,
                    emotion=style_msg.emotion,
                ))
                if len(dialogue_history) > _DIALOGUE_HISTORY_MAX:
                    dialogue_history.pop(0)

                # Per-request speed overrides session speed
                base_speed = item.speed if item.speed is not None else session_speed
                effective_speed = _resolve_effective_speed(base_speed, preset_cfg)

                # Producer-consumer bridge: synthesize_sentences runs in
                # a thread and pushes chunks into a stdlib queue; the async
                # consumer loop pulls them out.
                sync_q: stdlib_queue.Queue[np.ndarray | None] = stdlib_queue.Queue(maxsize=4)

                def _produce() -> None:
                    try:
                        for chunk in engine.synthesize_sentences(
                            text=item.text,
                            language=character.language,
                            spk_embed=spk_embed,
                            style=style,
                            speed=effective_speed,
                            cancel=cancel_event,
                            sentence_pause_ms=preset_cfg.sentence_pause_ms,
                            auto_style=preset_cfg.auto_style,
                        ):
                            # Use timeout to avoid deadlock when consumer
                            # stops reading (interrupt).  Check cancel between
                            # retries so the thread can exit.
                            while True:
                                if cancel_event.is_set():
                                    sync_q.put(None)
                                    return
                                try:
                                    sync_q.put(chunk, timeout=0.05)
                                    break
                                except stdlib_queue.Full:
                                    continue
                        sync_q.put(None)  # sentinel
                    except Exception:
                        logger.exception("TTS synthesis failed")
                        sync_q.put(None)

                # Speculative prefetch: pop next item, run G2P, re-enqueue.
                # PriorityQueue has no peek, so pop+re-put is the only way.
                # Best-effort: silently skip if queue is empty or on error.
                prefetch_future = None
                if not queue.empty():
                    try:
                        next_item = queue.get_nowait()
                        next_char = _characters.get(next_item.character_id)
                        if next_char:
                            prefetch_future = loop.run_in_executor(
                                None,
                                lambda ni=next_item, nc=next_char: engine.prefetch_g2p(
                                    ni.text, nc.language,
                                ),
                            )
                        # Re-enqueue immediately (order preserved by priority+timestamp)
                        await queue.put(next_item)
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        pass

                producer_future = loop.run_in_executor(None, _produce)

                chunk_idx = 0
                last_chunk: np.ndarray | None = None
                try:
                    while True:
                        # Check interrupt before pulling next chunk
                        if interrupt_event.is_set():
                            cancel_event.set()
                            # Apply fadeout to the last sent chunk
                            if last_chunk is not None and len(last_chunk) > 0:
                                n = min(FADEOUT_SAMPLES, len(last_chunk))
                                fadeout = np.linspace(1.0, 0.0, n, dtype=np.float32)
                                faded = last_chunk.copy()
                                faded[-n:] = faded[-n:] * fadeout
                                audio_b64 = base64.b64encode(
                                    faded.astype(np.float32).tobytes(),
                                ).decode("ascii")
                                await _send(WSAudioMessage(
                                    data=audio_b64,
                                    seq=item.seq,
                                    chunk_index=chunk_idx,
                                    is_last=True,
                                ).model_dump())
                            logger.info("Interrupted during streaming (seq=%d)", item.seq)
                            break

                        try:
                            chunk = await loop.run_in_executor(
                                None,
                                lambda: sync_q.get(timeout=0.05),
                            )
                        except stdlib_queue.Empty:
                            continue

                        if chunk is None:
                            # Sentinel 窶・mark last sent chunk as is_last
                            if last_chunk is not None:
                                audio_b64 = base64.b64encode(
                                    last_chunk.astype(np.float32).tobytes(),
                                ).decode("ascii")
                                await _send(WSAudioMessage(
                                    data=audio_b64,
                                    seq=item.seq,
                                    chunk_index=chunk_idx,
                                    is_last=True,
                                ).model_dump())
                            break

                        # Send previous chunk (not is_last yet)
                        if last_chunk is not None:
                            audio_b64 = base64.b64encode(
                                last_chunk.astype(np.float32).tobytes(),
                            ).decode("ascii")
                            await _send(WSAudioMessage(
                                data=audio_b64,
                                seq=item.seq,
                                chunk_index=chunk_idx,
                                is_last=False,
                            ).model_dump())
                            chunk_idx += 1

                        last_chunk = chunk

                except Exception as e:
                    logger.error("Consumer error: %s", e)
                    cancel_event.set()

                # Wait for producer to finish
                await producer_future

                # Wait for prefetch if it was started
                if prefetch_future is not None:
                    try:
                        await prefetch_future
                    except Exception:
                        pass  # best-effort

                # Drain any remaining items in sync_q
                while not sync_q.empty():
                    try:
                        sync_q.get_nowait()
                    except stdlib_queue.Empty:
                        break

                is_speaking = False
                await _send_queue_status()

        except asyncio.CancelledError:
            cancel_event.set()
            is_speaking = False

    # ------------------------------------------------------------------
    # Run receiver + consumer concurrently
    # ------------------------------------------------------------------
    consumer_task = asyncio.create_task(consumer())
    try:
        await receiver()
    finally:
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        logger.info("WebSocket client disconnected")


def _load_speaker_embed(character: CharacterProfile) -> "torch.Tensor":
    """Load speaker embedding from character's speaker file."""
    import torch

    if character.speaker_file and character.speaker_file.exists():
        from tmrvc_export.speaker_file import read_speaker_file
        spk_embed, _lora, _meta, _thumb = read_speaker_file(character.speaker_file)
        return torch.from_numpy(spk_embed).float()

    # Fallback: zero embedding
    logger.warning("No speaker file for '%s', using zero embedding", character.name)
    return torch.zeros(192)
