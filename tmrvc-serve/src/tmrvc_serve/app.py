"""FastAPI application for TMRVC TTS server.

Endpoints:
- POST /tts         — Generate audio from text (batch)
- POST /tts/stream  — Streaming audio generation
- WS   /ws/chat     — WebSocket for live chat TTS
- GET  /characters  — List available characters
- POST /characters  — Register a new character
- GET  /health      — Health check
"""

from __future__ import annotations

import base64
import io
import json
import logging
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
    TTSRequest,
    TTSResponse,
    WSAudioChunk,
    WSStyleInfo,
)
from tmrvc_serve.tts_engine import TTSEngine

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

    if api_key:
        from tmrvc_train.context_predictor import ContextStylePredictor
        _context_predictor = ContextStylePredictor(api_key=api_key)


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

    # Determine style
    style: StyleParams | None = None
    if req.emotion:
        style = StyleParams(emotion=req.emotion)
    elif _context_predictor and req.context:
        history = [
            DialogueTurn(speaker=t.speaker, text=t.text, emotion=t.emotion)
            for t in req.context
        ]
        try:
            style = await _context_predictor.predict(
                character, history, req.text, req.situation,
            )
        except Exception as e:
            logger.warning("Context prediction failed: %s", e)
            style = StyleParams.neutral()

    # Load speaker embedding
    spk_embed = _load_speaker_embed(character)

    audio, duration_sec = engine.synthesize(
        text=req.text,
        language=character.language,
        spk_embed=spk_embed,
        style=style,
        speed=req.speed,
    )

    audio_b64 = _audio_to_wav_base64(audio)

    return TTSResponse(
        audio_base64=audio_b64,
        sample_rate=SAMPLE_RATE,
        duration_sec=duration_sec,
        style_used=vars(style) if style else {},
    )


@app.websocket("/ws/chat")
async def chat_websocket(ws: WebSocket) -> None:
    """WebSocket endpoint for live chat TTS.

    Protocol:
    - Client sends JSON messages with "type" field
    - Server responds with audio chunks and style info
    """
    await ws.accept()
    logger.info("WebSocket client connected")

    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "detail": "Invalid JSON"})
                continue

            msg_type = msg.get("type")

            if msg_type == "response":
                # Manual response mode: generate TTS for given text
                character_id = msg.get("character_id", "")
                text = msg.get("text", "")

                if not text or not character_id:
                    await ws.send_json({"type": "error", "detail": "Missing text or character_id"})
                    continue

                character = _characters.get(character_id)
                if not character:
                    await ws.send_json({"type": "error", "detail": f"Unknown character: {character_id}"})
                    continue

                try:
                    engine = get_engine()
                    spk_embed = _load_speaker_embed(character)

                    # Style prediction
                    style = StyleParams.neutral()
                    if _context_predictor:
                        from tmrvc_train.context_predictor import ContextStylePredictor
                        style = _context_predictor.predict_rule_based(text, character)

                    # Send style info
                    await ws.send_json(
                        WSStyleInfo(
                            emotion=style.emotion,
                            vad=[style.valence, style.arousal, style.dominance],
                            reasoning=style.reasoning,
                        ).model_dump()
                    )

                    # Generate audio
                    audio, duration_sec = engine.synthesize(
                        text=text,
                        language=character.language,
                        spk_embed=spk_embed,
                        style=style,
                    )

                    # Send as single chunk
                    audio_b64 = base64.b64encode(audio.astype(np.float32).tobytes()).decode("ascii")
                    await ws.send_json(
                        WSAudioChunk(
                            data=audio_b64,
                            frame_index=0,
                            is_last=True,
                        ).model_dump()
                    )
                except Exception as e:
                    logger.error("TTS generation failed: %s", e)
                    await ws.send_json({"type": "error", "detail": f"TTS failed: {e}"})

            elif msg_type == "comment":
                # Chat comment — acknowledge
                await ws.send_json({
                    "type": "comment_ack",
                    "text": msg.get("text", ""),
                    "user": msg.get("user", ""),
                })

            else:
                await ws.send_json({"type": "error", "detail": f"Unknown message type: {msg_type}"})

    except WebSocketDisconnect:
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
