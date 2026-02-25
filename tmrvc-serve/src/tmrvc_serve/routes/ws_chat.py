"""WebSocket /ws/chat endpoint with priority queue."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import logging
import queue as stdlib_queue
import threading
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from tmrvc_core.dialogue_types import DialogueTurn, StyleParams
from tmrvc_serve._helpers import _iter_silence_chunks, _load_speaker_embed
from tmrvc_serve.schemas import (
    Priority,
    StylePreset,
    WSAudioMessage,
    WSError,
    WSQueueStatus,
    WSSkipped,
    WSStyleMessage,
)
from tmrvc_serve.style_resolver import (
    _STYLE_PRESET_TABLE,
    _apply_inline_stage_overlay,
    _predict_style_from_inputs,
    _resolve_effective_speed,
    _resolve_sentence_pause,
    _resolve_stage_speed,
    _resolve_style_preset,
)
from tmrvc_serve.tts_engine import FADEOUT_SAMPLES

logger = logging.getLogger(__name__)

_MAX_QUEUE_SIZE = 20
_DIALOGUE_HISTORY_MAX = 24

router = APIRouter()


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


@router.websocket("/ws/chat")
async def chat_websocket(ws: WebSocket) -> None:
    """WebSocket endpoint for live chat TTS.

    Protocol (new):
    - ``speak``      — enqueue TTS with priority + interrupt
    - ``cancel``     — drain queue and interrupt current speech
    - ``configure``  — update session character_id / speed
    """
    from fastapi import HTTPException

    from tmrvc_serve.app import _characters, _context_predictor, get_engine

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

    # Scene state (SSL): persists across speak requests, reset on scene_reset
    import torch as _torch
    session_scene_state: _torch.Tensor | None = None
    del _torch

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
    # Receiver task — reads client messages and dispatches
    # ------------------------------------------------------------------
    async def receiver() -> None:
        nonlocal seq_counter, session_character_id, session_speed
        nonlocal session_style_preset, session_situation
        nonlocal session_scene_state
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
                    if msg.get("scene_reset"):
                        session_scene_state = None
                        dialogue_history.clear()
                        logger.info("Scene state reset by client")

                else:
                    await _send(WSError(detail=f"Unknown message type: {msg_type}").model_dump())

        except WebSocketDisconnect:
            pass

    # ------------------------------------------------------------------
    # Consumer task — pulls from queue, synthesizes sentences, streams
    # ------------------------------------------------------------------
    async def consumer() -> None:
        nonlocal is_speaking, session_scene_state
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
                from tmrvc_core.text_utils import analyze_inline_stage_directions
                inline_stage = analyze_inline_stage_directions(
                    item.text, language=character.language,
                )
                spoken_text = inline_stage.spoken_text

                history_snapshot = dialogue_history[-_DIALOGUE_HISTORY_MAX:]
                style = await _predict_style_from_inputs(
                    character=character,
                    text=spoken_text,
                    emotion=item.emotion,
                    history=history_snapshot,
                    situation=item.situation,
                    hint=item.hint,
                    speaker=item.character_id,
                    context_predictor=_context_predictor,
                )
                style, preset_cfg = _resolve_style_preset(style, item.style_preset)
                style = _apply_inline_stage_overlay(style, inline_stage.style_overlay)
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
                    text=spoken_text,
                    emotion=style_msg.emotion,
                ))
                if len(dialogue_history) > _DIALOGUE_HISTORY_MAX:
                    dialogue_history.pop(0)

                # Per-request speed overrides session speed
                base_speed = item.speed if item.speed is not None else session_speed
                effective_speed = _resolve_stage_speed(
                    _resolve_effective_speed(base_speed, preset_cfg),
                    inline_stage.speed_scale,
                )
                effective_sentence_pause_ms = _resolve_sentence_pause(
                    preset_cfg.sentence_pause_ms,
                    inline_stage.sentence_pause_ms_delta,
                )

                # Producer-consumer bridge: synthesize_sentences runs in
                # a thread and pushes chunks into a stdlib queue; the async
                # consumer loop pulls them out.
                sync_q: stdlib_queue.Queue[np.ndarray | None] = stdlib_queue.Queue(maxsize=4)

                def _produce() -> None:
                    try:
                        for chunk in _iter_silence_chunks(inline_stage.leading_silence_ms, 100):
                            while True:
                                if cancel_event.is_set():
                                    sync_q.put(None)
                                    return
                                try:
                                    sync_q.put(chunk, timeout=0.05)
                                    break
                                except stdlib_queue.Full:
                                    continue

                        for chunk in engine.synthesize_sentences(
                            text=spoken_text,
                            language=character.language,
                            spk_embed=spk_embed,
                            style=style,
                            speed=effective_speed,
                            cancel=cancel_event,
                            sentence_pause_ms=effective_sentence_pause_ms,
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

                        for chunk in _iter_silence_chunks(inline_stage.trailing_silence_ms, 100):
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
                                    analyze_inline_stage_directions(
                                        ni.text, language=nc.language,
                                    ).spoken_text,
                                    nc.language,
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
                            # Sentinel — mark last sent chunk as is_last
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

                # Update scene state after successful synthesis
                if engine.scene_state_available and not cancel_event.is_set():
                    try:
                        z_prev = session_scene_state
                        if z_prev is None:
                            z_prev = engine.initial_scene_state()
                        z_t = await loop.run_in_executor(
                            None,
                            lambda: engine.update_scene_state(
                                spoken_text, character.language,
                                spk_embed, z_prev,
                            ),
                        )
                        session_scene_state = z_t
                    except Exception as e:
                        logger.warning("Scene state update failed: %s", e)

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
