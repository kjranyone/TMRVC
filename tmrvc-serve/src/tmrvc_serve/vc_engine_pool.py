"""Production-ready UCLM VC Engine pool with session isolation."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from tmrvc_core.dialogue_types import StyleParams
from tmrvc_serve.uclm_engine import EngineState, UCLMEngine

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Isolated state for a single UCLM VC session."""

    session_id: str
    spk_embed: np.ndarray
    engine_state: EngineState = field(default_factory=EngineState)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity = time.time()


class ModelPool:
    """Thread-safe UCLM model pool."""

    def __init__(
        self,
        uclm_checkpoint: Path,
        codec_checkpoint: Path,
        device: str = "cuda",
        max_instances: int = 2,
    ):
        self.uclm_checkpoint = Path(uclm_checkpoint)
        self.codec_checkpoint = Path(codec_checkpoint)
        self.device = device
        self.max_instances = max(1, int(max_instances))
        self._lock = threading.Lock()
        self._pool: list[UCLMEngine] = []
        self._available = threading.Semaphore(self.max_instances)
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True

    def acquire(self, timeout: float = 5.0) -> tuple[UCLMEngine, Callable[[], None]] | None:
        if not self._available.acquire(timeout=timeout):
            return None

        with self._lock:
            if self._pool:
                engine = self._pool.pop()
            else:
                engine = UCLMEngine(device=self.device)
                engine.load_models(
                    uclm_path=self.uclm_checkpoint, codec_path=self.codec_checkpoint
                )

        def release() -> None:
            with self._lock:
                self._pool.append(engine)
            self._available.release()

        return engine, release


class VCEnginePool:
    """Orchestrates multiple VC sessions with strict state isolation."""

    def __init__(
        self,
        uclm_checkpoint: Path | str,
        codec_checkpoint: Path | str,
        max_concurrent_sessions: int = 20,
        max_gpu_inference: int = 2,
        session_timeout_sec: float = 300.0,
        device: str = "cuda",
    ):
        self.uclm_checkpoint = Path(uclm_checkpoint)
        self.codec_checkpoint = Path(codec_checkpoint)
        self.max_concurrent_sessions = int(max_concurrent_sessions)
        self.session_timeout_sec = float(session_timeout_sec)

        self._sessions: dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._model_pool = ModelPool(
            self.uclm_checkpoint,
            self.codec_checkpoint,
            device=device,
            max_instances=max_gpu_inference,
        )
        self._cleanup_task: asyncio.Task | None = None

    def load_models(self) -> None:
        self._model_pool.load()

    async def start_cleanup_task(self) -> None:
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        if self._cleanup_task is None:
            return
        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
        self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(5.0)
                now = time.time()
                stale: list[str] = []
                with self._sessions_lock:
                    for sid, sess in self._sessions.items():
                        if now - sess.last_activity > self.session_timeout_sec:
                            stale.append(sid)
                    for sid in stale:
                        del self._sessions[sid]
                for sid in stale:
                    logger.info("Closed stale VC session: %s", sid)
        except asyncio.CancelledError:
            return

    async def create_session(self, session_id: str, spk_embed: np.ndarray) -> SessionState:
        with self._sessions_lock:
            if len(self._sessions) >= self.max_concurrent_sessions:
                raise RuntimeError("Max concurrent VC sessions reached.")
            state = SessionState(session_id=session_id, spk_embed=spk_embed.copy())
            self._sessions[session_id] = state
        logger.info("Created UCLM session %s.", session_id)
        return state

    def close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    @torch.no_grad()
    def process_frame(
        self,
        session: SessionState,
        audio_frame: np.ndarray,
        style: StyleParams | None = None,
    ) -> np.ndarray:
        acquired = self._model_pool.acquire()
        if not acquired:
            return audio_frame

        engine, release = acquired
        try:
            audio_t = (
                torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0).to(engine.device)
            )
            spk_t = torch.from_numpy(session.spk_embed).float().unsqueeze(0).to(engine.device)
            style = style or StyleParams.neutral()

            audio_out_t, new_engine_state = engine.vc_frame(
                audio_frame=audio_t,
                speaker_embed=spk_t,
                style=style,
                state=session.engine_state,
            )

            session.engine_state = new_engine_state
            session.touch()
            return audio_out_t.detach().cpu().numpy()
        finally:
            release()

    @property
    def active_sessions(self) -> int:
        with self._sessions_lock:
            return len(self._sessions)

    @property
    def is_ready(self) -> bool:
        return self._model_pool._loaded
