"""Production-ready UCLM VC Engine with session isolation and connection pooling."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_serve.uclm_engine import UCLMEngine

logger = logging.getLogger(__name__)

# UCLM v2 uses 10ms frames
FRAME_SIZE = 240


@dataclass
class SessionState:
    """Isolated state for a single UCLM VC session."""

    session_id: str
    spk_embed: np.ndarray | None = None
    token_kv_cache: torch.Tensor | None = None
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity = time.time()


class ModelPool:
    """Thread-safe UCLM model pool with lazy loading."""

    def __init__(
        self,
        uclm_checkpoint: Path,
        codec_checkpoint: Path,
        max_gpu_instances: int = 2,
        max_cpu_instances: int = 4,
        device: str = "cuda",
    ):
        self.uclm_checkpoint = Path(uclm_checkpoint)
        self.codec_checkpoint = Path(codec_checkpoint)
        self.max_gpu_instances = max_gpu_instances
        self.max_cpu_instances = max_cpu_instances
        self.primary_device = device

        self._lock = threading.Lock()
        self._gpu_pool: list[UCLMEngine] = []
        self._cpu_pool: list[UCLMEngine] = []
        self._gpu_available = threading.Semaphore(max_gpu_instances)
        self._cpu_available = threading.Semaphore(max_cpu_instances)

        self._models_loaded = False

    def load_models(self) -> None:
        """Load state dicts into memory."""
        if self._models_loaded:
            return

        uclm_ckpt = torch.load(self.uclm_checkpoint, map_location="cpu")
        self._uclm_state = uclm_ckpt["model"]

        codec_ckpt = torch.load(self.codec_checkpoint, map_location="cpu")
        self._codec_state = codec_ckpt["model"]

        self._models_loaded = True
        logger.info(
            "UCLM models loaded into pool (GPU=%d, CPU=%d)",
            self.max_gpu_instances,
            self.max_cpu_instances,
        )

    def acquire_gpu(self, timeout: float | None = None) -> tuple[UCLMEngine, callable] | None:
        """Acquire a GPU UCLMEngine instance."""
        if not self._gpu_available.acquire(timeout=timeout):
            return None

        with self._lock:
            if self._gpu_pool:
                engine = self._gpu_pool.pop()
            else:
                engine = UCLMEngine(device=self.primary_device)
                engine.load_from_state_dicts(self._uclm_state, self._codec_state)

        def release():
            with self._lock:
                self._gpu_pool.append(engine)
            self._gpu_available.release()

        return engine, release

    def acquire_cpu(self, timeout: float | None = None) -> tuple[UCLMEngine, callable] | None:
        """Acquire a CPU UCLMEngine instance."""
        if not self._cpu_available.acquire(timeout=timeout):
            return None

        with self._lock:
            if self._cpu_pool:
                engine = self._cpu_pool.pop()
            else:
                engine = UCLMEngine(device="cpu")
                engine.load_from_state_dicts(self._uclm_state, self._codec_state)

        def release():
            with self._lock:
                self._cpu_pool.append(engine)
            self._cpu_available.release()

        return engine, release


class VCEnginePool:
    """Production UCLM VC engine with session isolation and concurrency control."""

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
        self.max_concurrent_sessions = max_concurrent_sessions
        self.session_timeout = session_timeout_sec

        self._session_semaphore = asyncio.Semaphore(max_concurrent_sessions)
        self._sessions: dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()

        self._model_pool = ModelPool(
            uclm_checkpoint=self.uclm_checkpoint,
            codec_checkpoint=self.codec_checkpoint,
            max_gpu_instances=max_gpu_inference,
            device=device,
        )

        self._cleanup_task: asyncio.Task | None = None

    def load_models(self) -> None:
        """Initialize model pool."""
        self._model_pool.load_models()

    async def start_cleanup_task(self) -> None:
        """Start background session cleanup."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self) -> None:
        """Stop cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired sessions."""
        while True:
            await asyncio.sleep(60.0)
            self._cleanup_expired_sessions()

    def _cleanup_expired_sessions(self) -> None:
        """Remove sessions that have been inactive too long."""
        now = time.time()
        expired = []

        with self._sessions_lock:
            for session_id, state in list(self._sessions.items()):
                if now - state.last_activity > self.session_timeout:
                    expired.append(session_id)

            for session_id in expired:
                del self._sessions[session_id]
                logger.info("Cleaned up expired session: %s", session_id)

    async def create_session(
        self, session_id: str, spk_embed: np.ndarray
    ) -> SessionState:
        """Create a new isolated session."""
        await asyncio.wait_for(self._session_semaphore.acquire(), timeout=30.0)

        state = SessionState(
            session_id=session_id,
            spk_embed=spk_embed.copy(),
        )

        with self._sessions_lock:
            self._sessions[session_id] = state

        logger.info("Created UCLM session %s", session_id)
        return state

    def close_session(self, session_id: str) -> None:
        """Close and cleanup a session."""
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

        self._session_semaphore.release()

    @torch.no_grad()
    def process_frame(
        self,
        session: SessionState,
        audio_frame: np.ndarray,
        style: StyleParams | None = None,
        prefer_gpu: bool = True,
    ) -> np.ndarray:
        """Process a single 10ms frame using UCLM dual-stream engine."""
        if style is None:
            style = StyleParams.neutral()

        # Acquire model
        if prefer_gpu:
            acquired = self._model_pool.acquire_gpu(timeout=0.1)
        else:
            acquired = None

        if acquired is None:
            acquired = self._model_pool.acquire_cpu(timeout=1.0)
            if acquired is None:
                return audio_frame # Passthrough fallback

        engine, release = acquired

        try:
            # Prep inputs
            audio_t = torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0)
            audio_t = audio_t.to(engine.device)
            spk_t = torch.from_numpy(session.spk_embed).float().unsqueeze(0).to(engine.device)

            # Move kv_cache to device if it exists
            kv = session.token_kv_cache
            if kv is not None:
                kv = kv.to(engine.device)

            # Process
            audio_out, new_kv = engine.vc_frame(
                audio_t,
                spk_t,
                style,
                kv_cache=kv
            )

            # Save state (back to CPU for isolation/pooling flexibility)
            session.token_kv_cache = new_kv.cpu()
            session.touch()

            return audio_out.cpu().numpy()

        finally:
            release()

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    @property
    def is_ready(self) -> bool:
        return self._model_pool._models_loaded
