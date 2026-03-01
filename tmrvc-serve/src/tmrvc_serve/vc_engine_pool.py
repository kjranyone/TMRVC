"""Production-ready UCLM VC Engine with full State Contract (onnx-contract.md)."""

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

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_serve.uclm_engine import UCLMEngine, EngineState

logger = logging.getLogger(__name__)

# 10ms frame @ 24kHz
FRAME_SIZE = 240


@dataclass
class SessionState:
    """Isolated state for a single UCLM VC session."""
    session_id: str
    spk_embed: np.ndarray
    # Mirroring the full EngineState from uclm_engine
    engine_state: EngineState = field(default_factory=EngineState)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity = time.time()


class ModelPool:
    """Thread-safe UCLM model pool."""
    def __init__(self, checkpoint: Path, device: str = "cuda", max_instances: int = 2):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.max_instances = max_instances
        self._lock = threading.Lock()
        self._pool: list[UCLMEngine] = []
        self._available = threading.Semaphore(max_instances)
        self._loaded = False

    def load(self):
        if self._loaded: return
        # Pre-load could happen here, but we'll lazy-load instances
        self._loaded = True

    def acquire(self, timeout: float = 5.0) -> tuple[UCLMEngine, callable] | None:
        if not self._available.acquire(timeout=timeout): return None
        with self._lock:
            if self._pool:
                engine = self._pool.pop()
            else:
                engine = UCLMEngine(device=self.device)
                engine.load_from_combined_checkpoint(self.checkpoint)
        
        def release():
            with self._lock: self._pool.append(engine)
            self._available.release()
        return engine, release


class VCEnginePool:
    """Orchestrates multiple VC sessions with strict state isolation."""

    def __init__(
        self,
        uclm_checkpoint: Path | str,
        max_concurrent_sessions: int = 20,
        device: str = "cuda",
    ):
        self.checkpoint = Path(uclm_checkpoint)
        self.max_concurrent_sessions = max_concurrent_sessions
        self._sessions: dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()
        self._model_pool = ModelPool(self.checkpoint, device=device)

    def load_models(self):
        self._model_pool.load()

    async def create_session(self, session_id: str, spk_embed: np.ndarray) -> SessionState:
        state = SessionState(session_id=session_id, spk_embed=spk_embed.copy())
        with self._sessions_lock:
            self._sessions[session_id] = state
        logger.info("Created UCLM session %s with full state tracking.", session_id)
        return state

    def close_session(self, session_id: str):
        with self._sessions_lock:
            if session_id in self._sessions: del self._sessions[session_id]

    @torch.no_grad()
    def process_frame(
        self,
        session: SessionState,
        audio_frame: np.ndarray,
        style: StyleParams | None = None,
    ) -> np.ndarray:
        """Process 10ms frame using the session's persistent EngineState."""
        acquired = self._model_pool.acquire()
        if not acquired: return audio_frame # Fallback
        
        engine, release = acquired
        try:
            # Inputs to Tensor
            audio_t = torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0).to(engine.device)
            spk_t = torch.from_numpy(session.spk_embed).float().unsqueeze(0).to(engine.device)
            style = style or StyleParams.neutral()

            # Execute unified engine with Contract-compliant state passing
            audio_out_t, new_engine_state = engine.vc_frame(
                audio_frame=audio_t,
                speaker_embed=spk_t,
                style=style,
                state=session.engine_state
            )

            # Update session's state (already on correct device/format if needed)
            session.engine_state = new_engine_state
            session.touch()

            return audio_out_t.cpu().numpy()
        finally:
            release()

    @property
    def is_ready(self) -> bool:
        return self._model_pool._loaded
