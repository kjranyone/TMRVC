"""Production-ready VC Engine with session isolation and connection pooling.

Features:
- Per-session state management (KV-cache isolation)
- Connection pooling with configurable max concurrency
- Request queueing with backpressure
- GPU memory management
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from tmrvc_core.constants import FRAME_SIZE, SAMPLE_RATE

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Isolated state for a single VC session."""

    session_id: str
    spk_embed: np.ndarray | None = None
    codec_encoder_state: torch.Tensor | None = None
    codec_decoder_state: torch.Tensor | None = None
    token_kv_cache: list | None = None
    token_context: list = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_activity = time.time()


class ModelPool:
    """Thread-safe model pool with lazy loading.

    Supports:
    - Multiple model instances (for GPU parallelism)
    - CPU fallback when GPU exhausted
    - Reference counting for cleanup
    """

    def __init__(
        self,
        model_dir: Path,
        max_gpu_instances: int = 2,
        max_cpu_instances: int = 4,
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir)
        self.max_gpu_instances = max_gpu_instances
        self.max_cpu_instances = max_cpu_instances
        self.primary_device = device

        self._lock = threading.Lock()
        self._gpu_pool: list = []
        self._cpu_pool: list = []
        self._gpu_available = threading.Semaphore(max_gpu_instances)
        self._cpu_available = threading.Semaphore(max_cpu_instances)

        self._models_loaded = False

    def load_models(self) -> None:
        """Pre-load models into pool."""
        if self._models_loaded:
            return

        from tmrvc_train.models.streaming_codec import StreamingCodec, CodecConfig
        from tmrvc_train.models.token_model import TokenModel, TokenModelConfig

        # Load codec
        codec_ckpt = self.model_dir / "codec.pt"
        if codec_ckpt.exists():
            ckpt = torch.load(codec_ckpt, map_location="cpu", weights_only=False)
            self._codec_config = CodecConfig(**ckpt.get("config", {}))
            self._codec_state = ckpt["model"]

        # Load token model
        token_ckpt = self.model_dir / "token_model.pt"
        if token_ckpt.exists():
            ckpt = torch.load(token_ckpt, map_location="cpu", weights_only=False)
            self._token_config = TokenModelConfig(**ckpt.get("config", {}))
            self._token_state = ckpt["model"]

        self._models_loaded = True
        logger.info(
            "Models loaded into pool (GPU=%d, CPU=%d)",
            self.max_gpu_instances,
            self.max_cpu_instances,
        )

    def acquire_gpu(self, timeout: float | None = None) -> tuple | None:
        """Acquire a GPU model instance.

        Returns (codec, token_model, release_fn) or None if timeout.
        """
        if not self._gpu_available.acquire(timeout=timeout):
            return None

        with self._lock:
            if self._gpu_pool:
                instance = self._gpu_pool.pop()
            else:
                instance = self._create_instance(self.primary_device)

        def release():
            with self._lock:
                self._gpu_pool.append(instance)
            self._gpu_available.release()

        return (*instance, release)

    def acquire_cpu(self, timeout: float | None = None) -> tuple | None:
        """Acquire a CPU model instance."""
        if not self._cpu_available.acquire(timeout=timeout):
            return None

        with self._lock:
            if self._cpu_pool:
                instance = self._cpu_pool.pop()
            else:
                instance = self._create_instance("cpu")

        def release():
            with self._lock:
                self._cpu_pool.append(instance)
            self._cpu_available.release()

        return (*instance, release)

    def _create_instance(self, device: str) -> tuple:
        """Create a new model instance."""
        from tmrvc_train.models.streaming_codec import StreamingCodec
        from tmrvc_train.models.token_model import TokenModel

        device = torch.device(device)

        codec = StreamingCodec(self._codec_config).to(device)
        codec.load_state_dict(self._codec_state)
        codec.eval()

        token_model = TokenModel(self._token_config).to(device)
        token_model.load_state_dict(self._token_state)
        token_model.eval()

        return (codec, token_model, device)


class VCEnginePool:
    """Production VC engine with session isolation and concurrency control."""

    def __init__(
        self,
        model_dir: Path | str,
        max_concurrent_sessions: int = 10,
        max_gpu_inference: int = 2,
        session_timeout_sec: float = 300.0,
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir)
        self.max_concurrent_sessions = max_concurrent_sessions
        self.session_timeout = session_timeout_sec

        self._session_semaphore = asyncio.Semaphore(max_concurrent_sessions)
        self._sessions: dict[str, SessionState] = {}
        self._sessions_lock = threading.Lock()

        self._model_pool = ModelPool(
            model_dir=self.model_dir,
            max_gpu_instances=max_gpu_inference,
            device=device,
        )

        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_sessions)
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

        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))

    async def create_session(
        self, session_id: str, spk_embed: np.ndarray
    ) -> SessionState:
        """Create a new isolated session.

        Raises:
            asyncio.TimeoutError: If max concurrent sessions reached
        """
        await asyncio.wait_for(self._session_semaphore.acquire(), timeout=30.0)

        state = SessionState(
            session_id=session_id,
            spk_embed=spk_embed.copy(),
        )

        with self._sessions_lock:
            self._sessions[session_id] = state

        logger.info("Created session %s (active: %d)", session_id, len(self._sessions))
        return state

    def close_session(self, session_id: str) -> None:
        """Close and cleanup a session."""
        with self._sessions_lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

        self._session_semaphore.release()
        logger.info("Closed session %s", session_id)

    def get_session(self, session_id: str) -> SessionState | None:
        """Get session state."""
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session:
                session.touch()
            return session

    @torch.no_grad()
    def process_frame(
        self,
        session: SessionState,
        audio_frame: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 50,
        prefer_gpu: bool = True,
    ) -> np.ndarray:
        """Process a single audio frame.

        Uses session's isolated state. Acquires model from pool.
        """
        # Try GPU first, fallback to CPU
        if prefer_gpu:
            acquired = self._model_pool.acquire_gpu(timeout=0.1)
        else:
            acquired = None

        if acquired is None:
            acquired = self._model_pool.acquire_cpu(timeout=1.0)
            if acquired is None:
                logger.warning("No model available, returning passthrough")
                return audio_frame

        codec, token_model, device, release = acquired

        try:
            audio_t = torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0)
            audio_t = audio_t.to(device)

            spk_t = torch.from_numpy(session.spk_embed).float().unsqueeze(0).to(device)

            # Encode with session state
            indices, _, enc_state = codec.encode(audio_t, session.codec_encoder_state)
            session.codec_encoder_state = enc_state

            # Token context
            session.token_context.append(indices[:, :, -1].cpu())
            if len(session.token_context) > 10:
                session.token_context.pop(0)

            # Token model
            if len(session.token_context) >= 10:
                context = torch.cat(session.token_context[-10:], dim=-1)
                next_tokens, kv = token_model.generate_next_tokens(
                    context.unsqueeze(0),
                    spk_t,
                    session.token_kv_cache,
                    temperature=temperature,
                    top_k=top_k,
                )
                session.token_kv_cache = kv
            else:
                next_tokens = indices[:, :, -1]

            # Decode with session state
            audio_out, dec_state = codec.decode(
                next_tokens, session.codec_decoder_state
            )
            session.codec_decoder_state = dec_state

            return audio_out.squeeze().cpu().numpy()

        finally:
            release()

    def process_stream(
        self,
        session: SessionState,
        audio_generator: Generator[np.ndarray, None, None],
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        """Process audio stream with session state."""
        buffer = np.array([], dtype=np.float32)

        for chunk in audio_generator:
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= FRAME_SIZE:
                frame = buffer[:FRAME_SIZE]
                buffer = buffer[FRAME_SIZE:]

                output = self.process_frame(session, frame, temperature, top_k)
                yield output

    @property
    def active_sessions(self) -> int:
        return len(self._sessions)

    @property
    def is_ready(self) -> bool:
        return self._model_pool._models_loaded
