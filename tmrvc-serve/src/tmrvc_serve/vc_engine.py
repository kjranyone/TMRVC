"""Codec-Latent VC Engine for FastAPI streaming."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from pathlib import Path
from typing import Generator

import numpy as np
import torch

from tmrvc_core.constants import FRAME_SIZE, SAMPLE_RATE

logger = logging.getLogger(__name__)


class CodecLatentVCEngine:
    """Real-time Voice Conversion using Codec-Latent paradigm.

    Pipeline:
        audio_in (24kHz) → codec_encoder → tokens → token_model → codec_decoder → audio_out
    """

    def __init__(
        self,
        model_dir: Path | str,
        device: str = "cpu",
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self._models_loaded = False

        self.codec = None
        self.token_model = None

    def load_models(self) -> None:
        """Load ONNX models via Rust engine or PyTorch."""
        from tmrvc_train.models.streaming_codec import StreamingCodec, CodecConfig
        from tmrvc_train.models.token_model import TokenModel, TokenModelConfig

        codec_ckpt = self.model_dir / "codec.pt"
        token_ckpt = self.model_dir / "token_model.pt"

        if codec_ckpt.exists():
            ckpt = torch.load(codec_ckpt, map_location=self.device, weights_only=False)
            self.codec = StreamingCodec(CodecConfig())
            self.codec.load_state_dict(ckpt["model"])
            self.codec.eval()
            logger.info("Loaded codec from %s", codec_ckpt)

        if token_ckpt.exists():
            ckpt = torch.load(token_ckpt, map_location=self.device, weights_only=False)
            self.token_model = TokenModel(TokenModelConfig())
            self.token_model.load_state_dict(ckpt["model"])
            self.token_model.eval()
            logger.info("Loaded token_model from %s", token_ckpt)

        self._models_loaded = self.codec is not None and self.token_model is not None

    def is_ready(self) -> bool:
        return self._models_loaded

    @torch.no_grad()
    def process_frame(
        self,
        audio_frame: np.ndarray,
        spk_embed: np.ndarray,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> np.ndarray:
        """Process 20ms audio frame.

        Args:
            audio_frame: [480] float32 samples @ 24kHz
            spk_embed: [192] speaker embedding
            temperature: Sampling temperature
            top_k: Top-k sampling

        Returns:
            [480] float32 output audio
        """
        if not self._models_loaded:
            return audio_frame

        audio_t = torch.from_numpy(audio_frame).float().unsqueeze(0).unsqueeze(0)
        audio_t = audio_t.to(self.device)

        spk_t = torch.from_numpy(spk_embed).float().unsqueeze(0).to(self.device)

        # Encode: audio → tokens
        indices, _, _ = self.codec.encode(audio_t, None)

        # Token model: context → next tokens
        context_tokens = indices.unsqueeze(0)  # [1, 4, T]
        next_tokens, _ = self.token_model.generate_next_tokens(
            context_tokens, spk_t, None, temperature=temperature, top_k=top_k
        )

        # Decode: tokens → audio
        audio_out, _ = self.codec.decode(next_tokens, None)

        return audio_out.squeeze().cpu().numpy()

    def process_stream(
        self,
        audio_generator: Generator[np.ndarray, None, None],
        spk_embed: np.ndarray,
        chunk_size: int = 480,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        """Stream processing: yield output chunks.

        Args:
            audio_generator: Yields [N] audio chunks
            spk_embed: [192] speaker embedding
            chunk_size: Output chunk size in samples
            temperature: Sampling temperature
            top_k: Top-k sampling

        Yields:
            [chunk_size] output audio chunks
        """
        buffer = np.array([], dtype=np.float32)

        for chunk in audio_generator:
            buffer = np.concatenate([buffer, chunk])

            while len(buffer) >= FRAME_SIZE:
                frame = buffer[:FRAME_SIZE]
                buffer = buffer[FRAME_SIZE:]

                output = self.process_frame(frame, spk_embed, temperature, top_k)
                yield output


class AsyncVCProcessor:
    """Async wrapper for streaming VC with queue-based buffering."""

    def __init__(self, engine: CodecLatentVCEngine, spk_embed: np.ndarray):
        self.engine = engine
        self.spk_embed = spk_embed
        self.input_queue: queue.Queue[np.ndarray | None] = queue.Queue()
        self.output_queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self.input_queue.put(None)
        if self._thread:
            self._thread.join(timeout=1.0)

    def push_audio(self, audio: np.ndarray) -> None:
        """Push audio chunk to input queue."""
        self.input_queue.put(audio)

    async def get_audio(self) -> np.ndarray | None:
        """Get processed audio from output queue."""
        try:
            return await asyncio.wait_for(self.output_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None

    def _process_loop(self) -> None:
        buffer = np.array([], dtype=np.float32)

        while self._running:
            try:
                chunk = self.input_queue.get(timeout=0.01)
                if chunk is None:
                    break

                buffer = np.concatenate([buffer, chunk])

                while len(buffer) >= FRAME_SIZE:
                    frame = buffer[:FRAME_SIZE]
                    buffer = buffer[FRAME_SIZE:]

                    output = self.engine.process_frame(frame, self.spk_embed)

                    # Put in async queue (thread-safe)
                    asyncio.run_coroutine_threadsafe(
                        self.output_queue.put(output), asyncio.get_event_loop()
                    )

            except queue.Empty:
                continue
