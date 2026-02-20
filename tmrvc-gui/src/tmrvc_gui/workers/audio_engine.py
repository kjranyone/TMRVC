"""Python streaming voice-conversion engine for TMRVC Research Studio.

This module provides a QThread-based audio engine that mirrors the C++
``StreamingEngine`` architecture.  It uses ONNX Runtime for inference,
``sounddevice`` for audio I/O, and implements the same frame-by-frame
causal streaming pipeline described in ``docs/design/streaming-design.md``.

The engine is designed for **research and prototyping** -- the production
path is the C++ VST3 plugin.  Numeric behaviour should match the C++
engine to within the parity tolerances defined in ``onnx-contract.md``.
"""

from __future__ import annotations

import hashlib
import struct
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

# ---------------------------------------------------------------------------
# Project constants (from configs/constants.yaml via tmrvc_core)
# ---------------------------------------------------------------------------

from tmrvc_core.constants import (
    CONTENT_ENCODER_STATE_FRAMES,
    CONVERTER_STATE_FRAMES,
    D_CONTENT,
    D_CONVERTER_HIDDEN,
    D_SPEAKER,
    HOP_LENGTH,
    IR_ESTIMATOR_STATE_FRAMES,
    IR_UPDATE_INTERVAL,
    LORA_DELTA_SIZE,
    N_FFT,
    N_FREQ_BINS,
    N_ACOUSTIC_PARAMS,
    N_IR_PARAMS,
    N_MELS,
    N_VOICE_SOURCE_PARAMS,
    SAMPLE_RATE,
    VOCODER_STATE_FRAMES,
    WINDOW_LENGTH,
)

# Ring buffer capacity (samples)
RING_BUFFER_CAPACITY: int = 4096

# .tmrvc_speaker file constants
SPEAKER_MAGIC: bytes = b"TMSP"
SPEAKER_VERSION: int = 1

# State tensor shapes (batch, channels, context_frames)
STATE_SHAPES: dict[str, tuple[int, int, int]] = {
    "content_encoder": (1, D_CONTENT, CONTENT_ENCODER_STATE_FRAMES),
    "ir_estimator": (1, 128, IR_ESTIMATOR_STATE_FRAMES),
    "converter": (1, D_CONVERTER_HIDDEN, CONVERTER_STATE_FRAMES),
    "vocoder": (1, N_FREQ_BINS, VOCODER_STATE_FRAMES),
}


# ---------------------------------------------------------------------------
# RingBuffer -- lock-free SPSC ring buffer backed by a numpy array
# ---------------------------------------------------------------------------


class RingBuffer:
    """Lock-free single-producer single-consumer ring buffer.

    Backed by a contiguous :class:`numpy.ndarray` of ``float32`` values.
    The read and write positions advance monotonically and wrap using
    modular arithmetic.

    Parameters
    ----------
    capacity : int
        Maximum number of samples the buffer can hold.  Must be a
        power of two for efficient modular wrapping.

    Notes
    -----
    This Python implementation is **not** truly lock-free in the C++
    sense (the GIL serialises access), but it follows the same API
    contract as the C++ ``FixedRingBuffer`` to keep the code portable.
    """

    def __init__(self, capacity: int = RING_BUFFER_CAPACITY) -> None:
        # Round up to the next power of two for efficient masking
        self._capacity: int = 1 << (capacity - 1).bit_length()
        self._mask: int = self._capacity - 1
        self._buffer: np.ndarray = np.zeros(self._capacity, dtype=np.float32)
        self._read_pos: int = 0
        self._write_pos: int = 0

    @property
    def capacity(self) -> int:
        """Return the usable capacity (always one less than allocation)."""
        return self._capacity

    def available(self) -> int:
        """Return the number of samples available for reading."""
        return self._write_pos - self._read_pos

    def free_space(self) -> int:
        """Return the number of samples that can be written."""
        return self._capacity - self.available()

    def write(self, data: np.ndarray) -> int:
        """Write *data* into the buffer.

        Parameters
        ----------
        data : np.ndarray
            1-D ``float32`` array of samples to write.

        Returns
        -------
        int
            Number of samples actually written (may be less than
            ``len(data)`` if the buffer is nearly full).
        """
        n = min(len(data), self.free_space())
        if n == 0:
            return 0

        start = self._write_pos & self._mask
        end = start + n

        if end <= self._capacity:
            self._buffer[start:end] = data[:n]
        else:
            first = self._capacity - start
            self._buffer[start:] = data[:first]
            self._buffer[: n - first] = data[first:n]

        self._write_pos += n
        return n

    def read(self, count: int) -> np.ndarray:
        """Read and consume up to *count* samples from the buffer.

        Parameters
        ----------
        count : int
            Maximum number of samples to read.

        Returns
        -------
        np.ndarray
            1-D ``float32`` array of consumed samples.  May be shorter
            than *count* if fewer samples are available.
        """
        n = min(count, self.available())
        if n == 0:
            return np.empty(0, dtype=np.float32)

        start = self._read_pos & self._mask
        end = start + n

        if end <= self._capacity:
            out = self._buffer[start:end].copy()
        else:
            first = self._capacity - start
            out = np.empty(n, dtype=np.float32)
            out[:first] = self._buffer[start:]
            out[first:] = self._buffer[: n - first]

        self._read_pos += n
        return out

    def peek(self, count: int, offset: int = 0) -> np.ndarray:
        """Read *count* samples starting at *offset* without consuming.

        Parameters
        ----------
        count : int
            Number of samples to peek.
        offset : int
            Offset from the current read position.

        Returns
        -------
        np.ndarray
            1-D ``float32`` array.
        """
        avail = self.available()
        if offset + count > avail:
            count = max(0, avail - offset)
        if count == 0:
            return np.empty(0, dtype=np.float32)

        start = (self._read_pos + offset) & self._mask
        end = start + count

        if end <= self._capacity:
            return self._buffer[start:end].copy()
        else:
            first = self._capacity - start
            out = np.empty(count, dtype=np.float32)
            out[:first] = self._buffer[start:]
            out[first:] = self._buffer[: count - first]
            return out

    def reset(self) -> None:
        """Reset the buffer to empty, zeroing all data."""
        self._read_pos = 0
        self._write_pos = 0
        self._buffer[:] = 0.0


# ---------------------------------------------------------------------------
# PingPongState -- double-buffered state tensor management
# ---------------------------------------------------------------------------


class PingPongState:
    """Double-buffered (ping-pong) state tensor pair.

    Maintains two numpy arrays of identical shape and alternates
    which one is used as model input vs. output on each frame,
    avoiding in-place overwrites.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of each state buffer (e.g. ``(1, 256, 28)``).
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.buffer_a: np.ndarray = np.zeros(shape, dtype=np.float32)
        self.buffer_b: np.ndarray = np.zeros(shape, dtype=np.float32)
        self._current: int = 0  # 0 => A is input, 1 => B is input

    @property
    def input(self) -> np.ndarray:
        """Return the current input state buffer."""
        return self.buffer_a if self._current == 0 else self.buffer_b

    @property
    def output(self) -> np.ndarray:
        """Return the current output state buffer."""
        return self.buffer_b if self._current == 0 else self.buffer_a

    def swap(self) -> None:
        """Swap input and output roles (toggle ping-pong)."""
        self._current ^= 1

    def reset(self) -> None:
        """Zero-initialise both buffers (silence state)."""
        self.buffer_a[:] = 0.0
        self.buffer_b[:] = 0.0
        self._current = 0


# ---------------------------------------------------------------------------
# AudioEngine -- QThread-based streaming VC engine
# ---------------------------------------------------------------------------


class AudioEngine(QThread):
    """Python streaming voice-conversion engine.

    Mirrors the C++ ``StreamingEngine`` architecture: reads audio from
    an input device, processes it frame-by-frame through 4 ONNX models
    with causal state management, and writes the converted audio to an
    output device.

    Parameters
    ----------
    onnx_dir : Path
        Directory containing the 4 streaming ONNX model files
        (``content_encoder.onnx``, ``ir_estimator.onnx``,
        ``converter.onnx``, ``vocoder.onnx``).
    speaker_path : Path
        Path to a ``.tmrvc_speaker`` binary file containing the
        speaker embedding and LoRA delta.

    Signals
    -------
    level_updated(float, float)
        Emitted periodically with ``(input_db, output_db)`` levels.
    timing_updated(float)
        Emitted with inference time in milliseconds per frame.
    buffer_status(bool)
        Emitted when a buffer underrun occurs (*True*) or recovers
        (*False*).
    error(str)
        Emitted on unrecoverable errors.
    """

    level_updated = Signal(float, float)
    timing_updated = Signal(float)
    buffer_status = Signal(bool)
    error = Signal(str)

    def __init__(self, onnx_dir: Path, speaker_path: Path, parent=None) -> None:
        super().__init__(parent)

        self._onnx_dir = Path(onnx_dir)
        self._speaker_path = Path(speaker_path)

        # ---- ONNX sessions (loaded lazily) ----
        self._sessions: dict[str, object] = {}

        # ---- Speaker data ----
        self._spk_embed: np.ndarray = np.zeros((1, D_SPEAKER), dtype=np.float32)
        self._lora_delta: np.ndarray = np.zeros((1, LORA_DELTA_SIZE), dtype=np.float32)

        # ---- State tensors (ping-pong) ----
        self._states: dict[str, PingPongState] = {
            name: PingPongState(shape)
            for name, shape in STATE_SHAPES.items()
        }

        # ---- Ring buffers ----
        self._input_ring = RingBuffer(RING_BUFFER_CAPACITY)
        self._output_ring = RingBuffer(RING_BUFFER_CAPACITY)

        # ---- STFT context buffer (past window_length samples) ----
        self._context_buffer: np.ndarray = np.zeros(WINDOW_LENGTH, dtype=np.float32)

        # ---- Hann window (pre-computed) ----
        self._hann_window: np.ndarray = np.hanning(WINDOW_LENGTH).astype(np.float32)

        # ---- Mel filterbank (pre-computed) ----
        self._mel_filterbank: np.ndarray = self._build_mel_filterbank()

        # ---- IR state ----
        self._acoustic_params: np.ndarray = np.zeros((1, N_ACOUSTIC_PARAMS), dtype=np.float32)
        self._mel_accumulator: list[np.ndarray] = []
        self._frame_counter: int = 0

        # ---- Overlap-add buffer ----
        self._ola_buffer: np.ndarray = np.zeros(WINDOW_LENGTH, dtype=np.float32)

        # ---- Audio stream (sounddevice) ----
        self._stream: object | None = None
        self._input_device: int | str | None = None
        self._output_device: int | str | None = None
        self._buffer_size: int = 512

        # ---- Control parameters ----
        self._dry_wet: float = 1.0   # 0.0 = fully dry, 1.0 = fully wet
        self._output_gain: float = 1.0  # linear gain
        self._stopped: bool = True

        # ---- Voice source preset blending ----
        self._voice_source_preset: Optional[np.ndarray] = None  # [8] float32
        self._voice_source_alpha: float = 0.0  # 0=estimated, 1=full preset

    # ==================================================================
    # Model loading
    # ==================================================================

    def _load_models(self) -> None:
        """Load the 4 streaming ONNX models via ``onnxruntime``.

        Models loaded:
        - ``content_encoder.onnx`` -- per-frame (10 ms)
        - ``ir_estimator.onnx``    -- every 10 frames (~100 ms)
        - ``converter.onnx``       -- per-frame (10 ms)
        - ``vocoder.onnx``         -- per-frame (10 ms)

        The ``speaker_encoder`` is **not** loaded here; it is used
        only during offline enrollment.

        Raises
        ------
        FileNotFoundError
            If any required ONNX file is missing.
        RuntimeError
            If ``onnxruntime`` fails to create a session.
        """
        import onnxruntime as ort

        model_names = ["content_encoder", "ir_estimator", "converter", "vocoder"]

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 2
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        for name in model_names:
            path = self._onnx_dir / f"{name}.onnx"
            if not path.exists():
                raise FileNotFoundError(f"ONNX model not found: {path}")

            self._sessions[name] = ort.InferenceSession(
                str(path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

    # ==================================================================
    # Speaker loading
    # ==================================================================

    def _load_speaker(self) -> None:
        """Read a ``.tmrvc_speaker`` binary file.

        File layout (from ``onnx-contract.md`` section 6):

        ========  =========  ====================================
        Offset    Size       Field
        ========  =========  ====================================
        0x0000    4 bytes    Magic: ``"TMSP"`` (``0x544D5350``)
        0x0004    4 bytes    Version: ``uint32_le = 1``
        0x0008    4 bytes    ``spk_embed_size``: ``uint32_le = 192``
        0x000C    4 bytes    ``lora_delta_size``: ``uint32_le = 24576``
        0x0010    768 B      ``spk_embed``: ``float32[192]``
        0x0310    98304 B    ``lora_delta``: ``float32[24576]``
        0x18310   32 bytes   SHA-256 checksum of all preceding bytes
        ========  =========  ====================================

        Raises
        ------
        FileNotFoundError
            If the speaker file does not exist.
        ValueError
            If the magic, version, sizes, or checksum are invalid.
        """
        path = self._speaker_path
        if not path.exists():
            raise FileNotFoundError(f"Speaker file not found: {path}")

        data = path.read_bytes()

        # --- Header ---
        header_size = 16  # magic(4) + version(4) + embed_size(4) + delta_size(4)
        if len(data) < header_size:
            raise ValueError("Speaker file too short for header")

        magic = data[0:4]
        if magic != SPEAKER_MAGIC:
            raise ValueError(
                f"Invalid magic: expected {SPEAKER_MAGIC!r}, got {magic!r}"
            )

        version = struct.unpack("<I", data[4:8])[0]
        if version != SPEAKER_VERSION:
            raise ValueError(
                f"Unsupported speaker file version: {version}"
            )

        spk_embed_size = struct.unpack("<I", data[8:12])[0]
        lora_delta_size = struct.unpack("<I", data[12:16])[0]

        if spk_embed_size != D_SPEAKER:
            raise ValueError(
                f"spk_embed_size mismatch: expected {D_SPEAKER}, got {spk_embed_size}"
            )

        if lora_delta_size != LORA_DELTA_SIZE:
            raise ValueError(
                f"lora_delta_size mismatch: expected {LORA_DELTA_SIZE}, "
                f"got {lora_delta_size}"
            )

        # --- Payload sizes ---
        spk_bytes = spk_embed_size * 4   # float32
        lora_bytes = lora_delta_size * 4  # float32
        checksum_size = 32  # SHA-256
        expected_total = header_size + spk_bytes + lora_bytes + checksum_size

        if len(data) < expected_total:
            raise ValueError(
                f"Speaker file too short: expected {expected_total} bytes, "
                f"got {len(data)}"
            )

        # --- Checksum verification ---
        payload = data[: header_size + spk_bytes + lora_bytes]
        stored_checksum = data[
            header_size + spk_bytes + lora_bytes
            : header_size + spk_bytes + lora_bytes + checksum_size
        ]
        computed_checksum = hashlib.sha256(payload).digest()

        if stored_checksum != computed_checksum:
            raise ValueError("SHA-256 checksum mismatch in speaker file")

        # --- Extract arrays ---
        spk_start = header_size
        spk_end = spk_start + spk_bytes
        self._spk_embed = np.frombuffer(
            data[spk_start:spk_end], dtype=np.float32
        ).reshape(1, D_SPEAKER).copy()

        lora_start = spk_end
        lora_end = lora_start + lora_bytes
        self._lora_delta = np.frombuffer(
            data[lora_start:lora_end], dtype=np.float32
        ).reshape(1, LORA_DELTA_SIZE).copy()

    # ==================================================================
    # DSP helpers
    # ==================================================================

    @staticmethod
    def _build_mel_filterbank() -> np.ndarray:
        """Build a mel filterbank matrix using tmrvc_core's reference implementation.

        Returns a ``(N_MELS, N_FREQ_BINS)`` matrix identical to the one
        used in training and C++ inference.
        """
        from tmrvc_core.audio import create_mel_filterbank

        filterbank_torch = create_mel_filterbank()
        return filterbank_torch.numpy()

    def _compute_causal_stft(self, context: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute a single-frame causal STFT.

        Parameters
        ----------
        context : np.ndarray
            The past ``WINDOW_LENGTH`` (960) samples of audio, with the
            current hop at the right edge.

        Returns
        -------
        stft_real : np.ndarray
            Real part of the STFT, shape ``(N_FREQ_BINS,)``.
        stft_imag : np.ndarray
            Imaginary part of the STFT, shape ``(N_FREQ_BINS,)``.
        """
        # Apply Hann window
        windowed = np.zeros(N_FFT, dtype=np.float32)
        windowed[:WINDOW_LENGTH] = context * self._hann_window

        # Real FFT
        spectrum = np.fft.rfft(windowed)
        return spectrum.real.astype(np.float32), spectrum.imag.astype(np.float32)

    def _compute_mel(self, stft_real: np.ndarray, stft_imag: np.ndarray) -> np.ndarray:
        """Compute log-mel spectrogram for one frame.

        Parameters
        ----------
        stft_real : np.ndarray
            Real part, shape ``(N_FREQ_BINS,)``.
        stft_imag : np.ndarray
            Imaginary part, shape ``(N_FREQ_BINS,)``.

        Returns
        -------
        mel_frame : np.ndarray
            Log-mel features, shape ``(1, N_MELS, 1)``.
        """
        power = stft_real ** 2 + stft_imag ** 2
        mel_energy = self._mel_filterbank @ power  # (N_MELS,)
        log_mel = np.log(np.maximum(mel_energy, 1e-10))
        return log_mel.reshape(1, N_MELS, 1).astype(np.float32)

    def _compute_istft_ola(
        self, stft_mag: np.ndarray, stft_phase: np.ndarray
    ) -> np.ndarray:
        """Inverse STFT and overlap-add for one frame.

        Parameters
        ----------
        stft_mag : np.ndarray
            Predicted magnitude, shape ``(1, N_FREQ_BINS, 1)``.
        stft_phase : np.ndarray
            Predicted phase, shape ``(1, N_FREQ_BINS, 1)``.

        Returns
        -------
        hop_samples : np.ndarray
            Reconstructed audio for this hop, shape ``(HOP_LENGTH,)``.
        """
        mag = stft_mag.squeeze()   # (N_FREQ_BINS,)
        phase = stft_phase.squeeze()  # (N_FREQ_BINS,)

        # Reconstruct complex spectrum
        spectrum = mag * np.exp(1j * phase)

        # Inverse FFT
        time_signal = np.fft.irfft(spectrum, n=N_FFT).astype(np.float32)

        # Apply synthesis window (first WINDOW_LENGTH samples)
        windowed = time_signal[:WINDOW_LENGTH] * self._hann_window

        # Overlap-add: extract confirmed hop, shift, accumulate
        hop_output = self._ola_buffer[:HOP_LENGTH].copy()

        # Shift OLA buffer left by hop_length
        self._ola_buffer[:WINDOW_LENGTH - HOP_LENGTH] = (
            self._ola_buffer[HOP_LENGTH:WINDOW_LENGTH]
        )
        self._ola_buffer[WINDOW_LENGTH - HOP_LENGTH:] = 0.0

        # Add new windowed frame
        self._ola_buffer[:WINDOW_LENGTH] += windowed

        return hop_output

    @staticmethod
    def _estimate_f0_causal(context: np.ndarray) -> float:
        """Estimate fundamental frequency from a context buffer using autocorrelation.

        Uses a simple causal autocorrelation method suitable for
        research/prototyping.  Assumes speech F0 in the range 50--500 Hz.

        Parameters
        ----------
        context : np.ndarray
            Audio samples of length ``WINDOW_LENGTH`` (960 at 24 kHz).

        Returns
        -------
        float
            Estimated F0 in Hz, or 0.0 if unvoiced / below threshold.
        """
        f0_min, f0_max = 50.0, 500.0
        lag_max = int(SAMPLE_RATE / f0_min)   # 480
        lag_min = int(SAMPLE_RATE / f0_max)   # 48

        n = len(context)
        if n < lag_max:
            return 0.0

        # Energy-based voicing check
        energy = np.sum(context ** 2) / n
        if energy < 1e-6:
            return 0.0

        # Normalised autocorrelation over the valid lag range
        x = context - np.mean(context)
        autocorr = np.correlate(x, x, mode="full")
        autocorr = autocorr[n - 1:]  # keep non-negative lags only

        if autocorr[0] < 1e-10:
            return 0.0

        autocorr = autocorr / autocorr[0]

        # Find the highest peak in [lag_min, lag_max]
        search = autocorr[lag_min : lag_max + 1]
        if len(search) == 0:
            return 0.0

        peak_idx = int(np.argmax(search))
        peak_val = search[peak_idx]

        # Voicing threshold
        if peak_val < 0.3:
            return 0.0

        best_lag = lag_min + peak_idx
        return float(SAMPLE_RATE / best_lag)

    @staticmethod
    def _rms_to_db(signal: np.ndarray) -> float:
        """Convert an audio buffer to dB RMS level.

        Parameters
        ----------
        signal : np.ndarray
            Audio samples.

        Returns
        -------
        float
            RMS level in dB (minimum -100.0 dB).
        """
        rms = np.sqrt(np.mean(signal ** 2) + 1e-10)
        return max(20.0 * np.log10(rms), -100.0)

    # ==================================================================
    # ONNX inference helpers
    # ==================================================================

    def _run_content_encoder(
        self, mel_frame: np.ndarray, f0: np.ndarray
    ) -> np.ndarray:
        """Run the content encoder for one frame.

        Parameters
        ----------
        mel_frame : np.ndarray
            Shape ``(1, N_MELS, 1)``.
        f0 : np.ndarray
            Shape ``(1, 1, 1)``.

        Returns
        -------
        content : np.ndarray
            Shape ``(1, D_CONTENT, 1)``.
        """
        state = self._states["content_encoder"]
        session = self._sessions.get("content_encoder")

        if session is None:
            return np.zeros((1, D_CONTENT, 1), dtype=np.float32)

        outputs = session.run(
            None,
            {
                "mel_frame": mel_frame,
                "f0": f0,
                "state_in": state.input,
            },
        )
        content = outputs[0]
        np.copyto(state.output, outputs[1])
        state.swap()
        return content

    def _run_ir_estimator(self, mel_chunk: np.ndarray) -> np.ndarray:
        """Run the IR estimator on an accumulated mel chunk.

        Parameters
        ----------
        mel_chunk : np.ndarray
            Shape ``(1, N_MELS, IR_UPDATE_INTERVAL)``.

        Returns
        -------
        acoustic_params : np.ndarray
            Shape ``(1, N_ACOUSTIC_PARAMS)``.
        """
        state = self._states["ir_estimator"]
        session = self._sessions.get("ir_estimator")

        if session is None:
            return np.zeros((1, N_ACOUSTIC_PARAMS), dtype=np.float32)

        outputs = session.run(
            None,
            {
                "mel_chunk": mel_chunk,
                "state_in": state.input,
            },
        )
        acoustic_params = outputs[0]
        np.copyto(state.output, outputs[1])
        state.swap()
        return acoustic_params

    def _run_converter(
        self, content: np.ndarray, acoustic_params: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the converter for one frame.

        Parameters
        ----------
        content : np.ndarray
            Shape ``(1, D_CONTENT, 1)``.
        acoustic_params : np.ndarray or None
            Shape ``(1, N_ACOUSTIC_PARAMS)``.  If *None*, uses
            ``self._acoustic_params``.

        Returns
        -------
        pred_features : np.ndarray
            Shape ``(1, N_FREQ_BINS, 1)``.
        """
        state = self._states["converter"]
        session = self._sessions.get("converter")

        if session is None:
            return np.zeros((1, N_FREQ_BINS, 1), dtype=np.float32)

        if acoustic_params is None:
            acoustic_params = self._acoustic_params

        outputs = session.run(
            None,
            {
                "content": content,
                "spk_embed": self._spk_embed,
                "acoustic_params": acoustic_params,
                "state_in": state.input,
            },
        )
        pred_features = outputs[0]
        np.copyto(state.output, outputs[1])
        state.swap()
        return pred_features

    def _run_vocoder(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run the vocoder for one frame.

        Parameters
        ----------
        features : np.ndarray
            Shape ``(1, N_FREQ_BINS, 1)``.

        Returns
        -------
        stft_mag : np.ndarray
            Shape ``(1, N_FREQ_BINS, 1)``.
        stft_phase : np.ndarray
            Shape ``(1, N_FREQ_BINS, 1)``.
        """
        state = self._states["vocoder"]
        session = self._sessions.get("vocoder")

        if session is None:
            return (
                np.zeros((1, N_FREQ_BINS, 1), dtype=np.float32),
                np.zeros((1, N_FREQ_BINS, 1), dtype=np.float32),
            )

        outputs = session.run(
            None,
            {
                "features": features,
                "state_in": state.input,
            },
        )
        stft_mag = outputs[0]
        stft_phase = outputs[1]
        np.copyto(state.output, outputs[2])
        state.swap()
        return stft_mag, stft_phase

    # ==================================================================
    # Audio stream (sounddevice)
    # ==================================================================

    def start_stream(
        self,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        buffer_size: int = 512,
    ) -> None:
        """Configure and start the audio I/O stream.

        Parameters
        ----------
        input_device : int or str or None
            Input device index or name.  *None* for the system default.
        output_device : int or str or None
            Output device index or name.  *None* for the system default.
        buffer_size : int
            Number of samples per callback buffer (at ``SAMPLE_RATE``).
        """
        self._input_device = input_device
        self._output_device = output_device
        self._buffer_size = buffer_size

        import sounddevice as sd

        self._stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=buffer_size,
            device=(input_device, output_device),
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop_stream(self) -> None:
        """Stop and close the audio I/O stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass  # best-effort cleanup
            self._stream = None

    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Combined input/output callback for ``sounddevice.Stream``.

        Writes input samples to the input ring buffer and reads
        processed samples from the output ring buffer.

        Parameters
        ----------
        indata : np.ndarray
            Input audio from the device, shape ``(frames, 1)``.
        outdata : np.ndarray
            Output buffer to fill, shape ``(frames, 1)``.
        frames : int
            Number of frames in this callback.
        time_info : object
            Timing information from PortAudio.
        status : object
            Stream status flags.
        """
        # Write input samples to ring buffer
        self._input_ring.write(indata[:, 0])

        # Read processed samples from output ring buffer
        available = self._output_ring.available()
        if available >= frames:
            out_samples = self._output_ring.read(frames)
            outdata[:, 0] = out_samples
        else:
            # Buffer underrun: output silence and signal
            outdata[:] = 0.0
            self.buffer_status.emit(True)

    # ==================================================================
    # QThread main loop
    # ==================================================================

    def run(self) -> None:
        """Main processing loop (runs in a dedicated QThread).

        While not stopped:

        1. Wait for ``HOP_LENGTH`` (240) samples in the input ring buffer.
        2. Update the context buffer with the new hop.
        3. Compute causal STFT and log-mel spectrogram.
        4. Run the 4 ONNX models with state ping-pong.
        5. Perform iSTFT + overlap-add.
        6. Apply dry/wet mix and output gain.
        7. Write processed samples to the output ring buffer.
        8. Emit timing and level signals.
        """
        self._stopped = False

        # ---- Load models and speaker ----
        try:
            self._load_models()
        except Exception as exc:
            self.error.emit(f"Failed to load ONNX models: {exc}")
            self._stopped = True

        try:
            self._load_speaker()
        except Exception as exc:
            self.error.emit(f"Failed to load speaker file: {exc}")
            self._stopped = True

        # ---- Reset state ----
        for state in self._states.values():
            state.reset()
        self._acoustic_params[:] = 0.0
        self._mel_accumulator.clear()
        self._frame_counter = 0
        self._ola_buffer[:] = 0.0
        self._context_buffer[:] = 0.0

        # ---- Pre-fill output ring with 1 hop of silence ----
        silence = np.zeros(HOP_LENGTH, dtype=np.float32)
        self._output_ring.write(silence)

        # ---- Main loop ----
        while not self._stopped:
            # Wait for enough input samples
            if self._input_ring.available() < HOP_LENGTH:
                time.sleep(0.001)  # 1 ms poll interval
                continue

            frame_start = time.perf_counter()

            # ---- 1. Read hop from input ring ----
            hop_samples = self._input_ring.read(HOP_LENGTH)

            # Measure input level
            input_db = self._rms_to_db(hop_samples)

            # ---- 2. Update context buffer (shift left, append hop) ----
            self._context_buffer[:WINDOW_LENGTH - HOP_LENGTH] = (
                self._context_buffer[HOP_LENGTH:]
            )
            self._context_buffer[WINDOW_LENGTH - HOP_LENGTH:] = hop_samples

            # ---- 3. Causal STFT + mel ----
            stft_real, stft_imag = self._compute_causal_stft(self._context_buffer)
            mel_frame = self._compute_mel(stft_real, stft_imag)

            # ---- 4a. Content encoder ----
            f0_hz = self._estimate_f0_causal(self._context_buffer)
            f0 = np.array([[[f0_hz]]], dtype=np.float32)  # (1, 1, 1)
            content = self._run_content_encoder(mel_frame, f0)

            # ---- 4b. IR estimator (every IR_UPDATE_INTERVAL frames) ----
            self._mel_accumulator.append(mel_frame)
            self._frame_counter += 1

            if self._frame_counter >= IR_UPDATE_INTERVAL:
                mel_chunk = np.concatenate(
                    self._mel_accumulator[-IR_UPDATE_INTERVAL:], axis=2
                )  # (1, N_MELS, IR_UPDATE_INTERVAL)
                self._acoustic_params = self._run_ir_estimator(mel_chunk)
                self._mel_accumulator.clear()
                self._frame_counter = 0

            # ---- 4c. Converter (with voice source blend) ----
            blended_params = self._blend_voice_source(self._acoustic_params)
            pred_features = self._run_converter(content, blended_params)

            # ---- 4d. Vocoder ----
            stft_mag, stft_phase = self._run_vocoder(pred_features)

            # ---- 5. iSTFT + overlap-add ----
            wet_hop = self._compute_istft_ola(stft_mag, stft_phase)

            # ---- 6. Dry/wet mix + output gain ----
            mixed = (
                self._dry_wet * wet_hop
                + (1.0 - self._dry_wet) * hop_samples
            )
            output_hop = mixed * self._output_gain

            # Measure output level
            output_db = self._rms_to_db(output_hop)

            # ---- 7. Write to output ring buffer ----
            written = self._output_ring.write(output_hop.astype(np.float32))
            if written < HOP_LENGTH:
                self.buffer_status.emit(True)

            # ---- 8. Emit signals ----
            frame_ms = (time.perf_counter() - frame_start) * 1000.0
            self.timing_updated.emit(frame_ms)
            self.level_updated.emit(input_db, output_db)

    # ==================================================================
    # Control methods
    # ==================================================================

    def set_voice_source_preset(self, preset: np.ndarray | None) -> None:
        """Set the voice source preset for blending.

        Parameters
        ----------
        preset : np.ndarray or None
            Voice source preset of shape ``(N_VOICE_SOURCE_PARAMS,)``
            (8 floats), or ``None`` to disable.
        """
        if preset is not None:
            preset = np.asarray(preset, dtype=np.float32).ravel()
            assert preset.shape == (N_VOICE_SOURCE_PARAMS,)
        self._voice_source_preset = preset

    def set_voice_source_alpha(self, alpha: float) -> None:
        """Set the voice source blending strength.

        Parameters
        ----------
        alpha : float
            Blending ratio in ``[0.0, 1.0]``.  ``0.0`` uses the
            estimated values (no blending), ``1.0`` uses the preset fully.
        """
        self._voice_source_alpha = max(0.0, min(1.0, alpha))

    def _blend_voice_source(self, acoustic_params: np.ndarray) -> np.ndarray:
        """Blend estimated voice source params with preset.

        Returns a copy of ``acoustic_params`` with indices 24-31
        blended according to ``_voice_source_alpha``.  The original
        array is not modified.

        Parameters
        ----------
        acoustic_params : np.ndarray
            Shape ``(1, N_ACOUSTIC_PARAMS)``.

        Returns
        -------
        np.ndarray
            Blended copy, same shape.
        """
        if self._voice_source_preset is None or self._voice_source_alpha <= 0.0:
            return acoustic_params
        blended = acoustic_params.copy()
        alpha = self._voice_source_alpha
        blended[0, N_IR_PARAMS:] = (
            (1.0 - alpha) * acoustic_params[0, N_IR_PARAMS:]
            + alpha * self._voice_source_preset
        )
        return blended

    def set_dry_wet(self, ratio: float) -> None:
        """Set the dry/wet mix ratio.

        Parameters
        ----------
        ratio : float
            Mix ratio in ``[0.0, 1.0]``.  ``0.0`` is fully dry
            (passthrough), ``1.0`` is fully wet (converted).
        """
        self._dry_wet = max(0.0, min(1.0, ratio))

    def set_output_gain(self, gain_db: float) -> None:
        """Set the output gain.

        Parameters
        ----------
        gain_db : float
            Gain in decibels.  ``0.0`` dB = unity gain.
        """
        self._output_gain = 10.0 ** (gain_db / 20.0)

    def stop(self) -> None:
        """Request the processing loop to stop."""
        self._stopped = True
        self.stop_stream()

    def reset_states(self) -> None:
        """Zero all model state tensors and clear buffers.

        Call this when switching speakers or after a long pause to
        prevent state artifacts.
        """
        for state in self._states.values():
            state.reset()
        self._acoustic_params[:] = 0.0
        self._mel_accumulator.clear()
        self._frame_counter = 0
        self._ola_buffer[:] = 0.0
        self._context_buffer[:] = 0.0
        self._input_ring.reset()
        self._output_ring.reset()

        # Re-fill output ring with silence
        silence = np.zeros(HOP_LENGTH, dtype=np.float32)
        self._output_ring.write(silence)
