"""Python streaming voice-conversion engine using UCLM v2.

This module provides a QThread-based audio engine using UCLMEngine
for real-time unified TTS/VC processing in the GUI.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PySide6.QtCore import QThread, Signal

from tmrvc_core.constants import SAMPLE_RATE
from tmrvc_core.dialogue_types import StyleParams
from tmrvc_serve.uclm_engine import UCLMEngine
from tmrvc_gui.workers.ring_buffer import RING_BUFFER_CAPACITY, RingBuffer

# UCLM v2 uses 10ms frames (240 samples @ 24kHz)
FRAME_SIZE = 240


class AudioEngine(QThread):
    """Python streaming voice-conversion engine using UCLM v2.

    Signals
    -------
    level_updated(float, float)
        Emitted periodically with (input_db, output_db) levels.
    timing_updated(float)
        Emitted with inference time in milliseconds per frame.
    buffer_status(bool)
        Emitted when a buffer underrun occurs.
    error(str)
        Emitted on unrecoverable errors.
    """

    level_updated = Signal(float, float)
    timing_updated = Signal(float)
    buffer_status = Signal(bool)
    error = Signal(str)

    def __init__(self, uclm_checkpoint: Path, codec_checkpoint: Path, speaker_path: Path, parent=None) -> None:
        super().__init__(parent)

        self._uclm_checkpoint = Path(uclm_checkpoint)
        self._codec_checkpoint = Path(codec_checkpoint)
        self._speaker_path = Path(speaker_path)

        self._engine: Optional[UCLMEngine] = None
        self._spk_t: Optional[torch.Tensor] = None
        self._kv_cache: Optional[torch.Tensor] = None
        self._style = StyleParams.neutral()

        # Ring buffers
        self._input_ring = RingBuffer(RING_BUFFER_CAPACITY)
        self._output_ring = RingBuffer(RING_BUFFER_CAPACITY)

        # Audio stream (sounddevice)
        self._stream: object | None = None
        self._input_device: int | str | None = None
        self._output_device: int | str | None = None
        self._buffer_size: int = 512

        self._dry_wet: float = 1.0
        self._output_gain: float = 1.0
        self._stopped: bool = True

    def _load_resources(self) -> None:
        """Load UCLM engine and speaker embedding."""
        self._engine = UCLMEngine(
            uclm_checkpoint=self._uclm_checkpoint,
            codec_checkpoint=self._codec_checkpoint,
            device="cpu", # GUI uses CPU for stability
        )
        self._engine.load_models()

        # Load speaker
        if self._speaker_path.suffix == ".tmrvc_speaker":
            from tmrvc_export.speaker_file import read_speaker_file
            spk_np, _, _, _ = read_speaker_file(self._speaker_path)
            self._spk_t = torch.from_numpy(spk_np).float().unsqueeze(0)
        else:
            self._spk_t = torch.from_numpy(np.load(self._speaker_path)).float().unsqueeze(0)

    def start_stream(
        self,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        buffer_size: int = 512,
    ) -> None:
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
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(self, indata, outdata, frames, time_info, status):
        self._input_ring.write(indata[:, 0])
        available = self._output_ring.available()
        if available >= frames:
            outdata[:, 0] = self._output_ring.read(frames)
        else:
            outdata[:] = 0.0
            self.buffer_status.emit(True)

    def run(self) -> None:
        self._stopped = False
        try:
            self._load_resources()
        except Exception as e:
            self.error.emit(f"Failed to load engine: {e}")
            return

        self._kv_cache = None
        
        # Warmup ring buffer
        self._output_ring.write(np.zeros(FRAME_SIZE, dtype=np.float32))

        while not self._stopped:
            if self._input_ring.available() < FRAME_SIZE:
                time.sleep(0.001)
                continue

            t0 = time.perf_counter()
            
            # Read input frame
            in_frame = self._input_ring.read(FRAME_SIZE)
            in_t = torch.from_numpy(in_frame).float().unsqueeze(0).unsqueeze(0)
            
            # Measure input level
            input_db = self._rms_to_db(in_frame)

            # UCLMEngine VC Step
            out_t, next_kv = self._engine.vc_frame(
                audio_frame=in_t,
                speaker_embed=self._spk_t,
                style=self._style,
                kv_cache=self._kv_cache
            )
            self._kv_cache = next_kv
            
            out_frame = out_t.cpu().numpy()
            
            # Dry/Wet and Gain
            mixed = (self._dry_wet * out_frame + (1.0 - self._dry_wet) * in_frame)
            mixed *= self._output_gain
            
            output_db = self._rms_to_db(mixed)
            
            # Write to output
            self._output_ring.write(mixed.astype(np.float32))
            
            # Timing and level
            ms = (time.perf_counter() - t0) * 1000.0
            self.timing_updated.emit(ms)
            self.level_updated.emit(input_db, output_db)

    def set_style(self, style: StyleParams) -> None:
        self._style = style

    def set_dry_wet(self, ratio: float) -> None:
        self._dry_wet = max(0.0, min(1.0, ratio))

    def set_output_gain(self, gain_db: float) -> None:
        self._output_gain = 10.0 ** (gain_db / 20.0)

    def stop(self) -> None:
        self._stopped = True
        self.stop_stream()

    @staticmethod
    def _rms_to_db(signal: np.ndarray) -> float:
        rms = np.sqrt(np.mean(signal ** 2) + 1e-10)
        return max(20.0 * np.log10(rms), -100.0)
