"""Vocal event detection for enriched transcripts.

Detects non-linguistic vocal events that Whisper cannot transcribe:
- breathing (inhale/exhale)
- laughter
- sobbing / crying
- sighing
- voice breaks
- silence / pause

Uses DSP-based heuristics (no external model required) plus optional
emotion classification for acting directive tags.

These events are inserted into enriched transcripts as inline tags:
    [inhale] もう...[voice_break] いや...[sob]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000
HOP_LENGTH = 320  # 10ms at 24kHz (analysis rate, not codec rate)
FRAME_RATE = SAMPLE_RATE / HOP_LENGTH  # 75 Hz


@dataclass
class VocalEvent:
    """A detected vocal event."""
    event_type: str      # inhale, exhale, laugh, sob, sigh, voice_break, pause
    start_sec: float
    end_sec: float
    confidence: float    # 0-1
    tag: str             # inline tag string, e.g., "[laugh]"


@dataclass
class VocalEventReport:
    """All events detected in an utterance."""
    events: list[VocalEvent] = field(default_factory=list)
    emotion_label: str = ""
    emotion_confidence: float = 0.0

    def to_tags_at_positions(self, word_boundaries: list[tuple[float, float, str]] | None = None) -> list[tuple[float, str]]:
        """Convert events to (timestamp, tag) pairs for transcript insertion."""
        return [(e.start_sec, e.tag) for e in self.events if e.confidence > 0.5]


class VocalEventDetector:
    """DSP-based vocal event detector."""

    def __init__(
        self,
        silence_threshold_db: float = -45.0,
        min_pause_sec: float = 0.3,
        breath_band_hz: tuple[float, float] = (100, 1200),
        laugh_min_bursts: int = 5,
        voice_break_drop_db: float = 18.0,
        sob_cv_threshold: float = 1.2,
    ):
        self.silence_threshold = 10 ** (silence_threshold_db / 20)
        self.min_pause_frames = int(min_pause_sec * FRAME_RATE)
        self.breath_band = breath_band_hz
        self.laugh_min_bursts = laugh_min_bursts
        self.voice_break_drop_db = voice_break_drop_db
        self.sob_cv_threshold = sob_cv_threshold

    def detect(self, waveform: np.ndarray, sample_rate: int = SAMPLE_RATE) -> VocalEventReport:
        """Detect vocal events in a waveform.

        Args:
            waveform: [T_samples] mono audio
            sample_rate: sample rate

        Returns:
            VocalEventReport with detected events
        """
        report = VocalEventReport()

        if len(waveform) < sample_rate * 0.1:  # too short
            return report

        # Frame-level features
        hop = int(sample_rate / FRAME_RATE)
        n_frames = len(waveform) // hop

        if n_frames < 5:
            return report

        frames = waveform[:n_frames * hop].reshape(n_frames, hop)
        energy = np.sqrt(np.mean(frames ** 2, axis=1))  # RMS per frame
        energy_db = 20 * np.log10(energy + 1e-10)

        # 1. Pause detection (silence regions)
        is_silent = energy < self.silence_threshold
        report.events.extend(self._detect_pauses(is_silent, n_frames))

        # 2. Breathing detection (low-energy, specific spectral shape)
        report.events.extend(self._detect_breathing(frames, energy, sample_rate))

        # 3. Voice break detection (sudden pitch/energy discontinuity)
        report.events.extend(self._detect_voice_breaks(frames, energy, sample_rate))

        # 4. Laughter detection (periodic energy bursts)
        report.events.extend(self._detect_laughter(energy))

        # 5. Sobbing detection (irregular energy with pitch instability)
        report.events.extend(self._detect_sobbing(frames, energy, sample_rate))

        # Sort by time
        report.events.sort(key=lambda e: e.start_sec)

        return report

    def _detect_pauses(self, is_silent: np.ndarray, n_frames: int) -> list[VocalEvent]:
        events = []
        in_pause = False
        pause_start = 0

        for i in range(n_frames):
            if is_silent[i] and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent[i] and in_pause:
                pause_len = i - pause_start
                if pause_len >= self.min_pause_frames:
                    start_sec = pause_start / FRAME_RATE
                    end_sec = i / FRAME_RATE
                    events.append(VocalEvent(
                        event_type="pause",
                        start_sec=start_sec, end_sec=end_sec,
                        confidence=min(1.0, pause_len / (self.min_pause_frames * 3)),
                        tag="[pause]",
                    ))
                in_pause = False

        return events

    def _detect_breathing(
        self, frames: np.ndarray, energy: np.ndarray, sr: int,
    ) -> list[VocalEvent]:
        """Detect breathing by spectral shape: energy concentrated in low frequencies,
        low periodicity (no pitch), moderate energy."""
        events = []
        n_frames = len(energy)
        hop = frames.shape[1]

        for i in range(n_frames):
            if energy[i] < self.silence_threshold:
                continue
            if energy[i] > np.median(energy) * 0.8:
                continue  # too loud for breath

            # Check spectral concentration in breath band
            spectrum = np.abs(np.fft.rfft(frames[i]))
            freqs = np.fft.rfftfreq(hop, 1.0 / sr)

            breath_mask = (freqs >= self.breath_band[0]) & (freqs <= self.breath_band[1])
            if breath_mask.sum() == 0:
                continue

            breath_energy = np.sum(spectrum[breath_mask] ** 2)
            total_energy = np.sum(spectrum ** 2) + 1e-10
            breath_ratio = breath_energy / total_energy

            # High breath_ratio + low overall energy = likely breath
            if breath_ratio > 0.7 and energy[i] < np.median(energy) * 0.3:
                events.append(VocalEvent(
                    event_type="inhale",
                    start_sec=i / FRAME_RATE,
                    end_sec=(i + 1) / FRAME_RATE,
                    confidence=float(breath_ratio),
                    tag="[inhale]",
                ))

        # Merge adjacent breath frames
        return self._merge_adjacent(events, max_gap_sec=0.1)

    def _detect_voice_breaks(
        self, frames: np.ndarray, energy: np.ndarray, sr: int,
    ) -> list[VocalEvent]:
        """Detect voice breaks: sudden energy drop followed by recovery."""
        events = []
        n = len(energy)

        for i in range(2, n - 2):
            if energy[i] > 0 and energy[i-1] > 0:
                drop_db = 20 * np.log10(energy[i-1] / (energy[i] + 1e-10))
                recovery_db = 20 * np.log10(energy[min(i+2, n-1)] / (energy[i] + 1e-10))

                if drop_db > self.voice_break_drop_db and recovery_db > 10:
                    events.append(VocalEvent(
                        event_type="voice_break",
                        start_sec=i / FRAME_RATE,
                        end_sec=(i + 1) / FRAME_RATE,
                        confidence=min(1.0, drop_db / 20),
                        tag="[voice_break]",
                    ))

        return events

    def _detect_laughter(self, energy: np.ndarray) -> list[VocalEvent]:
        """Detect laughter: periodic energy bursts (3+ peaks in short window)."""
        events = []
        n = len(energy)
        window_frames = int(1.5 * FRAME_RATE)  # 1.5 sec window

        for start in range(0, n - window_frames, window_frames // 2):
            window = energy[start:start + window_frames]
            threshold = np.median(window) * 1.5

            # Count energy peaks
            peaks = 0
            in_peak = False
            for val in window:
                if val > threshold and not in_peak:
                    peaks += 1
                    in_peak = True
                elif val < threshold * 0.7:
                    in_peak = False

            if peaks >= self.laugh_min_bursts:
                events.append(VocalEvent(
                    event_type="laugh",
                    start_sec=start / FRAME_RATE,
                    end_sec=(start + window_frames) / FRAME_RATE,
                    confidence=min(1.0, peaks / 5),
                    tag="[laugh]",
                ))

        return self._merge_adjacent(events, max_gap_sec=0.5)

    def _detect_sobbing(
        self, frames: np.ndarray, energy: np.ndarray, sr: int,
    ) -> list[VocalEvent]:
        """Detect sobbing: irregular energy with high-frequency instability."""
        events = []
        n = len(energy)
        window_frames = int(2.0 * FRAME_RATE)

        for start in range(0, n - window_frames, window_frames // 2):
            window_e = energy[start:start + window_frames]

            # High energy variance (irregular)
            cv = np.std(window_e) / (np.mean(window_e) + 1e-10)

            # Check for pitch instability (zero-crossing rate variance)
            window_audio = frames[start:start + window_frames].flatten()
            zcr_per_frame = []
            frame_size = frames.shape[1]
            for i in range(min(window_frames, len(frames) - start)):
                f = frames[start + i]
                zcr = np.mean(np.abs(np.diff(np.sign(f))) > 0)
                zcr_per_frame.append(zcr)

            zcr_var = np.var(zcr_per_frame) if zcr_per_frame else 0

            # Sobbing: very high CV (irregular) + high ZCR variance
            if cv > self.sob_cv_threshold and zcr_var > 0.01:
                events.append(VocalEvent(
                    event_type="sob",
                    start_sec=start / FRAME_RATE,
                    end_sec=(start + window_frames) / FRAME_RATE,
                    confidence=min(1.0, cv / 1.5),
                    tag="[sob]",
                ))

        return self._merge_adjacent(events, max_gap_sec=0.5)

    @staticmethod
    def _merge_adjacent(events: list[VocalEvent], max_gap_sec: float) -> list[VocalEvent]:
        """Merge events of the same type that are close together."""
        if not events:
            return events

        merged = [events[0]]
        for e in events[1:]:
            prev = merged[-1]
            if e.event_type == prev.event_type and (e.start_sec - prev.end_sec) < max_gap_sec:
                prev.end_sec = e.end_sec
                prev.confidence = max(prev.confidence, e.confidence)
            else:
                merged.append(e)

        return merged


def enrich_transcript_with_events(
    transcript: str,
    events: list[VocalEvent],
    word_timestamps: list[tuple[float, float, str]] | None = None,
) -> str:
    """Insert vocal event tags into a transcript at appropriate positions.

    If word_timestamps are available, tags are inserted at the nearest word boundary.
    Otherwise, tags are prepended/appended based on timing.
    """
    if not events:
        return transcript

    high_conf_events = [e for e in events if e.confidence > 0.5]
    if not high_conf_events:
        return transcript

    if word_timestamps:
        # Insert tags at nearest word boundary
        result_parts = []
        words = list(word_timestamps)
        event_idx = 0

        for w_start, w_end, word in words:
            # Insert any events that occur before this word
            while event_idx < len(high_conf_events) and high_conf_events[event_idx].start_sec < w_start:
                result_parts.append(high_conf_events[event_idx].tag + " ")
                event_idx += 1
            result_parts.append(word)

        # Remaining events after last word
        while event_idx < len(high_conf_events):
            result_parts.append(" " + high_conf_events[event_idx].tag)
            event_idx += 1

        return "".join(result_parts)
    else:
        # Simple: prepend events at start, append at end
        pre_tags = [e.tag for e in high_conf_events if e.start_sec < 0.5]
        post_tags = [e.tag for e in high_conf_events if e.start_sec >= 0.5]

        parts = []
        if pre_tags:
            parts.append(" ".join(pre_tags) + " ")
        parts.append(transcript)
        if post_tags:
            parts.append(" " + " ".join(post_tags))

        return "".join(parts)


class EmotionClassifier:
    """Optional emotion classifier using cached wav2vec2 model."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            model_id = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
            self._processor = AutoFeatureExtractor.from_pretrained(model_id)
            self._model = AutoModelForAudioClassification.from_pretrained(model_id)
            self._model.eval()
            self._model.to(self.device)
            logger.info("Emotion classifier loaded: %s", model_id)
        except Exception as e:
            logger.warning("Emotion classifier not available: %s", e)

    def classify(self, waveform: np.ndarray, sample_rate: int = 24000) -> tuple[str, float]:
        """Classify utterance-level emotion.

        Returns: (emotion_label, confidence)
        """
        self._load()
        if self._model is None:
            return "", 0.0

        try:
            import torch
            import torchaudio.functional as AF

            # Resample to 16kHz if needed
            if sample_rate != 16000:
                waveform_t = torch.from_numpy(waveform).float()
                waveform_16k = AF.resample(waveform_t, sample_rate, 16000).numpy()
            else:
                waveform_16k = waveform

            inputs = self._processor(
                waveform_16k, sampling_rate=16000, return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                idx = probs.argmax().item()
                label = self._model.config.id2label.get(idx, f"class_{idx}")
                confidence = probs[idx].item()

            return label, confidence
        except Exception as e:
            logger.debug("Emotion classification failed: %s", e)
            return "", 0.0


def emotion_to_acting_tag(emotion: str) -> str:
    """Map emotion classifier output to acting directive tag."""
    mapping = {
        "angry": "[angry]",
        "disgust": "[disgusted]",
        "fear": "[fearful]",
        "happy": "[happy]",
        "sad": "[sad]",
        "surprise": "[surprised]",
        "neutral": "",
        "calm": "[calm]",
    }
    return mapping.get(emotion.lower(), "")
