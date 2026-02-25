"""Emotion dataset parsers for expressive TTS training.

Parses emotion labels from various corpora into a unified format
for StyleEncoder training (Phase 3).

Supported datasets:
- **Expresso** (Meta): 26 styles → 12-category mapping
- **JVNV**: 6 basic emotions → 12-category mapping
- **EmoV-DB**: 5 emotions → 12-category mapping
- **RAVDESS**: 8 emotions + intensity → 12-category mapping + VAD

Each parser returns a list of :class:`EmotionEntry` with unified labels.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from tmrvc_core.dialogue_types import EMOTION_CATEGORIES, EMOTION_TO_ID

logger = logging.getLogger(__name__)


@dataclass
class EmotionEntry:
    """A single utterance with emotion annotation."""

    utterance_id: str
    speaker_id: str
    audio_path: Path
    text: str
    emotion: str
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    dataset: str = ""

    def validate(self) -> bool:
        return self.emotion in EMOTION_CATEGORIES


# --- Expresso (Meta) ---
# 26 styles: read/default, read/happy, read/sad, read/angry, etc.
# + improvised styles

EXPRESSO_STYLE_MAP: dict[str, str] = {
    "default": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "confused": "surprised",
    "enunciated": "neutral",
    "laughing": "happy",
    "whisper": "whisper",
    # Improvised styles
    "narration-default": "neutral",
    "narration-happy": "happy",
    "narration-sad": "sad",
    "narration-angry": "angry",
    "narration-confused": "surprised",
    "narration-enunciated": "neutral",
    "narration-laughing": "happy",
    "narration-whisper": "whisper",
}


def parse_expresso(data_dir: str | Path) -> list[EmotionEntry]:
    """Parse Expresso dataset emotion labels.

    Expected structure::

        data_dir/
        ├── read/
        │   ├── ex01_default_00001.wav
        │   ├── ex01_happy_00001.wav
        │   └── ...
        ├── improvised/
        │   └── ...
        └── metadata.csv  (optional)

    Args:
        data_dir: Path to Expresso dataset root.

    Returns:
        List of EmotionEntry.
    """
    data_dir = Path(data_dir)
    entries: list[EmotionEntry] = []

    # Expresso file naming: {speaker}_{style}_{id}.wav
    pattern = re.compile(r"^(ex\d+)_([a-z_-]+)_(\d+)\.wav$")

    for wav_file in sorted(data_dir.rglob("*.wav")):
        match = pattern.match(wav_file.name)
        if not match:
            continue

        speaker_id = match.group(1)
        style = match.group(2)
        utt_num = match.group(3)

        emotion = EXPRESSO_STYLE_MAP.get(style)
        if emotion is None:
            logger.debug("Unknown Expresso style '%s', skipping", style)
            continue

        entries.append(EmotionEntry(
            utterance_id=f"{speaker_id}_{style}_{utt_num}",
            speaker_id=speaker_id,
            audio_path=wav_file,
            text="",  # Expresso doesn't include transcripts for all styles
            emotion=emotion,
            dataset="expresso",
        ))

    logger.info("Parsed %d entries from Expresso", len(entries))
    return entries


# --- JVNV ---
# 6 emotions: anger, disgust, fear, happiness, sadness, surprise

JVNV_EMOTION_MAP: dict[str, str] = {
    "anger": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "happiness": "happy",
    "sadness": "sad",
    "surprise": "surprised",
}


def parse_jvnv(data_dir: str | Path) -> list[EmotionEntry]:
    """Parse JVNV dataset emotion labels.

    Expected structure::

        data_dir/
        ├── jvnv_ver1/
        │   ├── JVNV001/
        │   │   ├── anger/
        │   │   │   ├── JVNV001_anger_001.wav
        │   │   │   └── ...
        │   │   ├── happiness/
        │   │   └── ...
        │   └── JVNV002/
        └── ...

    Args:
        data_dir: Path to JVNV dataset root.

    Returns:
        List of EmotionEntry.
    """
    data_dir = Path(data_dir)
    entries: list[EmotionEntry] = []

    for wav_file in sorted(data_dir.rglob("*.wav")):
        parts = wav_file.parts
        # Find emotion from directory name
        emotion_ja = None
        speaker_id = None
        for part in parts:
            if part in JVNV_EMOTION_MAP:
                emotion_ja = part
            if part.startswith("JVNV"):
                speaker_id = part

        if emotion_ja is None or speaker_id is None:
            continue

        emotion = JVNV_EMOTION_MAP[emotion_ja]
        utt_id = wav_file.stem

        entries.append(EmotionEntry(
            utterance_id=utt_id,
            speaker_id=speaker_id,
            audio_path=wav_file,
            text="",
            emotion=emotion,
            dataset="jvnv",
        ))

    logger.info("Parsed %d entries from JVNV", len(entries))
    return entries


# --- EmoV-DB ---
# 5 emotions: amused, angry, disgusted, neutral, sleepy

EMOV_EMOTION_MAP: dict[str, str] = {
    "amused": "happy",
    "angry": "angry",
    "disgusted": "disgusted",
    "neutral": "neutral",
    "sleepy": "bored",
}


def parse_emov_db(data_dir: str | Path) -> list[EmotionEntry]:
    """Parse EmoV-DB dataset.

    Expected structure::

        data_dir/
        ├── bea/
        │   ├── amused/
        │   │   ├── amused_1-15_0001.wav
        │   │   └── ...
        │   └── angry/
        └── sam/

    Args:
        data_dir: Path to EmoV-DB root.

    Returns:
        List of EmotionEntry.
    """
    data_dir = Path(data_dir)
    entries: list[EmotionEntry] = []

    for wav_file in sorted(data_dir.rglob("*.wav")):
        parts = wav_file.parts
        emotion_dir = None
        speaker_id = None

        for i, part in enumerate(parts):
            if part.lower() in EMOV_EMOTION_MAP:
                emotion_dir = part.lower()
                if i > 0:
                    speaker_id = parts[i - 1]

        if emotion_dir is None:
            continue
        if speaker_id is None:
            speaker_id = "unknown"

        emotion = EMOV_EMOTION_MAP[emotion_dir]

        entries.append(EmotionEntry(
            utterance_id=wav_file.stem,
            speaker_id=speaker_id,
            audio_path=wav_file,
            text="",
            emotion=emotion,
            dataset="emov_db",
        ))

    logger.info("Parsed %d entries from EmoV-DB", len(entries))
    return entries


# --- RAVDESS ---
# Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry,
#                06=fearful, 07=disgusted, 08=surprised
# Intensity: 01=normal, 02=strong
# File: {modality}-{vocal}-{emotion}-{intensity}-{statement}-{repetition}-{actor}.wav

RAVDESS_EMOTION_MAP: dict[int, str] = {
    1: "neutral",
    2: "tender",     # calm → tender
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgusted",
    8: "surprised",
}

# Approximate VAD values for RAVDESS emotions
RAVDESS_VAD: dict[int, tuple[float, float, float]] = {
    1: (0.0, 0.0, 0.0),      # neutral
    2: (0.3, -0.3, 0.0),     # calm
    3: (0.7, 0.5, 0.3),      # happy
    4: (-0.6, -0.3, -0.3),   # sad
    5: (-0.5, 0.7, 0.5),     # angry
    6: (-0.6, 0.5, -0.5),    # fearful
    7: (-0.5, 0.2, 0.3),     # disgusted
    8: (0.0, 0.7, 0.0),      # surprised
}


def parse_ravdess(data_dir: str | Path) -> list[EmotionEntry]:
    """Parse RAVDESS dataset.

    Expected structure::

        data_dir/
        ├── Actor_01/
        │   ├── 03-01-01-01-01-01-01.wav
        │   └── ...
        └── Actor_02/

    Args:
        data_dir: Path to RAVDESS root.

    Returns:
        List of EmotionEntry.
    """
    data_dir = Path(data_dir)
    entries: list[EmotionEntry] = []

    pattern = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})\.wav$")

    for wav_file in sorted(data_dir.rglob("*.wav")):
        match = pattern.match(wav_file.name)
        if not match:
            continue

        modality = int(match.group(1))
        vocal_channel = int(match.group(2))
        emotion_code = int(match.group(3))
        intensity = int(match.group(4))
        actor = int(match.group(7))

        # Only speech (modality=03) or song (01)
        # Accept all modalities that have audio

        emotion = RAVDESS_EMOTION_MAP.get(emotion_code)
        if emotion is None:
            continue

        vad = RAVDESS_VAD.get(emotion_code, (0.0, 0.0, 0.0))
        # Scale by intensity
        intensity_scale = 1.5 if intensity == 2 else 1.0

        entries.append(EmotionEntry(
            utterance_id=wav_file.stem,
            speaker_id=f"Actor_{actor:02d}",
            audio_path=wav_file,
            text="",
            emotion=emotion,
            valence=_clamp(vad[0] * intensity_scale, -1.0, 1.0),
            arousal=_clamp(vad[1] * intensity_scale, -1.0, 1.0),
            dominance=_clamp(vad[2] * intensity_scale, -1.0, 1.0),
            dataset="ravdess",
        ))

    logger.info("Parsed %d entries from RAVDESS", len(entries))
    return entries


def parse_dataset(name: str, data_dir: str | Path) -> list[EmotionEntry]:
    """Parse any supported emotion dataset by name.

    Args:
        name: Dataset name (expresso, jvnv, emov_db, ravdess).
        data_dir: Path to dataset root.

    Returns:
        List of EmotionEntry.
    """
    parsers = {
        "expresso": parse_expresso,
        "jvnv": parse_jvnv,
        "emov_db": parse_emov_db,
        "ravdess": parse_ravdess,
    }
    if name not in parsers:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(parsers.keys())}")
    return parsers[name](data_dir)


def compute_prosody(
    audio: "np.ndarray",
    sr: int,
) -> list[float]:
    """Compute prosody features from audio: [speaking_rate, energy, pitch_range].

    All values are normalized to roughly [0, 1] range for MSE regression.

    Args:
        audio: 1-D float32 audio array.
        sr: Sample rate.

    Returns:
        [speaking_rate, energy, pitch_range] as floats.
    """
    import numpy as np

    # 1. Speaking rate proxy: syllable-rate estimated from energy envelope peaks
    #    Simple approach: count energy peaks per second
    frame_len = int(sr * 0.025)  # 25ms frames
    hop = int(sr * 0.010)  # 10ms hop
    n_frames = max(1, (len(audio) - frame_len) // hop + 1)

    energy_frames = np.array([
        np.sqrt(np.mean(audio[i * hop : i * hop + frame_len] ** 2))
        for i in range(n_frames)
    ])

    # Normalize energy for peak detection
    e_max = energy_frames.max()
    if e_max > 1e-8:
        e_norm = energy_frames / e_max
    else:
        return [0.0, 0.0, 0.0]

    # Count peaks above 30% of max as syllable-like events
    threshold = 0.3
    above = e_norm > threshold
    # Count rising edges
    peaks = 0
    for i in range(1, len(above)):
        if above[i] and not above[i - 1]:
            peaks += 1

    duration_sec = len(audio) / sr
    # Typical syllable rate: 3-8 per second → normalize to [0,1] with 6 syl/s as midpoint
    rate_raw = peaks / max(duration_sec, 0.1)
    speaking_rate = min(1.0, rate_raw / 10.0)

    # 2. Energy: RMS in dB, normalized
    rms = np.sqrt(np.mean(audio ** 2))
    rms_db = 20 * np.log10(max(rms, 1e-8))
    # Typical speech range: -40 to -10 dB → normalize to [0, 1]
    energy = _clamp((rms_db + 40) / 30, 0.0, 1.0)

    # 3. Pitch range: use simple autocorrelation for F0 estimation
    #    More robust than FFT for short segments
    f0_values = _estimate_f0_autocorr(audio, sr)
    if len(f0_values) > 2:
        f0_lo = np.percentile(f0_values, 10)
        f0_hi = np.percentile(f0_values, 90)
        # Pitch range in semitones
        if f0_lo > 0:
            semitone_range = 12 * np.log2(f0_hi / f0_lo)
        else:
            semitone_range = 0.0
        # Typical range: 3-20 semitones → normalize to [0, 1]
        pitch_range = _clamp(semitone_range / 24.0, 0.0, 1.0)
    else:
        pitch_range = 0.0

    return [round(speaking_rate, 4), round(energy, 4), round(pitch_range, 4)]


def _estimate_f0_autocorr(
    audio: "np.ndarray",
    sr: int,
    frame_len_sec: float = 0.04,
    hop_sec: float = 0.02,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
) -> list[float]:
    """Estimate F0 values using autocorrelation method.

    Returns a list of detected F0 values (Hz) for voiced frames.
    """
    import numpy as np

    frame_len = int(sr * frame_len_sec)
    hop = int(sr * hop_sec)
    min_lag = max(1, int(sr / f0_max))
    max_lag = int(sr / f0_min)

    f0_values: list[float] = []

    for start in range(0, len(audio) - frame_len, hop):
        frame = audio[start : start + frame_len]
        # Check if frame has enough energy
        if np.sqrt(np.mean(frame ** 2)) < 1e-4:
            continue

        # Normalized autocorrelation
        frame = frame - frame.mean()
        norm = np.sum(frame ** 2)
        if norm < 1e-10:
            continue

        corr = np.correlate(frame, frame, mode="full")
        corr = corr[len(frame) - 1 :]  # Only positive lags
        corr = corr / norm

        # Search for peak in valid lag range
        search = corr[min_lag : min(max_lag + 1, len(corr))]
        if len(search) == 0:
            continue

        peak_idx = np.argmax(search)
        peak_val = search[peak_idx]

        # Voiced if autocorrelation peak > 0.3
        if peak_val > 0.3:
            lag = peak_idx + min_lag
            f0 = sr / lag
            f0_values.append(f0)

    return f0_values


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
