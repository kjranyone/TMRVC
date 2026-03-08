"""Stage 6: Prosody / Event Recovery - Extract acting-relevant signals.

Recovers pause spans, pitch (F0) statistics, energy statistics, speech rate,
and computes canonical 8-D voice_state pseudo-labels with per-dimension
confidence and observed masks.

The canonical voice_state dimensions are:
  0: pitch_level
  1: pitch_range
  2: energy_level
  3: pressedness
  4: spectral_tilt
  5: breathiness
  6: voice_irregularity
  7: openness
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from tmrvc_core.voice_state import CANONICAL_VOICE_STATE_IDS

from ..models import CurationRecord, Provenance, RecordStatus
from ..providers import ProviderRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOICE_STATE_DIM = 8
VOICE_STATE_NAMES = list(CANONICAL_VOICE_STATE_IDS)
SILENCE_DB = -50.0
MIN_PAUSE_SEC = 0.15  # minimum pause duration to record
F0_MIN_HZ = 50.0
F0_MAX_HZ = 600.0


def _load_audio_mono(path: str, start: float, end: float) -> Tuple[np.ndarray, int]:
    """Load audio segment as mono float32."""
    info = sf.info(path)
    sr = info.samplerate
    start_frame = int(start * sr)
    n_frames = int((end - start) * sr)
    audio, sr = sf.read(path, start=start_frame, frames=n_frames,
                        dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


# ---------------------------------------------------------------------------
# Pause detection
# ---------------------------------------------------------------------------

def extract_pause_spans(
    audio: np.ndarray, sr: int,
    frame_sec: float = 0.025, hop_sec: float = 0.010,
    silence_db: float = SILENCE_DB, min_pause_sec: float = MIN_PAUSE_SEC,
) -> List[Dict[str, Any]]:
    """Detect pause spans using frame-level energy thresholding."""
    frame_len = max(1, int(frame_sec * sr))
    hop_len = max(1, int(hop_sec * sr))
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)

    pauses: List[Dict[str, Any]] = []
    in_silence = False
    silence_start = 0.0

    for i in range(n_frames):
        start_idx = i * hop_len
        frame = audio[start_idx: start_idx + frame_len]
        energy_db = 10.0 * np.log10(np.mean(frame ** 2) + 1e-12)
        t = i * hop_sec

        if energy_db < silence_db:
            if not in_silence:
                silence_start = t
                in_silence = True
        else:
            if in_silence:
                duration = t - silence_start
                if duration >= min_pause_sec:
                    pauses.append({
                        "start_sec": round(silence_start, 4),
                        "end_sec": round(t, 4),
                        "duration_sec": round(duration, 4),
                    })
                in_silence = False

    # Close trailing silence
    if in_silence:
        t_end = n_frames * hop_sec
        duration = t_end - silence_start
        if duration >= min_pause_sec:
            pauses.append({
                "start_sec": round(silence_start, 4),
                "end_sec": round(t_end, 4),
                "duration_sec": round(duration, 4),
            })

    return pauses


# ---------------------------------------------------------------------------
# Pitch (F0) extraction - autocorrelation method
# ---------------------------------------------------------------------------

def _autocorrelation_f0(frame: np.ndarray, sr: int,
                        f0_min: float, f0_max: float) -> float:
    """Estimate F0 from a single frame using autocorrelation.

    Returns F0 in Hz, or 0.0 if unvoiced.
    """
    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)
    if max_lag >= len(frame) or min_lag < 1:
        return 0.0

    # Normalized autocorrelation
    frame = frame - np.mean(frame)
    norm = np.sum(frame ** 2)
    if norm < 1e-10:
        return 0.0

    autocorr = np.correlate(frame, frame, mode="full")
    autocorr = autocorr[len(frame) - 1:]  # keep non-negative lags
    autocorr = autocorr / (norm + 1e-12)

    # Find peak in valid lag range
    search = autocorr[min_lag: max_lag + 1]
    if len(search) == 0:
        return 0.0

    peak_idx = np.argmax(search)
    peak_val = search[peak_idx]

    # Voicing threshold
    if peak_val < 0.3:
        return 0.0

    lag = min_lag + peak_idx
    if lag == 0:
        return 0.0

    return float(sr / lag)


def extract_pitch_stats(
    audio: np.ndarray, sr: int,
    frame_sec: float = 0.03, hop_sec: float = 0.01,
) -> Dict[str, Any]:
    """Extract F0 statistics over the audio segment."""
    frame_len = max(1, int(frame_sec * sr))
    hop_len = max(1, int(hop_sec * sr))
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)

    f0_values = []
    voiced_count = 0

    for i in range(n_frames):
        start_idx = i * hop_len
        frame = audio[start_idx: start_idx + frame_len]
        if len(frame) < frame_len:
            break
        f0 = _autocorrelation_f0(frame, sr, F0_MIN_HZ, F0_MAX_HZ)
        if f0 > 0:
            f0_values.append(f0)
            voiced_count += 1

    voiced_ratio = voiced_count / n_frames if n_frames > 0 else 0.0

    if f0_values:
        f0_arr = np.array(f0_values)
        return {
            "f0_mean_hz": round(float(np.mean(f0_arr)), 2),
            "f0_std_hz": round(float(np.std(f0_arr)), 2),
            "f0_min_hz": round(float(np.min(f0_arr)), 2),
            "f0_max_hz": round(float(np.max(f0_arr)), 2),
            "f0_median_hz": round(float(np.median(f0_arr)), 2),
            "voiced_ratio": round(voiced_ratio, 4),
            "n_voiced_frames": voiced_count,
            "n_total_frames": n_frames,
        }
    else:
        return {
            "f0_mean_hz": 0.0,
            "f0_std_hz": 0.0,
            "f0_min_hz": 0.0,
            "f0_max_hz": 0.0,
            "f0_median_hz": 0.0,
            "voiced_ratio": 0.0,
            "n_voiced_frames": 0,
            "n_total_frames": n_frames,
        }


# ---------------------------------------------------------------------------
# Energy statistics
# ---------------------------------------------------------------------------

def extract_energy_stats(audio: np.ndarray, sr: int) -> Dict[str, Any]:
    """Extract energy statistics over the audio segment."""
    if len(audio) == 0:
        return {"rms": 0.0, "rms_db": -100.0, "energy_std": 0.0, "peak": 0.0}

    rms = float(np.sqrt(np.mean(audio ** 2)))
    rms_db = 20.0 * np.log10(rms + 1e-12)
    peak = float(np.max(np.abs(audio)))

    # Per-frame energy variation
    frame_len = max(1, int(0.025 * sr))
    hop_len = max(1, int(0.010 * sr))
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)
    frame_energies = []
    for i in range(n_frames):
        s = i * hop_len
        frame = audio[s: s + frame_len]
        frame_energies.append(float(np.sqrt(np.mean(frame ** 2))))

    energy_std = float(np.std(frame_energies)) if frame_energies else 0.0

    return {
        "rms": round(rms, 6),
        "rms_db": round(float(rms_db), 2),
        "energy_std": round(energy_std, 6),
        "peak": round(peak, 6),
    }


# ---------------------------------------------------------------------------
# Speech rate estimation
# ---------------------------------------------------------------------------

def estimate_speech_rate(
    audio: np.ndarray, sr: int, transcript: Optional[str] = None
) -> Dict[str, Any]:
    """Estimate speech rate.

    If transcript is available, computes characters-per-second as a proxy
    for phonemes-per-second. Otherwise, estimates syllable rate from
    energy envelope modulations.
    """
    duration = len(audio) / sr if sr > 0 else 0.0
    if duration < 0.1:
        return {"speech_rate_estimate": 0.0, "method": "too_short"}

    if transcript and len(transcript.strip()) > 0:
        # Use character count as phoneme proxy
        # Rough heuristic: ~1.2 phonemes per character for most languages
        n_chars = len(transcript.strip())
        chars_per_sec = n_chars / duration
        phonemes_per_sec = chars_per_sec * 1.2
        return {
            "speech_rate_estimate": round(phonemes_per_sec, 2),
            "chars_per_sec": round(chars_per_sec, 2),
            "method": "transcript_based",
        }

    # Syllable rate from energy envelope modulations
    # Downsample energy envelope and count peaks
    frame_len = max(1, int(0.025 * sr))
    hop_len = max(1, int(0.010 * sr))
    n_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)

    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop_len
        frame = audio[s: s + frame_len]
        envelope[i] = float(np.sqrt(np.mean(frame ** 2)))

    # Simple peak counting on smoothed envelope
    if len(envelope) > 5:
        # Smooth with running mean
        kernel_size = min(5, len(envelope))
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(envelope, kernel, mode="same")

        # Count peaks above mean
        mean_e = np.mean(smoothed)
        above = smoothed > mean_e
        # Count transitions from below to above
        transitions = np.sum(np.diff(above.astype(int)) > 0)
        syllable_rate = transitions / duration if duration > 0 else 0.0
        # Rough conversion: ~1 syllable = ~2-3 phonemes
        phonemes_per_sec = syllable_rate * 2.5
    else:
        phonemes_per_sec = 0.0

    return {
        "speech_rate_estimate": round(phonemes_per_sec, 2),
        "method": "envelope_modulation",
    }


# ---------------------------------------------------------------------------
# Spectral tilt (breathiness proxy)
# ---------------------------------------------------------------------------

def _estimate_spectral_tilt(audio: np.ndarray, sr: int) -> float:
    """Estimate spectral tilt as a breathiness proxy.

    Negative tilt (energy decreasing with frequency) is normal speech.
    Flatter or positive tilt suggests breathiness.
    Returns normalized value in [0, 1] where 1 = very breathy.
    """
    n_fft = min(2048, len(audio))
    if n_fft < 64:
        return 0.5

    # Average spectrum
    hop = n_fft // 2
    n_wins = max(1, (len(audio) - n_fft) // hop)
    spectra = []
    for i in range(min(n_wins, 50)):
        s = i * hop
        frame = audio[s: s + n_fft] * np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frame))
        spectra.append(spec)

    avg_spec = np.mean(spectra, axis=0)
    avg_spec_db = 20.0 * np.log10(avg_spec + 1e-12)

    # Linear regression of spectrum (dB) vs frequency bin index
    x = np.arange(len(avg_spec_db), dtype=np.float64)
    if len(x) < 2:
        return 0.5

    x_mean = np.mean(x)
    y_mean = np.mean(avg_spec_db)
    slope = float(np.sum((x - x_mean) * (avg_spec_db - y_mean)) /
                  (np.sum((x - x_mean) ** 2) + 1e-12))

    # Typical speech slope is around -0.5 to -2.0 dB/bin
    # Flatten to [0, 1]: more negative = less breathy, flatter = more breathy
    breathiness = float(np.clip((slope + 2.0) / 2.0, 0.0, 1.0))
    return breathiness


# ---------------------------------------------------------------------------
# Jitter/shimmer (voice-irregularity proxy)
# ---------------------------------------------------------------------------

def _estimate_jitter_shimmer(
    audio: np.ndarray, sr: int
) -> Tuple[float, float]:
    """Estimate jitter (F0 perturbation) and shimmer (amplitude perturbation).

    Returns (jitter_ratio, shimmer_ratio) both in [0, 1].
    """
    frame_len = int(0.03 * sr)
    hop_len = int(0.01 * sr)
    if frame_len < 10 or len(audio) < frame_len:
        return 0.0, 0.0

    n_frames = 1 + (len(audio) - frame_len) // hop_len
    periods = []
    amplitudes = []

    for i in range(n_frames):
        s = i * hop_len
        frame = audio[s: s + frame_len]
        f0 = _autocorrelation_f0(frame, sr, F0_MIN_HZ, F0_MAX_HZ)
        if f0 > 0:
            periods.append(1.0 / f0)
            amplitudes.append(float(np.max(np.abs(frame))))

    if len(periods) < 3:
        return 0.0, 0.0

    # Jitter: average absolute period-to-period difference / mean period
    period_diffs = [abs(periods[i + 1] - periods[i]) for i in range(len(periods) - 1)]
    jitter = float(np.mean(period_diffs) / (np.mean(periods) + 1e-12))

    # Shimmer: average absolute amplitude-to-amplitude difference / mean amplitude
    amp_diffs = [abs(amplitudes[i + 1] - amplitudes[i]) for i in range(len(amplitudes) - 1)]
    shimmer = float(np.mean(amp_diffs) / (np.mean(amplitudes) + 1e-12))

    # Clip to [0, 1]
    jitter = float(np.clip(jitter, 0.0, 1.0))
    shimmer = float(np.clip(shimmer, 0.0, 1.0))

    return jitter, shimmer


# ---------------------------------------------------------------------------
# 8-D voice_state computation
# ---------------------------------------------------------------------------

def compute_voice_state(
    audio: np.ndarray,
    sr: int,
    pitch_stats: Dict[str, Any],
    energy_stats: Dict[str, Any],
    speech_rate_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute 8-D voice_state pseudo-labels with confidence and observed mask.

    Returns:
        Dict with 'voice_state' (8 floats), 'voice_state_observed_mask' (8 bools),
        'voice_state_confidence' (8 floats), and 'voice_state_names'.
    """
    voice_state = np.zeros(VOICE_STATE_DIM, dtype=np.float32)
    observed_mask = np.zeros(VOICE_STATE_DIM, dtype=bool)
    confidence = np.zeros(VOICE_STATE_DIM, dtype=np.float32)

    f0_std = pitch_stats.get("f0_std_hz", 0.0)
    f0_mean = pitch_stats.get("f0_mean_hz", 0.0)
    voiced_ratio = pitch_stats.get("voiced_ratio", 0.0)
    rms = energy_stats.get("rms", 0.0)
    energy_norm = float(np.clip(rms / 0.2, 0.0, 1.0))
    spectral_tilt = _estimate_spectral_tilt(audio, sr)
    jitter, shimmer = _estimate_jitter_shimmer(audio, sr)
    voice_irregularity = 0.5 * jitter + 0.5 * shimmer
    breathiness = float(np.clip(0.7 * spectral_tilt + 0.3 * (1.0 - voiced_ratio), 0.0, 1.0))

    # Dim 0: pitch_level (normalised F0 mean)
    if f0_mean > 0:
        pitch_level = float(np.clip(np.log2(max(f0_mean, 50.0) / 50.0) / 4.0, 0.0, 1.0))
        voice_state[0] = pitch_level
        observed_mask[0] = True
        confidence[0] = 0.7
    else:
        voice_state[0] = 0.5
        observed_mask[0] = False
        confidence[0] = 0.1

    # Dim 1: pitch_range (F0 coefficient of variation)
    if f0_mean > 0:
        f0_cv = f0_std / (f0_mean + 1e-6)
        voice_state[1] = float(np.clip(f0_cv / 0.35, 0.0, 1.0))
        observed_mask[1] = True
        confidence[1] = 0.6
    else:
        voice_state[1] = 0.3
        observed_mask[1] = False
        confidence[1] = 0.1

    # Dim 2: energy_level (direct RMS measurement)
    voice_state[2] = energy_norm
    observed_mask[2] = True
    confidence[2] = 0.8

    # Dim 3: pressedness (weak proxy from F0 level + energy + inverse breathiness)
    if f0_mean > 0:
        voice_state[3] = float(
            np.clip(
                0.4 * voice_state[0] + 0.35 * energy_norm + 0.25 * (1.0 - breathiness),
                0.0,
                1.0,
            )
        )
        observed_mask[3] = True
        confidence[3] = 0.35
    else:
        voice_state[3] = 0.35
        observed_mask[3] = False
        confidence[3] = 0.1

    # Dim 4: spectral_tilt
    voice_state[4] = spectral_tilt
    observed_mask[4] = True
    confidence[4] = 0.6

    # Dim 5: breathiness
    voice_state[5] = breathiness
    observed_mask[5] = True
    confidence[5] = 0.45

    # Dim 6: voice_irregularity (jitter/shimmer)
    voice_state[6] = voice_irregularity
    observed_mask[6] = jitter > 0 or shimmer > 0
    confidence[6] = 0.4 if observed_mask[6] else 0.1

    # Dim 7: openness (not robustly observable here; keep neutral unless stronger provider exists)
    voice_state[7] = 0.5
    observed_mask[7] = False
    confidence[7] = 0.1

    return {
        "voice_state": [round(float(v), 4) for v in voice_state],
        "voice_state_observed_mask": [bool(m) for m in observed_mask],
        "voice_state_confidence": [round(float(c), 4) for c in confidence],
        "voice_state_names": list(VOICE_STATE_NAMES),
    }


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run_prosody_recovery(
    record: CurationRecord,
    registry: Optional[ProviderRegistry] = None,
) -> Optional[CurationRecord]:
    """Process a single record through Stage 6: Prosody / Event Recovery.

    Extracts:
    - Pause spans
    - Pitch (F0) statistics
    - Energy statistics
    - Speech rate
    - 8-D voice_state pseudo-labels with per-dimension confidence

    Args:
        record: The curation record to process.
        registry: Optional provider registry for external extraction.

    Returns:
        Updated CurationRecord.
    """
    if record.status == RecordStatus.REJECTED:
        return record

    # --- Try external event extraction provider ---
    if registry is not None:
        provider = registry.get_primary("event_extraction")
        if provider is not None and provider.is_available():
            try:
                output = provider.process(record)
                for key, value in output.fields.items():
                    if key == "attributes" and isinstance(value, dict):
                        record.attributes.update(value)
                    elif hasattr(record, key):
                        setattr(record, key, value)
                if output.provenance:
                    record.providers["prosody_recovery"] = output.provenance
                return record
            except Exception as e:
                logger.warning(
                    "Event extraction provider failed for %s: %s",
                    record.record_id, e,
                )

    # --- Builtin extraction ---
    try:
        audio, sr = _load_audio_mono(
            record.source_path,
            record.segment_start_sec,
            record.segment_end_sec,
        )
    except Exception as e:
        logger.warning("Prosody recovery: cannot load %s: %s", record.record_id, e)
        record.providers["prosody_recovery"] = Provenance(
            stage="prosody_recovery",
            provider="builtin_prosody",
            version="1.0.0",
            timestamp=time.time(),
            confidence=0.0,
            metadata={"error": str(e)},
        )
        return record

    # --- Pause detection ---
    pause_spans = extract_pause_spans(audio, sr)
    record.attributes["pause_events"] = pause_spans
    record.attributes["n_pauses"] = len(pause_spans)
    total_pause_dur = sum(p["duration_sec"] for p in pause_spans)
    record.attributes["total_pause_duration_sec"] = round(total_pause_dur, 4)

    # --- Pitch extraction ---
    pitch_stats = extract_pitch_stats(audio, sr)
    record.attributes["pitch_stats"] = pitch_stats

    # --- Energy extraction ---
    energy_stats = extract_energy_stats(audio, sr)
    record.attributes["energy_stats"] = energy_stats

    # --- Speech rate ---
    speech_rate_info = estimate_speech_rate(audio, sr, record.transcript)
    record.attributes["speech_rate"] = speech_rate_info

    # --- 8-D voice_state ---
    voice_state_result = compute_voice_state(
        audio, sr, pitch_stats, energy_stats, speech_rate_info,
    )
    record.attributes["voice_state"] = voice_state_result["voice_state"]
    record.attributes["voice_state_observed_mask"] = voice_state_result["voice_state_observed_mask"]
    record.attributes["voice_state_confidence"] = voice_state_result["voice_state_confidence"]
    record.attributes["voice_state_names"] = voice_state_result["voice_state_names"]

    # Compute summary metrics for downstream scoring
    observed_count = sum(voice_state_result["voice_state_observed_mask"])
    observed_ratio = observed_count / VOICE_STATE_DIM
    avg_confidence = float(np.mean(voice_state_result["voice_state_confidence"]))

    record.attributes["voice_state_observed_ratio"] = round(observed_ratio, 4)
    record.attributes["voice_state_confidence_summary"] = round(avg_confidence, 4)
    record.attributes["voice_state_target_source"] = "builtin_prosody"

    # SNR rough estimate (useful for downstream scoring)
    if "snr_db" not in record.attributes:
        # Rough SNR: ratio of speech energy to silence energy
        speech_rms = energy_stats.get("rms", 0.0)
        if pause_spans and speech_rms > 0:
            pause_energies = []
            for p in pause_spans:
                p_start = int(p["start_sec"] * sr)
                p_end = int(p["end_sec"] * sr)
                if p_end > p_start and p_end <= len(audio):
                    p_audio = audio[p_start:p_end]
                    pause_energies.append(float(np.sqrt(np.mean(p_audio ** 2))))
            if pause_energies:
                noise_rms = float(np.mean(pause_energies))
                if noise_rms > 1e-10:
                    snr_db = 20.0 * np.log10(speech_rms / noise_rms)
                    record.attributes["snr_db"] = round(float(snr_db), 2)

    # Overall prosody confidence
    prosody_confidence = round(avg_confidence, 4)

    record.providers["prosody_recovery"] = Provenance(
        stage="prosody_recovery",
        provider="builtin_prosody",
        version="1.0.0",
        timestamp=time.time(),
        confidence=prosody_confidence,
        metadata={
            "n_pauses": len(pause_spans),
            "f0_mean_hz": pitch_stats.get("f0_mean_hz", 0.0),
            "voiced_ratio": pitch_stats.get("voiced_ratio", 0.0),
            "voice_state_observed_ratio": observed_ratio,
            "voice_state_avg_confidence": avg_confidence,
            "speech_rate_method": speech_rate_info.get("method", "unknown"),
        },
    )

    return record
