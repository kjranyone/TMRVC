"""Audio preprocessing: resample, loudness normalisation, silence trimming, segmentation."""

from __future__ import annotations

import gc
import logging
from typing import Iterator

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

from tmrvc_core.constants import (
    LOUDNESS_TARGET_LUFS,
    SAMPLE_RATE,
    SEGMENT_MAX_SEC,
    SEGMENT_MIN_SEC,
)

logger = logging.getLogger(__name__)


def _is_cuda_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "cuda" in msg and "out of memory" in msg


def _transcribe_text_with_fallback(
    audio_path: str | object,
    whisper,
    language: str,
    device: str,
    models: dict | None = None,
) -> str:
    """Run Whisper transcription with OOM fallback.

    Strategy:
    1. Keep current model/settings for quality.
    2. On CUDA OOM, clear cache and retry with lower-memory GPU compute type.
    3. If it still OOMs, fall back to CPU int8 for this worker.
    """
    from faster_whisper import WhisperModel

    path = str(audio_path)
    try:
        segments, _ = whisper.transcribe(path, language=language)
        return "".join(seg.text for seg in segments).strip()
    except RuntimeError as e:
        if not _is_cuda_oom_error(e) or device != "cuda":
            raise

        logger.warning(
            "Whisper CUDA OOM on %s. Retrying with lower-memory settings.",
            path,
        )
        torch.cuda.empty_cache()
        gc.collect()

        try:
            lowmem_whisper = None
            if models is not None:
                lowmem_whisper = models.get("_whisper_lowmem")
            if lowmem_whisper is None:
                lowmem_whisper = WhisperModel(
                    "large-v3-turbo",
                    device="cuda",
                    compute_type="int8_float16",
                )
                if models is not None:
                    models["_whisper_lowmem"] = lowmem_whisper
                    models["whisper"] = lowmem_whisper
            segments, _ = lowmem_whisper.transcribe(path, language=language)
            return "".join(seg.text for seg in segments).strip()
        except RuntimeError as e2:
            if not _is_cuda_oom_error(e2):
                raise

            logger.warning(
                "Whisper still OOM on GPU for %s. Falling back to CPU int8.",
                path,
            )
            torch.cuda.empty_cache()
            gc.collect()

            cpu_whisper = None
            if models is not None:
                cpu_whisper = models.get("_whisper_cpu")
            if cpu_whisper is None:
                cpu_whisper = WhisperModel(
                    "large-v3-turbo",
                    device="cpu",
                    compute_type="int8",
                )
                if models is not None:
                    models["_whisper_cpu"] = cpu_whisper

            segments, _ = cpu_whisper.transcribe(path, language=language)
            return "".join(seg.text for seg in segments).strip()


# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------


def load_and_resample(
    path: str | object,
    target_sr: int = SAMPLE_RATE,
) -> tuple[torch.Tensor, int]:
    """Load an audio file and resample to *target_sr*.

    Uses soundfile for reading to avoid torchcodec dependency.

    Returns:
        ``(waveform, target_sr)`` where ``waveform`` is ``[1, T]`` float32.
    """
    data, sr = sf.read(str(path), dtype="float32")
    # soundfile returns [T] for mono, [T, C] for multi-channel
    waveform = torch.from_numpy(data)
    del data  # break numpy↔tensor shared memory immediately
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T]
    else:
        waveform = waveform.T  # [C, T]
    # Mix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = AF.resample(waveform, sr, target_sr)
    return waveform.clone(), target_sr  # clone to detach from resampler internals


# ---------------------------------------------------------------------------
# Loudness normalisation
# ---------------------------------------------------------------------------


def normalize_loudness(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = LOUDNESS_TARGET_LUFS,
) -> torch.Tensor:
    """Normalise integrated loudness to *target_lufs* (ITU-R BS.1770-4).

    Args:
        waveform: ``[1, T]`` float32.

    Returns:
        Loudness-normalised ``[1, T]`` tensor.
    """
    audio_np = waveform.squeeze(0).numpy()
    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(audio_np)

    if np.isinf(current_lufs):
        logger.warning("Silent audio detected, skipping loudness normalisation")
        return waveform

    normalised = pyln.normalize.loudness(audio_np, current_lufs, target_lufs)
    return torch.from_numpy(normalised).float().unsqueeze(0)


# ---------------------------------------------------------------------------
# Silence trimming (energy-based, no external VAD dependency)
# ---------------------------------------------------------------------------


def trim_silence(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    threshold_db: float = -40.0,
    frame_ms: float = 10.0,
    min_speech_ms: float = 200.0,
) -> torch.Tensor:
    """Trim leading/trailing silence using a simple energy-based VAD.

    This avoids a hard dependency on Silero VAD at import time.  For
    production preprocessing the CLI can optionally use Silero via
    ``--vad silero``.

    Args:
        waveform: ``[1, T]`` float32.
        threshold_db: Energy threshold relative to full scale.
        frame_ms: Analysis frame length in milliseconds.
        min_speech_ms: Minimum speech region to keep.

    Returns:
        Trimmed ``[1, T']`` tensor (or the original if entirely below threshold).
    """
    frame_len = int(sample_rate * frame_ms / 1000.0)
    audio = waveform.squeeze(0)
    n_frames = audio.shape[0] // frame_len

    if n_frames == 0:
        return waveform

    # RMS energy per frame
    frames = audio[: n_frames * frame_len].view(n_frames, frame_len)
    rms = frames.pow(2).mean(dim=1).sqrt()
    rms_db = 20.0 * torch.log10(rms.clamp(min=1e-10))

    voiced = rms_db > threshold_db
    indices = torch.where(voiced)[0]

    if len(indices) == 0:
        return waveform

    start_frame = indices[0].item()
    end_frame = indices[-1].item() + 1

    # Enforce minimum speech duration
    min_frames = int(min_speech_ms / frame_ms)
    if (end_frame - start_frame) < min_frames:
        return waveform

    start_sample = start_frame * frame_len
    end_sample = min(end_frame * frame_len, audio.shape[0])
    return audio[start_sample:end_sample].unsqueeze(0)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def segment_utterance(
    waveform: torch.Tensor,
    sample_rate: int = SAMPLE_RATE,
    min_sec: float = SEGMENT_MIN_SEC,
    max_sec: float = SEGMENT_MAX_SEC,
) -> Iterator[torch.Tensor]:
    """Split a waveform into segments of *min_sec* to *max_sec* duration.

    Segments shorter than *min_sec* at the tail are discarded.

    Yields:
        ``[1, T_seg]`` tensors.
    """
    total_samples = waveform.shape[-1]
    min_samples = int(min_sec * sample_rate)
    max_samples = int(max_sec * sample_rate)

    if total_samples <= max_samples:
        if total_samples >= min_samples:
            yield waveform
        return

    offset = 0
    while offset < total_samples:
        remaining = total_samples - offset
        seg_len = min(max_samples, remaining)
        if seg_len < min_samples:
            break
        seg = waveform[..., offset : offset + seg_len]
        yield seg
        offset += seg_len


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def preprocess_audio(
    path: str | object,
    target_sr: int = SAMPLE_RATE,
    target_lufs: float = LOUDNESS_TARGET_LUFS,
    trim: bool = True,
) -> tuple[torch.Tensor, int]:
    """Load, resample, normalise loudness, and optionally trim silence.

    Returns:
        ``(waveform, sample_rate)`` where ``waveform`` is ``[1, T]``.
    """
    waveform, sr = load_and_resample(path, target_sr)
    waveform = normalize_loudness(waveform, sr, target_lufs)
    if trim:
        waveform = trim_silence(waveform, sr)
    return waveform, sr


# ---------------------------------------------------------------------------
# Single utterance preprocessing (for pipeline)
# ---------------------------------------------------------------------------


def preprocess_single_utterance(
    utt,
    cache_dir,
    dataset: str,
    split: str,
    device: str,
    language: str = "ja",
    models: dict = None,
) -> bool:
    """Process a single utterance and save to cache.

    This function extracts all UCLM features for one utterance:
    - Dual-stream tokens (A + B)
    - Voice state (explicit + SSL)
    - Speaker embedding
    - ASR transcription

    Args:
        utt: Utterance object from dataset adapter
        cache_dir: Cache directory path
        dataset: Dataset name
        split: Dataset split (train/val/test)
        device: Device for computation (cuda/cpu)
        language: Language for ASR
        models: Pre-initialized models dict (optional)

    Returns:
        True if successful, False if skipped/failed
    """
    from pathlib import Path
    import torchaudio.transforms as T
    from faster_whisper import WhisperModel

    from tmrvc_core.audio import compute_mel
    from tmrvc_core.constants import SAMPLE_RATE
    from tmrvc_core.types import UCLMFeatureSet
    from tmrvc_data.cache import FeatureCache
    from tmrvc_data.codec import UCLMCodecWrapper
    from tmrvc_data.voice_state import SSLVoiceStateEstimator
    from tmrvc_data.speaker import SpeakerEncoder

    cache_dir = Path(cache_dir)
    cache = FeatureCache(cache_dir)

    # Check if already cached (idempotency)
    if cache.exists(dataset, split, utt.speaker_id, utt.utterance_id):
        logger.debug("Already cached: %s", utt.utterance_id)
        return True

    # Duration check
    info = sf.info(str(utt.audio_path))
    if info.duration < 0.1 or info.duration > 30.0:
        logger.debug(
            "Skipping %s due to duration (%.2fs)",
            utt.utterance_id,
            info.duration,
        )
        return False

    # Initialize models if not provided
    if models is None:
        codec = UCLMCodecWrapper(None, device=device)
        vs_estimator = SSLVoiceStateEstimator(device=device)
        spk_encoder = SpeakerEncoder(device=device)
        compute_type = "float16" if device == "cuda" else "int8"
        whisper = WhisperModel(
            "large-v3-turbo", device=device, compute_type=compute_type
        )
    else:
        codec = models["codec"]
        vs_estimator = models["vs_estimator"]
        spk_encoder = models["spk_encoder"]
        whisper = models["whisper"]

    # Load and process audio
    waveform, sr = preprocess_audio(str(utt.audio_path), target_sr=SAMPLE_RATE)
    waveform_t = waveform.unsqueeze(0).to(device)

    # Extract features
    a_tokens, b_logits = codec.encode(waveform_t)
    b_tokens = b_logits.argmax(dim=-1)

    mel = compute_mel(waveform_t.squeeze(1)).to(device)
    f0 = torch.zeros(1, 1, mel.shape[-1], device=device)

    waveform_16k = T.Resample(SAMPLE_RATE, 16000).to(device)(waveform_t.squeeze(1))
    vs_dict = vs_estimator(waveform_16k, waveform_t.squeeze(1), mel, f0)

    text = _transcribe_text_with_fallback(
        audio_path=utt.audio_path,
        whisper=whisper,
        language=language,
        device=device,
        models=models,
    )

    spk_embed = spk_encoder.extract(waveform_t.squeeze(1))

    # Frame alignment verification (CRITICAL)
    T_target = a_tokens.shape[-1]
    T_mel = mel.shape[-1]

    assert T_mel == T_target, (
        f"Frame mismatch: mel={T_mel}, codec={T_target}. "
        f"This indicates a bug in MelSpectrogram or codec implementation."
    )

    explicit_state = vs_dict["explicit_state"].detach().cpu()
    if explicit_state.dim() == 3:
        explicit_state = explicit_state.squeeze(0)
    T_explicit = explicit_state.shape[0]

    assert T_explicit == T_target, (
        f"Frame mismatch: explicit_state={T_explicit}, codec={T_target}. "
        f"This indicates a bug in VoiceStateEstimator implementation."
    )
    explicit_state = explicit_state.transpose(0, 1)

    ssl_state = vs_dict["ssl_state"].detach().cpu()
    if ssl_state.dim() == 3:
        ssl_state = ssl_state.squeeze(0)

    ssl_state = (
        torch.nn.functional.interpolate(
            ssl_state.unsqueeze(0).transpose(1, 2),
            size=T_target,
            mode="linear",
            align_corners=False,
        )
        .transpose(1, 2)
        .squeeze(0)
    )
    ssl_state = ssl_state.transpose(0, 1)

    b_tokens_aligned = b_tokens.detach().cpu().squeeze(0)
    T_b = b_tokens_aligned.shape[-1]
    assert T_b == T_target, (
        f"Frame mismatch: b_tokens={T_b}, codec={T_target}. "
        f"This indicates a bug in codec implementation."
    )

    from tmrvc_data.g2p import text_to_phonemes
    g2p_result = text_to_phonemes(text, language=language)
    phoneme_ids = g2p_result.phoneme_ids.detach().cpu()

    # Save to cache
    features = UCLMFeatureSet(
        codec_tokens_a=a_tokens.detach().cpu().squeeze(0),
        codec_tokens_b=b_tokens_aligned,
        voice_state_explicit=explicit_state,
        voice_state_ssl=ssl_state,
        spk_embed=spk_embed.detach().cpu().squeeze(0),
        phoneme_ids=phoneme_ids,
        durations=None,
        text=text,
        utterance_id=utt.utterance_id,
        speaker_id=utt.speaker_id,
        n_frames=T_target,
        waveform=waveform.detach(),
    )

    cache.save(features, dataset, split)
    return True
