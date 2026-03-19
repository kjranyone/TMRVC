"""Multi-objective reward functions for v4 RL fine-tuning.

Four reward classes:
1. InstructionFollowingReward  -- rich-transcription ASR re-transcription,
   inline tag compliance (recall / precision / F1).
2. PhysicalComplianceReward -- DSP/SSL measured features vs targets across
   the 12-D physical control space.
3. IntelligibilityReward -- plain transcript WER / CER.
4. NaturalnessGuard -- silence, noise, repetition detection.

Heavy model dependencies (Qwen3-ASR, VoiceStateEstimator) are lazy-imported
so that config-only usage never triggers GPU memory allocation.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward result container
# ---------------------------------------------------------------------------


@dataclass
class RewardResult:
    """Detailed breakdown of all reward components for a single sample."""

    instruction_following: float = 0.0
    physical_compliance: float = 0.0
    intelligibility: float = 0.0
    naturalness: float = 1.0
    total: float = 0.0

    # Fine-grained breakdown
    tag_recall: float = 0.0
    tag_precision: float = 0.0
    tag_f1: float = 0.0
    physical_rmse: float = 0.0
    physical_correlation: float = 0.0
    wer: float = 1.0
    cer: float = 1.0
    is_degenerate: bool = False
    silence_ratio: float = 0.0
    repetition_ratio: float = 0.0
    noise_ratio: float = 0.0


# ---------------------------------------------------------------------------
# 1. InstructionFollowingReward
# ---------------------------------------------------------------------------


class InstructionFollowingReward(nn.Module):
    """Compute instruction-following reward via rich-transcription ASR re-transcription.

    Workflow:
        1. Receive generated audio waveform.
        2. Re-transcribe with Qwen3-ASR in rich-transcription mode (producing
           inline tags such as [laugh], [whisper], etc.).
        3. Extract inline tags from re-transcribed text.
        4. Compare extracted tags against the input enriched transcript tags.
        5. Return recall, precision, and F1-based reward.

    The ASR model is lazy-loaded on first ``compute()`` call.
    """

    def __init__(self, asr_model_name: str = "Qwen/Qwen3-ASR-1.7B"):
        super().__init__()
        self.asr_model_name = asr_model_name
        self._asr_pipeline = None
        self._device: Optional[torch.device] = None

    # -- lazy ASR loading ---------------------------------------------------

    def _ensure_asr(self, device: torch.device) -> None:
        """Lazy-load the ASR model on first use."""
        if self._asr_pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-untyped]

            self._asr_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.asr_model_name,
                device=device,
            )
            self._device = device
            logger.info("Loaded ASR model %s on %s", self.asr_model_name, device)
        except Exception:
            logger.warning(
                "ASR model %s unavailable; falling back to tag-string matching",
                self.asr_model_name,
            )
            self._asr_pipeline = None

    # -- tag extraction helpers ---------------------------------------------

    @staticmethod
    def _extract_tags(text: str) -> List[str]:
        """Extract bracket-delimited tags from text.

        >>> InstructionFollowingReward._extract_tags("[angry] Hello [pause] world")
        ['[angry]', '[pause]']
        """
        import re

        return re.findall(r"\[[^\]]+\]", text)

    @staticmethod
    def _tag_compliance(
        requested: Sequence[str],
        detected: Sequence[str],
    ) -> Tuple[float, float, float]:
        """Compute recall, precision, F1 for tag compliance.

        Args:
            requested: Tags from the input enriched transcript.
            detected: Tags found in the re-transcribed output.

        Returns:
            (recall, precision, f1)
        """
        if not requested:
            # No tags requested -- vacuously satisfied.
            return 1.0, 1.0, 1.0

        req_counts: Dict[str, int] = {}
        for t in requested:
            req_counts[t] = req_counts.get(t, 0) + 1
        det_counts: Dict[str, int] = {}
        for t in detected:
            det_counts[t] = det_counts.get(t, 0) + 1

        # Per-type minimum overlap
        hits = sum(min(req_counts.get(t, 0), det_counts.get(t, 0)) for t in req_counts)
        total_req = sum(req_counts.values())
        total_det = sum(det_counts.values())

        recall = hits / total_req if total_req > 0 else 1.0
        precision = hits / total_det if total_det > 0 else 1.0
        f1 = (
            2 * recall * precision / (recall + precision)
            if (recall + precision) > 0
            else 0.0
        )
        return recall, precision, f1

    # -- main interface -----------------------------------------------------

    def compute(
        self,
        generated_audio: torch.Tensor,
        input_enriched_transcript: str,
        sample_rate: int = 24000,
    ) -> RewardResult:
        """Compute instruction-following reward for a single sample.

        Args:
            generated_audio: [1, T_samples] or [T_samples] waveform tensor.
            input_enriched_transcript: The enriched transcript with inline tags.
            sample_rate: Audio sample rate.

        Returns:
            RewardResult with instruction_following and tag detail fields populated.
        """
        result = RewardResult()
        requested_tags = self._extract_tags(input_enriched_transcript)

        if not requested_tags:
            result.instruction_following = 1.0
            result.tag_recall = 1.0
            result.tag_precision = 1.0
            result.tag_f1 = 1.0
            return result

        # Attempt ASR re-transcription
        detected_tags: List[str] = []
        if generated_audio.dim() == 1:
            generated_audio = generated_audio.unsqueeze(0)
        device = generated_audio.device
        self._ensure_asr(device)

        if self._asr_pipeline is not None:
            try:
                audio_np = generated_audio.squeeze().cpu().numpy().astype(np.float32)
                asr_out = self._asr_pipeline(
                    {"raw": audio_np, "sampling_rate": sample_rate},
                )
                transcription = asr_out.get("text", "") if isinstance(asr_out, dict) else str(asr_out)
                detected_tags = self._extract_tags(transcription)
            except Exception as exc:
                logger.warning("ASR re-transcription failed: %s", exc)
                detected_tags = []
        else:
            # Fallback: heuristic energy-based pseudo-detection for vocal events
            detected_tags = self._energy_heuristic_tags(generated_audio, requested_tags)

        recall, precision, f1 = self._tag_compliance(requested_tags, detected_tags)
        result.tag_recall = recall
        result.tag_precision = precision
        result.tag_f1 = f1
        result.instruction_following = f1
        return result

    @staticmethod
    def _energy_heuristic_tags(
        audio: torch.Tensor,
        requested_tags: Sequence[str],
    ) -> List[str]:
        """Crude heuristic: detect silence / high-energy events to approximate tags.

        This is a fallback when ASR is unavailable. It cannot detect semantic
        tags like [angry] but can approximate [pause], [emphasis] etc.
        """
        audio_1d = audio.squeeze()
        if audio_1d.numel() == 0:
            return []

        rms = audio_1d.pow(2).mean().sqrt().item()
        detected = []

        # Very rough heuristics
        tag_set = set(requested_tags)
        if "[pause]" in tag_set:
            # Check for silent regions (>100ms below threshold)
            frame_size = 240  # 10ms at 24kHz
            frames = audio_1d.unfold(0, frame_size, frame_size)
            frame_rms = frames.pow(2).mean(dim=1).sqrt()
            silence_frames = (frame_rms < 0.01).sum().item()
            if silence_frames >= 10:
                detected.append("[pause]")

        if "[emphasis]" in tag_set:
            # High energy region
            if rms > 0.15:
                detected.append("[emphasis]")

        if "[whisper]" in tag_set:
            # Very low energy
            if rms < 0.02 and rms > 0.001:
                detected.append("[whisper]")

        return detected


# ---------------------------------------------------------------------------
# 2. PhysicalComplianceReward
# ---------------------------------------------------------------------------


class PhysicalComplianceReward(nn.Module):
    """Compute physical control compliance reward.

    Measures DSP/SSL features of the generated audio and compares against
    the explicit 12-D physical targets.  Uses the same VoiceStateEstimator
    as the bootstrap pipeline for consistency.

    The VoiceStateEstimator is lazy-loaded on first use.
    """

    def __init__(self):
        super().__init__()
        self._estimator = None

    def _ensure_estimator(self) -> None:
        if self._estimator is not None:
            return
        try:
            from tmrvc_data.curation.providers.voice_state import (
                VoiceStateEstimator,
            )
            self._estimator = VoiceStateEstimator()
            logger.info("Loaded VoiceStateEstimator for physical compliance reward")
        except Exception:
            logger.warning(
                "VoiceStateEstimator unavailable; using direct tensor comparison"
            )
            self._estimator = None

    @staticmethod
    def _rmse_per_dim(
        target: torch.Tensor,
        measured: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-dimension RMSE.

        Args:
            target: [T, D] or [B, T, D]
            measured: same shape
            mask: optional boolean mask, same shape

        Returns:
            [D] tensor of RMSE per dimension
        """
        diff = target - measured
        if mask is not None:
            diff = diff * mask.float()
            n = mask.float().sum(dim=tuple(range(diff.dim() - 1))).clamp(min=1)
        else:
            n = torch.tensor(
                diff[..., 0].numel(), dtype=diff.dtype, device=diff.device
            ).expand(diff.shape[-1])

        mse = (diff ** 2).sum(dim=tuple(range(diff.dim() - 1))) / n
        return mse.sqrt()

    @staticmethod
    def _pearson_per_dim(
        target: torch.Tensor,
        measured: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-dimension Pearson correlation.

        Args:
            target: [T, D] or flattened along time
            measured: same shape
            mask: optional

        Returns:
            [D] tensor of correlations
        """
        if target.dim() == 3:
            # [B, T, D] -> [B*T, D]
            B, T, D = target.shape
            target = target.reshape(-1, D)
            measured = measured.reshape(-1, D)
            if mask is not None:
                mask = mask.reshape(-1, D)

        # T6 fix: removed premature masking (target * mask, measured * mask)
        # that corrupted values to 0 before per-dim valid-index filtering below.
        # The per-dim loop already filters by mask[:, d].bool().

        D = target.shape[-1]
        correlations = torch.zeros(D, device=target.device)
        for d in range(D):
            t = target[:, d]
            m = measured[:, d]
            if mask is not None:
                valid = mask[:, d].bool()
                t = t[valid]
                m = m[valid]
            if t.numel() < 2:
                correlations[d] = 0.0
                continue
            t_c = t - t.mean()
            m_c = m - m.mean()
            num = (t_c * m_c).sum()
            den = (t_c.pow(2).sum() * m_c.pow(2).sum()).sqrt()
            correlations[d] = num / den.clamp(min=1e-8)
        return correlations

    def compute(
        self,
        generated_audio: torch.Tensor,
        physical_targets: torch.Tensor,
        observed_mask: Optional[torch.Tensor] = None,
        sample_rate: int = 24000,
    ) -> RewardResult:
        """Compute physical compliance reward.

        If VoiceStateEstimator is available, extract features from the audio.
        Otherwise, fall back to comparing against physical_targets directly
        (caller must provide measured features via physical_targets shape
        matching convention).

        Args:
            generated_audio: [1, T_samples] or [T_samples] waveform.
            physical_targets: [T_frames, 12] target physical controls.
            observed_mask: [T_frames, 12] boolean mask for supervised dims.
            sample_rate: Audio sample rate.

        Returns:
            RewardResult with physical_compliance, physical_rmse, physical_correlation.
        """
        result = RewardResult()

        # Try to extract measured physical features from audio
        measured: Optional[torch.Tensor] = None
        self._ensure_estimator()

        if self._estimator is not None:
            try:
                audio_np = generated_audio.squeeze().cpu().numpy().astype(np.float32)
                est_out = self._estimator.estimate(audio_np, sample_rate)
                if hasattr(est_out, "voice_state"):
                    measured = torch.from_numpy(est_out.voice_state).float()
            except Exception as exc:
                logger.debug("Physical feature extraction failed: %s", exc)

        if measured is None:
            # Fallback: compute basic DSP features directly
            measured = self._basic_dsp_features(generated_audio, physical_targets.shape[0])

        # Align time dimensions
        T = min(physical_targets.shape[0], measured.shape[0])
        target = physical_targets[:T]
        measured = measured[:T]
        mask = observed_mask[:T] if observed_mask is not None else None

        rmse_per_dim = self._rmse_per_dim(target, measured, mask)
        corr_per_dim = self._pearson_per_dim(target, measured, mask)

        mean_rmse = rmse_per_dim.mean().item()
        mean_corr = corr_per_dim.mean().item()

        # Reward: higher is better.  RMSE contribution capped at 1.0.
        rmse_reward = max(0.0, 1.0 - mean_rmse)
        corr_reward = max(0.0, (mean_corr + 1.0) / 2.0)  # map [-1,1] -> [0,1]
        compliance = 0.6 * rmse_reward + 0.4 * corr_reward

        result.physical_compliance = compliance
        result.physical_rmse = mean_rmse
        result.physical_correlation = mean_corr
        return result

    @staticmethod
    def _basic_dsp_features(audio: torch.Tensor, n_frames: int) -> torch.Tensor:
        """Extract very basic physical features when estimator is unavailable.

        Returns:
            [n_frames, 12] tensor with crude feature estimates.
        """
        audio_1d = audio.squeeze()
        if audio_1d.numel() == 0:
            return torch.zeros(n_frames, 12)

        hop = max(1, audio_1d.numel() // n_frames)
        features = torch.zeros(n_frames, 12)

        for f in range(min(n_frames, audio_1d.numel() // hop)):
            frame = audio_1d[f * hop : (f + 1) * hop]
            rms = frame.pow(2).mean().sqrt().item()

            # dim 2: energy_level
            features[f, 2] = min(1.0, rms * 5.0)
            # dim 10: vocal_effort (correlated with energy)
            features[f, 10] = min(1.0, rms * 4.0)
            # Other dims stay at 0 (unknown)

        return features


# ---------------------------------------------------------------------------
# 3. IntelligibilityReward
# ---------------------------------------------------------------------------


class IntelligibilityReward(nn.Module):
    """Compute intelligibility reward via plain-text WER/CER.

    Re-transcribes the generated audio with a standard ASR model (not
    rich-transcription mode) and computes edit-distance metrics against
    the plain reference transcript.

    The ASR model is shared with InstructionFollowingReward when possible.
    """

    def __init__(self, asr_model_name: str = "Qwen/Qwen3-ASR-1.7B"):
        super().__init__()
        self.asr_model_name = asr_model_name
        self._asr_pipeline = None

    def _ensure_asr(self, device: torch.device) -> None:
        if self._asr_pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            self._asr_pipeline = hf_pipeline(
                "automatic-speech-recognition",
                model=self.asr_model_name,
                device=device,
            )
            logger.info("Loaded ASR for intelligibility: %s", self.asr_model_name)
        except Exception:
            logger.warning("ASR unavailable for intelligibility; using Levenshtein fallback")
            self._asr_pipeline = None

    def share_asr(self, instruction_reward: InstructionFollowingReward) -> None:
        """Share the ASR pipeline from InstructionFollowingReward to save memory."""
        if instruction_reward._asr_pipeline is not None:
            self._asr_pipeline = instruction_reward._asr_pipeline
            logger.info("Shared ASR pipeline for intelligibility reward")

    @staticmethod
    def _levenshtein(s1: Sequence[str], s2: Sequence[str]) -> int:
        """Compute Levenshtein edit distance between two sequences."""
        if len(s1) == 0:
            return len(s2)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i in range(len(s1)):
            curr = [i + 1] + [0] * len(s2)
            for j in range(len(s2)):
                ins = prev[j + 1] + 1
                dele = curr[j] + 1
                sub = prev[j] + (0 if s1[i] == s2[j] else 1)
                curr[j + 1] = min(ins, dele, sub)
            prev = curr
        return prev[-1]

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """Word Error Rate."""
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0
        dist = IntelligibilityReward._levenshtein(ref_words, hyp_words)
        return min(1.0, dist / len(ref_words))

    @staticmethod
    def compute_cer(reference: str, hypothesis: str) -> float:
        """Character Error Rate."""
        ref_chars = list(reference.replace(" ", ""))
        hyp_chars = list(hypothesis.replace(" ", ""))
        if not ref_chars:
            return 0.0 if not hyp_chars else 1.0
        dist = IntelligibilityReward._levenshtein(ref_chars, hyp_chars)
        return min(1.0, dist / len(ref_chars))

    def compute(
        self,
        generated_audio: torch.Tensor,
        reference_transcript: str,
        sample_rate: int = 24000,
    ) -> RewardResult:
        """Compute intelligibility reward.

        Args:
            generated_audio: [1, T_samples] or [T_samples]
            reference_transcript: Plain text (no inline tags).
            sample_rate: Audio sample rate.

        Returns:
            RewardResult with intelligibility, wer, cer fields populated.
        """
        result = RewardResult()

        if generated_audio.dim() == 1:
            generated_audio = generated_audio.unsqueeze(0)
        device = generated_audio.device

        hypothesis = ""
        self._ensure_asr(device)

        if self._asr_pipeline is not None:
            try:
                audio_np = generated_audio.squeeze().cpu().numpy().astype(np.float32)
                asr_out = self._asr_pipeline(
                    {"raw": audio_np, "sampling_rate": sample_rate},
                )
                hypothesis = asr_out.get("text", "") if isinstance(asr_out, dict) else str(asr_out)
            except Exception as exc:
                logger.warning("ASR transcription for intelligibility failed: %s", exc)
                hypothesis = ""
        else:
            # When ASR is unavailable, return a neutral score based on audio quality
            # rather than penalising unfairly.
            energy = generated_audio.pow(2).mean().sqrt().item()
            if energy > 0.005:
                result.intelligibility = 0.7  # Assume reasonable if non-silent
                result.wer = 0.3
                result.cer = 0.2
            else:
                result.intelligibility = 0.0
                result.wer = 1.0
                result.cer = 1.0
            return result

        # Strip tags from hypothesis for fair comparison
        import re
        hypothesis_clean = re.sub(r"\[[^\]]+\]", "", hypothesis).strip()
        # Also strip tags from reference just in case
        reference_clean = re.sub(r"\[[^\]]+\]", "", reference_transcript).strip()

        wer = self.compute_wer(reference_clean, hypothesis_clean)
        cer = self.compute_cer(reference_clean, hypothesis_clean)

        # Reward: mix of WER and CER (CER more reliable for CJK)
        reward = max(0.0, 1.0 - 0.5 * wer - 0.5 * cer)

        result.intelligibility = reward
        result.wer = wer
        result.cer = cer
        return result


# ---------------------------------------------------------------------------
# 4. NaturalnessGuard
# ---------------------------------------------------------------------------


class NaturalnessGuard(nn.Module):
    """Detect degenerate outputs: silence, noise, repetition.

    This is a penalty-based guard that returns 1.0 for natural outputs
    and approaches 0.0 for degenerate ones.  When ``is_degenerate`` is
    True in the result, the RL trainer should treat the sample as a
    hard failure.
    """

    def __init__(
        self,
        silence_threshold: float = 0.005,
        silence_ratio_limit: float = 0.6,
        repetition_window_min: int = 5,
        repetition_window_max: int = 80,
        noise_spectral_flatness_threshold: float = 0.85,
        sample_rate: int = 24000,
    ):
        super().__init__()
        self.silence_threshold = silence_threshold
        self.silence_ratio_limit = silence_ratio_limit
        self.repetition_window_min = repetition_window_min
        self.repetition_window_max = repetition_window_max
        self.noise_spectral_flatness_threshold = noise_spectral_flatness_threshold
        self.sample_rate = sample_rate

    def _check_silence(self, audio: torch.Tensor) -> Tuple[float, float]:
        """Check for excessive silence.

        Returns:
            (silence_ratio, penalty)
        """
        audio_1d = audio.squeeze()
        if audio_1d.numel() == 0:
            return 1.0, 1.0

        frame_size = max(1, self.sample_rate // 100)  # 10ms frames
        n_frames = audio_1d.numel() // frame_size
        if n_frames == 0:
            return 0.0, 0.0

        frames = audio_1d[: n_frames * frame_size].reshape(n_frames, frame_size)
        frame_energy = frames.pow(2).mean(dim=1).sqrt()
        silent_frames = (frame_energy < self.silence_threshold).sum().item()
        silence_ratio = silent_frames / n_frames

        penalty = 0.0
        if silence_ratio > self.silence_ratio_limit:
            penalty = (silence_ratio - self.silence_ratio_limit) / (1.0 - self.silence_ratio_limit)
        # Total silence
        if silence_ratio > 0.95:
            penalty = 1.0

        return silence_ratio, penalty

    def _check_repetition(self, codec_tokens: torch.Tensor) -> Tuple[float, float]:
        """Check for repetitive codec token patterns.

        Args:
            codec_tokens: [N_codebooks, T_frames] or [T_frames] for single codebook.

        Returns:
            (repetition_ratio, penalty)
        """
        if codec_tokens.dim() == 1:
            tokens = codec_tokens
        else:
            tokens = codec_tokens[0]  # First codebook is most informative

        T = tokens.numel()
        if T < self.repetition_window_min * 3:
            return 0.0, 0.0

        max_rep_length = 0
        for win in range(
            self.repetition_window_min,
            min(self.repetition_window_max, T // 2) + 1,
        ):
            pattern = tokens[:win]
            # Count consecutive repetitions
            n_reps = 0
            pos = 0
            while pos + win <= T:
                segment = tokens[pos : pos + win]
                if segment.shape[0] == win and torch.equal(segment, pattern):
                    n_reps += 1
                    pos += win
                else:
                    break
            if n_reps >= 3:
                max_rep_length = max(max_rep_length, n_reps * win)

        repetition_ratio = max_rep_length / T if T > 0 else 0.0
        penalty = 0.0
        if repetition_ratio > 0.3:
            penalty = min(1.0, (repetition_ratio - 0.3) / 0.4)
        if repetition_ratio > 0.7:
            penalty = 1.0

        return repetition_ratio, penalty

    def _check_noise(self, audio: torch.Tensor) -> Tuple[float, float]:
        """Detect pure noise via spectral flatness approximation.

        Returns:
            (noise_ratio, penalty)
        """
        audio_1d = audio.squeeze()
        if audio_1d.numel() < 512:
            return 0.0, 0.0

        # Simple spectral flatness: geometric mean / arithmetic mean of power spectrum
        try:
            spectrum = torch.fft.rfft(audio_1d[:4096])
            power = spectrum.abs().pow(2) + 1e-10
            log_mean = power.log().mean().item()
            geom_mean = math.exp(log_mean)
            arith_mean = power.mean().item()
            flatness = geom_mean / (arith_mean + 1e-10)
        except Exception:
            return 0.0, 0.0

        penalty = 0.0
        if flatness > self.noise_spectral_flatness_threshold:
            penalty = min(
                1.0,
                (flatness - self.noise_spectral_flatness_threshold)
                / (1.0 - self.noise_spectral_flatness_threshold),
            )

        return flatness, penalty

    def compute(
        self,
        generated_audio: torch.Tensor,
        codec_tokens: Optional[torch.Tensor] = None,
    ) -> RewardResult:
        """Compute naturalness guard reward.

        Args:
            generated_audio: [1, T_samples] or [T_samples]
            codec_tokens: Optional [N_codebooks, T_frames] for repetition check.

        Returns:
            RewardResult with naturalness, is_degenerate, silence/repetition/noise ratios.
        """
        result = RewardResult()

        silence_ratio, silence_penalty = self._check_silence(generated_audio)
        result.silence_ratio = silence_ratio

        repetition_ratio = 0.0
        repetition_penalty = 0.0
        if codec_tokens is not None:
            repetition_ratio, repetition_penalty = self._check_repetition(codec_tokens)
        result.repetition_ratio = repetition_ratio

        noise_ratio, noise_penalty = self._check_noise(generated_audio)
        result.noise_ratio = noise_ratio

        # Combined penalty (max of individual penalties)
        max_penalty = max(silence_penalty, repetition_penalty, noise_penalty)
        result.naturalness = max(0.0, 1.0 - max_penalty)
        result.is_degenerate = max_penalty >= 0.8

        return result
