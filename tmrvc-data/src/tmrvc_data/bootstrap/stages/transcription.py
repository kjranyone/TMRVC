"""Stage 7: ASR transcription with word timestamps.

Uses Qwen3-ASR-1.7B (or Whisper fallback) to transcribe each utterance.
Produces text_transcript, word timestamps, detected language, and
per-segment confidence.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from tmrvc_data.bootstrap.contracts import (
    BootstrapConfig,
    BootstrapStage,
    BootstrapUtterance,
)

logger = logging.getLogger(__name__)


class TranscriptionStage:
    """ASR transcription using Qwen3-ASR-1.7B or Whisper fallback.

    Lazy-loads the ASR model on first use.  Falls back through:
    1. Qwen3-ASR-1.7B (transformers)
    2. faster-whisper
    3. whisper (openai)
    """

    def __init__(self, config: Optional[BootstrapConfig] = None) -> None:
        self.config = config or BootstrapConfig()
        self._model = None
        self._processor = None
        self._backend: Optional[str] = None

    def process(
        self, utterances: List[BootstrapUtterance],
    ) -> List[BootstrapUtterance]:
        """Transcribe all non-rejected utterances."""
        for utt in utterances:
            if utt.is_rejected:
                utt.stage_completed = BootstrapStage.TRANSCRIPTION
                continue

            try:
                audio, sr = self._load_segment_audio(utt)
                result = self._transcribe(audio, sr)

                utt.text_transcript = result.get("transcript", "")
                utt.transcript_confidence = result.get("confidence", 0.0)
                utt.language = result.get("language", "")

            except Exception as exc:
                logger.warning(
                    "Transcription failed for %s: %s", utt.utterance_id, exc,
                )
                utt.text_transcript = ""
                utt.transcript_confidence = 0.0
                utt.warnings.append(f"transcription_error:{exc}")

            utt.stage_completed = BootstrapStage.TRANSCRIPTION

        logger.info("Transcription: processed %d utterances", len(utterances))
        return utterances

    # ------------------------------------------------------------------
    # ASR backends
    # ------------------------------------------------------------------

    def _transcribe(
        self, audio: np.ndarray, sr: int,
    ) -> Dict[str, Any]:
        """Transcribe audio using the best available backend."""
        # 1. Qwen3-ASR
        try:
            return self._transcribe_qwen3(audio, sr)
        except (ImportError, Exception) as exc:
            logger.debug("Qwen3-ASR unavailable: %s", exc)

        # 2. faster-whisper
        try:
            return self._transcribe_faster_whisper(audio, sr)
        except (ImportError, Exception) as exc:
            logger.debug("faster-whisper unavailable: %s", exc)

        # 3. openai whisper
        try:
            return self._transcribe_whisper(audio, sr)
        except (ImportError, Exception) as exc:
            logger.debug("whisper unavailable: %s", exc)

        logger.warning(
            "No ASR backend available. Install one of: "
            "transformers (for Qwen3-ASR), faster-whisper, or openai-whisper"
        )
        return {"transcript": "", "confidence": 0.0, "language": ""}

    def _transcribe_qwen3(
        self, audio: np.ndarray, sr: int,
    ) -> Dict[str, Any]:
        """Transcribe using Qwen3-ASR-1.7B."""
        import torch

        if self._model is None or self._backend != "qwen3":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            model_id = "Qwen/Qwen3-ASR-1.7B"
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32,
            ).to(self.config.device)
            self._backend = "qwen3"

        # Qwen3-ASR expects 16 kHz
        if sr != 16000:
            audio = self._resample(audio, sr, 16000)

        inputs = self._processor(
            audio, sampling_rate=16000, return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                return_timestamps=True,
                return_dict_in_generate=True,
                output_scores=True,
            )

        transcript = self._processor.batch_decode(
            generated.sequences, skip_special_tokens=True,
        )[0].strip()

        # Estimate confidence from generation scores
        if hasattr(generated, "scores") and generated.scores:
            log_probs = []
            for score in generated.scores:
                probs = torch.softmax(score, dim=-1)
                max_prob = probs.max(dim=-1).values
                log_probs.append(float(max_prob.mean().cpu()))
            confidence = float(np.mean(log_probs)) if log_probs else 0.5
        else:
            confidence = 0.5

        # Language detection from the transcript content
        language = self._detect_language(transcript)

        return {
            "transcript": transcript,
            "confidence": round(min(1.0, max(0.0, confidence)), 4),
            "language": language,
        }

    def _transcribe_faster_whisper(
        self, audio: np.ndarray, sr: int,
    ) -> Dict[str, Any]:
        """Transcribe using faster-whisper."""
        if self._model is None or self._backend != "faster_whisper":
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.config.whisper_model,
                device=self.config.device,
                compute_type="float16" if self.config.device == "cuda" else "int8",
            )
            self._backend = "faster_whisper"

        if sr != 16000:
            audio = self._resample(audio, sr, 16000)

        segments, info = self._model.transcribe(
            audio,
            language=self.config.whisper_language,
            word_timestamps=True,
            beam_size=5,
        )

        full_text_parts = []
        confidences = []
        for segment in segments:
            full_text_parts.append(segment.text.strip())
            if segment.avg_logprob is not None:
                import math
                conf = math.exp(segment.avg_logprob)
                confidences.append(min(1.0, max(0.0, conf)))

        transcript = " ".join(full_text_parts)
        confidence = float(np.mean(confidences)) if confidences else 0.5

        return {
            "transcript": transcript,
            "confidence": round(confidence, 4),
            "language": info.language or "",
        }

    def _transcribe_whisper(
        self, audio: np.ndarray, sr: int,
    ) -> Dict[str, Any]:
        """Transcribe using openai-whisper."""
        if self._model is None or self._backend != "whisper":
            import whisper

            self._model = whisper.load_model(self.config.whisper_model)
            self._backend = "whisper"

        if sr != 16000:
            audio = self._resample(audio, sr, 16000)

        # Whisper expects float32 at 16kHz
        result = self._model.transcribe(
            audio.astype(np.float32),
            language=self.config.whisper_language,
            word_timestamps=True,
        )

        transcript = result.get("text", "").strip()
        language = result.get("language", "")

        # Extract confidence from segments
        confidences = []
        for seg in result.get("segments", []):
            avg_logprob = seg.get("avg_logprob", -1.0)
            import math
            conf = math.exp(avg_logprob)
            confidences.append(min(1.0, max(0.0, conf)))

        confidence = float(np.mean(confidences)) if confidences else 0.5

        return {
            "transcript": transcript,
            "confidence": round(confidence, 4),
            "language": language,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_language(text: str) -> str:
        """Simple heuristic language detection from transcript text."""
        if not text:
            return ""

        # Check for CJK characters
        cjk_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        hiragana_count = sum(1 for ch in text if '\u3040' <= ch <= '\u309f')
        katakana_count = sum(1 for ch in text if '\u30a0' <= ch <= '\u30ff')
        hangul_count = sum(1 for ch in text if '\uac00' <= ch <= '\ud7af')
        total = len(text.replace(" ", ""))

        if total == 0:
            return ""

        if (hiragana_count + katakana_count) / total > 0.1:
            return "ja"
        if hangul_count / total > 0.3:
            return "ko"
        if cjk_count / total > 0.3:
            return "zh"

        # Default to English for Latin-script text
        latin_count = sum(1 for ch in text if 'a' <= ch.lower() <= 'z')
        if latin_count / total > 0.5:
            return "en"

        return ""

    @staticmethod
    def _load_segment_audio(utt: BootstrapUtterance) -> tuple[np.ndarray, int]:
        """Load the audio segment for an utterance."""
        path = Path(utt.audio_path)

        try:
            import soundfile as sf
            data, sr = sf.read(str(path), dtype="float32")
        except Exception:
            try:
                import torchaudio
                waveform, sr = torchaudio.load(str(path))
                data = waveform.numpy().squeeze()
            except Exception:
                from scipy.io import wavfile
                sr, data = wavfile.read(str(path))
                if data.dtype != np.float32:
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max

        if data.ndim > 1:
            data = np.mean(data, axis=0)

        if utt.segment is not None and utt.segment.end_sec > 0:
            start = int(utt.segment.start_sec * sr)
            end = int(utt.segment.end_sec * sr)
            data = data[start:end]

        return data, sr

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio."""
        if orig_sr == target_sr:
            return audio
        try:
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(orig_sr, target_sr)
            return resample_poly(audio, target_sr // g, orig_sr // g).astype(np.float32)
