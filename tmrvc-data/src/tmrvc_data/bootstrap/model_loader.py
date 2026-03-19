"""Centralized model loader for the v4 bootstrap pipeline.

Lazy-loads all external models on first use. Shares model instances
across pipeline stages to avoid redundant GPU memory allocation.

Loading pattern follows the codebase convention:
- Lazy initialization (load on first call)
- Device management via string ("cuda"/"cpu")
- .eval() and @torch.inference_mode() for all inference
- OOM fallback: GPU -> CPU
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class BootstrapModelLoader:
    """Lazy model loader for all bootstrap pipeline external dependencies.

    Usage:
        loader = BootstrapModelLoader(device="cuda")

        # Models are loaded on first access
        text = loader.transcribe(audio_path, language="ja")
        embed = loader.extract_speaker_embedding(waveform)
        vs = loader.extract_voice_state(waveform)
    """

    def __init__(
        self,
        device: str = "cuda",
        whisper_model: str = "large-v3",
        annotation_model: str = "Qwen/Qwen3.5-9B",
    ):
        self.device = device
        self.whisper_model_name = whisper_model
        self.annotation_model_name = annotation_model

        # Lazy-loaded model instances
        self._whisper = None
        self._speaker_encoder = None
        self._voice_state_estimator = None
        self._wavlm_extractor = None
        self._codec = None
        self._annotation_llm = None
        self._annotation_tokenizer = None
        self._diarization = None

    # --- ASR ---

    def _load_whisper(self):
        if self._whisper is not None:
            return
        try:
            from faster_whisper import WhisperModel
            compute_type = "float16" if self.device == "cuda" else "int8"
            self._whisper = WhisperModel(
                self.whisper_model_name,
                device=self.device,
                compute_type=compute_type,
            )
            logger.info("Loaded Whisper '%s' on %s", self.whisper_model_name, self.device)
        except Exception as e:
            logger.warning("Failed to load Whisper on %s: %s. Trying CPU.", self.device, e)
            try:
                from faster_whisper import WhisperModel
                self._whisper = WhisperModel(
                    self.whisper_model_name, device="cpu", compute_type="int8",
                )
                logger.info("Loaded Whisper '%s' on CPU (fallback)", self.whisper_model_name)
            except Exception as e2:
                logger.error("Whisper loading failed completely: %s", e2)
                self._whisper = None

    def transcribe(self, audio_path: str, language: Optional[str] = None) -> tuple[str, float, str]:
        """Transcribe audio file.

        Returns: (text, confidence, detected_language)
        """
        self._load_whisper()
        if self._whisper is None:
            logger.warning("No ASR model available, returning empty transcript")
            return "", 0.0, ""

        try:
            segments, info = self._whisper.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            text = "".join(seg.text for seg in segments).strip()
            detected_lang = info.language if hasattr(info, "language") else (language or "")
            confidence = info.language_probability if hasattr(info, "language_probability") else 0.0
            return text, min(1.0, max(0.0, confidence)), detected_lang
        except Exception as e:
            logger.error("Transcription failed: %s", e)
            return "", 0.0, ""

    # --- Speaker Encoder ---

    def _load_speaker_encoder(self):
        if self._speaker_encoder is not None:
            return
        try:
            from tmrvc_data.speaker import SpeakerEncoder
            self._speaker_encoder = SpeakerEncoder(device=self.device)
            logger.info("Loaded SpeakerEncoder on %s", self.device)
        except Exception as e:
            logger.error("SpeakerEncoder loading failed: %s", e)
            self._speaker_encoder = None

    def extract_speaker_embedding(self, waveform: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """Extract speaker embedding [192].

        Args:
            waveform: numpy array [T_samples]
            sample_rate: sample rate

        Returns:
            Speaker embedding [192] as numpy
        """
        self._load_speaker_encoder()
        if self._speaker_encoder is None:
            logger.warning("No speaker encoder, returning zeros")
            return np.zeros(192, dtype=np.float32)

        try:
            waveform_t = torch.from_numpy(waveform).float().unsqueeze(0)
            embed = self._speaker_encoder.extract(waveform_t, sample_rate=sample_rate)
            if isinstance(embed, torch.Tensor):
                return embed.cpu().numpy().astype(np.float32)
            return np.array(embed, dtype=np.float32)
        except Exception as e:
            logger.error("Speaker embedding extraction failed: %s", e)
            return np.zeros(192, dtype=np.float32)

    # --- Voice State ---

    def _load_voice_state_estimator(self):
        if self._voice_state_estimator is not None:
            return
        try:
            from tmrvc_data.voice_state import VoiceStateEstimator
            self._voice_state_estimator = VoiceStateEstimator(device=self.device)
            logger.info("Loaded VoiceStateEstimator on %s", self.device)
        except Exception as e:
            logger.error("VoiceStateEstimator loading failed: %s", e)
            self._voice_state_estimator = None

    def extract_voice_state(
        self, waveform: np.ndarray, sample_rate: int = 24000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract 12-D voice state.

        Returns: (targets [T,12], observed_mask [T,12], confidence [T,12])
        """
        self._load_voice_state_estimator()

        n_frames = max(1, len(waveform) // 240)  # 10ms frames at 24kHz
        d_phys = 12

        if self._voice_state_estimator is None:
            return (
                np.zeros((n_frames, d_phys), dtype=np.float32),
                np.zeros((n_frames, d_phys), dtype=bool),
                np.zeros((n_frames, d_phys), dtype=np.float32),
            )

        try:
            waveform_t = torch.from_numpy(waveform).float().unsqueeze(0)
            # Compute mel for the estimator
            from tmrvc_core.audio import compute_mel
            mel = compute_mel(waveform_t).to(self.device)
            f0 = torch.zeros(1, 1, mel.shape[-1], device=torch.device(self.device))

            result = self._voice_state_estimator.estimate(mel, f0)

            if isinstance(result, torch.Tensor):
                targets = result.squeeze(0).cpu().numpy()
            elif isinstance(result, dict):
                targets = result.get("explicit_state", result.get("voice_state"))
                if isinstance(targets, torch.Tensor):
                    targets = targets.squeeze(0).cpu().numpy()
            else:
                targets = np.array(result, dtype=np.float32)

            T = targets.shape[0]
            observed_mask = np.ones((T, d_phys), dtype=bool)
            # New v4 dims (8-11) may have lower confidence
            observed_mask[:, 8:] = False  # aperiodicity, formant_shift, vocal_effort, creak not yet estimated
            confidence = np.ones((T, d_phys), dtype=np.float32) * 0.8
            confidence[:, 8:] = 0.0

            return targets.astype(np.float32), observed_mask, confidence

        except Exception as e:
            logger.error("Voice state extraction failed: %s", e)
            return (
                np.zeros((n_frames, d_phys), dtype=np.float32),
                np.zeros((n_frames, d_phys), dtype=bool),
                np.zeros((n_frames, d_phys), dtype=np.float32),
            )

    # --- Codec ---

    def _load_codec(self):
        if self._codec is not None:
            return
        try:
            from tmrvc_data.mimi_codec import MimiCodecWrapper
            self._codec = MimiCodecWrapper(device=self.device)
            logger.info("Loaded Mimi codec on %s", self.device)
        except Exception as e:
            logger.error("Mimi codec loading failed: %s", e)
            self._codec = None

    def encode_audio(self, waveform: np.ndarray, sample_rate: int = 24000) -> tuple[np.ndarray, np.ndarray]:
        """Encode audio to codec tokens using Mimi.

        Returns: (acoustic_tokens [8, T_codec], control_tokens [4, T_control])
            T_codec is at 12.5 Hz (hop=1920), T_control is at 100 Hz (hop=240).
            Control tokens are placeholder zeros (Mimi does not produce them).
        """
        self._load_codec()

        n_samples = len(waveform)
        hop_codec = 1920   # Mimi 12.5 Hz
        hop_control = 240  # 100 Hz control rate
        n_codec_frames = max(1, n_samples // hop_codec)
        n_control_frames = max(1, n_samples // hop_control)

        if self._codec is None:
            return (
                np.zeros((8, n_codec_frames), dtype=np.int64),
                np.zeros((4, n_control_frames), dtype=np.int64),
            )

        try:
            waveform_t = torch.from_numpy(waveform).float().unsqueeze(0).unsqueeze(0)  # [1, 1, T]
            codes = self._codec.encode(waveform_t)  # [1, 8, T_codec]
            a_np = codes.squeeze(0).numpy().astype(np.int64)

            # Control tokens are not produced by Mimi; return zeros at 100 Hz
            n_control = max(1, n_samples // hop_control)
            b_np = np.zeros((4, n_control), dtype=np.int64)

            return a_np, b_np
        except Exception as e:
            logger.error("Mimi codec encoding failed: %s", e)
            return (
                np.zeros((8, n_codec_frames), dtype=np.int64),
                np.zeros((4, n_control_frames), dtype=np.int64),
            )

    # --- G2P ---

    def text_to_phonemes(self, text: str, language: str = "ja") -> np.ndarray:
        """Convert text to phoneme IDs."""
        try:
            from tmrvc_data.g2p import text_to_phonemes
            result = text_to_phonemes(text, language=language)
            ids = result.phoneme_ids
            if isinstance(ids, torch.Tensor):
                return ids.cpu().numpy().astype(np.int64)
            return np.array(ids, dtype=np.int64)
        except Exception as e:
            logger.error("G2P failed: %s", e)
            return np.array([], dtype=np.int64)

    # --- Diarization ---

    def _load_diarization(self):
        if self._diarization is not None:
            return
        try:
            from pyannote.audio import Pipeline as PyannotePipeline
            self._diarization = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
            )
            if self.device == "cuda" and torch.cuda.is_available():
                self._diarization.to(torch.device("cuda"))
            logger.info("Loaded pyannote diarization pipeline")
        except Exception as e:
            logger.warning("Diarization pipeline not available: %s", e)
            self._diarization = None

    def diarize(self, audio_path: str) -> list[dict]:
        """Run speaker diarization.

        Returns: list of {speaker: str, start: float, end: float}
        """
        self._load_diarization()
        if self._diarization is None:
            return []

        try:
            output = self._diarization(audio_path)
            segments = []
            for turn, _, speaker in output.itertracks(yield_label=True):
                segments.append({
                    "speaker": speaker,
                    "start": turn.start,
                    "end": turn.end,
                })
            return segments
        except Exception as e:
            logger.error("Diarization failed: %s", e)
            return []

    # --- Semantic Annotation LLM ---

    def _load_annotation_llm(self):
        if self._annotation_llm is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._annotation_tokenizer = AutoTokenizer.from_pretrained(
                self.annotation_model_name, trust_remote_code=True,
            )
            self._annotation_llm = AutoModelForCausalLM.from_pretrained(
                self.annotation_model_name,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            self._annotation_llm.eval()
            logger.info("Loaded annotation LLM '%s'", self.annotation_model_name)
        except Exception as e:
            logger.warning("Annotation LLM not available: %s", e)
            self._annotation_llm = None
            self._annotation_tokenizer = None

    def annotate_semantic(self, text: str, language: str = "") -> dict:
        """Generate semantic acting annotations for a transcript.

        Returns: dict with scene_summary, dialogue_intent, emotion_description, acting_hint
        """
        self._load_annotation_llm()
        if self._annotation_llm is None:
            return {
                "scene_summary": "",
                "dialogue_intent": "",
                "emotion_description": "",
                "acting_hint": "",
            }

        prompt = (
            f"Analyze this speech transcript and provide acting annotations.\n"
            f"Language: {language}\n"
            f"Transcript: {text}\n\n"
            f"Respond in JSON with keys: scene_summary, dialogue_intent, emotion_description, acting_hint"
        )

        try:
            inputs = self._annotation_tokenizer(prompt, return_tensors="pt").to(
                self._annotation_llm.device
            )
            with torch.inference_mode():
                outputs = self._annotation_llm.generate(
                    **inputs, max_new_tokens=256, temperature=0.1, do_sample=False,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self._annotation_tokenizer.decode(generated, skip_special_tokens=True)

            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "scene_summary": "",
                    "dialogue_intent": "",
                    "emotion_description": response[:200],
                    "acting_hint": "",
                }
        except Exception as e:
            logger.error("Semantic annotation failed: %s", e)
            return {
                "scene_summary": "",
                "dialogue_intent": "",
                "emotion_description": "",
                "acting_hint": "",
            }

    def cleanup(self):
        """Free GPU memory by unloading all models."""
        self._whisper = None
        self._speaker_encoder = None
        self._voice_state_estimator = None
        self._wavlm_extractor = None
        self._codec = None
        self._annotation_llm = None
        self._annotation_tokenizer = None
        self._diarization = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All bootstrap models unloaded")
