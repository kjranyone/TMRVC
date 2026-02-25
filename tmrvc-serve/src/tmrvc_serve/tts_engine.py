"""TTS inference engine wrapping the full pipeline.

Loads TTS models (TextEncoder, DurationPredictor, F0Predictor, ContentSynthesizer)
and VC backend (Converter, Vocoder) to produce audio from text.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Generator
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from tmrvc_core.constants import (
    D_CONTENT,
    D_HISTORY,
    D_SCENE_STATE,
    D_SPEAKER,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    PHONEME_VOCAB_SIZE,
    N_STYLE_PARAMS,
    SAMPLE_RATE,
    TOKENIZER_VOCAB_SIZE,
    WINDOW_LENGTH,
)
from tmrvc_core.dialogue_types import CharacterProfile, DialogueTurn, StyleParams

logger = logging.getLogger(__name__)

# Crossfade/fadeout constants for sentence-level streaming
CROSSFADE_SAMPLES = 2400   # 100ms @ 24kHz
FADEOUT_SAMPLES = 1200     # 50ms @ 24kHz
SENTENCE_PAUSE_SAMPLES = 2880  # 120ms @ 24kHz (natural inter-sentence pause)
G2P_CACHE_MAX_SIZE = 256


@dataclass
class SynthesisMetrics:
    """Per-call timing breakdown for synthesis pipeline stages."""

    g2p_ms: float = 0.0
    text_encoder_ms: float = 0.0
    duration_predictor_ms: float = 0.0
    f0_predictor_ms: float = 0.0
    content_synthesizer_ms: float = 0.0
    converter_ms: float = 0.0
    vocoder_ms: float = 0.0
    istft_ms: float = 0.0
    total_ms: float = 0.0
    output_frames: int = 0
    output_duration_ms: float = 0.0
    cancelled: bool = False

    @property
    def rtf(self) -> float:
        """Real-time factor: total_ms / output_duration_ms. <1 means faster than real-time."""
        if self.output_duration_ms <= 0:
            return 0.0
        return self.total_ms / self.output_duration_ms


@dataclass
class StreamMetrics:
    """Aggregate metrics for a sentence-streaming call."""

    sentence_count: int = 0
    first_chunk_ms: float = 0.0
    total_ms: float = 0.0
    per_sentence: list[SynthesisMetrics] = field(default_factory=list)

    @property
    def avg_sentence_ms(self) -> float:
        if not self.per_sentence:
            return 0.0
        return sum(m.total_ms for m in self.per_sentence) / len(self.per_sentence)


class TTSEngine:
    """End-to-end TTS inference engine.

    Loads pre-trained TTS front-end and VC back-end models.
    Produces audio from text + speaker embedding + style params.

    Args:
        tts_checkpoint: Path to TTS checkpoint (.pt).
        vc_checkpoint: Path to VC/distill checkpoint (.pt) for Converter+Vocoder.
        device: Torch device string.
    """

    def __init__(
        self,
        tts_checkpoint: Path | str | None = None,
        vc_checkpoint: Path | str | None = None,
        device: str = "cpu",
        text_frontend: Literal["phoneme", "tokenizer", "auto"] = "tokenizer",
    ) -> None:
        self.device = torch.device(device)
        self._models_loaded = False
        self._tts_checkpoint = tts_checkpoint
        self._vc_checkpoint = vc_checkpoint
        self._warmed_up = False
        if text_frontend not in {"phoneme", "tokenizer", "auto"}:
            raise ValueError(
                f"Unsupported text_frontend: {text_frontend}. "
                "Expected one of: phoneme, tokenizer, auto."
            )
        self._text_frontend: Literal["phoneme", "tokenizer", "auto"] = text_frontend
        self._text_vocab_size = self._resolve_frontend_vocab_size(text_frontend)

        # Models (lazily loaded)
        self._text_encoder: torch.nn.Module | None = None
        self._duration_predictor: torch.nn.Module | None = None
        self._f0_predictor: torch.nn.Module | None = None
        self._content_synthesizer: torch.nn.Module | None = None
        self._converter: torch.nn.Module | None = None
        self._vocoder: torch.nn.Module | None = None
        self._scene_state_update: torch.nn.Module | None = None

        # G2P cache for speculative prefetch (LRU-bounded)
        self._g2p_cache: OrderedDict[tuple[str, str], object] = OrderedDict()

        # Per-instance lookahead pool (1 worker per engine instance)
        self._lookahead_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="tts-lookahead",
        )

        # Last metrics (accessible after synthesize / synthesize_sentences)
        self.last_metrics: SynthesisMetrics | None = None
        self.last_stream_metrics: StreamMetrics | None = None

    @staticmethod
    def _resolve_frontend_vocab_size(frontend: str) -> int:
        if frontend == "tokenizer":
            return TOKENIZER_VOCAB_SIZE
        if frontend == "auto":
            return max(PHONEME_VOCAB_SIZE, TOKENIZER_VOCAB_SIZE)
        return PHONEME_VOCAB_SIZE

    @property
    def models_loaded(self) -> bool:
        return self._models_loaded

    def load_models(self) -> None:
        """Load all models from checkpoints."""
        from tmrvc_train.models.text_encoder import TextEncoder
        from tmrvc_train.models.duration_predictor import DurationPredictor
        from tmrvc_train.models.f0_predictor import F0Predictor
        from tmrvc_train.models.content_synthesizer import ContentSynthesizer

        self._text_encoder = TextEncoder(vocab_size=self._text_vocab_size).to(self.device).eval()
        self._duration_predictor = DurationPredictor().to(self.device).eval()
        self._f0_predictor = F0Predictor().to(self.device).eval()
        self._content_synthesizer = ContentSynthesizer().to(self.device).eval()

        if self._tts_checkpoint:
            ckpt = torch.load(self._tts_checkpoint, map_location=self.device, weights_only=False)
            if not isinstance(ckpt, dict):
                raise RuntimeError("Invalid TTS checkpoint format: expected dict")
            state = ckpt
            if "text_encoder" not in state:
                raise RuntimeError("Invalid TTS checkpoint: missing text_encoder")

            missing = [
                key for key in ("text_frontend", "text_vocab_size")
                if key not in state
            ]
            if missing:
                raise RuntimeError(
                    "Legacy TTS checkpoints are not supported. Missing metadata: "
                    + ", ".join(missing)
                )

            ckpt_frontend = str(state["text_frontend"])
            if ckpt_frontend not in {"phoneme", "tokenizer"}:
                raise RuntimeError(
                    f"Unsupported checkpoint text_frontend: {ckpt_frontend}"
                )
            ckpt_vocab_size = int(state["text_vocab_size"])

            text_state = state["text_encoder"]
            embed_weight = text_state.get("phoneme_embed.weight")
            if embed_weight is None:
                raise RuntimeError(
                    "Invalid TTS checkpoint: missing text_encoder phoneme_embed.weight"
                )
            if ckpt_vocab_size != int(embed_weight.shape[0]):
                raise RuntimeError(
                    "Invalid TTS checkpoint metadata: text_vocab_size does not "
                    "match text_encoder embedding rows"
                )

            configured_frontend = getattr(self, "_text_frontend", "tokenizer")
            if configured_frontend != "auto" and configured_frontend != ckpt_frontend:
                raise RuntimeError(
                    f"Checkpoint/frontend mismatch: configured={configured_frontend}, "
                    f"checkpoint={ckpt_frontend}"
                )

            self._text_frontend = ckpt_frontend
            if ckpt_vocab_size != int(self._text_encoder.phoneme_embed.num_embeddings):
                logger.info(
                    "Rebuilding TextEncoder vocab size %d -> %d from checkpoint",
                    int(self._text_encoder.phoneme_embed.num_embeddings),
                    ckpt_vocab_size,
                )
                self._text_encoder = TextEncoder(
                    vocab_size=ckpt_vocab_size,
                ).to(self.device).eval()
            self._text_vocab_size = ckpt_vocab_size

            self._text_encoder.load_state_dict(state["text_encoder"])
            self._duration_predictor.load_state_dict(state["duration_predictor"])
            self._f0_predictor.load_state_dict(state["f0_predictor"])
            self._content_synthesizer.load_state_dict(state["content_synthesizer"])

            if state.get("enable_ssl") and "ssl_state_update" in state:
                from tmrvc_train.models.scene_state import SceneStateUpdate
                self._scene_state_update = SceneStateUpdate().to(self.device).eval()
                self._scene_state_update.load_state_dict(state["ssl_state_update"])
                logger.info("Loaded SceneStateUpdate from TTS checkpoint")

            logger.info(
                "Loaded TTS models from %s (frontend=%s, vocab=%d)",
                self._tts_checkpoint, self._text_frontend, self._text_vocab_size,
            )

        # VC backend (Converter + Vocoder) — load if checkpoint provided
        if self._vc_checkpoint:
            self._load_vc_backend()

        self._models_loaded = True
        logger.info("TTS engine ready on %s", self.device)

    def warmup(self) -> None:
        """Run dummy inference to pre-compile JIT kernels.

        Call after load_models() to eliminate first-inference latency spike.
        """
        if self._warmed_up or not self._models_loaded:
            return
        logger.info("Warming up TTS engine...")
        t0 = time.perf_counter()
        dummy_embed = torch.zeros(192, device=self.device)
        try:
            self.synthesize("warmup", "ja", dummy_embed, speed=1.0)
        except Exception as e:
            # Warmup must not block service startup. Runtime inference still
            # reports backend errors explicitly if they occur.
            logger.warning("Warmup skipped due to frontend dependency issue: %s", e)
            return
        elapsed = (time.perf_counter() - t0) * 1000
        self._warmed_up = True
        logger.info("Warmup complete in %.1fms", elapsed)

    @property
    def scene_state_available(self) -> bool:
        """Whether scene state tracking is available (SSL model loaded)."""
        return self._scene_state_update is not None

    def initial_scene_state(self) -> torch.Tensor:
        """Create zero initial scene state ``[1, D_SCENE_STATE]``."""
        return torch.zeros(1, D_SCENE_STATE, device=self.device)

    @torch.no_grad()
    def update_scene_state(
        self,
        text: str,
        language: str,
        spk_embed: torch.Tensor,
        z_prev: torch.Tensor,
    ) -> torch.Tensor:
        """Compute updated scene state after an utterance.

        Runs text frontend + text encoder to get utterance encoding,
        then applies SceneStateUpdate GRU.

        Args:
            text: The spoken text.
            language: Language code.
            spk_embed: ``[192]`` speaker embedding.
            z_prev: ``[1, D_SCENE_STATE]`` previous scene state.

        Returns:
            ``[1, D_SCENE_STATE]`` updated scene state.
        """
        if self._scene_state_update is None:
            return z_prev

        # Compute utterance encoding (mean-pooled text features)
        cache_key = (text, language)
        g2p_result = self._g2p_cache.pop(cache_key, None)
        if g2p_result is None:
            g2p_result = self._run_text_frontend(text, language=language)
        input_ids, language_id = self._extract_ids_and_lang(g2p_result, language=language)
        phoneme_ids = input_ids.unsqueeze(0).to(self.device)
        language_ids = torch.tensor([language_id], dtype=torch.long, device=self.device)
        text_features = self._text_encoder(phoneme_ids, language_ids)  # [1, 256, L]
        u_t = text_features.mean(dim=2)  # [1, 256]

        # Dialogue history summary (zeros — trained without history encoder)
        h_t = torch.zeros(1, D_HISTORY, device=self.device)

        # Speaker embedding
        s = spk_embed.to(self.device).unsqueeze(0) if spk_embed.dim() == 1 else spk_embed.to(self.device)

        z_t = self._scene_state_update(z_prev.to(self.device), u_t, h_t, s)
        return z_t

    def _load_vc_backend(self) -> None:
        """Load Converter and Vocoder from VC checkpoint."""
        from tmrvc_train.models.converter import (
            ConverterStudent,
        )
        from tmrvc_train.models.vocoder import VocoderStudent

        self._converter = ConverterStudent(
            n_acoustic_params=N_STYLE_PARAMS,
        ).to(self.device).eval()
        self._vocoder = VocoderStudent().to(self.device).eval()

        ckpt = torch.load(self._vc_checkpoint, map_location=self.device, weights_only=False)
        converter_state = ckpt.get("converter")
        if converter_state is not None:
            film_weight = converter_state.get("blocks.0.film.proj.weight")
            if film_weight is None:
                raise RuntimeError("Invalid converter state dict: missing blocks.0.film.proj.weight")

            cond_dim = int(film_weight.shape[1])
            tts_cond_dim = D_SPEAKER + N_STYLE_PARAMS

            if cond_dim != tts_cond_dim:
                raise RuntimeError(
                    f"Unsupported converter conditioning size: {cond_dim} "
                    f"(expected style-conditioned {tts_cond_dim}). "
                    "Legacy VC converter checkpoints are not supported."
                )
            self._converter.load_state_dict(converter_state)
        else:
            logger.warning("VC checkpoint has no converter state; using randomly initialized converter.")

        if "vocoder" in ckpt:
            self._vocoder.load_state_dict(ckpt["vocoder"])
        else:
            logger.warning("VC checkpoint has no vocoder state; using randomly initialized vocoder.")
        logger.info("Loaded VC backend from %s", self._vc_checkpoint)

    def _run_text_frontend(self, text: str, language: str) -> object:
        """Run configured text frontend and return frontend result object."""
        frontend = getattr(self, "_text_frontend", "phoneme")
        text_vocab_size = int(
            getattr(
                self,
                "_text_vocab_size",
                self._resolve_frontend_vocab_size(frontend),
            ),
        )

        if frontend == "tokenizer":
            from tmrvc_data.text_tokenizer import text_to_tokens
            return text_to_tokens(text, language=language)

        if frontend == "phoneme":
            from tmrvc_data.g2p import text_to_phonemes
            return text_to_phonemes(text, language=language)

        # auto mode: prefer tokenizer when vocab can represent it, else phoneme.
        if text_vocab_size >= TOKENIZER_VOCAB_SIZE:
            try:
                from tmrvc_data.text_tokenizer import text_to_tokens
                return text_to_tokens(text, language=language)
            except Exception as e:
                logger.warning("Tokenizer frontend failed in auto mode, fallback to phoneme: %s", e)
        from tmrvc_data.g2p import text_to_phonemes
        return text_to_phonemes(text, language=language)

    @staticmethod
    def _extract_ids_and_lang(frontend_result: object, language: str) -> tuple[torch.Tensor, int]:
        """Extract input IDs and language ID from frontend result."""
        token_ids = getattr(frontend_result, "token_ids", None)
        phoneme_ids = getattr(frontend_result, "phoneme_ids", None)
        ids = token_ids if token_ids is not None else phoneme_ids
        if ids is None:
            raise RuntimeError("Text frontend result has neither token_ids nor phoneme_ids")

        language_id = getattr(frontend_result, "language_id", None)
        if language_id is None:
            # Fallback for custom mocks.
            lang_id_map = {"ja": 0, "en": 1, "zh": 2, "ko": 3}
            if language not in lang_id_map:
                raise ValueError(f"Unsupported language: {language}")
            language_id = lang_id_map[language]
        return ids, int(language_id)

    def prefetch_g2p(self, text: str, language: str) -> None:
        """Run text frontend for the given text and cache the result.

        Intended for speculative prefetch while previous audio is streaming.
        Uses LRU eviction when cache exceeds ``G2P_CACHE_MAX_SIZE``.
        """
        cache_key = (text, language)
        if cache_key in self._g2p_cache:
            self._g2p_cache.move_to_end(cache_key)
            return
        self._g2p_cache[cache_key] = self._run_text_frontend(text, language=language)
        while len(self._g2p_cache) > G2P_CACHE_MAX_SIZE:
            self._g2p_cache.popitem(last=False)  # evict oldest

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        language: str,
        spk_embed: torch.Tensor,
        style: StyleParams | None = None,
        speed: float = 1.0,
        cancel: threading.Event | None = None,
    ) -> tuple[np.ndarray, float] | None:
        """Synthesize audio from text.

        Args:
            text: Input text.
            language: Language code ('ja', 'en', 'zh', 'ko').
            spk_embed: ``[192]`` speaker embedding.
            style: Style parameters (None = neutral).
            speed: Speed factor (>1 = faster, <1 = slower).
            cancel: If set, abort between pipeline stages and return None.

        Returns:
            Tuple of (audio_samples [N], duration_sec), or None if cancelled.
        """
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")

        metrics = SynthesisMetrics()
        t_total = time.perf_counter()

        # G2P (use cache if available)
        t0 = time.perf_counter()
        cache_key = (text, language)
        g2p_result = self._g2p_cache.pop(cache_key, None)
        if g2p_result is None:
            g2p_result = self._run_text_frontend(text, language=language)
        input_ids, language_id = self._extract_ids_and_lang(g2p_result, language=language)
        phoneme_ids = input_ids.unsqueeze(0).to(self.device)  # [1, L]
        metrics.g2p_ms = (time.perf_counter() - t0) * 1000

        if cancel and cancel.is_set():
            metrics.cancelled = True
            self.last_metrics = metrics
            return None

        # Language ID
        language_ids = torch.tensor([language_id], dtype=torch.long, device=self.device)

        # Text encoding
        t0 = time.perf_counter()
        text_features = self._text_encoder(phoneme_ids, language_ids)  # [1, 256, L]
        metrics.text_encoder_ms = (time.perf_counter() - t0) * 1000

        if cancel and cancel.is_set():
            metrics.cancelled = True
            self.last_metrics = metrics
            return None

        # Style vector
        if style is None:
            style = StyleParams.neutral()
        style_vec = torch.tensor(
            style.to_vector(), dtype=torch.float32, device=self.device,
        ).unsqueeze(0)  # [1, 32]

        # Duration prediction
        t0 = time.perf_counter()
        durations = self._duration_predictor(text_features, style_vec)  # [1, L]
        durations = durations / speed
        durations = torch.round(durations).long().clamp(min=1)
        metrics.duration_predictor_ms = (time.perf_counter() - t0) * 1000

        if cancel and cancel.is_set():
            metrics.cancelled = True
            self.last_metrics = metrics
            return None

        # Length regulate (expand phoneme features to frame-level)
        from tmrvc_train.models.f0_predictor import length_regulate
        expanded = length_regulate(text_features, durations.float())  # [1, 256, T]

        # F0 prediction
        t0 = time.perf_counter()
        f0, voiced_prob = self._f0_predictor(expanded, style_vec)  # [1, 1, T]
        metrics.f0_predictor_ms = (time.perf_counter() - t0) * 1000

        if cancel and cancel.is_set():
            metrics.cancelled = True
            self.last_metrics = metrics
            return None

        # Content synthesis
        t0 = time.perf_counter()
        content = self._content_synthesizer(expanded)  # [1, 256, T]
        metrics.content_synthesizer_ms = (time.perf_counter() - t0) * 1000

        if cancel and cancel.is_set():
            metrics.cancelled = True
            self.last_metrics = metrics
            return None

        T = content.shape[-1]
        metrics.output_frames = T
        duration_sec = T * HOP_LENGTH / SAMPLE_RATE
        metrics.output_duration_ms = duration_sec * 1000

        # If VC backend loaded, generate audio
        if self._converter is not None and self._vocoder is not None:
            spk = spk_embed.to(self.device).unsqueeze(0)  # [1, 192]
            # Zero acoustic params for now (TTS mode).
            from tmrvc_train.models.style_encoder import StyleEncoder
            acoustic_params = torch.zeros(1, 32, device=self.device)
            style_params = StyleEncoder.combine_style_params(acoustic_params, style_vec)

            # Converter conditioning can be VC(32) or TTS(64) depending on checkpoint.
            expected_cond_dim = D_SPEAKER + N_STYLE_PARAMS
            actual_cond_dim = int(self._converter.blocks[0].film.proj.in_features)
            if actual_cond_dim != expected_cond_dim:
                raise RuntimeError(
                    f"Unexpected converter conditioning width: {actual_cond_dim} "
                    f"(expected {expected_cond_dim})"
                )

            if cancel and cancel.is_set():
                metrics.cancelled = True
                self.last_metrics = metrics
                return None

            # Converter: content + spk_embed + cond → STFT features
            t0 = time.perf_counter()
            pred_features, _ = self._converter(content, spk, style_params)  # [1, 513, T]
            metrics.converter_ms = (time.perf_counter() - t0) * 1000

            if cancel and cancel.is_set():
                metrics.cancelled = True
                self.last_metrics = metrics
                return None

            # Vocoder: STFT features → magnitude + phase
            t0 = time.perf_counter()
            stft_mag, stft_phase, _ = self._vocoder(pred_features)  # [1, 513, T]
            metrics.vocoder_ms = (time.perf_counter() - t0) * 1000

            # iSTFT reconstruction
            t0 = time.perf_counter()
            stft_complex = stft_mag * torch.exp(1j * stft_phase)  # [1, 513, T]
            window = torch.hann_window(WINDOW_LENGTH, device=self.device)
            expected_length = T * HOP_LENGTH
            audio = torch.istft(
                stft_complex,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=WINDOW_LENGTH,
                window=window,
                center=True,
                length=expected_length,
            )  # [1, N_samples]
            audio_np = audio.squeeze(0).cpu().numpy()
            metrics.istft_ms = (time.perf_counter() - t0) * 1000
        else:
            # No VC backend — return silence placeholder
            n_samples = T * HOP_LENGTH
            audio_np = np.zeros(n_samples, dtype=np.float32)
            logger.warning("No VC backend loaded; returning silence.")

        metrics.total_ms = (time.perf_counter() - t_total) * 1000
        self.last_metrics = metrics

        logger.debug(
            "Synthesized %d frames (%.0fms audio) in %.1fms (RTF=%.2f) "
            "[g2p=%.1f enc=%.1f dur=%.1f f0=%.1f cs=%.1f conv=%.1f voc=%.1f istft=%.1f]",
            metrics.output_frames, metrics.output_duration_ms, metrics.total_ms,
            metrics.rtf,
            metrics.g2p_ms, metrics.text_encoder_ms, metrics.duration_predictor_ms,
            metrics.f0_predictor_ms, metrics.content_synthesizer_ms,
            metrics.converter_ms, metrics.vocoder_ms, metrics.istft_ms,
        )

        return audio_np, duration_sec

    def synthesize_sentences(
        self,
        text: str,
        language: str,
        spk_embed: torch.Tensor,
        style: StyleParams | None = None,
        speed: float = 1.0,
        chunk_duration_ms: int = 100,
        cancel: threading.Event | None = None,
        sentence_pause_ms: int = 120,
        auto_style: bool = True,
    ) -> Generator[np.ndarray, None, None]:
        """Sentence-level streaming synthesis with lookahead pipeline.

        Splits text into sentences, synthesizes each independently with
        1-sentence lookahead (next sentence synthesized while current is
        being yielded), inserts natural silence, and applies crossfade
        at boundaries.

        Args:
            text: Input text.
            language: Language code.
            spk_embed: Speaker embedding [192].
            style: Base style parameters (overridden per-sentence if auto_style=True).
            speed: Speed factor.
            chunk_duration_ms: Chunk size in ms for yielded audio.
            cancel: If set, abort and stop yielding.
            sentence_pause_ms: Silence between sentences in ms (0 to disable).
            auto_style: Infer per-sentence style from text content.

        Yields:
            numpy arrays of shape [chunk_samples] (float32).
        """
        from tmrvc_core.text_utils import infer_sentence_style, segment_sentences

        t_stream_start = time.perf_counter()
        stream_metrics = StreamMetrics()

        sentences = segment_sentences(text, language)
        stream_metrics.sentence_count = len(sentences)
        chunk_samples = int(SAMPLE_RATE * chunk_duration_ms / 1000)
        pause_samples = int(SAMPLE_RATE * sentence_pause_ms / 1000)

        # Determine per-sentence styles
        if auto_style and style is not None:
            styles = [infer_sentence_style(s, language, style) for s in sentences]
        elif auto_style:
            base = StyleParams.neutral()
            styles = [infer_sentence_style(s, language, base) for s in sentences]
        else:
            styles = [style] * len(sentences)

        prev_tail: np.ndarray | None = None
        first_chunk_emitted = False

        # Lookahead: submit next sentence synthesis while yielding current.
        # We use a single-thread executor to avoid GIL contention with
        # the main synthesis (which also uses this thread).
        lookahead_future: Future | None = None

        def _synth_sentence(sent: str, sent_style: StyleParams | None) -> tuple[np.ndarray, float] | None:
            return self.synthesize(sent, language, spk_embed, sent_style, speed, cancel=cancel)

        for sent_idx, sentence in enumerate(sentences):
            if cancel and cancel.is_set():
                if lookahead_future is not None:
                    lookahead_future.cancel()
                return

            # Get current sentence's audio
            if lookahead_future is not None:
                # Wait for lookahead to complete
                result = lookahead_future.result()
                lookahead_future = None
            else:
                # First sentence (or no lookahead): synthesize synchronously
                result = _synth_sentence(sentence, styles[sent_idx])

            if result is None:
                return
            audio, _ = result

            # Collect metrics
            if self.last_metrics is not None:
                stream_metrics.per_sentence.append(self.last_metrics)

            is_last = sent_idx == len(sentences) - 1

            # Launch lookahead for next sentence
            if not is_last and (cancel is None or not cancel.is_set()):
                next_idx = sent_idx + 1
                next_sent = sentences[next_idx]
                next_style = styles[next_idx]
                # Prefetch G2P for the next sentence
                self.prefetch_g2p(next_sent, language)
                # Submit synthesis to thread pool
                lookahead_future = self._lookahead_pool.submit(
                    _synth_sentence, next_sent, next_style,
                )

            # Apply crossfade or silence gap with previous sentence's tail
            if prev_tail is not None:
                if pause_samples > 0:
                    # With inter-sentence pause: emit prev_tail with fadeout,
                    # then silence, then let next sentence start fresh.
                    # Do NOT crossfade audio-through-silence (would create
                    # an unintended fade-in from silence).
                    n_fade = min(FADEOUT_SAMPLES, len(prev_tail))
                    faded_tail = prev_tail.copy()
                    faded_tail[-n_fade:] *= np.linspace(
                        1.0, 0.0, n_fade, dtype=np.float32,
                    )
                    to_emit = np.concatenate([
                        faded_tail,
                        np.zeros(pause_samples, dtype=np.float32),
                    ])
                    for start in range(0, len(to_emit), chunk_samples):
                        if cancel and cancel.is_set():
                            return
                        chunk = to_emit[start:start + chunk_samples]
                        if not first_chunk_emitted:
                            stream_metrics.first_chunk_ms = (time.perf_counter() - t_stream_start) * 1000
                            first_chunk_emitted = True
                        yield chunk
                    prev_tail = None
                elif len(audio) >= CROSSFADE_SAMPLES:
                    # No pause: crossfade adjacent sentences directly
                    if len(prev_tail) >= CROSSFADE_SAMPLES:
                        emit_prefix = prev_tail[:-CROSSFADE_SAMPLES]
                        tail_xfade = prev_tail[-CROSSFADE_SAMPLES:]
                    else:
                        emit_prefix = np.array([], dtype=np.float32)
                        tail_xfade = prev_tail

                    n_xfade = len(tail_xfade)
                    fade_out = np.linspace(1.0, 0.0, n_xfade, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, n_xfade, dtype=np.float32)
                    audio[:n_xfade] = (
                        tail_xfade * fade_out + audio[:n_xfade] * fade_in
                    )

                    for start in range(0, len(emit_prefix), chunk_samples):
                        if cancel and cancel.is_set():
                            return
                        chunk = emit_prefix[start:start + chunk_samples]
                        if not first_chunk_emitted:
                            stream_metrics.first_chunk_ms = (time.perf_counter() - t_stream_start) * 1000
                            first_chunk_emitted = True
                        yield chunk
                    prev_tail = None
                else:
                    # Audio shorter than crossfade window: blend what we can
                    n = min(len(prev_tail), len(audio))
                    fade_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
                    audio[:n] = prev_tail[:n] * fade_out + audio[:n] * fade_in
                    if len(prev_tail) > n:
                        remainder = prev_tail[n:]
                        for start in range(0, len(remainder), chunk_samples):
                            if cancel and cancel.is_set():
                                return
                            yield remainder[start:start + chunk_samples]
                    prev_tail = None

            # Determine tail for next crossfade
            if is_last:
                emit = audio
            else:
                if len(audio) > CROSSFADE_SAMPLES:
                    emit = audio[:-CROSSFADE_SAMPLES]
                    prev_tail = audio[-CROSSFADE_SAMPLES:].copy()
                else:
                    prev_tail = audio.copy()
                    continue

            # Yield in chunks
            for start in range(0, len(emit), chunk_samples):
                if cancel and cancel.is_set():
                    return
                chunk = emit[start:start + chunk_samples]
                if not first_chunk_emitted:
                    stream_metrics.first_chunk_ms = (time.perf_counter() - t_stream_start) * 1000
                    first_chunk_emitted = True
                yield chunk

        # Flush any remaining tail
        if prev_tail is not None:
            for start in range(0, len(prev_tail), chunk_samples):
                chunk = prev_tail[start:start + chunk_samples]
                if not first_chunk_emitted:
                    stream_metrics.first_chunk_ms = (time.perf_counter() - t_stream_start) * 1000
                    first_chunk_emitted = True
                yield chunk

        stream_metrics.total_ms = (time.perf_counter() - t_stream_start) * 1000
        self.last_stream_metrics = stream_metrics

        logger.debug(
            "Stream: %d sentences, first_chunk=%.1fms, total=%.1fms, avg_sentence=%.1fms",
            stream_metrics.sentence_count,
            stream_metrics.first_chunk_ms,
            stream_metrics.total_ms,
            stream_metrics.avg_sentence_ms,
        )

    def synthesize_chunks(
        self,
        text: str,
        language: str,
        spk_embed: torch.Tensor,
        style: StyleParams | None = None,
        speed: float = 1.0,
        chunk_duration_ms: int = 100,
    ) -> Generator[np.ndarray, None, None]:
        """Synthesize audio and yield fixed-size PCM chunks.

        Runs the full synthesis pipeline, then splits the result into
        chunks of ``chunk_duration_ms`` milliseconds. The last chunk may
        be shorter.

        Yields:
            numpy arrays of shape ``[chunk_samples]`` (float32).
        """
        result = self.synthesize(text, language, spk_embed, style, speed)
        if result is None:
            return
        audio, _ = result
        chunk_samples = int(SAMPLE_RATE * chunk_duration_ms / 1000)
        for start in range(0, len(audio), chunk_samples):
            yield audio[start : start + chunk_samples]
