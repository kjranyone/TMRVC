"""Run external baseline inference for A/B comparison.

Usage:
    python scripts/eval/run_baseline.py --baseline cosyvoice2 --prompt-set eval/prompts.json --output-dir eval/baseline_outputs/

Supports: CosyVoice2, CosyVoice3, F5-TTS, MaskGCT, Qwen3-TTS.
Each runner loads the model from a configurable path and runs inference.
Outputs are saved with metadata for Evaluation Arena consumption.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class BaselineSettings:
    """Inference settings for a baseline run."""

    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    cfg_scale: float = 1.0
    decoding_mode: str = "auto"
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """Result of a single baseline synthesis."""

    audio: np.ndarray  # [samples] float32 waveform
    sample_rate: int
    baseline_id: str
    registry_baseline_id: str
    prompt_id: str
    settings: dict[str, Any]
    duration_sec: float
    inference_time_sec: float


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaselineRunner(ABC):
    """Abstract base class for external baseline inference runners."""

    baseline_id: str = "unknown"
    registry_baseline_id: str = ""  # must match baseline_id in external-baseline-registry.md

    @abstractmethod
    def load_model(self, model_path: str | None = None) -> None:
        """Load the baseline model from disk or download default weights.

        Args:
            model_path: Optional explicit path to model checkpoint.
                        If None, use the default model location.
        """
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
        settings: BaselineSettings | None = None,
    ) -> np.ndarray:
        """Run TTS/VC inference and return audio waveform.

        Args:
            text: Text to synthesize.
            reference_audio: Optional reference audio for zero-shot voice cloning.
            reference_sr: Sample rate of reference audio.
            settings: Inference settings.

        Returns:
            Float32 waveform array.
        """
        ...

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate of this baseline."""
        ...

    def run_prompt(
        self,
        text: str,
        prompt_id: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
        settings: BaselineSettings | None = None,
    ) -> SynthesisResult:
        """Run inference and wrap in SynthesisResult with timing metadata."""
        if settings is None:
            settings = BaselineSettings()

        t0 = time.perf_counter()
        audio = self.synthesize(text, reference_audio, reference_sr, settings)
        t1 = time.perf_counter()

        return SynthesisResult(
            audio=audio,
            sample_rate=self.sample_rate,
            baseline_id=self.baseline_id,
            registry_baseline_id=self.registry_baseline_id,
            prompt_id=prompt_id,
            settings=asdict(settings),
            duration_sec=len(audio) / self.sample_rate,
            inference_time_sec=t1 - t0,
        )


# ---------------------------------------------------------------------------
# CosyVoice 2 runner
# ---------------------------------------------------------------------------

class CosyVoiceRunner(BaselineRunner):
    """CosyVoice 2 baseline runner.

    Requires: ``pip install cosyvoice``
    Model weights: https://github.com/FunAudioLLM/CosyVoice
    """

    baseline_id = "cosyvoice2"

    def __init__(self) -> None:
        self._model = None
        self._sr = 22050

    def load_model(self, model_path: str | None = None) -> None:
        try:
            from cosyvoice import CosyVoice  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "CosyVoice is not installed. Install with:\n"
                "  git clone https://github.com/FunAudioLLM/CosyVoice.git\n"
                "  cd CosyVoice && pip install -e .\n"
                "Then download model weights per the CosyVoice README."
            )

        model_dir = model_path or "pretrained_models/CosyVoice-300M"
        logger.info("Loading CosyVoice from %s", model_dir)
        self._model = CosyVoice(model_dir)

    def synthesize(
        self,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
        settings: BaselineSettings | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if reference_audio is not None:
            # Zero-shot voice cloning mode
            import torchaudio  # type: ignore[import-untyped]
            import torch
            import tempfile

            # Save reference audio to temp file for CosyVoice API
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                ref_tensor = torch.from_numpy(reference_audio).unsqueeze(0).float()
                torchaudio.save(f.name, ref_tensor, reference_sr)
                result = self._model.inference_zero_shot(text, "", f.name)
        else:
            result = self._model.inference_sft(text)

        # CosyVoice returns a generator of dicts with 'tts_speech' key
        chunks = []
        for chunk in result:
            chunks.append(chunk["tts_speech"].numpy().flatten())
        return np.concatenate(chunks).astype(np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sr


# ---------------------------------------------------------------------------
# F5-TTS runner
# ---------------------------------------------------------------------------

class F5TTSRunner(BaselineRunner):
    """F5-TTS baseline runner.

    Requires: ``pip install f5-tts``
    Paper: arXiv:2410.06885
    """

    baseline_id = "f5tts"

    def __init__(self) -> None:
        self._model = None
        self._sr = 24000

    def load_model(self, model_path: str | None = None) -> None:
        try:
            from f5_tts.api import F5TTS  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "F5-TTS is not installed. Install with:\n"
                "  pip install f5-tts\n"
                "See https://github.com/SWivid/F5-TTS for details."
            )

        logger.info("Loading F5-TTS model")
        kwargs = {}
        if model_path is not None:
            kwargs["ckpt_file"] = model_path
        self._model = F5TTS(**kwargs)

    def synthesize(
        self,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
        settings: BaselineSettings | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if reference_audio is not None:
            import tempfile
            import soundfile as sf  # type: ignore[import-untyped]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, reference_audio, reference_sr)
                audio, sr, _ = self._model.infer(
                    ref_file=f.name,
                    ref_text="",
                    gen_text=text,
                    seed=settings.seed if settings else 42,
                )
        else:
            audio, sr, _ = self._model.infer(
                ref_file="",
                ref_text="",
                gen_text=text,
                seed=settings.seed if settings else 42,
            )

        self._sr = sr
        return np.asarray(audio, dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sr


# ---------------------------------------------------------------------------
# MaskGCT runner
# ---------------------------------------------------------------------------

class MaskGCTRunner(BaselineRunner):
    """MaskGCT baseline runner.

    Requires MaskGCT to be installed from source.
    See https://github.com/open-mmlab/Amphion for details.
    """

    baseline_id = "maskgct"

    def __init__(self) -> None:
        self._pipeline = None
        self._sr = 24000

    def load_model(self, model_path: str | None = None) -> None:
        try:
            from models.tts.maskgct.maskgct_utils import prepare_models  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "MaskGCT is not installed. Install from Amphion:\n"
                "  git clone https://github.com/open-mmlab/Amphion.git\n"
                "  cd Amphion && pip install -e .\n"
                "Then download MaskGCT checkpoints per the Amphion README."
            )

        logger.info("Loading MaskGCT models")
        self._pipeline = prepare_models(model_path)

    def synthesize(
        self,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
        settings: BaselineSettings | None = None,
    ) -> np.ndarray:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        try:
            from models.tts.maskgct.maskgct_utils import maskgct_inference  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError("MaskGCT inference utils not available.")

        if reference_audio is None:
            raise ValueError("MaskGCT requires reference audio for zero-shot synthesis.")

        import tempfile
        import soundfile as sf  # type: ignore[import-untyped]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, reference_audio, reference_sr)
            audio = maskgct_inference(
                self._pipeline,
                text,
                f.name,
                target_len=None,
            )

        return np.asarray(audio, dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sr


# ---------------------------------------------------------------------------
# Qwen3-TTS runner
# ---------------------------------------------------------------------------

class Qwen3TTSRunner(BaselineRunner):
    """Qwen3-TTS baseline runner.

    Requires: transformers, Qwen3-TTS model weights from HuggingFace/ModelScope.
    Paper: arXiv:2601.15621
    """

    baseline_id = "qwen3tts_12hz_1p7b"
    registry_baseline_id = "primary_qwen3_tts_12hz_1p7b_base_hf_fd4b254"

    def __init__(self) -> None:
        self._model = None
        self._processor = None
        self._codec = None
        self._sr = 24000

    def load_model(self, model_path: str | None = None) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "transformers is not installed. Install with:\n"
                "  pip install transformers torch\n"
            )

        import torch

        model_id = model_path or "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        logger.info("Loading Qwen3-TTS from %s", model_id)

        try:
            self._processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                attn_implementation="sdpa",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Qwen3-TTS from '{model_id}'. "
                f"Ensure the model is downloaded. Error: {e}"
            )

    def synthesize(
        self,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
        settings: BaselineSettings | None = None,
    ) -> np.ndarray:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import torch

        if reference_audio is not None:
            # Voice-clone mode via create_voice_clone_prompt
            inputs = self._processor.create_voice_clone_prompt(
                text=text,
                reference_audio=torch.from_numpy(reference_audio).unsqueeze(0).float(),
                reference_sr=reference_sr,
                return_tensors="pt",
            )
        else:
            inputs = self._processor(text=text, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=settings.temperature if settings else 1.0,
                top_k=settings.top_k if settings else 50,
            )

        # Decode codec tokens to waveform
        audio = outputs.audio.squeeze().cpu().numpy()
        return audio.astype(np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sr


# ---------------------------------------------------------------------------
# CosyVoice 3 runner
# ---------------------------------------------------------------------------

class CosyVoice3Runner(BaselineRunner):
    """CosyVoice3 baseline runner (zero-shot mode).

    Requires: ``pip install cosyvoice``
    """

    baseline_id = "cosyvoice3_0p5b"
    registry_baseline_id = "secondary_fun_cosyvoice3_0p5b_2512_hf_29e01c4"

    def __init__(self) -> None:
        self._model = None
        self._sr = 22050

    def load_model(self, model_path: str | None = None) -> None:
        try:
            from cosyvoice import CosyVoice  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "CosyVoice is not installed. Install with:\n"
                "  git clone https://github.com/FunAudioLLM/CosyVoice.git\n"
                "  cd CosyVoice && pip install -e .\n"
            )

        model_id = model_path or "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
        logger.info("Loading CosyVoice3 from %s", model_id)
        self._model = CosyVoice(model_id)

    def synthesize(
        self,
        text: str,
        reference_audio: np.ndarray | None = None,
        reference_sr: int = 24000,
        settings: BaselineSettings | None = None,
    ) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if reference_audio is None:
            raise ValueError("CosyVoice3 requires reference audio for zero-shot synthesis.")

        import tempfile
        import torch
        import torchaudio  # type: ignore[import-untyped]

        stream = False
        if settings and settings.extra.get("stream"):
            stream = True

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_tensor = torch.from_numpy(reference_audio).unsqueeze(0).float()
            torchaudio.save(f.name, ref_tensor, reference_sr)
            result = self._model.inference_zero_shot(
                text, "", f.name, stream=stream,
            )

        chunks = []
        for chunk in result:
            chunks.append(chunk["tts_speech"].numpy().flatten())
        return np.concatenate(chunks).astype(np.float32)

    @property
    def sample_rate(self) -> int:
        return self._sr


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BASELINE_REGISTRY: dict[str, type[BaselineRunner]] = {
    "cosyvoice2": CosyVoiceRunner,
    "cosyvoice3_0p5b": CosyVoice3Runner,
    "f5tts": F5TTSRunner,
    "maskgct": MaskGCTRunner,
    "qwen3tts": Qwen3TTSRunner,
    "qwen3tts_12hz_1p7b": Qwen3TTSRunner,
}


def create_baseline_runner(baseline_id: str) -> BaselineRunner:
    """Create a baseline runner by ID.

    Args:
        baseline_id: One of the registered baseline IDs.

    Returns:
        An unloaded BaselineRunner instance. Call load_model() before use.

    Raises:
        ValueError: If baseline_id is not registered.
    """
    if baseline_id not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown baseline: {baseline_id!r}. Available: {available}"
        )
    return BASELINE_REGISTRY[baseline_id]()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _save_result(result: SynthesisResult, output_dir: Path) -> Path:
    """Save a synthesis result to disk with metadata."""
    import soundfile as sf  # type: ignore[import-untyped]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Audio file
    audio_path = output_dir / f"{result.baseline_id}_{result.prompt_id}.wav"
    sf.write(str(audio_path), result.audio, result.sample_rate)

    # Metadata
    meta = {
        "baseline_id": result.baseline_id,
        "registry_baseline_id": result.registry_baseline_id,
        "prompt_id": result.prompt_id,
        "settings": result.settings,
        "duration_sec": result.duration_sec,
        "inference_time_sec": result.inference_time_sec,
        "sample_rate": result.sample_rate,
        "audio_file": audio_path.name,
    }
    meta_path = output_dir / f"{result.baseline_id}_{result.prompt_id}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return audio_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run external baseline inference for A/B comparison."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        choices=sorted(BASELINE_REGISTRY.keys()),
        help="Baseline system to run.",
    )
    parser.add_argument(
        "--eval-set",
        default=None,
        help="Path to manifest.jsonl (frozen eval set). Each line: {item_id, target_text, reference_audio_id?, language_id}.",
    )
    parser.add_argument(
        "--prompt-set",
        default=None,
        help="Path to JSON file with prompts (legacy). Each entry: {id, text, reference_audio?}.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write baseline outputs.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional path to model checkpoint.",
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    parser.add_argument(
        "--top-k", type=int, default=50, help="Top-k sampling."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )

    args = parser.parse_args()

    if args.eval_set is None and args.prompt_set is None:
        parser.error("one of --eval-set or --prompt-set is required")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load prompts from either eval-set manifest or legacy prompt-set JSON
    prompts: list[dict] = []
    if args.eval_set is not None:
        with open(args.eval_set) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entry: dict[str, Any] = {
                    "id": row["item_id"],
                    "text": row["target_text"],
                    "language_id": row.get("language_id"),
                }
                if row.get("reference_audio_id"):
                    entry["reference_audio"] = row["reference_audio_id"]
                prompts.append(entry)
    else:
        with open(args.prompt_set) as f:
            prompts = json.load(f)

    # Create and load runner
    runner = create_baseline_runner(args.baseline)
    runner.load_model(args.model_path)

    settings = BaselineSettings(
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)

    # Run inference
    for prompt in prompts:
        prompt_id = prompt["id"]
        text = prompt["text"]

        # Load reference audio if specified
        reference_audio = None
        reference_sr = 24000
        if "reference_audio" in prompt:
            import soundfile as sf  # type: ignore[import-untyped]

            reference_audio, reference_sr = sf.read(prompt["reference_audio"])

        logger.info("Synthesizing prompt %s with %s", prompt_id, args.baseline)
        result = runner.run_prompt(
            text=text,
            prompt_id=prompt_id,
            reference_audio=reference_audio,
            reference_sr=reference_sr,
            settings=settings,
        )

        audio_path = _save_result(result, output_dir)
        logger.info(
            "  -> %s (%.2fs audio, %.3fs inference)",
            audio_path,
            result.duration_sec,
            result.inference_time_sec,
        )

    logger.info("Done. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
