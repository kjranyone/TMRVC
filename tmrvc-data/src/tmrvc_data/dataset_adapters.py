"""Dataset adapters: uniform interface over supported corpora."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Iterator

from tmrvc_core.types import Utterance

logger = logging.getLogger(__name__)


class DatasetAdapter:
    """Base class for corpus-specific adapters."""

    name: str = ""

    def iter_utterances(self, root: Path, split: str = "train") -> Iterator[Utterance]:
        raise NotImplementedError


class VCTKAdapter(DatasetAdapter):
    """VCTK Corpus (48 kHz, 109 speakers, English).

    Supports two layouts:

    VCTK 0.92 (preferred)::

        root/
            wav48_silence_trimmed/
                p225/
                    p225_001_mic1.flac

    Older VCTK::

        root/
            wav48/
                p225/
                    p225_001.wav
    """

    name = "vctk"

    def iter_utterances(
        self, root: Path, split: str = "train"
    ) -> Iterator[Utterance]:
        wav_dir = root / "wav48_silence_trimmed"
        if not wav_dir.exists():
            wav_dir = root / "wav48"
        if not wav_dir.exists():
            raise FileNotFoundError(f"VCTK wav directory not found under {root}")

        # Detect format: 0.92 uses *_mic1.flac, older uses *.wav
        use_092 = wav_dir.name == "wav48_silence_trimmed"

        for spk_dir in sorted(wav_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            speaker_id = spk_dir.name

            if use_092:
                pattern = "*_mic1.flac"
            else:
                pattern = "*.wav"

            for wav_path in sorted(spk_dir.glob(pattern)):
                if use_092:
                    utt_id = wav_path.stem.replace("_mic1", "")
                else:
                    utt_id = wav_path.stem
                import soundfile as sf

                info = sf.info(str(wav_path))
                yield Utterance(
                    utterance_id=f"vctk_{utt_id}",
                    speaker_id=f"vctk_{speaker_id}",
                    dataset="vctk",
                    audio_path=wav_path,
                    duration_sec=info.duration,
                    sample_rate=info.samplerate,
                    language="en",
                )


class JVSAdapter(DatasetAdapter):
    """JVS Corpus (24 kHz, 100 speakers, Japanese).

    Expected layout::

        root/
            jvs001/
                parallel100/
                    wav24kHz16bit/
                        VOICEACTRESS100_001.wav
                        ...
                nonpara30/
                    wav24kHz16bit/
                        ...
    """

    name = "jvs"

    def iter_utterances(
        self, root: Path, split: str = "train"
    ) -> Iterator[Utterance]:
        for spk_dir in sorted(root.iterdir()):
            if not spk_dir.is_dir() or not spk_dir.name.startswith("jvs"):
                continue
            speaker_id = spk_dir.name

            # Enumerate all subdirectories with wav files
            for subset in ["parallel100", "nonpara30"]:
                wav_dir = spk_dir / subset / "wav24kHz16bit"
                if not wav_dir.exists():
                    continue
                for wav_path in sorted(wav_dir.glob("*.wav")):
                    import soundfile as sf

                    info = sf.info(str(wav_path))
                    utt_id = f"{speaker_id}_{subset}_{wav_path.stem}"
                    yield Utterance(
                        utterance_id=f"jvs_{utt_id}",
                        speaker_id=f"jvs_{speaker_id}",
                        dataset="jvs",
                        audio_path=wav_path,
                        duration_sec=info.duration,
                        sample_rate=info.samplerate,
                        language="ja",
                    )


class LibriTTSRAdapter(DatasetAdapter):
    """LibriTTS-R (24 kHz, ~2456 speakers, English).

    Expected layout::

        root/
            train-clean-100/
                19/
                    198/
                        19_198_000000_000000.wav
    """

    name = "libritts_r"

    def iter_utterances(
        self, root: Path, split: str = "train"
    ) -> Iterator[Utterance]:
        split_dirs = {
            "train": ["train-clean-100", "train-clean-360", "train-other-500"],
            "dev": ["dev-clean", "dev-other"],
            "test": ["test-clean", "test-other"],
        }
        dirs = split_dirs.get(split, [split])

        for split_name in dirs:
            split_dir = root / split_name
            if not split_dir.exists():
                logger.warning("Split dir %s not found, skipping", split_dir)
                continue

            for spk_dir in sorted(split_dir.iterdir()):
                if not spk_dir.is_dir():
                    continue
                speaker_id = spk_dir.name

                for chapter_dir in sorted(spk_dir.iterdir()):
                    if not chapter_dir.is_dir():
                        continue
                    for wav_path in sorted(chapter_dir.glob("*.wav")):
                        import soundfile as sf

                        info = sf.info(str(wav_path))
                        yield Utterance(
                            utterance_id=f"libritts_{wav_path.stem}",
                            speaker_id=f"libritts_{speaker_id}",
                            dataset="libritts_r",
                            audio_path=wav_path,
                            duration_sec=info.duration,
                            sample_rate=info.samplerate,
                            language="en",
                        )


class TsukuyomiAdapter(DatasetAdapter):
    """Generic recursive adapter for Tsukuyomi-chan training audio.

    Expected layout:
      - Single-speaker:
          root/*.wav
          root/**/*.flac
      - Multi-folder:
          root/<speaker-ish-folder>/**/*.wav

    Notes:
      - The first directory component (if present) is used as speaker id.
      - Unsupported characters are normalized to ``_`` for stable cache keys.
    """

    name = "tsukuyomi"
    _AUDIO_EXTS = {".wav", ".flac", ".ogg"}

    @staticmethod
    def _sanitize(value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", value).strip("_")
        return cleaned or "unknown"

    def iter_utterances(
        self, root: Path, split: str = "train"
    ) -> Iterator[Utterance]:
        if not root.exists():
            raise FileNotFoundError(f"Tsukuyomi root not found: {root}")

        audio_files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in self._AUDIO_EXTS
        )
        if not audio_files:
            raise FileNotFoundError(
                f"No audio files found under {root} (expected: {sorted(self._AUDIO_EXTS)})"
            )

        for wav_path in audio_files:
            rel = wav_path.relative_to(root)
            rel_no_ext = rel.with_suffix("")

            # Use top-level folder as speaker id when available; otherwise single-speaker default.
            speaker_raw = rel.parts[0] if len(rel.parts) > 1 else "tsukuyomi"
            speaker_id = f"tsukuyomi_{self._sanitize(speaker_raw)}"

            utt_raw = "_".join(rel_no_ext.parts)
            utt_id = f"tsukuyomi_{self._sanitize(utt_raw)}"

            import soundfile as sf

            info = sf.info(str(wav_path))
            yield Utterance(
                utterance_id=utt_id,
                speaker_id=speaker_id,
                dataset="tsukuyomi",
                audio_path=wav_path,
                duration_sec=info.duration,
                sample_rate=info.samplerate,
                language="ja",
            )


class GenericAdapter(DatasetAdapter):
    """Generic adapter: auto-detect speaker_id/**.wav structure.

    Expected layout::

        root/
        +-- speaker_A/
        |   +-- 001.wav
        |   +-- 002.wav
        +-- speaker_B/
            +-- 001.wav

    Single-speaker (flat)::

        root/
        +-- 001.wav
        +-- 002.wav

    speaker_id = directory name (or dataset name for flat layout).
    """

    name = "generic"
    _AUDIO_EXTS = {".wav", ".flac", ".ogg"}

    def __init__(
        self,
        dataset_name: str = "generic",
        language: str = "en",
        speaker_map_path: str | Path | None = None,
    ) -> None:
        self._dataset_name = dataset_name
        self._language = language
        self._speaker_map: dict[str, str] | None = None
        if speaker_map_path is not None:
            with open(speaker_map_path, encoding="utf-8") as f:
                data = json.load(f)
            self._speaker_map = data["mapping"]

    @staticmethod
    def _sanitize(value: str) -> str:
        cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", value).strip("_")
        return cleaned or "unknown"

    def iter_utterances(
        self, root: Path, split: str = "train"
    ) -> Iterator[Utterance]:
        if not root.exists():
            raise FileNotFoundError(f"Root directory not found: {root}")

        audio_files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in self._AUDIO_EXTS
        )
        if not audio_files:
            raise FileNotFoundError(
                f"No audio files found under {root} (expected: {sorted(self._AUDIO_EXTS)})"
            )

        prefix = self._sanitize(self._dataset_name)

        for wav_path in audio_files:
            rel = wav_path.relative_to(root)
            rel_no_ext = rel.with_suffix("")

            # Speaker map overrides folder-based speaker detection
            if self._speaker_map is not None:
                filename = rel.parts[-1]  # just the filename
                spk_label = self._speaker_map.get(filename)
                if spk_label is None or spk_label == "spk_noise":
                    continue
                speaker_id = f"{prefix}_{self._sanitize(spk_label)}"
            else:
                # Use top-level folder as speaker id when available
                speaker_raw = rel.parts[0] if len(rel.parts) > 1 else prefix
                speaker_id = f"{prefix}_{self._sanitize(speaker_raw)}"

            utt_raw = "_".join(rel_no_ext.parts)
            utt_id = f"{prefix}_{self._sanitize(utt_raw)}"

            import soundfile as sf

            info = sf.info(str(wav_path))
            yield Utterance(
                utterance_id=utt_id,
                speaker_id=speaker_id,
                dataset=self._dataset_name,
                audio_path=wav_path,
                duration_sec=info.duration,
                sample_rate=info.samplerate,
                language=self._language,
            )


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "vctk": VCTKAdapter,
    "jvs": JVSAdapter,
    "libritts_r": LibriTTSRAdapter,
    "tsukuyomi": TsukuyomiAdapter,
    "generic": GenericAdapter,
}


def get_adapter(
    dataset_name: str,
    *,
    adapter_type: str | None = None,
    language: str = "en",
    speaker_map_path: str | Path | None = None,
) -> DatasetAdapter:
    """Get a dataset adapter by name or explicit type.

    Args:
        dataset_name: Name of the dataset (used as key in ADAPTERS if
            *adapter_type* is not given).
        adapter_type: Explicit adapter type override.  When provided,
            ``"generic"`` creates a :class:`GenericAdapter` with
            *dataset_name* and *language*.
        language: Language hint passed to GenericAdapter (ignored for
            built-in adapters).
        speaker_map_path: Path to ``_speaker_map.json`` from
            ``cluster_speakers.py``.  Only used by :class:`GenericAdapter`.
    """
    type_key = adapter_type or dataset_name
    cls = ADAPTERS.get(type_key)
    if cls is None:
        raise ValueError(
            f"Unknown dataset/type: {type_key!r}. Available: {list(ADAPTERS)}"
        )
    if cls is GenericAdapter:
        return GenericAdapter(
            dataset_name=dataset_name,
            language=language,
            speaker_map_path=speaker_map_path,
        )
    return cls()
