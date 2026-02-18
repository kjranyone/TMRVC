"""Dataset adapters: uniform interface over VCTK, JVS, LibriTTS-R."""

from __future__ import annotations

import logging
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


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "vctk": VCTKAdapter,
    "jvs": JVSAdapter,
    "libritts_r": LibriTTSRAdapter,
}


def get_adapter(dataset_name: str) -> DatasetAdapter:
    """Get a dataset adapter by name."""
    cls = ADAPTERS.get(dataset_name)
    if cls is None:
        raise ValueError(
            f"Unknown dataset: {dataset_name!r}. Available: {list(ADAPTERS)}"
        )
    return cls()
