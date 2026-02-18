"""``tmrvc-extract-features`` — extract features from already-preprocessed audio.

This is a lighter variant of ``tmrvc-preprocess`` that skips the audio
preprocessing step and directly extracts features from cached audio or
from a directory of preprocessed .wav files.

Usage::

    tmrvc-extract-features --audio-dir /data/preprocessed/vctk \\
        --cache-dir /data/cache --dataset vctk
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from tmrvc_core.audio import compute_mel
from tmrvc_core.constants import HOP_LENGTH, SAMPLE_RATE
from tmrvc_core.types import FeatureSet
from tmrvc_data.cache import FeatureCache
from tmrvc_data.features import ContentVecExtractor, create_f0_extractor
from tmrvc_data.preprocessing import load_and_resample
from tmrvc_data.speaker import SpeakerEncoder

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-extract-features",
        description="Extract features from preprocessed audio files.",
    )
    parser.add_argument(
        "--audio-dir",
        required=True,
        type=Path,
        help="Directory with preprocessed .wav files (speaker/utt.wav layout).",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Output cache directory.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name label.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name.",
    )
    parser.add_argument(
        "--f0-method",
        default="torchcrepe",
        choices=["torchcrepe", "rmvpe"],
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cache = FeatureCache(args.cache_dir)
    content_extractor = ContentVecExtractor(device=args.device)
    f0_extractor = create_f0_extractor(args.f0_method, device=args.device)
    spk_encoder = SpeakerEncoder(device=args.device)

    processed = 0
    audio_dir = args.audio_dir

    for spk_dir in sorted(audio_dir.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker_id = spk_dir.name

        for wav_path in sorted(spk_dir.glob("*.wav")):
            utt_id = wav_path.stem

            if args.skip_existing and cache.exists(
                args.dataset, args.split, speaker_id, utt_id
            ):
                continue

            try:
                waveform, sr = load_and_resample(wav_path)

                mel = compute_mel(waveform).squeeze(0)  # [80, T]
                n_frames = mel.shape[1]

                content = content_extractor.extract(waveform, sr)
                if content.shape[1] != n_frames:
                    content = torch.nn.functional.interpolate(
                        content.unsqueeze(0),
                        size=n_frames,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)

                f0 = f0_extractor.extract(waveform, sr)
                if f0.shape[1] != n_frames:
                    f0 = torch.nn.functional.interpolate(
                        f0.unsqueeze(0),
                        size=n_frames,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)

                spk_embed = spk_encoder.extract(waveform, sr)

                features = FeatureSet(
                    mel=mel,
                    content=content,
                    f0=f0,
                    spk_embed=spk_embed,
                    utterance_id=utt_id,
                    speaker_id=speaker_id,
                    n_frames=n_frames,
                )
                cache.save(features, args.dataset, args.split)
                processed += 1

                if processed % 50 == 0:
                    logger.info("Extracted features for %d utterances", processed)

            except Exception:
                logger.error("Failed: %s/%s", speaker_id, utt_id, exc_info=True)

    logger.info("Done. Extracted features for %d utterances.", processed)


if __name__ == "__main__":
    main()
