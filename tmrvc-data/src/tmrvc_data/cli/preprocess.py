"""``tmrvc-preprocess`` — full preprocessing pipeline.

Usage::

    tmrvc-preprocess --dataset vctk --raw-dir /data/vctk --cache-dir /data/cache
    tmrvc-preprocess --dataset jvs  --raw-dir /data/jvs  --cache-dir /data/cache
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import torch

from tmrvc_core.audio import compute_mel
from tmrvc_core.constants import HOP_LENGTH, SAMPLE_RATE
from tmrvc_core.types import FeatureSet
from tmrvc_data.cache import FeatureCache
from tmrvc_data.dataset_adapters import get_adapter
from tmrvc_data.features import ContentVecExtractor, create_f0_extractor
from tmrvc_data.preprocessing import preprocess_audio, segment_utterance
from tmrvc_data.speaker import SpeakerEncoder

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-preprocess",
        description="Preprocess raw audio and extract features to cache.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["vctk", "jvs", "libritts_r", "tsukuyomi"],
        help="Dataset name.",
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        type=Path,
        help="Path to raw dataset root.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Path to feature cache directory.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name (default: train).",
    )
    parser.add_argument(
        "--f0-method",
        default="torchcrepe",
        choices=["torchcrepe", "rmvpe"],
        help="F0 extraction method.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for model inference (default: cpu).",
    )
    parser.add_argument(
        "--max-utterances",
        type=int,
        default=0,
        help="Max utterances to process (0=all, for debugging).",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=1.0,
        help="Fraction of utterances to process (0.0-1.0, default: 1.0=all).",
    )
    parser.add_argument(
        "--segment-min-sec",
        type=float,
        default=None,
        help="Override minimum segment duration in seconds (default: constants.yaml).",
    )
    parser.add_argument(
        "--segment-max-sec",
        type=float,
        default=None,
        help="Override maximum segment duration in seconds (default: constants.yaml).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip utterances already in cache.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    adapter = get_adapter(args.dataset)
    cache = FeatureCache(args.cache_dir)

    # Collect utterances (needed for subset sampling)
    all_utterances = list(adapter.iter_utterances(args.raw_dir, args.split))
    total = len(all_utterances)
    if args.subset < 1.0:
        k = max(1, int(total * args.subset))
        all_utterances = sorted(
            random.sample(all_utterances, k),
            key=lambda u: u.utterance_id,
        )
        logger.info("Subset %.0f%%: %d / %d utterances selected",
                     args.subset * 100, k, total)
    else:
        logger.info("Total utterances: %d", total)

    # Segment duration overrides
    seg_kwargs: dict = {}
    if args.segment_min_sec is not None:
        seg_kwargs["min_sec"] = args.segment_min_sec
    if args.segment_max_sec is not None:
        seg_kwargs["max_sec"] = args.segment_max_sec

    logger.info("Loading extractors on %s ...", args.device)
    content_extractor = ContentVecExtractor(device=args.device)
    f0_extractor = create_f0_extractor(args.f0_method, device=args.device)
    spk_encoder = SpeakerEncoder(device=args.device)

    processed = 0
    skipped = 0
    errors = 0

    for utt in all_utterances:
        if 0 < args.max_utterances <= processed:
            break

        if args.skip_existing and cache.exists(
            args.dataset, args.split, utt.speaker_id, utt.utterance_id
        ):
            skipped += 1
            continue

        try:
            # 1. Load and preprocess
            waveform, sr = preprocess_audio(str(utt.audio_path))

            # 2. Segment if too long
            for seg_idx, segment in enumerate(segment_utterance(waveform, **seg_kwargs)):
                seg_id = (
                    f"{utt.utterance_id}_seg{seg_idx}" if seg_idx > 0
                    else utt.utterance_id
                )

                # 3. Extract features
                mel = compute_mel(segment)  # [1, 80, T]
                mel = mel.squeeze(0)        # [80, T]
                n_frames = mel.shape[1]

                content = content_extractor.extract(segment, sr)  # [768, T']
                # Align to mel frame count
                if content.shape[1] != n_frames:
                    content = torch.nn.functional.interpolate(
                        content.unsqueeze(0),
                        size=n_frames,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)

                f0 = f0_extractor.extract(segment, sr)  # [1, T']
                if f0.shape[1] != n_frames:
                    f0 = torch.nn.functional.interpolate(
                        f0.unsqueeze(0),
                        size=n_frames,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)

                spk_embed = spk_encoder.extract(segment, sr)  # [192]

                # 4. Save to cache
                features = FeatureSet(
                    mel=mel,
                    content=content,
                    f0=f0,
                    spk_embed=spk_embed,
                    utterance_id=seg_id,
                    speaker_id=utt.speaker_id,
                    n_frames=n_frames,
                )
                cache.save(features, args.dataset, args.split)
                processed += 1

                if processed % 100 == 0:
                    logger.info(
                        "Processed %d utterances (skipped %d, errors %d)",
                        processed, skipped, errors,
                    )

        except Exception:
            logger.error("Failed to process %s", utt.utterance_id, exc_info=True)
            errors += 1

    logger.info(
        "Done. Processed=%d, Skipped=%d, Errors=%d",
        processed, skipped, errors,
    )


if __name__ == "__main__":
    main()
