#!/usr/bin/env python3
"""Apply pseudo-emotion labels to unlabeled datasets.

Two-step pipeline:
1. Train a lightweight emotion classifier on labeled data
2. Apply it to unlabeled data with confidence filtering

Usage::

    # Step 1: Train classifier on labeled emotion datasets
    python scripts/apply_pseudo_labels.py train \\
        --cache-dir data/cache \\
        --datasets expresso,jvnv,emov_db,ravdess \\
        --output checkpoints/emotion_cls.pt \\
        --device xpu

    # Step 2: Apply pseudo-labels to unlabeled datasets
    python scripts/apply_pseudo_labels.py label \\
        --cache-dir data/cache \\
        --classifier checkpoints/emotion_cls.pt \\
        --datasets vctk,jvs \\
        --confidence 0.8 \\
        --device xpu
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pseudo-labeling pipeline for emotion data augmentation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Train sub-command
    train_p = sub.add_parser("train", help="Train emotion classifier on labeled data.")
    train_p.add_argument("--cache-dir", required=True, type=Path)
    train_p.add_argument(
        "--datasets", required=True,
        help="Comma-separated labeled emotion datasets (e.g. expresso,jvnv,emov_db,ravdess).",
    )
    train_p.add_argument("--output", required=True, type=Path, help="Output checkpoint path.")
    train_p.add_argument("--max-steps", type=int, default=10000)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--device", default="cpu")
    train_p.add_argument("-v", "--verbose", action="store_true")

    # Label sub-command
    label_p = sub.add_parser("label", help="Apply pseudo-labels to unlabeled data.")
    label_p.add_argument("--cache-dir", required=True, type=Path)
    label_p.add_argument("--classifier", required=True, type=Path, help="Trained classifier checkpoint.")
    label_p.add_argument(
        "--datasets", required=True,
        help="Comma-separated target datasets (e.g. vctk,jvs).",
    )
    label_p.add_argument("--split", default="train")
    label_p.add_argument("--confidence", type=float, default=0.8, help="Confidence threshold.")
    label_p.add_argument("--batch-size", type=int, default=32)
    label_p.add_argument("--overwrite", action="store_true")
    label_p.add_argument("--device", default="cpu")
    label_p.add_argument("-v", "--verbose", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> None:
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8")

    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.command == "train":
        from tmrvc_data.pseudo_label import train_emotion_classifier

        datasets = [d.strip() for d in args.datasets.split(",")]
        train_emotion_classifier(
            cache_dir=args.cache_dir,
            datasets=datasets,
            output_path=args.output,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

    elif args.command == "label":
        from tmrvc_data.pseudo_label import PseudoLabeler

        labeler = PseudoLabeler(
            classifier_ckpt=args.classifier,
            confidence_threshold=args.confidence,
            device=args.device,
        )
        datasets = [d.strip() for d in args.datasets.split(",")]
        for ds in datasets:
            logger.info("Labeling dataset: %s", ds)
            stats = labeler.label_dataset(
                cache_dir=args.cache_dir,
                dataset=ds,
                split=args.split,
                overwrite=args.overwrite,
                batch_size=args.batch_size,
            )
            logger.info(
                "%s: labeled %d/%d (%.1f%%), low_conf=%d",
                ds, stats.labeled, stats.total,
                stats.label_rate * 100, stats.skipped_low_confidence,
            )


if __name__ == "__main__":
    main()
