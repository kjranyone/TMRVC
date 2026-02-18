"""``tmrvc-verify-cache`` — verify integrity of the feature cache.

Usage::

    tmrvc-verify-cache --cache-dir /data/cache --dataset vctk
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from tmrvc_data.cache import FeatureCache

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tmrvc-verify-cache",
        description="Verify feature cache integrity.",
    )
    parser.add_argument(
        "--cache-dir",
        required=True,
        type=Path,
        help="Feature cache root directory.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name to verify.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split name.",
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
    result = cache.verify(args.dataset, args.split)

    print(f"Dataset: {args.dataset} / {args.split}")
    print(f"  Total entries: {result['total']}")
    print(f"  Valid:         {result['valid']}")
    print(f"  Invalid:       {result['invalid']}")

    if result["invalid"] > 0:
        logger.warning("%d invalid entries found!", result["invalid"])
        sys.exit(1)
    else:
        print("All entries valid.")


if __name__ == "__main__":
    main()
