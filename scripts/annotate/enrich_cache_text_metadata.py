#!/usr/bin/env python3
"""Backfill text metadata into cache/meta.json from raw corpora."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

LANG_ID = {"ja": 0, "en": 1, "zh": 2, "ko": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill cache meta.json with text/language fields from raw corpora.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["jvs", "vctk", "tsukuyomi"],
        choices=["jvs", "vctk", "tsukuyomi"],
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _parse_jvs_transcript_file(path: Path) -> dict[str, str]:
    """Parse JVS transcript file robustly (works for 1-line/line-separated variants)."""
    if not path.exists():
        return {}
    text = _read_text_file(path)
    if not text:
        return {}

    pattern = re.compile(r"([A-Za-z0-9_-]+):")
    matches = list(pattern.finditer(text))
    result: dict[str, str] = {}
    for i, m in enumerate(matches):
        key = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        value = text[start:end].strip()
        if value:
            result[key] = value
    return result


def build_jvs_map(raw_dir: Path) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Build mapping for JVS cache keys and fallback VOICEACTRESS IDs."""
    root = raw_dir / "jvs_corpus" / "jvs_ver1"
    if not root.exists():
        return {}, {}

    per_cache_key: dict[tuple[str, str], str] = {}
    by_voiceactress_id: dict[str, str] = {}

    for spk_dir in sorted(root.iterdir()):
        if not spk_dir.is_dir() or not spk_dir.name.startswith("jvs"):
            continue
        spk = spk_dir.name
        for subset in ("parallel100", "nonpara30"):
            tx = _parse_jvs_transcript_file(spk_dir / subset / "transcripts_utf8.txt")
            for utt_id, sentence in tx.items():
                cache_speaker = f"jvs_{spk}"
                cache_utt = f"jvs_{spk}_{subset}_{utt_id}"
                per_cache_key[(cache_speaker, cache_utt)] = sentence
                if utt_id.startswith("VOICEACTRESS100_"):
                    by_voiceactress_id.setdefault(utt_id, sentence)
    return per_cache_key, by_voiceactress_id


def build_vctk_map(raw_dir: Path) -> dict[tuple[str, str], str]:
    root_candidates = [
        raw_dir / "VCTK-Corpus" / "VCTK-Corpus" / "txt",
        raw_dir / "VCTK-Corpus" / "txt",
    ]
    txt_root = next((p for p in root_candidates if p.exists()), None)
    if txt_root is None:
        return {}

    mapping: dict[tuple[str, str], str] = {}
    for spk_dir in sorted(txt_root.iterdir()):
        if not spk_dir.is_dir():
            continue
        spk = spk_dir.name
        for txt_file in sorted(spk_dir.glob("*.txt")):
            utt = txt_file.stem
            sentence = _read_text_file(txt_file)
            if not sentence:
                continue
            mapping[(f"vctk_{spk}", f"vctk_{utt}")] = sentence
    return mapping


def _load_meta(meta_path: Path) -> dict:
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _dump_meta(meta_path: Path, meta: dict) -> None:
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)


def update_dataset(
    cache_dir: Path,
    dataset: str,
    split: str,
    *,
    jvs_map: dict[tuple[str, str], str],
    vctk_map: dict[tuple[str, str], str],
    voiceact_map: dict[str, str],
    dry_run: bool,
) -> dict[str, int]:
    base = cache_dir / dataset / split
    stats = {"total": 0, "updated": 0, "missing_text": 0, "already": 0}
    if not base.exists():
        return stats

    for spk_dir in sorted(base.iterdir()):
        if not spk_dir.is_dir():
            continue
        speaker = spk_dir.name
        for utt_dir in sorted(spk_dir.iterdir()):
            if not utt_dir.is_dir():
                continue
            meta_path = utt_dir / "meta.json"
            if not meta_path.exists():
                continue
            stats["total"] += 1
            utter = utt_dir.name
            meta = _load_meta(meta_path)

            text: str | None = None
            language: str | None = None
            if dataset == "jvs":
                text = jvs_map.get((speaker, utter))
                language = "ja"
            elif dataset == "vctk":
                text = vctk_map.get((speaker, utter))
                language = "en"
            elif dataset == "tsukuyomi":
                m = re.search(r"(VOICEACTRESS100_\d{3})", utter)
                if m:
                    text = voiceact_map.get(m.group(1))
                language = "ja"

            if not text:
                stats["missing_text"] += 1
                continue

            needs_write = (
                meta.get("text") != text
                or meta.get("language") != language
                or meta.get("language_id") != LANG_ID[language]
            )
            if not needs_write:
                stats["already"] += 1
                continue

            meta["text"] = text
            meta["language"] = language
            meta["language_id"] = LANG_ID[language]
            if not dry_run:
                _dump_meta(meta_path, meta)
            stats["updated"] += 1
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    jvs_map, voiceact_map = build_jvs_map(args.raw_dir)
    vctk_map = build_vctk_map(args.raw_dir)
    logger.info(
        "Loaded transcript maps: jvs=%d, voiceact=%d, vctk=%d",
        len(jvs_map), len(voiceact_map), len(vctk_map),
    )

    total = {"total": 0, "updated": 0, "missing_text": 0, "already": 0}
    for ds in args.datasets:
        st = update_dataset(
            args.cache_dir,
            ds,
            args.split,
            jvs_map=jvs_map,
            vctk_map=vctk_map,
            voiceact_map=voiceact_map,
            dry_run=args.dry_run,
        )
        logger.info(
            "[%s] total=%d updated=%d already=%d missing=%d",
            ds, st["total"], st["updated"], st["already"], st["missing_text"],
        )
        for k in total:
            total[k] += st[k]

    logger.info(
        "Done total=%d updated=%d already=%d missing=%d dry_run=%s",
        total["total"], total["updated"], total["already"], total["missing_text"], args.dry_run,
    )


if __name__ == "__main__":
    main()
