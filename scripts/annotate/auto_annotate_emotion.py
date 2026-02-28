#!/usr/bin/env python3
"""Auto-annotate emotion labels using pre-trained models.

Supports:
- Hugging Face audio classification models (e.g., "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
- Japanese emotion models (e.g., "library/bert-japanese-emotion")

Usage:
    uv run python scripts/auto_annotate_emotion.py \
        --cache-dir data/cache --dataset custom_speaker \
        --audio-dir data/wav --language ja --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import tqdm

logger = logging.getLogger(__name__)

EMOTION_MAP_EN = {
    "anger": 0,
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "happiness": 3,
    "joy": 3,
    "sad": 4,
    "sadness": 4,
    "surprise": 5,
    "neutral": 6,
    "calm": 6,
    "excited": 7,
    "frustrated": 8,
    "anxious": 9,
    "apologetic": 10,
    "confident": 11,
}

EMOTION_MAP_JA = {
    "怒り": 0,
    "いかり": 0,
    "嫌悪": 1,
    "けんお": 1,
    "恐怖": 2,
    "きょうふ": 2,
    "喜び": 3,
    "よろこび": 3,
    "幸福": 3,
    "悲しみ": 4,
    "かなしみ": 4,
    "悲": 4,
    "驚き": 5,
    "おどろき": 5,
    "中立": 6,
    "通常": 6,
    "普通": 6,
    "neutral": 6,
    "興奮": 7,
    "不満": 8,
    "不安": 9,
    "謝罪": 10,
    "自信": 11,
}


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate emotion labels")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--language", default="ja", choices=["ja", "en"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default=None, help="Custom HF model name")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Select model based on language
    if args.model:
        model_name = args.model
    elif args.language == "ja":
        model_name = "library/bert-japanese-emotion"
        use_audio = False
    else:
        model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
        use_audio = True

    logger.info("Loading model: %s", model_name)

    if "wav2vec2" in model_name or "audio" in model_name:
        from transformers import pipeline

        classifier = pipeline(
            "audio-classification",
            model=model_name,
            device=0 if args.device == "cuda" else -1,
        )
        use_audio = True
    else:
        from transformers import pipeline

        classifier = pipeline(
            "text-classification",
            model=model_name,
            device=0 if args.device == "cuda" else -1,
        )
        use_audio = False

    emotion_map = EMOTION_MAP_JA if args.language == "ja" else EMOTION_MAP_EN

    cache_root = args.cache_dir / args.dataset / args.split
    if not cache_root.exists():
        logger.error("Cache root not found: %s", cache_root)
        return

    meta_paths = sorted(cache_root.rglob("meta.json"))
    logger.info("Found %d cache entries", len(meta_paths))

    updated = 0
    errors = 0

    for meta_path in tqdm.tqdm(meta_paths, desc="Annotating"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

        if args.skip_existing and "emotion_id" in meta:
            continue

        utt_id = meta.get("utterance_id", "")
        stem = utt_id.split("_")[-1]
        audio_path = args.audio_dir / f"{stem}.wav"

        if not audio_path.exists():
            errors += 1
            continue

        try:
            if use_audio:
                result = classifier(str(audio_path))
                label = result[0]["label"].lower()
            else:
                text = meta.get("text", "")
                if not text:
                    continue
                result = classifier(text)
                label = result[0]["label"].lower()

            emotion_id = emotion_map.get(label, 6)

            confidence = result[0].get("score", 1.0)

            meta["emotion_id"] = emotion_id
            meta["emotion_label"] = label
            meta["emotion_confidence"] = round(confidence, 4)

            if len(result) > 1:
                vad = [
                    0.5 + 0.3 * (emotion_id in [3, 7]),
                    0.3 + 0.4 * confidence,
                    0.5,
                ]
                meta["vad"] = [round(v, 3) for v in vad]

            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            updated += 1

        except Exception as e:
            logger.warning("Error annotating %s: %s", utt_id, e)
            errors += 1

    logger.info("Done. Updated %d entries, %d errors", updated, errors)


if __name__ == "__main__":
    main()
