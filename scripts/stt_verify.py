#!/usr/bin/env python3
"""STT verification: transcribe generated audio and check against target text.

Usage:
    .venv/bin/python scripts/stt_verify.py output/v4d_85k_greedy
"""
from __future__ import annotations
import sys
from pathlib import Path

# Target texts (must match generate_v4_sample.py style dict)
TARGETS = {
    "neutral": ("本当にありがとうございます。", "ja"),
    "angry": ("いい加減にしてよ！", "ja"),
    "whisper": ("ねえ、ちょっと聞いて。", "ja"),
    "sad": ("もう会えないのかな。", "ja"),
    "excited": ("すごい！信じられない！", "ja"),
    "tender": ("大丈夫だよ、心配しないで。", "ja"),
    "professional": ("本日の会議を始めさせていただきます。", "ja"),
    "creaky_dramatic": ("それは...嘘でしょう。", "ja"),
    "mesugaki": ("ざぁ〜こ♡ ざぁ〜こ♡ お兄さん弱すぎじゃない？", "ja"),
}


def main():
    if len(sys.argv) < 2:
        print("usage: stt_verify.py <dir>")
        sys.exit(1)

    output_dir = Path(sys.argv[1])
    from faster_whisper import WhisperModel
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    whisper = WhisperModel("large-v3", device=device, compute_type=compute)

    print(f"STT verification for {output_dir}\n")
    print(f"{'name':<20} {'target':<30} {'recognized':<30} {'conf':<6}")
    print("-" * 90)

    any_recognized = False
    for name, (target_text, lang) in TARGETS.items():
        wav_path = output_dir / f"{name}.wav"
        if not wav_path.exists():
            continue
        segs, info = whisper.transcribe(str(wav_path), beam_size=5, vad_filter=False, language=lang)
        seg_list = list(segs)
        text = "".join(s.text for s in seg_list).strip()
        conf = 0.0
        if seg_list:
            conf = sum(s.avg_logprob for s in seg_list) / len(seg_list)
            conf = max(0.0, min(1.0, 1.0 + conf / 2.0))
        is_speech = len(text) > 0 and conf > 0.3
        if is_speech:
            any_recognized = True
        marker = "✓" if is_speech else "✗"
        print(f"{name:<20} {target_text[:28]:<30} {text[:28]:<30} {conf:.2f} {marker}")

    print()
    if any_recognized:
        print("STT can recognize at least one sample as speech.")
    else:
        print("STT recognizes NONE as speech. Output is noise.")


if __name__ == "__main__":
    main()
