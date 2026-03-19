# Track: v4 Idempotent Workflow

## Status: Active (2026-03-18)

## Problem

現在のパイプラインは冪等性がない:

- キャッシュ生成と学習が一体化 (`train_v4_full.py`)
- vocal event annotation が後付けPass
- 各フェーズの完了状態を追跡していない
- 途中で止まると再開できない

## 設計原則

1. **各フェーズが独立して冪等に実行可能**
2. **utterance 単位で処理状態を管理** — meta.json に処理済みフラグ
3. **どこで止まっても `dev.py` から再開可能**
4. **フェーズ間の依存関係が明示的**

## ワークフロー定義

```
dev.py 1 (bootstrap)
  ├── Phase A: Audio Ingest + ASR + G2P
  │     入力: data/raw/**/*.wav
  │     出力: meta.json + phoneme_ids.npy + spk_embed.npy + voice_state.npy
  │     冪等: meta.json に "phase_a_done": true
  │
  ├── Phase B: Codec Tokenization (EnCodec)
  │     入力: Phase A の waveform
  │     出力: codec_tokens.npy
  │     冪等: codec_tokens.npy の存在チェック
  │
  ├── Phase C: Vocal Event Detection (DSP)
  │     入力: waveform
  │     出力: meta.json に vocal_events 追加
  │     冪等: meta.json に "phase_c_done": true
  │
  └── Phase D: LLM Annotation + Enriched Transcript
        入力: transcript + vocal_events
        出力: meta.json に acting_annotations + enriched_transcript
        冪等: meta.json に "phase_d_done": true
        VRAM管理: Phase A-C のモデルをアンロード後に実行

dev.py 4 (確認)
  ├── tier summary
  ├── enriched transcript preview
  ├── vocal event stats
  └── cache integrity check

dev.py 2 (train)
  ├── キャッシュ済みデータのみ使用 (bootstrap 不実行)
  ├── 入力: data/cache/v4full/ (Phase A-D 完了済み utterances のみ)
  └── 出力: checkpoints/v4_full/
```

## Phase 状態管理

各 utterance の meta.json に処理状態を記録:

```json
{
  "utterance_id": "v4full_000001",
  "phases": {
    "a_ingest": {"done": true, "timestamp": "2026-03-18T12:00:00"},
    "b_codec": {"done": true, "timestamp": "2026-03-18T12:01:00"},
    "c_vocal_events": {"done": true, "timestamp": "2026-03-18T12:02:00"},
    "d_llm_annotation": {"done": true, "timestamp": "2026-03-18T12:05:00"}
  }
}
```

`dev.py 1` は各 utterance の phases を見て、未完了フェーズのみ実行する。

## VRAM 管理

RTX 2080 Ti (22GB) の制約:

- Phase A: Whisper (2GB) + SpeakerEncoder (0.5GB) + VoiceState (0.5GB) = ~3GB
- Phase B: EnCodec (0.1GB) = ~0.1GB — Phase A と同時可能
- Phase C: DSP only (0GB GPU) — CPU のみ
- Phase D: Qwen3.5-4B 4bit (4GB) — Phase A-C のモデルを先にアンロード

Phase A+B → unload → Phase C (CPU) → Phase D (LLM)

## Vocal Event Taxonomy

33イベント、6カテゴリ (`tmrvc_core.vocal_events`):

- respiratory (4): inhale, exhale, gasp, held_breath
- phonation (5): voice_break, creak, falsetto, tremor, strained
- emotional (8): laugh, chuckle, sob, whimper, sigh, groan, scream, exclaim
- articulatory (6): click, lip_smack, swallow, cough, throat_clear, sniff
- prosodic (6): pause, long_pause, hesitation, emphasis, prolonged, rush
- paralinguistic (4): hmm, uh_huh, tsk, shh

31/33 が DSP 検出可能。2 (hesitation, uh_huh) は LLM 推論。

## Text Encoder Vocabulary 更新

acting_tags.py の語彙を vocal_events.py と統合:

- acting directives: [angry], [whisper], [calm], etc. (既存16タグ)
- vocal events: [inhale], [laugh], [sob], etc. (新規33タグ)
- prosodic markers: 既存のものは vocal_events に吸収
- free-form: [act:...] (既存)

合計: ~50+ タグ

## Exit Criteria

- `dev.py 1` が任意のタイミングで中断・再開可能
- 各 utterance の meta.json に phase 完了状態が記録される
- `dev.py 2` はキャッシュの phase_a_done + phase_b_done が true の utterance のみを学習に使用
- vocal event annotation は enriched transcript に反映される
