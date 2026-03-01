# データセット作成フロー設計

## 概要

雑多な音声ファイルを TMRVC 学習用データセットに変換するパイプライン。
UCLM v2 では `acoustic_tokens(A_t)`、`control_tokens(B_t)`、`delta_voice_state` を追加保存する。

## 入力

```
data/raw_my_voices/           # 任意のディレクトリ
├── character_a/
│   ├── voice_001.wav         # 48kHz, stereo, 音量バラバラ
│   ├── voice_002.mp3         # 44.1kHz, mono
│   └── ...
├── character_b/
│   └── ...
└── flat/
    ├── 001.wav               # 話者不明
    └── 002.wav
```

## 出力

```
data/cache/my_voices/train/
├── my_voices_character_a/
│   ├── my_voices_voice_001/
│   │   ├── mel.npy           # [80, T] 24kHz log-mel
│   │   ├── content.npy       # [768, T] ContentVec
│   │   ├── f0.npy            # [1, T] Hz
│   │   ├── spk_embed.npy     # [192] speaker embedding
│   │   └── meta.json         # アノテーション
│   └── ...
└── my_voices_character_b/
    └── ...
```

## パイプライン構成

### Stage 1: スキャン & フィルタリング

```python
# 入力: 音声ファイル群
# 出力: 有効なファイルリスト

フィルタ条件:
- min_duration: 0.5秒 (短すぎる=ノイズ)
- max_duration: 30秒 (長すぎる=処理時間過多)
- min_rms: 0.005 (無音除外)
- max_rms: 0.99 (クリップ除外)
```

### Stage 2: 音声正規化

```python
# 入力: 生音声
# 出力: 正規化済み音声

処理:
1. モノラル化
2. 24kHz リサンプリング
3. Loudness正規化 (-23 LUFS)
4. クリッピング防止
```

### Stage 3: 特徴量抽出

```python
# 入力: 正規化済み音声
# 出力: mel, content, f0, spk_embed

モデル:
- ContentVec: HuBERT-based content extractor
- F0: pyworld (cheaptrick)
- Speaker: SpeechBrain ECAPA-TDNN (192-dim)
```

### Stage 4: 自動アノテーション

```python
# 入力: 音声 + 特徴量
# 出力: text, emotion_id, vad, prosody

モデル:
- Whisper large-v3: 文字起こし
- wav2vec2-emotion: 感情分類 (12クラス)
- ルールベース: VAD推定、韻律特徴量
```

### Stage 5: データセット保存

```python
# 入力: 特徴量 + アノテーション
# 出力: キャッシュディレクトリ

構造:
{cache_dir}/{dataset}/train/{speaker_id}/{utt_id}/
├── mel.npy
├── content.npy
├── f0.npy
├── spk_embed.npy
└── meta.json
```

## CLI インターフェース

```bash
# 基本実行
uv run python scripts/data/prepare_dataset.py \
    --input data/raw_my_voices \
    --output data/cache \
    --name my_voices \
    --language ja \
    --device cuda

# 話者マップ使用（フラットディレクトリ用）
uv run python scripts/data/prepare_dataset.py \
    --input data/raw_voices \
    --output data/cache \
    --name game_voices \
    --speaker-map data/raw_voices/_speaker_map.json \
    --device cuda

# 再開実行（既存キャッシュをスキップ）
uv run python scripts/data/prepare_dataset.py \
    --input data/raw_my_voices \
    --output data/cache \
    --name my_voices \
    --resume

# ドライラン（ファイル数確認のみ）
uv run python scripts/data/prepare_dataset.py \
    --input data/raw_my_voices \
    --dry-run
```

## 設定ファイル

```yaml
# configs/prepare_dataset.yaml
input:
  audio_dir: data/raw_my_voices
  speaker_map: null  # optional

output:
  cache_dir: data/cache
  dataset_name: my_voices

filter:
  min_duration: 0.5
  max_duration: 30.0
  min_rms: 0.005
  max_rms: 0.99
  extensions: [".wav", ".flac", ".ogg", ".mp3"]

normalize:
  target_sr: 24000
  target_lufs: -23.0
  mono: true

annotate:
  whisper_model: large-v3
  emotion_model: ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
  language: ja

device: cuda
```

## meta.json スキーマ

```json
{
  "utterance_id": "my_voices_voice_001",
  "speaker_id": "my_voices_character_a",
  "n_frames": 246,
  "duration_sec": 2.46,
  "sample_rate": 24000,
  
  "text": "こんにちは、元気ですか？",
  "language_id": 0,
  
  "emotion_id": 6,
  "emotion_label": "neutral",
  "emotion_confidence": 0.85,
  
  "vad": [0.5, 0.3, 0.5],
  "prosody": [1.0, 0.5, 0.5],
  
  "source_path": "data/raw_my_voices/character_a/voice_001.wav",
  "pipeline_version": "1.0"
}
```

## エラーハンドリング

| エラー | 処理 |
|--------|------|
| ファイル読み込み失敗 | スキップ、ログ出力 |
| 短すぎる/長すぎる | スキップ、統計に記録 |
| 無音検出 | スキップ |
| Whisper失敗 | text="" で保存、継続 |
| 感情分類失敗 | emotion_id=6 (neutral) で保存 |

## 進捗表示

```
[Stage 1/5] Scanning...          Found 8116 files, 7992 valid
[Stage 2/5] Normalizing...       ████████░░ 80% (6394/7992)
[Stage 3/5] Extracting...        ██████████ 100% (7992/7992)
[Stage 4/5] Transcribing...      ██████░░░░ 60% (4795/7992)
[Stage 5/5] Saving...            ██████████ 100% (7992/7992)

Summary:
  Processed: 7992
  Skipped:   124 (too short: 89, too long: 35)
  Errors:    23
  Duration:  2h 15m
```

## 次のステップ

データセット作成後:

```bash
# マニフェスト確認
cat data/cache/_manifests/my_voices_train.json

# UCLM 学習開始
uv run tmrvc-train-uclm \
    --cache-dir data/cache \
    --datasets my_voices \
    --device cuda
```
