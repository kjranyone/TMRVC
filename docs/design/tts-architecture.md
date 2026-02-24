# TMRVC-TTS アーキテクチャ設計

Kojiro Tanaka — TTS extension design
Created: 2026-02-24 (Asia/Tokyo)

> **Goal:** 既存 VC パイプラインの後半 (Converter + Vocoder) を流用し、
> テキスト→音声合成に拡張する。VC と TTS で共通の content[256,T] 表現を介して接続。

---

## 1. 設計方針: VC 基盤の再利用

VC パイプラインの「前半」(音声→content/f0 抽出) をテキスト→content/f0 生成に置き換え、
「後半」(Converter + Vocoder) はそのまま流用する。

```
[VC]  Source Audio → ContentEncoder → content[256,T]
                                        ↓
[TTS] Text → TextEncoder → Duration → ContentSynthesizer → content[256,T]
                                        ↓
                              + spk_embed[192] + style_params[64]
                                        ↓
                              Converter (流用) → Vocoder (流用) → Audio
```

この設計により:
- Converter/Vocoder の学習コストを VC と共有
- VC の音質改善が自動的に TTS にも反映
- ONNX エクスポート・ストリーミングエンジンを共有可能

## 2. 新規コンポーネント一覧

| モジュール | 入力 | 出力 | パラメータ数 | ファイル |
|---|---|---|---|---|
| **TextEncoder** | `phoneme_ids[B,L]` | `text_features[B,256,L]` | ~4M | `models/text_encoder.py` |
| **DurationPredictor** | `text_features + style` | `durations[B,L]` | ~0.5M | `models/duration_predictor.py` |
| **F0Predictor** | `text_features (expanded) + style` | `f0[B,1,T], voiced[B,1,T]` | ~1M | `models/f0_predictor.py` |
| **ContentSynthesizer** | `text_features (expanded)` | `content[B,256,T]` | ~1.5M | `models/content_synthesizer.py` |
| **StyleEncoder** | `mel_ref` or text prompt | `style[B,32]` | ~3M | `models/style_encoder.py` |
| **合計** | | | **~10M** | |

既存コンポーネント (流用):

| モジュール | パラメータ数 | TTS での役割 |
|---|---|---|
| **Converter** | ~4M | content → STFT features (FiLM 拡張) |
| **Vocoder** | ~2M | STFT features → waveform |
| **SpeakerEncoder** | ~1.7M | 話者埋め込み抽出 (凍結) |

## 3. TextEncoder

### 3.1 アーキテクチャ

音素ベースの Transformer エンコーダ。日英の統一 IPA 音素体系 (~200 音素) を使用。

```
phoneme_ids[B, L]
    → Embedding(vocab=200, d=256) + LangEmbedding(n_lang=4, d=256)
    + SinusoidalPositionalEncoding(d=256)
    → TransformerEncoder (6層, d=256, 4head, ff=1024, GELU, norm_first=True)
    → LayerNorm
    → text_features[B, 256, L]
```

### 3.2 G2P フロントエンド

| 言語 | バックエンド | 出力形式 |
|---|---|---|
| 日本語 | `pyopenjtalk` (fullcontext → phoneme) | IPA 準拠 |
| 英語 | `phonemizer` (espeak-ng backend) | IPA |

- G2P は `tmrvc-data/src/tmrvc_data/g2p.py` に実装
- 音素ボキャブラリ: `PHONE2ID` dict、特殊トークン `<pad>=0, <unk>=1, <bos>=2, <eos>=3, <sil>=4, <breath>=5`
- 言語 ID: `ja=0, en=1, zh=2, other=3`

### 3.3 定数 (`configs/constants.yaml`)

```yaml
d_text_encoder: 256
n_text_encoder_layers: 6
n_text_encoder_heads: 4
text_encoder_ff_dim: 1024
phoneme_vocab_size: 200
n_languages: 4
```

## 4. DurationPredictor

### 4.1 アーキテクチャ

FastSpeech2 方式の明示的デュレーション予測。
VTuber・演技ではタイミング制御が重要なため、暗黙的 (attention-based) ではなく明示的予測を採用。

```
text_features[B, 256, L]
    → Conv1d(256, 256, k=3, same_pad) + ReLU + Dropout
    → Conv1d(256, 256, k=3, same_pad) + ReLU + Dropout
    + FiLM(style[32])
    → Linear(256, 1) + Softplus
    → durations[B, L]  (positive frame counts)
```

- **Ground truth**: Montreal Forced Aligner (MFA) による強制アラインメント
- style 条件付けで話速・テンポを制御可能
- **注意**: Conv1d 出力 `[B,C,T]` に LayerNorm は使わない (ストリーミング時に T=1 vs T>1 で正規化が変わるため)

### 4.2 Length Regulation

デュレーション (音素ごとのフレーム数) に基づいて音素レベル特徴量をフレームレベルに展開:

```python
length_regulate(text_features[B, 256, L], durations[B, L]) → [B, 256, T]
# T = sum(durations[b]) per batch
```

学習時は GT デュレーション (teacher forcing)、推論時は予測デュレーションを使用。

## 5. F0Predictor

### 5.1 アーキテクチャ

スタイル条件付き F0 予測。感情で韻律パターンが大きく変わるため FiLM で制御。

```
text_features[B, 256, T] (length regulated)
    → Conv1d(256, 128, k=1)
    → CausalConvNeXtBlock x 4 (d=128, k=3, dilation=[1,1,2,2])
      + FiLM(style[32]) per block
    → Conv1d(128, 2, k=1)
    → f0[B,1,T] (Softplus → positive Hz)
    → voiced_prob[B,1,T] (Sigmoid → [0,1])
```

- 出力フォーマットは CREPE/RMVPE と同一 → Converter への入力が VC/TTS で共通
- `d_f0_predictor: 128`

## 6. ContentSynthesizer

### 6.1 アーキテクチャ

テキスト特徴量を ContentEncoder の出力空間 (256d) に変換する「ブリッジ」。

```
text_features[B, 256, T] (length regulated)
    → Conv1d(256, 256, k=1) + SiLU
    → CausalConvNeXtBlock x 4 (d=256, k=3, dilation=[1,1,2,4])
    → Conv1d(256, 256, k=1)
    → content[B, 256, T]
```

### 6.2 アラインメント制約

ContentSynthesizer の出力が ContentEncoder の出力と同じ分布を持つ必要がある:

```
L_content = MSE(ContentSynthesizer(text_feat), ContentEncoder(mel, f0))
```

これにより Converter は VC モードでも TTS モードでも同じ content 表現を受け取れる。

- `d_content_synthesizer: 256` (= `d_content`)

## 7. StyleEncoder

### 7.1 acoustic_params の拡張

既存の `acoustic_params[32]` を `style_params[64]` に拡張:

```
現在: acoustic_params[32] = IR(24d) + voice_source(8d)
拡張: style_params[64]   = acoustic_params(32d) + emotion_style(32d)
```

### 7.2 emotion_style[32d] の内訳

| Index | 次元 | 内容 |
|---|---|---|
| 0-2 | 3d | Valence / Arousal / Dominance (連続値) |
| 3-5 | 3d | VAD の不確実性 |
| 6-8 | 3d | 話速 / エネルギー / ピッチレンジ |
| 9-20 | 12d | 感情カテゴリ softmax (12カテゴリ) |
| 21-28 | 8d | 学習済み潜在表現 (ラベルで表現できないニュアンス) |
| 29-31 | 3d | 予備 |

12 感情カテゴリ: happy, sad, angry, fearful, surprised, disgusted, neutral, bored, excited, tender, sarcastic, whisper

### 7.3 AudioStyleEncoder

```
mel_ref[B, 80, T]
    → unsqueeze(1) → [B, 1, 80, T]
    → Conv2d x 4 (ch=[32,64,128,256], stride=2, BN, SiLU)
    → GlobalAvgPool(dim=time)
    → Flatten → MLP(256*5 → 256 → 32)
    → style[B, 32]
```

### 7.4 入力モード (Phase 3+)

| モード | 入力 | 実装 | Phase |
|---|---|---|---|
| **音声参照** | mel_ref | AudioStyleEncoder (CNN) | 3 |
| **テキスト指示** | "怒りを抑えて冷たく" | Embedding + SmallTransformer | 3+ |
| **LLM 出力** | 構造化 JSON | 直接ベクトルマッピング | 4 |

### 7.5 後方互換性

- VC モード: `StyleEncoder.make_vc_style_params(acoustic_params)` → `style[32:64] = 0`
- FiLM 初期化 (gamma=1, beta=0) により、追加次元がゼロなら恒等変換
- VC チェックポイントからの移行: `converter_from_vc_checkpoint()` で FiLM 重みをパディング

## 8. Converter FiLM 拡張

### 8.1 d_cond の変更

```python
# VC モード:  d_cond = d_speaker(192) + n_acoustic_params(32)  = 224
# TTS モード: d_cond = d_speaker(192) + n_style_params(64)     = 256
```

### 8.2 重み移行

`converter_from_vc_checkpoint()` により VC → TTS の重み移行:

1. 新しい ConverterStudent を `n_acoustic_params=64` で構築
2. input_proj / output_proj はそのままコピー
3. FiLM 重み: `migrate_film_weights()` で既存 224 列をコピー、新 32 列はゼロ初期化
4. ゼロ初期化により TTS モードでも VC と同一出力 (emotion_style=0 の場合)

### 8.3 定数

```yaml
d_style: 32
n_style_params: 64        # n_acoustic_params(32) + d_style(32)
n_emotion_categories: 12
```

## 9. TTS パイプライン全体フロー

### 9.1 推論時

```
テキスト
    │
    ├─▶ G2P (pyopenjtalk / phonemizer)
    │       → phoneme_ids[L], language_id
    │
    ├─▶ TextEncoder
    │       → text_features[256, L]
    │
    ├─▶ DurationPredictor (+ style)
    │       → durations[L]
    │
    ├─▶ Length Regulate
    │       → text_features[256, T]  (T = sum(durations))
    │
    ├─▶ F0Predictor (+ style)
    │       → f0[1, T], voiced_prob[1, T]
    │
    ├─▶ ContentSynthesizer
    │       → content[256, T]
    │
    ├─▶ Converter (VC 流用, + spk_embed + style_params)
    │       → stft_features[513, T]
    │
    └─▶ Vocoder (VC 流用)
            → audio waveform
```

### 9.2 学習時

```
(phoneme_ids, durations_gt, mel, content_gt, f0_gt, spk_embed)
    │
    ├─▶ TextEncoder(phoneme_ids, lang_id)
    │       → text_features[256, L]
    │
    ├─▶ DurationPredictor(text_features)
    │       → pred_durations → L_duration = MSE(log1p(pred), log1p(gt))
    │
    ├─▶ Length Regulate (GT durations, teacher forcing)
    │       → expanded_features[256, T]
    │
    ├─▶ F0Predictor(expanded_features)
    │       → pred_f0 → L_f0 = MSE(pred, gt)
    │       → pred_voiced → L_voiced = BCE(pred, gt_voiced)
    │
    └─▶ ContentSynthesizer(expanded_features)
            → pred_content → L_content = MSE(pred, content_gt)
```

Converter / Vocoder は **凍結**。ContentEncoder は GT content の参照ターゲットとしてのみ使用。

## 10. テンソル形状一覧

| テンソル | 形状 | 定数 | 説明 |
|---|---|---|---|
| `phoneme_ids` | `[B, L]` | `phoneme_vocab_size=200` | 音素 ID |
| `language_ids` | `[B]` | `n_languages=4` | 言語 ID |
| `text_features` | `[B, 256, L]` | `d_text_encoder=256` | 音素レベル特徴量 |
| `durations` | `[B, L]` | — | フレーム数/音素 (正の実数) |
| `f0` | `[B, 1, T]` | — | F0 in Hz |
| `voiced_prob` | `[B, 1, T]` | — | 有声確率 [0,1] |
| `content` | `[B, 256, T]` | `d_content=256` | content features |
| `style` | `[B, 32]` | `d_style=32` | emotion style |
| `style_params` | `[B, 64]` | `n_style_params=64` | acoustic + emotion |
| `spk_embed` | `[B, 192]` | `d_speaker=192` | 話者埋め込み |

## 11. ファイル配置

```
tmrvc-data/src/tmrvc_data/
    g2p.py                  # G2P フロントエンド
    alignment.py            # MFA ラッパー

tmrvc-train/src/tmrvc_train/
    models/
        text_encoder.py         # TextEncoder
        duration_predictor.py   # DurationPredictor
        f0_predictor.py         # F0Predictor + length_regulate()
        content_synthesizer.py  # ContentSynthesizer
        style_encoder.py        # StyleEncoder + AudioStyleEncoder
        converter.py            # + migrate_film_weights(), converter_from_vc_checkpoint()
    tts_trainer.py              # TTSTrainer + TTSTrainerConfig

tmrvc-core/src/tmrvc_core/
    types.py                # + TTSFeatureSet, TTSBatch
    constants.py            # + TTS 定数

configs/
    constants.yaml          # + tts セクション
    train_tts.yaml          # TTS 学習設定

tests/python/
    test_tts_models.py      # 32 テスト
    test_g2p.py             # 15 テスト
```

## 12. 整合性チェックリスト

- [x] `d_text_encoder=256` = `d_content=256` = ContentSynthesizer 入出力次元 (`constants.yaml`)
- [x] `n_style_params=64` = `n_acoustic_params(32) + d_style(32)` (`constants.yaml`)
- [x] Converter FiLM `d_cond` = `d_speaker(192) + n_style_params(64) = 256` (`converter.py`)
- [x] `migrate_film_weights()` でゼロ初期化 → VC 互換出力を検証済み (`test_tts_models.py`)
- [x] ContentSynthesizer 出力 `[B, 256, T]` = ContentEncoder 出力 `[B, 256, T]` (`model-architecture.md §2`)
- [x] F0Predictor 出力 `[B, 1, T]` = CREPE/RMVPE F0 フォーマット (`onnx-contract.md §3`)
- [x] DurationPredictor に LayerNorm を使わない (ストリーミング互換、`MEMORY.md` 制約)
- [x] `phoneme_vocab_size=200` が G2P ボキャブラリ `len(PHONEME_LIST)` 以上 (`g2p.py`)
- [x] TTSBatch の `frame_lengths` = TrainingBatch の `lengths` と同等のマスキング用途 (`types.py`)
- [x] 全テスト通過: 297 passed, 3 skipped (既存テスト退行なし)

---

**関連資料:**

- `docs/design/architecture.md` — VC 全体アーキテクチャ
- `docs/design/model-architecture.md` — VC モデル詳細 (ContentEncoder, Converter, Vocoder)
- `docs/design/onnx-contract.md` — ONNX I/O テンソル仕様
- `docs/design/tts-training-plan.md` — TTS 学習ロードマップ
- `docs/design/tts-style-design.md` — StyleEncoder・LLM 統合・VTuber 設計
