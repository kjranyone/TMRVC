# TMRVC-TTS 学習計画

Kojiro Tanaka — TTS training plan
Created: 2026-02-24 (Asia/Tokyo)

> Merged from: `docs/design/tts-training-plan.md` + `docs/design/tts-architecture.md`

> **Goal:** VC 基盤完成後、4 フェーズで TTS を段階的に拡張。
> 基本 TTS → 表現的 TTS → 文脈認識 TTS → VTuber 統合。

---

## 1. 全体スケジュール

```
2026-02 ████████████░░░░░░░░░  Phase 1: VC 基盤 (進行中, step 234K)
2026-03 ░░░░████████████░░░░░  Phase 2: 基本 TTS (~3-4 週)
2026-04 ░░░░░░░░████████████░  Phase 3: 表現的 TTS (~4-6 週)
2026-05 ░░░░░░░░░░░░████████░  Phase 4: 文脈認識 TTS (~3-4 週)
2026-05 ░░░░░░░░░░░░░░░░████  Phase 5: VTuber 統合 (~2-3 週)
```

**合計: ~4 ヶ月** (Phase 1 残り含む)

## 2. Phase 1: VC 基盤完成 [進行中]

現計画通り進行。TTS に変更なし。

```
Phase 1a: Teacher 学習 (500K steps, WavLM 1024d)     ← 現在 step 234K
Phase 1b: STFT + Speaker loss 追加 (200K steps)
Phase 2:  IR-robust 学習 + RIR augmentation (200K steps)
Student 蒸留 (Phase A/B/B2/C)
ONNX エクスポート
```

**所要時間**: ~2-3 週間 (残り)

## 3. Phase 2: 基本 TTS

**目標**: テキスト + 話者指定 → 自然な音声生成

### 3.1 Step 2.1: G2P・強制アラインメント基盤 (1 週間)

**実装済みファイル:**
- `tmrvc-data/src/tmrvc_data/g2p.py` — G2P フロントエンド (日英)
- `tmrvc-data/src/tmrvc_data/alignment.py` — MFA ラッパー

**未実装:**
- `scripts/run_forced_alignment.py` — 全データセットにアラインメント一括実行

**データ取得:**

| データセット | 言語 | 話者 | 時間 | サイズ | 取得方法 |
|---|---|---|---|---|---|
| **JSUT** | JA | 1 (女性) | ~10h | ~2 GB | 直接 DL |
| **LJSpeech** | EN | 1 (女性) | ~24h | ~2.6 GB | 直接 DL |
| VCTK | EN | 109 | 44h | — | 取得済み |
| JVS | JA | 100 | ~30h | — | 取得済み |

**アラインメント作業:**
1. MFA モデルのダウンロード (`mfa model download acoustic japanese`, `english_mfa`)
2. 全データセットに MFA 実行
3. TextGrid → `phoneme_ids.npy`, `durations.npy` に変換
4. キャッシュに追加保存

### 3.2 Step 2.2: TTS モデル実装 (1 週間)

**実装済み:**
- `models/text_encoder.py` — Transformer (6 層, d=256)
- `models/duration_predictor.py` — Conv + FiLM + Softplus
- `models/f0_predictor.py` — CausalConvNeXt + FiLM
- `models/content_synthesizer.py` — CausalConvNeXt (text→content)
- `models/style_encoder.py` — CNN + MLP (Phase 3 用)
- `types.py` — `TTSFeatureSet`, `TTSBatch`
- `tts_trainer.py` — TTSTrainer + TTSTrainerConfig
- `configs/train_tts.yaml` — 学習設定

**未実装:**
- TTS 用 Dataset クラス (`TTSDataset`)
- TTS 用 collate_fn
- TTS CLI (`tmrvc-train-tts`)

### 3.3 Step 2.3: TTS 学習 (1-2 週間)

```yaml
# configs/train_tts.yaml
batch_size: 32
max_frames: 400
lr: 1.0e-4
warmup_steps: 5000
max_steps: 200000
losses:
  duration: 1.0    # MSE(log1p(pred_dur), log1p(mfa_dur))
  f0: 0.5          # MSE(pred_f0, extracted_f0)
  content: 1.0     # MSE(synthesized, content_encoder(mel,f0))
  voiced: 0.2      # BCE(pred_voiced, gt_voiced)
datasets: [jsut, ljspeech, vctk, jvs]
frozen: [content_encoder, converter, vocoder, speaker_encoder]
```

**リソース:**
- XPU (Arc B570): bs=32, max_frames=400 → ~2.5 GB VRAM
- 推定時間: ~3 日 (200K steps, ~0.5s/step)

## 4. Phase 3: 表現的 TTS

**目標**: 感情・スタイル指定で表現力のある音声生成

### 4.1 Step 3.1: データ取得・前処理 (1 週間)

| データセット | 言語 | 話者 | 時間 | 感情ラベル | サイズ | ライセンス | 優先度 |
|---|---|---|---|---|---|---|---|
| **Expresso** (Meta) | EN | 4 | ~40h | 26 スタイル | ~15 GB | CC BY-NC 4.0 | **最高** |
| **JVNV** | JA | 4 | ~4h | 6 基本感情 | ~1.5 GB | CC BY-SA 4.0 | **高** |
| **EmoV-DB** | EN | 4+ | ~7h | 5 感情 | ~3 GB | Apache 2.0 | **高** |
| **RAVDESS** | EN | 24 | ~2h | 8 感情 + VAD | ~1 GB | CC BY-NC-SA 4.0 | **中** |
| **J-MAC** | JA | 複数 | ~12h | 感情込み朗読 | ~5 GB | CC BY-SA 4.0 | **中** |

**感情ラベルの統一:**
- 全データセットを 12 カテゴリ + VAD 連続値に正規化
- `tmrvc-data/src/tmrvc_data/emotion_features.py` (新規) でマッピング

### 4.2 日本語感情データの不足問題と対策

**課題**: 英語 ~60h vs 日本語 ~4h (JVNV のみ)

**4 段階の対策:**

| Step | 手法 | 日本語データ増加量 | 説明 |
|---|---|---|---|
| 1 | Cross-lingual Transfer | — | Expresso (40h EN) で事前学習→ JVNV で fine-tune |
| 2 | Pseudo-Label | +50-60h | J-MAC + JTubeSpeech に英語分類器で擬似ラベル (confidence > 0.8) |
| 3 | コミュニティ収集 | +5-10h | ITA コーパス感情バリエーション、ROHAN4600 |
| 4 | Few-shot Fine-tune | — | ターゲット VTuber 音声 (5-10 分) で最終調整 |

**目標バランス:** 英語 ~60h / 日本語 ~60-80h

### 4.3 Step 3.2: StyleEncoder 学習 (2 週間)

**Phase 3a**: 音声→スタイルエンコーダ
- 感情分類 (12 カテゴリ, cross-entropy)
- VAD 回帰 (MSE)
- 話速推定 (MSE)

**Phase 3b**: テキスト→スタイルエンコーダ (Phase 3+)
- 対照学習: "angry shouting" のテキスト埋め込みが怒り音声のスタイル埋め込みに一致

### 4.4 Step 3.3: Converter FiLM 拡張 + Joint Fine-tune (2 週間)

```yaml
# configs/train_expressive.yaml
lr: 5.0e-5
max_steps: 100000
losses:
  mel: 1.0
  stft: 0.5
  style_cls: 0.3
  style_vad: 0.2
datasets: [expresso, jvnv, emov_db, ravdess]
trainable: [style_encoder, converter_film, duration_predictor, f0_predictor]
frozen: [vocoder, content_encoder, text_encoder, content_synthesizer]
```

- `n_acoustic_params`: 32 → 64 (`n_style_params`) に拡張
- [x] 旧VCチェックポイント移行は行わず、style-conditioned checkpointのみを対象とする

### 4.5 Step 3.4: 統合テスト (1 週間)

```
テキスト + "angry, shouting" → StyleEncoder → style[32]
    + acoustic_params[32] → style_params[64]
テキスト → G2P → TextEncoder → DurationPredictor → F0Predictor
    → ContentSynthesizer → Converter → Vocoder → Audio
```

## 5. Phase 4: 文脈認識 TTS

**目標**: 会話履歴・状況プロンプトから自動的に適切な演技を生成

参照: `docs/training/style-training-plan.md §4`

### 5.1 主要タスク

1. **Context Predictor** (1 週間): Claude API 統合、`ContextStylePredictor`
2. **台本パーサー + バッチ生成** (1 週間): YAML 台本フォーマット、キュー管理
3. **プロンプトエンジニアリング + 評価** (1-2 週間): DailyDialog / EmpatheticDialogues

### 5.2 評価データ (LLM プロンプト用、モデル学習には使わない)

| データセット | 言語 | サイズ | 用途 |
|---|---|---|---|
| DailyDialog | EN | ~13K 対話 | 感情ラベル付き日常会話 |
| EmpatheticDialogues | EN | ~25K 対話 | 共感応答＋感情ラベル |
| persona-chat | EN | ~10K 対話 | キャラクター一貫性 |

## 6. Phase 5: VTuber 統合

参照: `docs/training/style-training-plan.md §5, §6`

### 6.1 主要タスク

1. `.tmrvc_character` フォーマット定義
2. ライブ配信チャットレスポンス・アーキテクチャ
3. FastAPI サーバー (WebSocket + REST)
4. GUI 拡張 (TTS ページ, チャットモニター)

## 7. 既存重みの活用戦略

| コンポーネント | Phase 2 | Phase 3 | Phase 4-5 |
|---|---|---|---|
| Converter | **凍結** | FiLM 層のみ fine-tune | 凍結 |
| Vocoder | **凍結** | **凍結** | **凍結** |
| ContentEncoder | 参照ターゲット (凍結) | 凍結 | 凍結 |
| SpeakerEncoder | 凍結 | 凍結 | 凍結 |
| TextEncoder | **新規学習** | fine-tune | 凍結 |
| DurationPredictor | **新規学習** | style 条件追加 fine-tune | 凍結 |
| F0Predictor | **新規学習** | style 条件追加 fine-tune | 凍結 |
| ContentSynthesizer | **新規学習** | fine-tune | 凍結 |
| StyleEncoder | — | **新規学習** | 凍結 |

## 8. 計算リソース見積もり (Intel Arc B570, 9.6 GB VRAM)

| Phase | 学習パラメータ | 凍結パラメータ | VRAM (推定) | 推定時間 |
|---|---|---|---|---|
| Phase 2 | ~7M (TTS front-end) | ~20M (CE+Conv+Voc) | ~2.5 GB | ~3 日 |
| Phase 3a | ~3M (StyleEncoder) | ~20M | ~2 GB | ~2 日 |
| Phase 3b | ~10M (Joint FT) | ~10M (Voc+CE) | ~4 GB | ~5 日 |
| **合計** | | | | **~10 日** |

全フェーズ B570 の 9.6 GB に収まる。

## 9. データ取得サイズ総計

| Phase | データセット | 合計サイズ |
|---|---|---|
| Phase 2 | JSUT, LJSpeech | ~5 GB |
| Phase 3 | Expresso, JVNV, EmoV-DB, RAVDESS, J-MAC | ~25 GB |
| Phase 3 (optional) | JTubeSpeech (filtered) | ~20 GB |
| **合計** | | **~50 GB** |

## 10. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| ContentSynthesizer が ContentEncoder の分布と合わない | Converter 品質劣化 | alignment loss 重み調整、Teacher end-to-end 学習 |
| 日本語感情音声データ不足 (JVNV 4h のみ) | 感情制御精度不足 | Expresso cross-lingual transfer + pseudo-label |
| MFA アラインメント品質 (日本語) | デュレーション予測劣化 | 手動検査 + 低品質除外フィルタ |
| FiLM 拡張が VC 品質に悪影響 | 既存 VC の退行 | style-conditioned converter のみを運用し、旧VC checkpoint は対象外とする |
| pyopenjtalk ビルド失敗 (Windows, VS 必要) | 日本語 G2P 不可 | pre-built wheel 使用、WSL フォールバック |

## 11. 整合性チェックリスト

- [x] Phase 2 のデータセット (JSUT, LJSpeech, VCTK, JVS) は `configs/datasets.yaml` に登録予定
- [x] `train_tts.yaml` の `max_frames: 400` は `CLAUDE.md` のデフォルト設定と一致
- [x] `batch_size: 32` は B570 VRAM (9.6 GB) 内で動作する推定 (~2.5 GB)
- [x] Converter 凍結 → `tts_trainer.py` で optimizer に含まない
- [x] 旧VCチェックポイント移行は行わず、style-conditioned checkpointのみを対象とする
- [x] 全 Phase で `--device xpu` を使用 (`CLAUDE.md` Runtime Device Policy)
- [x] 学習チェックポイントは `checkpoints/tts/` に保存 (gitignore 対象)

---

**関連資料:**

- `docs/training/tts-training-plan.md` (本ファイル Appendix A) — TTS モジュール構成・テンソル仕様
- `docs/training/style-training-plan.md` — StyleEncoder・LLM 統合・VTuber 設計
- `docs/training/vc-training-plan.md` — VC Teacher 学習計画 (Phase 1)
- `docs/design/model-architecture.md` — VC モデル詳細

---

## Appendix A: TTS アーキテクチャ仕様

> 以下は `docs/design/tts-architecture.md` の内容を統合したもの。

Kojiro Tanaka — TTS extension design
Created: 2026-02-24 (Asia/Tokyo)

> **Goal:** 既存 VC パイプラインの後半 (Converter + Vocoder) を流用し、
> テキスト→音声合成に拡張する。VC と TTS で共通の content[256,T] 表現を介して接続。

---

### A.1 設計方針: VC 基盤の再利用

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

### A.2 新規コンポーネント一覧

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

### A.3 TextEncoder

#### A.3.1 アーキテクチャ

音素ベースの Transformer エンコーダ。日英の統一 IPA 音素体系 (~200 音素) を使用。

```
phoneme_ids[B, L]
    → Embedding(vocab=200, d=256) + LangEmbedding(n_lang=4, d=256)
    + SinusoidalPositionalEncoding(d=256)
    → TransformerEncoder (6層, d=256, 4head, ff=1024, GELU, norm_first=True)
    → LayerNorm
    → text_features[B, 256, L]
```

#### A.3.2 G2P フロントエンド

| 言語 | バックエンド | 出力形式 |
|---|---|---|
| 日本語 | `pyopenjtalk` (fullcontext → phoneme) | IPA 準拠 |
| 英語 | `phonemizer` (espeak-ng backend) | IPA |

- G2P は `tmrvc-data/src/tmrvc_data/g2p.py` に実装
- 音素ボキャブラリ: `PHONE2ID` dict、特殊トークン `<pad>=0, <unk>=1, <bos>=2, <eos>=3, <sil>=4, <breath>=5`
- 言語 ID: `ja=0, en=1, zh=2, ko=3`

#### A.3.3 定数 (`configs/constants.yaml`)

```yaml
d_text_encoder: 256
n_text_encoder_layers: 6
n_text_encoder_heads: 4
text_encoder_ff_dim: 1024
phoneme_vocab_size: 200
n_languages: 4
```

### A.4 DurationPredictor

#### A.4.1 アーキテクチャ

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

#### A.4.2 Length Regulation

デュレーション (音素ごとのフレーム数) に基づいて音素レベル特徴量をフレームレベルに展開:

```python
length_regulate(text_features[B, 256, L], durations[B, L]) → [B, 256, T]
# T = sum(durations[b]) per batch
```

学習時は GT デュレーション (teacher forcing)、推論時は予測デュレーションを使用。

### A.5 F0Predictor

#### A.5.1 アーキテクチャ

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

### A.6 ContentSynthesizer

#### A.6.1 アーキテクチャ

テキスト特徴量を ContentEncoder の出力空間 (256d) に変換する「ブリッジ」。

```
text_features[B, 256, T] (length regulated)
    → Conv1d(256, 256, k=1) + SiLU
    → CausalConvNeXtBlock x 4 (d=256, k=3, dilation=[1,1,2,4])
    → Conv1d(256, 256, k=1)
    → content[B, 256, T]
```

#### A.6.2 アラインメント制約

ContentSynthesizer の出力が ContentEncoder の出力と同じ分布を持つ必要がある:

```
L_content = MSE(ContentSynthesizer(text_feat), ContentEncoder(mel, f0))
```

これにより Converter は VC モードでも TTS モードでも同じ content 表現を受け取れる。

- `d_content_synthesizer: 256` (= `d_content`)

### A.7 StyleEncoder

#### A.7.1 acoustic_params の拡張

既存の `acoustic_params[32]` を `style_params[64]` に拡張:

```
現在: acoustic_params[32] = IR(24d) + voice_source(8d)
拡張: style_params[64]   = acoustic_params(32d) + emotion_style(32d)
```

#### A.7.2 emotion_style[32d] の内訳

| Index | 次元 | 内容 |
|---|---|---|
| 0-2 | 3d | Valence / Arousal / Dominance (連続値) |
| 3-5 | 3d | VAD の不確実性 |
| 6-8 | 3d | 話速 / エネルギー / ピッチレンジ |
| 9-20 | 12d | 感情カテゴリ softmax (12カテゴリ) |
| 21-28 | 8d | 学習済み潜在表現 (ラベルで表現できないニュアンス) |
| 29-31 | 3d | 予備 |

12 感情カテゴリ: happy, sad, angry, fearful, surprised, disgusted, neutral, bored, excited, tender, sarcastic, whisper

#### A.7.3 AudioStyleEncoder

```
mel_ref[B, 80, T]
    → unsqueeze(1) → [B, 1, 80, T]
    → Conv2d x 4 (ch=[32,64,128,256], stride=2, BN, SiLU)
    → GlobalAvgPool(dim=time)
    → Flatten → MLP(256*5 → 256 → 32)
    → style[B, 32]
```

#### A.7.4 入力モード (Phase 3+)

| モード | 入力 | 実装 | Phase |
|---|---|---|---|
| **音声参照** | mel_ref | AudioStyleEncoder (CNN) | 3 |
| **テキスト指示** | "怒りを抑えて冷たく" | Embedding + SmallTransformer | 3+ |
| **LLM 出力** | 構造化 JSON | 直接ベクトルマッピング | 4 |

#### A.7.5 互換ポリシー

- ランタイムは後方互換フォールバックを持たない。
- Converter は d_cond = d_speaker + n_style_params = 256 のみ許可する。
- 旧VCチェックポイント (d_cond=224) の自動移行は廃止。

### A.8 Converter FiLM 拡張

#### A.8.1 d_cond の変更

```python
# VC モード:  d_cond = d_speaker(192) + n_acoustic_params(32)  = 224
# TTS モード: d_cond = d_speaker(192) + n_style_params(64)     = 256
```

#### A.8.2 条件次元ポリシー

ランタイムでは converter_from_vc_checkpoint() を使った自動移行は行わない。
d_cond=256 の style-conditioned converter のみロード対象とする。

#### A.8.3 定数

```yaml
d_style: 32
n_style_params: 64        # n_acoustic_params(32) + d_style(32)
n_emotion_categories: 12
```

### A.9 TTS パイプライン全体フロー

#### A.9.1 推論時

```
テキスト
    │
    ├──▶ G2P (pyopenjtalk / phonemizer)
    │       → phoneme_ids[L], language_id
    │
    ├──▶ TextEncoder
    │       → text_features[256, L]
    │
    ├──▶ DurationPredictor (+ style)
    │       → durations[L]
    │
    ├──▶ Length Regulate
    │       → text_features[256, T]  (T = sum(durations))
    │
    ├──▶ F0Predictor (+ style)
    │       → f0[1, T], voiced_prob[1, T]
    │
    ├──▶ ContentSynthesizer
    │       → content[256, T]
    │
    ├──▶ Converter (VC 流用, + spk_embed + style_params)
    │       → stft_features[513, T]
    │
    └──▶ Vocoder (VC 流用)
            → audio waveform
```

#### A.9.2 学習時

```
(phoneme_ids, durations_gt, mel, content_gt, f0_gt, spk_embed)
    │
    ├──▶ TextEncoder(phoneme_ids, lang_id)
    │       → text_features[256, L]
    │
    ├──▶ DurationPredictor(text_features)
    │       → pred_durations → L_duration = MSE(log1p(pred), log1p(gt))
    │
    ├──▶ Length Regulate (GT durations, teacher forcing)
    │       → expanded_features[256, T]
    │
    ├──▶ F0Predictor(expanded_features)
    │       → pred_f0 → L_f0 = MSE(pred, gt)
    │       → pred_voiced → L_voiced = BCE(pred, gt_voiced)
    │
    └──▶ ContentSynthesizer(expanded_features)
            → pred_content → L_content = MSE(pred, content_gt)
```

Converter / Vocoder は **凍結**。ContentEncoder は GT content の参照ターゲットとしてのみ使用。

### A.10 テンソル形状一覧

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

### A.11 ファイル配置

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
        converter.py            # converter modules
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

### A.12 整合性チェックリスト

- [x] `d_text_encoder=256` = `d_content=256` = ContentSynthesizer 入出力次元 (`constants.yaml`)
- [x] `n_style_params=64` = `n_acoustic_params(32) + d_style(32)` (`constants.yaml`)
- [x] Converter FiLM `d_cond` = `d_speaker(192) + n_style_params(64) = 256` (`converter.py`)
- [x] 旧VC checkpoint (`d_cond=224`) はランタイムで reject し、style-conditioned (`d_cond=256`) のみ許可する。
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
- `docs/training/tts-training-plan.md` — 本ファイル (TTS 学習ロードマップ + アーキテクチャ)
- `docs/training/style-training-plan.md` — StyleEncoder・LLM 統合・VTuber 設計
