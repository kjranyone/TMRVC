# TMRVC-TTS 学習ロードマップ

Kojiro Tanaka — TTS training plan
Created: 2026-02-24 (Asia/Tokyo)

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

参照: `docs/design/tts-style-design.md §4`

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

参照: `docs/design/tts-style-design.md §5, §6`

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

- `docs/design/tts-architecture.md` — TTS モジュール構成・テンソル仕様
- `docs/design/tts-style-design.md` — StyleEncoder・LLM 統合・VTuber 設計
- `docs/design/training-plan.md` — VC Teacher 学習計画 (Phase 1)
- `docs/design/model-architecture.md` — VC モデル詳細
