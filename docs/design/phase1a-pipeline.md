# Phase 1a 学習パイプライン — 実行手順書

Created: 2026-02-21
Phase 0 完了: step 100K, flow loss 1.83 → 1.07 (cosine LR decay)

---

## 0. Phase 0 → Phase 1a の変更点

| 項目 | Phase 0 | Phase 1a |
|---|---|---|
| Content Teacher | ContentVec (768d) | **WavLM-large layer 7 (1024d)** |
| データセット | VCTK + tsukuyomi | **VCTK + JVS + tsukuyomi + LibriTTS-R** |
| ステップ数 | 100K | **500K** |
| 学習率 | 2e-4 | **1e-4** |
| Warmup | 0 | **5,000 steps** |
| 損失 | Flow only | Flow only (変更なし) |
| チェックポイント | 新規 | **新規** (d_content 変更のため resume 不可) |
| segment_min_sec | 5.0 | **2.0** (VCTK 発話量 7x 増) |
| マルチデータセット | 非対応 | **対応済み** (`--dataset vctk,jvs,...`) |

**Resume 不可の理由:**
`content_proj` の入力次元が `[512, 768, 1]` → `[512, 1024, 1]` に変わるため、
Phase 0 のチェックポイントから重みを引き継げない。Phase 1a は新規学習となる。

---

## 1. データ取得チェックリスト

### 1.1 現在の状態

```
data/raw/
├── VCTK-Corpus/     ✅ 取得済み (48kHz, 109話者, ~44h)
├── tsukuyomi/       ✅ 取得済み (96kHz, 1話者, ~0.3h)
├── jvs_corpus/      ❌ 未取得
└── libritts_r/      ❌ 未取得

data/cache/
├── vctk/            ✅ ContentVec (768d) で前処理済み → WavLM で再前処理が必要
└── tsukuyomi/       ✅ ContentVec (768d) で前処理済み → WavLM で再前処理が必要
```

### 1.2 JVS コーパスの取得

- **URL:** https://sites.google.com/site/shinaborutnk/research-topics/jvs_corpus
- **サイズ:** ~9 GB (ZIP)
- **話者数:** 100 (JVS001〜JVS100)
- **SR:** 24 kHz
- **ライセンス:** CC BY-SA 4.0
- **配置先:** `data/raw/jvs_corpus/`

```
data/raw/jvs_corpus/
└── jvs_ver1/
    ├── jvs001/
    │   ├── parallel100/wav24kHz16bit/  ← VOICEACTRESS100 (100文)
    │   ├── nonpara30/wav24kHz16bit/    ← 非パラレル朗読 (30文)
    │   └── ...
    └── jvs100/
```

### 1.3 LibriTTS-R の取得

- **URL:** https://www.openslr.org/141/
- **サイズ:** ~65 GB (tar.gz x 6)
- **話者数:** 2,456
- **SR:** 24 kHz
- **ライセンス:** CC BY 4.0
- **配置先:** `data/raw/libritts_r/`

必要なサブセット:
```bash
# train-clean-100 (~54h, 247話者)
# train-clean-360 (~192h, 904話者)
# train-other-500 (~310h, 1,160話者)
# 合計: ~556h, 2,311話者
```

**Phase 1a 推奨:** まず `train-clean-100` + `train-clean-360` (~246h) で開始し、
品質が不足なら `train-other-500` を追加する。

---

## 2. 前処理 (WavLM 1024d)

### 2.1 概要

Phase 1a では ContentVec (768d) → WavLM-large layer 7 (1024d) に切り替える。
既存の VCTK/tsukuyomi キャッシュは再前処理が必要。

```
Phase 0:  content.npy = [768, T]  (ContentVec)
Phase 1a: content.npy = [1024, T] (WavLM-large layer 7)
```

### 2.2 WavLM Extractor の仕様

| 項目 | 値 |
|---|---|
| モデル | `microsoft/wavlm-large` (HuggingFace) |
| パラメータ数 | 317M |
| 入力 SR | 16 kHz (24kHz から自動リサンプル) |
| 出力次元 | 1024d |
| 出力ホップ | 20ms @ 16kHz → 10ms に線形補間 |
| 抽出レイヤー | layer 7 (中間層、content-prosody バランス最適) |
| VRAM 使用量 | ~2 GB (推論時) |

### 2.3 前処理コマンド

```bash
# 1. datasets.yaml を編集: raw_dir を実際のパスに設定、enabled: true にする
#    (content_teacher は CLI で指定するため YAML には不要)

# 2. 個別実行 (推奨 — 進捗を確認しやすい)
# VCTK (既存キャッシュを上書き)
uv run tmrvc-preprocess \
  --dataset vctk \
  --raw-dir data/raw/VCTK-Corpus \
  --cache-dir data/cache \
  --content-teacher wavlm \
  --device xpu -v

# JVS
uv run tmrvc-preprocess \
  --dataset jvs \
  --raw-dir data/raw/jvs_corpus \
  --cache-dir data/cache \
  --content-teacher wavlm \
  --device xpu -v

# つくよみちゃん
uv run tmrvc-preprocess \
  --dataset tsukuyomi \
  --raw-dir data/raw/tsukuyomi \
  --cache-dir data/cache \
  --content-teacher wavlm \
  --device xpu -v

# LibriTTS-R
uv run tmrvc-preprocess \
  --dataset libritts_r \
  --raw-dir data/raw/libritts_r \
  --cache-dir data/cache \
  --content-teacher wavlm \
  --device xpu -v

# 3. 一括実行 (datasets.yaml の enabled フラグに従う)
uv run python scripts/prepare_datasets.py \
  --config configs/datasets.yaml \
  --device xpu
```

### 2.4 前処理の所要時間見積もり (XPU)

| データセット | 発話数 | 推定時間 |
|---|---|---|
| VCTK | ~44,000 | ~2-3h |
| JVS | ~13,000 | ~1h |
| tsukuyomi | ~100 | ~1min |
| LibriTTS-R (clean-100+360) | ~100,000 | ~5-8h |
| **合計** | **~157,000** | **~8-12h** |

### 2.5 前処理後のキャッシュ構造

```
data/cache/
├── _manifests/
│   ├── vctk_train.json
│   ├── jvs_train.json
│   ├── tsukuyomi_train.json
│   └── libritts_r_train.json
├── vctk/train/{speaker_id}/{utt_id}/
│   ├── mel.npy       # [80, T] log-mel (変更なし)
│   ├── content.npy    # [1024, T] WavLM (← 768d から変更)
│   ├── f0.npy         # [1, T] Hz (変更なし)
│   ├── spk_embed.npy  # [192] ECAPA (変更なし)
│   └── meta.json
├── jvs/train/...
├── tsukuyomi/train/...
└── libritts_r/train/...
```

---

## 3. マルチデータセット学習 (実装が必要)

### 3.1 現状の制約

`create_dataloader(dataset=...)` は **単一データセット名のみ** を受け付ける。
Phase 1a の `datasets: [vctk, jvs, tsukuyomi, libritts_r]` を直接渡すことはできない。

### 3.2 実装方針

`TMRVCDataset` と `create_dataloader` をマルチデータセット対応に拡張する。

```python
# Option A: dataset パラメータをリスト対応にする (推奨)
# create_dataloader(dataset="vctk,jvs,libritts_r", ...) or
# create_dataloader(dataset=["vctk", "jvs", "libritts_r"], ...)

# 内部で各データセットの entries を連結し、
# BalancedSpeakerSampler で均等にサンプリングする。
# speaker_id は "{dataset}_{speaker}" 形式で一意化。
```

**必要な変更箇所:**

| ファイル | 変更内容 |
|---|---|
| `tmrvc-data/src/tmrvc_data/dataset.py` | `TMRVCDataset` で複数 dataset の entries を連結 |
| `tmrvc-data/src/tmrvc_data/dataset.py` | `create_dataloader()` の `dataset` パラメータをリスト対応 |
| `tmrvc-train/src/tmrvc_train/cli/train_teacher.py` | `--dataset` をカンマ区切り or 複数指定対応 |
| `configs/train_teacher.yaml` | Phase 1a の `datasets` リストを CLI で自動利用 |

### 3.3 話者バランシング

マルチデータセットでは話者数の偏りが大きい:

| データセット | 話者数 | 発話数/話者 |
|---|---|---|
| VCTK | 109 | ~400 |
| JVS | 100 | ~130 |
| tsukuyomi | 1 | ~100 |
| LibriTTS-R | 2,456 | ~20-50 |

`BalancedSpeakerSampler` が話者単位で均等にサンプリングするため、
データセット間のバランスは自動的に取られる。ただし:

- tsukuyomi (1 話者) は `speaker_groups` で weight を上げる必要がある
- LibriTTS-R は話者が多い分、1 話者あたりの出現頻度が低くなる → 正しい挙動

---

## 4. 学習実行

### 4.1 設定 (configs/train_teacher.yaml Phase 1a)

```yaml
"1a":
  lr: 1.0e-4
  max_steps: 500000
  warmup_steps: 5000
  lambda_stft: 0.0       # Flow loss のみ
  lambda_spk: 0.0
  lambda_ir: 0.0
  datasets: [vctk, jvs, tsukuyomi, libritts_r]
```

### 4.2 CLI コマンド

```bash
# マルチデータセット対応後:
uv run tmrvc-train-teacher \
  --config configs/train_teacher.yaml \
  --cache-dir data/cache \
  --dataset vctk,jvs,tsukuyomi,libritts_r \
  --phase 1a \
  --device xpu \
  --max-frames 400 \
  --num-workers 0 \
  --save-every 10000 \
  -v

# speaker_groups で tsukuyomi を重視する場合:
# configs/train_teacher.yaml に追加:
#   speaker_groups:
#     moe:
#       speakers: ["tsukuyomi/*"]
#       weight: 10
```

### 4.3 学習パラメータ

| パラメータ | 値 | 備考 |
|---|---|---|
| Learning rate | 1e-4 | Phase 0 の半分 |
| Warmup | 5,000 steps | Linear warmup → cosine decay |
| Max steps | 500,000 | Phase 0 の 5 倍 |
| Batch size | 64 | B570 で 1.2 GB VRAM |
| Max frames | 400 | XPU カーネル固定化 |
| d_content | 1024 (default) | WavLM-large |
| Optimizer | AdamW (wd=0.01) | 変更なし |
| 保存間隔 | 10,000 steps | ~2.8h ごと (XPU) |

### 4.4 所要時間見積もり (XPU, B570)

```
1 step = ~0.6s (max_frames=400, batch=64)
500K steps = ~300,000s = ~83h = ~3.5 日

中間チェックポイント: 10K steps ごと → 50 個のチェックポイント
チェックポイントサイズ: ~140 MB (17.2M params × 2 (model + optimizer))
```

### 4.5 品質チェックポイント

| ステップ | 期待 flow loss | 確認事項 |
|---|---|---|
| 10K | ~1.8 | Loss が下降傾向か |
| 50K | ~1.3 | Phase 0 の 100K 水準に近づいているか |
| 100K | ~1.1 | Phase 0 最終値を下回っているか (データ量増加による) |
| 200K | ~0.9 | 収束傾向の確認 |
| 300K | ~0.85 | Plateau チェック — LR decay 効果の確認 |
| 500K | ~0.80 | 最終値 — サンプル生成で品質確認 |

**早期終了条件:**
- 50K steps で loss > 2.0 → 何か問題がある (データ、設定)
- 100K → 200K で loss 改善 < 0.05 → cosine LR の効果を確認、必要なら Phase 1b に移行

### 4.6 サンプル生成 (品質確認)

```bash
uv run python scripts/eval_teacher_sample.py \
  --checkpoint checkpoints/teacher_step{N}.pt \
  --cache-dir data/cache \
  --dataset vctk \
  --speaker vctk_p225 \
  --output-dir eval_samples \
  --steps 32 \
  --sway 1.0 \
  --device xpu
```

---

## 5. 実行順序まとめ

```
Step 1: データ取得
  ├── JVS コーパスのダウンロード → data/raw/jvs_corpus/
  └── LibriTTS-R のダウンロード → data/raw/libritts_r/

Step 2: マルチデータセット実装
  └── create_dataloader() を複数 dataset 対応に拡張

Step 3: WavLM 前処理 (全データセット)
  ├── VCTK:       tmrvc-preprocess --content-teacher wavlm (~2-3h)
  ├── JVS:        tmrvc-preprocess --content-teacher wavlm (~1h)
  ├── tsukuyomi:  tmrvc-preprocess --content-teacher wavlm (~1min)
  └── LibriTTS-R: tmrvc-preprocess --content-teacher wavlm (~5-8h)

Step 4: Phase 1a 学習
  └── tmrvc-train-teacher --phase 1a --dataset vctk,jvs,tsukuyomi,libritts_r (~3.5日)

Step 5: 品質評価
  ├── Flow loss の確認 (目標: ~0.80)
  ├── サンプル生成 + 主観評価
  └── SECS 計算 (目標: > 0.85)

Step 6: 判断
  ├── 品質十分 → Phase 1b へ (STFT + speaker loss 追加)
  └── 品質不足 → データ追加 or ハイパーパラメータ調整
```

---

## 6. Phase 0 からの教訓

1. **Cosine LR scheduling は必須** — Phase 0 で plateau (loss ~1.46) を打破した
2. **XPU では max_frames 固定** — カーネル再コンパイル回避で 10x 高速化
3. **チェックポイントのキー移行** — film_ir → film_acoustic, shape パディング対応済み
4. **Loss の解釈** — flow loss 1.0 は mel 空間 (range [-23, +6]) での MSE。
   サンプル生成で実際の音質を確認する方が信頼性が高い
5. **Griffin-Lim サンプル** — Vocoder 不要で手軽に品質確認できる
