# TMRVC Teacher Training Plan

Kojiro Tanaka — training plan
Created: 2026-02-16 (Asia/Tokyo)

> **原則:** 小さく始めて速く検証し、品質天井を確認してからスケール。
> Cloud GPU (Lambda / RunPod 等) を予算に応じて使い分ける。

---

## 1. コーパス構成

### 1.1 音声データセット (段階的追加)

| Tier | データセット | 言語 | 話者数 | SR | 時間 | ライセンス | 追加タイミング |
|---|---|---|---|---|---|---|---|
| **T1 (必須)** | VCTK | 英語 | 109 | 48 kHz | ~44h | CC BY 4.0 | Phase 0 から |
| **T1 (必須)** | JVS | 日本語 | 100 | 24 kHz | ~30h | CC BY-SA 4.0 | Phase 0 から |
| **T2 (標準)** | LibriTTS-R | 英語 | 2,456 | 24 kHz | ~585h | CC BY 4.0 | Phase 1 から |
| **T3 (拡張)** | Emilia (EN+JA subset) | 多言語 | ~10K+ | 24 kHz | ~5-10Kh | Apache 2.0 | Phase 2 で品質不足の場合 |
| **評価用** | JSUT | 日本語 | 1 | 48 kHz | ~10h | CC BY-SA 4.0 | 評価のみ |

**選定理由:**
- VCTK: 48kHz の高品質多話者データ。話者ごとの発話数が多く VC 学習に最適
- JVS: 日本語対応に不可欠。100 話者 × parallel/nonpara 読み上げ
- LibriTTS-R: 話者多様性の確保 (2,456 話者)。24kHz だが content/speaker の汎化に有効
- Emilia: Seed-VC が使用した大規模コーパス。品質天井の最大化に必要だが、高コスト

### 1.2 RIR データセット (IR-aware 学習用)

| データセット | RIR 数 | 用途 | 取得 |
|---|---|---|---|
| **AIR Database** | ~170 | Training augmentation | OpenSLR #20 |
| **BUT ReverbDB** | ~1,500 | Training augmentation | 公式サイト |
| **ACE Challenge** | ~200 | IR Estimator の GT ラベル | Zenodo |

### 1.3 データ前処理パイプライン

```
Raw Audio
    │
    ├─▶ Resample to 24 kHz
    ├─▶ Loudness normalization (-23 LUFS)
    ├─▶ Silence trimming (Silero VAD, threshold -40 dB)
    ├─▶ Segment into 5-15 sec utterances
    │
    ├─▶ Feature extraction (offline, cached to disk):
    │     ├── ContentVec features (768-dim)  ← Teacher の content teacher
    │     ├── F0 contour (RMVPE, continuous Hz)
    │     ├── Speaker embedding (ECAPA-TDNN, 192-dim)
    │     └── Mel spectrogram (80-bin, 24kHz)
    │
    └─▶ IR Augmentation (online, during training):
          ├── Random RIR convolution (AIR / BUT ReverbDB, p=0.5)
          ├── Random EQ (low/high shelf, ±6 dB, p=0.3)
          ├── Random noise addition (SNR 15-40 dB, p=0.3)
          └── Compute IR params from augmented audio (IR Estimator GT)
```

---

## 2. Teacher アーキテクチャの選択

### 2.1 現行設計 vs 参考設計

| 項目 | 現行 (model-architecture.md §6) | 参考 (system_design.md §3) |
|---|---|---|
| Backbone | U-Net + cross-attention, ~80M | DiT + U-Net skip, ~200M |
| Content | HuBERT-base (768d) | WavLM-large layer 7 (1024d) |
| Pitch | Continuous F0 (1d scalar) | Pitch VQVAE (128d) |
| Speaker | Static embed (192d, FiLM) | Time-varying timbre tokens (64 tokens, cross-attn) |
| Diffusion | v-prediction | OT-CFM v-prediction |

### 2.2 段階的アップグレード戦略

Teacher は GPU 学習・推論のため、パラメータ数制約はない。
品質天井が最終的な Student 品質を決定するため、**Teacher には最高の構成を使う**。

ただし、一気に最大構成にせず、段階的に複雑化する:

```
Step 1 (Phase 0): U-Net 80M + ContentVec + continuous F0
  → アーキテクチャの検証、学習パイプラインのデバッグ

Step 2 (Phase 1): 同じ構成で本格学習
  → 品質天井を測定 (SECS, UTMOS)

Step 3 (Phase 2 以降、品質不足の場合のみ):
  → WavLM-large に切り替え (content 品質向上)
  → Pitch VQVAE 追加 (F0 の robustness 向上)
  → DiT backbone に変更 (品質天井の最大化)
  → Time-varying timbre tokens (話者類似度向上)
```

**判断基準:**
- Phase 1 完了後に SECS ≥ 0.88 → Step 2 のまま蒸留に進む
- Phase 1 完了後に SECS < 0.88 → Step 3 のアップグレードを検討

### 2.3 Content Teacher の選択

| 選択肢 | Params | 品質 | 可用性 | 推奨 |
|---|---|---|---|---|
| **ContentVec** | 95M | 良好 (content 特化) | HuggingFace | Phase 0-1 |
| HuBERT-base | 95M | 良 | HuggingFace | 代替 |
| WavLM-large layer 7 | 315M | 最高 | HuggingFace | Phase 2+ (品質不足時) |

ContentVec は HuBERT-base から content 抽出に特化して学習されたモデルで、
speaker 情報の漏洩が少ない。VC の content encoder teacher としては HuBERT-base より適切。

---

## 3. 学習フェーズ

### Phase 0: アーキテクチャ検証 (1-2 日, ~$20-50)

```
目的: 学習パイプラインの動作確認、ハイパーパラメータの粗い調整
GPU:  1x A100 spot (~$1.5/hr)

データ: VCTK + JVS (T1, ~74h)
モデル: U-Net 80M + ContentVec + F0 + ECAPA speaker

学習:
  - Steps: 50K-100K
  - Batch: 32 (accumulation で調整)
  - lr: 2e-4, cosine schedule
  - 損失: Flow matching loss (v-prediction)

検証:
  - [ ] 学習 loss が収束するか
  - [ ] 生成 mel が入力に対応した形状になるか
  - [ ] 話者変換が機能するか (VCTK 話者間)
  - [ ] F0 追従が機能するか
  - 目標: 最低限の変換動作確認 (SECS > 0.7 程度)
```

### Phase 1: Base Teacher 学習 (3-7 日, ~$100-300)

```
目的: 本格的な Teacher の学習。蒸留の入力となる品質の確保。
GPU:  1x A100 ($1.5-2/hr)

データ: VCTK + JVS + LibriTTS-R (T1+T2, ~660h)
  ※ LibriTTS-R は 24kHz → mel 80-bin で統一
  ※ VCTK は 48→24kHz にリサンプル

学習:
  Phase 1a: Base flow matching
    - Steps: 300K-500K
    - Batch: 64 (gradient accumulation)
    - lr: 1e-4, warmup 5K steps, cosine decay
    - 損失: L_flow = E[||v_θ(x_t, t, cond) - v||²]
    - 時間: ~2-4 日

  Phase 1b: Perceptual loss 追加
    - Phase 1a から fine-tune
    - Steps: 100K-200K
    - 追加損失:
      - L_stft: Multi-res STFT loss (λ=0.5)
        - FFT sizes: [256, 512, 1024]
        - Spectral convergence + log magnitude L1
      - L_spk: ECAPA speaker consistency loss (λ=0.3)
        - cos_sim(ECAPA(generated), ECAPA(target))
    - lr: 5e-5
    - 時間: ~1-2 日

品質目標:
  - SECS ≥ 0.88 (10-step sampling)
  - UTMOS ≥ 3.8
  - 生成音声の明瞭度が十分 (主観評価)
```

### Phase 2: IR-robust 化 (2-3 日, ~$50-100)

```
目的: 残響・マイク特性に頑健な Teacher にする。
GPU:  1x A100

データ: Phase 1 と同じ + RIR augmentation (online)
  - RIR: AIR + BUT ReverbDB からランダム畳み込み (p=0.5)
  - EQ: shelf ±6dB (p=0.3)
  - Noise: SNR 15-40dB (p=0.3)

モデル変更:
  - IR conditioning path を追加
  - IR Estimator (lightweight CNN) を同時に学習
  - IR params (24-dim) を FiLM で Teacher に注入

学習:
  - Phase 1 checkpoint から fine-tune
  - Steps: 100K-200K
  - 追加損失: L_ir = MSE(predicted_ir_params, gt_ir_params) (λ=0.1)
  - lr: 5e-5
  - 時間: ~2-3 日

検証:
  - [ ] Dry 条件: SECS ≥ 0.88 (Phase 1 から劣化なし)
  - [ ] Reverberant 条件 (RT60=0.5s): SECS ≥ 0.84
  - [ ] IR params の推定精度: RT60 RMSE < 0.2s
```

### Phase 3: 品質向上 (必要に応じて, ~$100-500)

Phase 1-2 の品質が不足する場合のみ実行。

```
Option A: データ増量 (+$200-500)
  - Emilia (EN+JA subset, 5-10Kh) を追加
  - 2-4x A100 で 200K-300K steps
  - 話者多様性の大幅向上

Option B: Content encoder 強化 (+$100-200)
  - ContentVec → WavLM-large layer 7 に変更
  - Content features: 768d → 1024d
  - Teacher の入力 projection を再学習
  - 100K-200K steps fine-tune

Option C: Architecture upgrade (+$200-500)
  - U-Net 80M → DiT 200M (system_design.md の構成)
  - Pitch VQVAE + time-varying timbre tokens 追加
  - 500K+ steps (実質 re-train)
  - 品質天井: SECS 0.92+ を目指す
```

---

## 4. Teacher の学習設定詳細

### 4.1 Flow Matching (v-prediction)

```python
# Forward diffusion
def forward_process(x_0, t):
    """x_0: clean mel, t: timestep in [0, 1]"""
    noise = torch.randn_like(x_0)
    alpha_t = 1.0 - t
    sigma_t = t
    x_t = alpha_t * x_0 + sigma_t * noise
    v_target = noise - x_0  # v = α'ε - σ'x_0 = ε - x_0 for linear schedule
    return x_t, v_target

# Training step
def train_step(model, batch):
    content, f0, spk_embed, ir_params, mel_target = batch
    t = torch.rand(batch_size, 1, 1)  # uniform [0, 1]
    x_t, v_target = forward_process(mel_target, t)

    v_pred = model(x_t, t, content, f0, spk_embed, ir_params)
    loss_flow = F.mse_loss(v_pred, v_target)
    return loss_flow
```

### 4.2 Teacher の推論 (sampling)

```python
# Euler ODE solver
def sample(model, content, f0, spk_embed, ir_params, steps=10):
    x = torch.randn(1, 80, T)  # start from noise
    dt = 1.0 / steps

    for i in range(steps):
        t = 1.0 - i * dt
        v = model(x, t, content, f0, spk_embed, ir_params)
        x = x - v * dt  # Euler step (reverse direction)

    return x  # predicted mel
```

### 4.3 Conditioning の入力方法

```
┌── Content (ContentVec 768d) ──▶ Linear(768 → 256) ──▶ cross-attention K/V
│
├── F0 (1d, log scale) ──▶ Linear(1 → 256) ──▶ FiLM (γ, β)
│
├── Speaker (ECAPA 192d) ──▶ Linear(192 → 256) ──▶ FiLM (γ, β)
│     ※ 全フレーム同じ値 (utterance-level)
│
├── IR params (24d) ──▶ Linear(24 → 256) ──▶ FiLM (γ, β)
│     ※ Phase 2 で追加。Phase 0-1 では入力なし
│
└── Timestep (1d) ──▶ sinusoidal embedding (256d) ──▶ AdaLN or FiLM
```

### 4.4 損失関数の構成

| Phase | 損失 | 重み | 備考 |
|---|---|---|---|
| 0, 1a | L_flow (v-prediction MSE) | 1.0 | 基本 |
| 1b+ | + L_stft (multi-res STFT) | 0.5 | mel → wav → STFT で計算 (vocoder 経由) |
| 1b+ | + L_spk (speaker consistency) | 0.3 | ECAPA cos sim |
| 2+ | + L_ir (IR param prediction) | 0.1 | auxiliary head |

### 4.5 Vocoder (Teacher 用)

Teacher の学習・評価時に mel → waveform 変換が必要:

| 選択肢 | 用途 | 備考 |
|---|---|---|
| **Vocos (pre-trained, 24kHz)** | L_stft 計算、評価 | 軽量で十分 |
| BigVGAN-v2 | 最高品質評価 | 評価のみ使用 (重い) |
| Griffin-Lim | デバッグ | 品質低いが依存なし |

推奨: **Vocos (pre-trained)** を凍結して使用。L_stft の勾配は mel を通じて Teacher に伝搬。

---

## 5. VC 学習の仕組み: ペアデータなしでどう学ぶか

Voice Conversion は **パラレルデータ (同じ文を異なる話者が読んだペア) がなくても学習できる**。

### 5.1 Self-reconstruction + Any-to-any 学習

```
学習データの各発話について:
  1. 話者 A の音声から content, F0 を抽出 (ContentVec, RMVPE)
  2. 同じ話者 A の音声から speaker embedding を抽出 (ECAPA)
  3. 同じ話者 A の mel spectrogram を target とする

  Teacher の学習:
    input:  (content_A, f0_A, spk_embed_A, t, noise) → predict v
    target: mel_A (元の音声の mel)

  つまり「話者 A の content + 話者 A の speaker = 話者 A の mel」を復元する
  = Self-reconstruction タスク
```

**なぜこれで VC ができるか:**
- ContentVec は話者非依存の content 表現を出力する (学習時に話者情報が除去されている)
- Speaker embedding は content に依存しない話者特徴を表現する
- 学習後、推論時に `(content_A, f0_A, spk_embed_B)` を入力すれば、
  「話者 A の内容を話者 B の声で生成」が実現される
- Flow matching は conditional generation を学ぶため、
  条件 (content, speaker) の組み合わせを自由に変えられる

### 5.2 Any-to-any の品質を高めるための工夫

```
Training augmentation for better disentanglement:

  1. Cross-speaker conditioning (p=0.2):
     content_A + spk_embed_B → Teacher generates mel
     損失: L_spk(generated, target_B) のみ (content loss なし)
     → 異なる話者の組み合わせを明示的に学習

  2. F0 perturbation (p=0.3):
     F0 をランダムに ±2 semitone シフト
     → F0 と timbre の分離を促進

  3. Content dropout (p=0.1):
     Content features の一部をゼロマスク
     → Speaker embedding への依存を強化

  4. Speaker embedding perturbation (p=0.1):
     ECAPA embedding に小さな noise を加算
     → Speaker space の汎化
```

### 5.3 VCTK の parallel 発話の活用

VCTK は全話者が同じ文セットを読んでいるため、pseudo-parallel ペアが作れる:

```
話者 p225 の "Please call Stella" → content_225, mel_225
話者 p226 の "Please call Stella" → content_226, mel_226

Cross-reconstruction loss (補助的):
  (content_225, spk_embed_226) → Teacher → should ≈ mel_226
  L_parallel = L1(generated_mel, mel_226)

※ Content が本当に話者非依存なら content_225 ≈ content_226 になるはず
※ この loss が大きい = content に話者情報が漏洩している → 改善が必要
```

---

## 6. コスト見積もり

### 6.1 Cloud GPU 料金参考 (2026 年初)

| Provider | GPU | $/hr (spot) | $/hr (on-demand) |
|---|---|---|---|
| Lambda | A100 80GB | ~$1.10 | ~$1.99 |
| RunPod | A100 80GB | ~$1.20 | ~$1.64 |
| Lambda | H100 80GB | ~$2.00 | ~$2.99 |

### 6.2 フェーズ別コスト見積もり

| Phase | GPU | 時間 | コスト (spot) | 累計 |
|---|---|---|---|---|
| Phase 0 (検証) | 1x A100 | ~12-24h | ~$15-30 | ~$15-30 |
| Phase 1 (Base) | 1x A100 | ~72-168h | ~$80-185 | ~$95-215 |
| Phase 2 (IR) | 1x A100 | ~48-72h | ~$55-80 | ~$150-295 |
| **Phase 0-2 合計** | | | | **~$150-300** |
| Phase 3 Option A (データ増) | 2x A100 | ~72-120h | ~$160-265 | ~$310-560 |
| Phase 3 Option C (DiT) | 4x A100 | ~120-240h | ~$530-1060 | ~$680-1355 |

### 6.3 推奨予算プラン

| プラン | 予算 | 期待品質 | 推奨 |
|---|---|---|---|
| **Lean** | ~$150-300 | SECS ~0.87-0.90 | 最初はこれで開始 |
| **Standard** | ~$300-600 | SECS ~0.89-0.92 | 品質不足時にスケール |
| **Premium** | ~$700-1500 | SECS ~0.91-0.93 | SOTA 品質を目指す場合 |

---

## 7. SOTA 学習テクニック (2024-2025)

### 7.1 Sway Sampling (F5-TTS)

推論時の timestep spacing を非一様にする。中間域 (denoising が最も重要な領域) にステップを集中させることで、同じステップ数でも品質が向上する。

```python
# sway_coefficient=1.0 を推奨 (0=uniform)
timesteps = 1.0 - indices.pow(1.0 + sway_coefficient)
```

### 7.2 OT-CFM (Matcha-TTS)

Forward process で noise と data のペアリングを Optimal Transport で最適化。
Minibatch 内で transport cost が最小になるペアを `linear_sum_assignment` で計算。
軌道が直線化し、少ないステップで高品質な生成が可能になる。

### 7.3 CFG-free Training

学習時に確率 `p_uncond` で conditioning をゼロ化 (content, spk_embed, f0)。
推論時に `cfg_scale > 1.0` で 2-pass CFG を使えるが、`cfg_scale=1.0` なら通常と同じ。
学習の安定性が向上し、品質の ceiling が上がる。

### 7.4 Reflow (VoiceFlow)

Teacher 学習後に ODE solver で (noise, clean) ペアを生成し、そのペアで Teacher を再学習。
軌道が直線化され、1-step 蒸留がより容易になる。

```
scripts/generate_reflow_pairs.py で事前にペア生成 → phase=reflow で再学習
```

### 7.5 Global Timbre Memory (TVTSyn)

静的な speaker FiLM を時変の timbre cross-attention に置換。
`spk_embed[192] → memory[8, 48] → cross-attention with content[384, T]`
ONNX I/O は変更なし (内部で展開)。`ConverterStudentGTM` として実装。

### 7.6 DMD2 + Metric Optimization (NeurIPS 2024 + ICML 2025)

**Phase B2 (DMD2):** GAN discriminator による分布レベルの蒸留。
Regression loss を廃止し、MelDiscriminator で real/fake を判別。
Two time-scale update で discriminator を 2x 更新。

**Phase C (Metric Optimization):**
蒸留済み Student を凍結 speaker encoder の SV loss + multi-res STFT loss で直接最適化。
Student が Teacher を超える可能性がある。

---

## 8. 蒸留への接続

Teacher の学習が完了したら、model-architecture.md §2.4 に従って蒸留を実行する。

```
Teacher (80-200M, 10-step) が完成したら:

1. Teacher で蒸留用データを生成:
   各発話 × 各話者ペアで mel_teacher を生成
   → (content, f0, spk_embed, ir_params, mel_teacher) の組を保存

2. Student (7.7M, causal CNN) を蒸留:
   Phase A: ODE trajectory pre-training (v-prediction matching)
   Phase B/B2: DMD/DMD2 (distribution matching)
   Phase C: Metric Optimization (SV + STFT direct optimization)
   + L_stft + L_spk

3. Few-shot adaptation:
   Target speaker の音声 → spk_embed + LoRA delta
   → .tmrvc_speaker ファイルとして保存
```

---

## 8. チェックリスト

### Phase 0 完了条件
- [ ] 学習 loss が monotonic に減少
- [ ] 生成 mel が妥当な形状
- [ ] Self-reconstruction SECS > 0.7
- [ ] Any-to-any 変換が動作 (主観確認)

### Phase 1 完了条件
- [ ] SECS ≥ 0.88 (10-step, VCTK test set)
- [ ] UTMOS ≥ 3.8
- [ ] 日本語音声 (JVS) でも変換が動作
- [ ] 明瞭度が十分 (主観評価)

### Phase 2 完了条件
- [ ] Dry 条件: Phase 1 から SECS 劣化 < 0.02
- [ ] Reverberant 条件: SECS ≥ 0.84
- [ ] IR Estimator: RT60 RMSE < 0.2s

### 蒸留開始条件
- [ ] 上記すべてクリア
- [ ] Teacher の 10-step sampling が安定 (NaN/Inf なし)
- [ ] 蒸留用データ生成パイプラインが動作

---

## 9. Student 妥当性検証とアブレーション計画

### 9.1 目的

`Latency-Quality` スペクトラムで Student の構成妥当性を検証し、
低遅延と高品質の両端で破綻しない設定を特定する。

### 9.2 実験マトリクス

| Axis | 候補 |
|---|---|
| lookahead_hops | 0 / 2 / 4 / 6 |
| converter depth | 6 / 8 / 10 blocks |
| hidden dim | 320 / 384 / 448 |
| vocoder profile | live / hq |
| f0 window | 20 / 40 / 80 ms |
| ir update interval | 10 / 7 / 5 frames |

### 9.3 評価セット

- Clean VC セット (VCTK/JVS)
- Reverberant/noisy VC セット (IR augmentation)
- 早口・子音密度の高い文セット (活舌評価用)
- 疑問文/抑揚強め文セット (イントネーション評価用)

### 9.4 計測項目

- レイテンシ: reported latency, p50/p95 frame time, overrun rate
- 明瞭性: WER, 子音誤り率
- ピッチ: F0 RMSE, F0 corr, V/UV error
- 話者性/音質: SECS, UTMOS, MCD

### 9.5 採択ルール

1. Live (lookahead=0) で overrun rate < 1% を満たすこと。
2. Quality (lookahead=6) で Live 比の明瞭性/ピッチ指標が改善すること。
3. Mix (lookahead=3) がレイテンシと品質の Pareto front 上にあること。
4. パラメータ増加に対する品質改善が閾値未満なら小型構成を採択すること。

### 9.6 成果物

- Student Ablation Report (表 + 図 + 推奨構成)
- 採択構成の ONNX エクスポート一式
- `Latency-Quality` 既定値 (`q_default`) と adaptive しきい値