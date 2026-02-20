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
| **T1 (必須)** | つくよみちゃんコーパス | 日本語 | 1 | 96 kHz | ~0.3h (100文) | 独自 (§1.4) | Phase 0 から |
| **T2 (標準)** | LibriTTS-R | 英語 | 2,456 | 24 kHz | ~585h | CC BY 4.0 | Phase 1 から |
| **T3 (拡張)** | Emilia (EN+JA subset) | 多言語 | ~10K+ | 24 kHz | ~5-10Kh | Apache 2.0 | Phase 2 で品質不足の場合 |
| **評価用** | JSUT | 日本語 | 1 | 48 kHz | ~10h | CC BY-SA 4.0 | 評価のみ |

**選定理由:**
- VCTK: 48kHz の高品質多話者データ。話者ごとの発話数が多く VC 学習に最適
- JVS: 日本語対応に不可欠。100 話者 × parallel/nonpara 読み上げ
- つくよみちゃんコーパス: 萌え声・アニメ声スタイルの日本語データ。JVS と同じ声優統計コーパス 100 文を高音ウィスパー系の声で読み上げ。Voice source parameters (breathiness, formant_shift 等) の学習に不可欠。96kHz 高品質録音
- LibriTTS-R: 話者多様性の確保 (2,456 話者)。24kHz だが content/speaker の汎化に有効
- Emilia: Seed-VC が使用した大規模コーパス。品質天井の最大化に必要だが、高コスト

### 1.2 つくよみちゃんコーパス ライセンスノート

- **配布元:** https://tyc.rei-yumesaki.net/material/corpus/
- **ライセンス:** 独自 (著作権法第 30 条の 4 に基づく情報解析目的利用)
  - 個人・法人、営利・非営利、研究・開発を問わず利用可能
  - 声質を使用した音声変換ソフトの公開 (有料含む) 可能
  - CC BY-SA のコピーレフト (継承) は不要
- **クレジット:** 必須。学習済みモデル配布時に以下を明記:
  - 「VOICEVOX:つくよみちゃん」ではなく **「つくよみちゃんコーパス」** と記載
  - 公式サイト (https://tyc.rei-yumesaki.net/) へのリンク
- **コーパス再配布:** 禁止 (ダウンロードは公式サイトから)
- **仕様:** 96 kHz / 32-bit float、JVS 準拠 100 文
- **前処理:** 96 kHz → 24 kHz にリサンプル後、通常パイプラインに投入

### 1.3 RIR データセット (IR-aware 学習用)

| データセット | RIR 数 | 用途 | 取得 |
|---|---|---|---|
| **AIR Database** | ~170 | Training augmentation | OpenSLR #20 |
| **BUT ReverbDB** | ~1,500 | Training augmentation | 公式サイト |
| **ACE Challenge** | ~200 | IR Estimator の GT ラベル | Zenodo |

### 1.4 データ前処理パイプライン

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

### 2.1 現行設計 (品質天井最大化版)

| 項目 | 設定 | 根拠 |
|---|---|---|
| **Backbone** | U-Net + cross-attention, ~80M | GPU推論のためパラメータ制約なし |
| **Content** | WavLM-large layer 7 (1024d) | 最高品質、speaker leakage 少ない |
| **Content VQ** | Factorized VQ bottleneck (2×8192) | 残留 speaker 情報の除去 |
| **Pitch** | Continuous F0 (1d scalar) | 十分実績あり |
| **Speaker** | GTM (8×48) + LoRA | 時変 timbre、few-shot 対応 |
| **Diffusion** | OT-CFM v-prediction | 軌道直線化、1-step 蒸留容易 |

### 2.2 段階的アップグレード戦略 (更新)

```
Step 1 (Phase 0): U-Net 17M + ContentVec + Rectified Flow
  → アーキテクチャの検証、学習パイプラインのデバッグ
  → 目標: SECS > 0.75

Step 2 (Phase 1): WavLM-large + OT-CFM + VQ bottleneck (本番構成)
  → 品質天井の最大化
  → OT-CFM で軌道直線化
  → VQ で speaker leakage 対策
  → 目標: SECS ≥ 0.90, UTMOS ≥ 4.0

Step 3 (Phase 2): IR-robust 化 + Voice Source 蒸留
  → RIR augmentation + Voice Source external distillation
  → 目標: 残響条件下 SECS ≥ 0.86
```

**判断基準 (更新):**
- Phase 0 完了後: SECS > 0.75 なら Phase 1 に進む
- Phase 1 完了後: SECS ≥ 0.90 なら蒸留に進む
- SECS < 0.90 の場合: データ増量 (Emilia) または構成見直し

### 2.3 Content Teacher の選択 (更新)

| 選択肢 | Params | 出力次元 | 品質 | 推奨 |
|---|---|---|---|---|
| ContentVec | 95M | 768d | 良好 | Phase 0 (検証用) |
| HuBERT-base | 95M | 768d | 良 | 代替 |
| **WavLM-large layer 7** | **317M** | **1024d** | **最高** | **Phase 1+ (本番)** |

WavLM-large は multi-layer 表現を持ち、layer 7 (中間層) が content と prosody の
バランスに最適。ContentVec よりも speaker leakage が少なく、VC に適する。

---

## 3. 学習フェーズ

### Phase 0: アーキテクチャ検証 (1-2 日, ~$20-50)

```
目的: 学習パイプラインの動作確認、ハイパーパラメータの粗い調整
GPU:  1x A100 spot (~$1.5/hr)

データ: VCTK + JVS (T1, ~74h)
モデル: U-Net 17M + ContentVec + F0 + ECAPA speaker

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

### Phase 1: Base Teacher 学習 (3-7 日, ~$150-400)

```
目的: 本格的な Teacher の学習。蒸留の入力となる品質の確保。
GPU:  1x A100 ($1.5-2/hr)

データ: VCTK + JVS + LibriTTS-R + つくよみちゃん (T1+T2, ~660h)
  ※ LibriTTS-R は 24kHz → mel 80-bin で統一
  ※ VCTK は 48→24kHz にリサンプル
  ※ つくよみちゃんは voice source 多様性に寄与

モデル構成 (品質天井最大化):
  - Content: WavLM-large layer 7 (1024d) → projection → 256d
  - VQ Bottleneck: Factorized VQ (2 codebooks × 8192 entries × 128d)
  - Flow: OT-CFM with optimal transport pairing
  - Speaker: GTM (8×48) cross-attention

学習:
  Phase 1a: Base OT-CFM training
    - Steps: 300K-500K
    - Batch: 64 (gradient accumulation)
    - lr: 1e-4, warmup 5K steps, cosine decay
    - 損失: 
        L_flow = OT-CFM velocity MSE
        L_commit = λ_commit × VQ commitment loss (λ=0.25)
    - 時間: ~3-5 日

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

品質目標 (更新):
  - SECS ≥ 0.90 (10-step sampling)
  - UTMOS ≥ 4.0
  - Speaker leakage: ↓20% vs ContentVec (VQ 効果)
  - 生成音声の明瞭度が十分 (主観評価)
```

### Phase 2: IR-robust 化 + Voice Source 蒸留 (3-5 日, ~$100-200)

```
目的: 残響・マイク特性に頑健な Teacher にする + Voice Source の明示的学習。
GPU:  1x A100

データ: Phase 1 と同じ + RIR augmentation (online)
  - RIR: AIR + BUT ReverbDB からランダム畳み込み (p=0.5)
  - EQ: shelf ±6dB (p=0.3)
  - Noise: SNR 15-40dB (p=0.3)

モデル変更:
  - Acoustic conditioning path を追加
  - Acoustic Estimator (lightweight CNN, 32-dim output) を同時に学習
  - Voice Source 外部蒸留 (下記参照)

学習:
  - Phase 1 checkpoint から fine-tune
  - Steps: 100K-200K
  - 損失:
      L_ir = MSE(predicted_acoustic_params[0:24], gt_ir_params)
      L_voice_distill = MSE(predicted[24:32], external_estimator(audio))
      λ_ir = 0.1, λ_voice = 0.2
  - lr: 5e-5
  - 時間: ~3-5 日

Voice Source 外部蒸留:
  外部の事前学習済み voice source 推定器 (例: NKF-stack, 
  またはカスタム訓練した breathiness/tension 推定 CNN) を用いて
  Teacher 側の Voice Source params を教師あり学習する。
  
  # 推定器 (凍結)
  external_voice_estimator = load_pretrained_voice_source_model()
  
  # 学習時
  with torch.no_grad():
      voice_gt = external_voice_estimator(audio)  # [B, 8]
  voice_pred = acoustic_estimator(audio)[:, 24:32]
  L_voice = MSE(voice_pred, voice_gt)

検証:
  - [ ] Dry 条件: SECS ≥ 0.90 (Phase 1 から劣化なし)
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
  - U-Net 17M → DiT ~200M (system_design.md の構成)
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
    content, f0, spk_embed, acoustic_params, mel_target = batch
    t = torch.rand(batch_size, 1, 1)  # uniform [0, 1]
    x_t, v_target = forward_process(mel_target, t)

    v_pred = model(x_t, t, content, f0, spk_embed, acoustic_params)
    loss_flow = F.mse_loss(v_pred, v_target)
    return loss_flow
```

### 4.2 Teacher の推論 (sampling)

```python
# Euler ODE solver
def sample(model, content, f0, spk_embed, acoustic_params, steps=10):
    x = torch.randn(1, 80, T)  # start from noise
    dt = 1.0 / steps

    for i in range(steps):
        t = 1.0 - i * dt
        v = model(x, t, content, f0, spk_embed, acoustic_params)
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
├── Acoustic params (32d) ──▶ Linear(32 → 256) ──▶ FiLM (γ, β)
│     ※ Phase 2 で追加。Phase 0-1 では入力なし
│     ※ 24 IR (環境) + 8 voice source (声質)
│     ※ 蒸留時に VoiceSourceStatsTracker で話者別統計を収集
│     ※ 推論時は voice_source_preset とブレンド可能 (alpha 制御)
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

### 7.7 Voice Source Presets (データ駆動プリセット)

蒸留 Phase A/B/B2/C の全ステップで `VoiceSourceStatsTracker` が `acoustic_params[24..31]`
の話者別 running mean を収集する。チェックポイント保存時に `.voice_source_stats.json` を自動出力。

```python
# 蒸留後: 萌え声グループのプリセット生成
compute_group_preset("stats.json", patterns=["moe/*"], output_path="moe_preset.json")
# → {"preset": [0.72, 0.45, ...], "matched_speakers": [...], "n_speakers": 5}
```

推論時は `.tmrvc_speaker` metadata の `voice_source_preset[8]` と推定値を alpha ブレンド:
```
blended[24+i] = lerp(estimated[24+i], preset[i], alpha)
```
- `alpha=0`: 推定値そのまま（既存動作と同一）
- `alpha=1`: プリセット完全適用（「萌え寄せ」）

詳細: `docs/design/acoustic-condition-pathway.md` §Voice Source Presets

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
Teacher (17-200M, 10-step) が完成したら:

1. Teacher で蒸留用データを生成:
   各発話 × 各話者ペアで mel_teacher を生成
   → (content, f0, spk_embed, acoustic_params, mel_teacher) の組を保存

2. Student (7.7M, causal CNN) を蒸留:
   Phase A: ODE trajectory pre-training (v-prediction matching)
   Phase B/B2: DMD/DMD2 (distribution matching)
   Phase C: Metric Optimization (SV + STFT direct optimization)
   + L_stft + L_spk
   ※ 全 Phase で VoiceSourceStatsTracker が voice source params の話者別統計を自動収集

3. Voice source presets 生成:
   蒸留完了後に compute_group_preset() で目的のスタイルグループの
   voice source params 平均値を .tmrvc_speaker metadata に格納

4. Few-shot adaptation:
   Target speaker の音声 → spk_embed + LoRA delta + voice_source_preset
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

### 蒸留完了後条件
- [ ] VoiceSourceStatsTracker の統計 JSON が出力されている
- [ ] 目的グループの voice_source_preset が compute_group_preset() で生成可能

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