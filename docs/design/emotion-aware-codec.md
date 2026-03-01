# Emotion-Aware Neural Codec for UCLM

Kojiro Tanaka — Codec Design
Created: 2026-03-01 (Asia/Tokyo)
Updated: 2026-03-01 (Token Spec v2)

> **Core Insight:** 汎用 EnCodec では感情・Non-verbal を十分に制御できない。Emotion/Non-verbal を音響 RVQ と分離した 2 ストリーム設計が必要。

---

## 1. 動機

### 1.1 EnCodec の限界

| 特徴 | EnCodec (汎用) | 必要な機能 |
|---|---|---|
| **感情** | 再生時の制御が弱い | Control Stream で明示制御 |
| **Non-verbal** | 可変長イベント表現が弱い | Event token + duration/intensity |
| **時間変化** | フレーム独立寄り | Causal temporal control |
| **声質制御** | 固定 | Voice State FiLM |

### 1.2 目標

```
「あああ～～～」 + control[moan, valence<0, 1200ms] → 悲しい嗚咽音声
「ははは」 + control[laugh, arousal>0, 300ms] → 自然な笑い声
```

---

## 2. Token Spec v2 (重要)

### 2.1 設計原則

1. RVQ codebook の語彙は常に `0..1023` のみを使用する
2. Non-verbal は RVQ ID に混ぜず、Control Stream で表現する
3. Non-verbal は単一 ID ではなく `開始/継続/終了 + duration + intensity` で表現する
4. UCLM は Acoustic Stream と Control Stream を同時生成する

### 2.2 全体構成

```
Audio In [B,1,240]
    |
    v
Encoder (causal)
    |
    +--> Acoustic RVQ --------------------> A_t [B,8,T]   (0..1023 per codebook)
    |
    +--> Control Tokenizer ---------------> B_t [B,4,T]   (event tuple)

VCEncoder (Information Bottleneck)
    input:  A_t [B,8,T]
    output: content_features [B,T,d_model]  (speaker/style stripped via VQ)

VoiceStateEncoder
    input:  explicit_state [B,T,8], ssl_state [B,T,128], delta_state [B,T,8]
    output: state_cond [B,T,d_model]

UCLM Core (CodecTransformer, dual-stream generation)
    input:  content_features, state_cond, speaker_embed
    output: A_t' logits [B,8,T,1024], B_t' logits [B,4,T,64]

Decoder
    input:  RVQ(A_t') + ControlEncoder(B_t') + VoiceStateFiLM(voice_state)
    output: audio [B,1,T']
```

**Note**: VCEncoder は UCLM の前置段階として A_t から speaker/style 情報を VQ bottleneck で除去し、純粋な content_features を生成する。これにより Voice Conversion 時の speaker leakage を防ぐ。

### 2.3 ストリーム定義

#### Stream A: Acoustic RVQ

- 目的: 音響再構成 (pitch, timbre, spectral, fine temporal details)
- 形式: `A_t = [q0, q1, ..., q7]`
- 制約: `qi in [0, 1023]`
- 備考: 特殊トークンは導入しない

#### Stream B: Control Event Tuple

- 目的: 感情・Non-verbal・イベント時間構造の制御
- 形式: `B_t = [op_id, type_id, dur_id, int_id]`
- 各スロットの意味:
  - `op_id`: `start/hold/end/none`
  - `type_id`: `laugh/sob/sigh/breath/moan/silence`
  - `dur_id`: 50ms 単位の継続長ビン
  - `int_id`: 強度ビン (0..7)

### 2.4 Control Vocabulary

| Range | Token Group | Count | Notes |
|---|---|---|---|
| `0..3` | special (`<ctrl_none>`, `<ctrl_pad>`, `<ctrl_bos>`, `<ctrl_eos>`) | 4 | 制御系列管理 |
| `4..7` | op (`<op_none>`, `<op_start>`, `<op_hold>`, `<op_end>`) | 4 | イベント進行 |
| `8..13` | type (`<laugh>`, `<sob>`, `<sigh>`, `<breath>`, `<moan>`, `<silence>`) | 6 | Non-verbal種別 |
| `14..53` | duration bins (`<dur_1>` ... `<dur_40>`) | 40 | 50ms x 1..40 (=50..2000ms) |
| `54..61` | intensity bins (`<int_0>` ... `<int_7>`) | 8 | 強度量子化 |
| `62..63` | reserved | 2 | 将来拡張 |

**Control vocab size = 64**

### 2.5 Voice State FiLM Conditioning

```python
class VoiceStateFiLM(nn.Module):
    """Feature-wise Linear Modulation with voice_state.

    voice_state [B, 8] -> gamma, beta [B, d_model]
    y = gamma * x + beta
    """

    def __init__(self, d_voice_state: int = 8, d_model: int = 512):
        super().__init__()
        self.proj = nn.Linear(d_voice_state, d_model * 2)

    def forward(self, x: torch.Tensor, voice_state: torch.Tensor) -> torch.Tensor:
        gamma_beta = self.proj(voice_state)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
```

### 2.6 Control Encoder (Decoder 側)

```python
class ControlEncoder(nn.Module):
    """Encode control tuple tokens for decoder-side conditioning.

    input B_t: [B, 4] -> output: [B, d_model, 1]
    """

    def __init__(self, vocab_size: int = 64, d_model: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model * 4, d_model)

    def forward(self, ctrl_tokens: torch.Tensor) -> torch.Tensor:
        # ctrl_tokens: [B, 4]
        e = self.embed(ctrl_tokens)              # [B, 4, d_model]
        e = e.flatten(start_dim=1)               # [B, 4*d_model]
        return self.proj(e).unsqueeze(-1)        # [B, d_model, 1]
```

---

## 3. Non-verbal Event 生成仕様

### 3.1 イベント表現

単一の `<moan_long>` のような固定トークンは使わない。

例: 1200ms の嗚咽

```
Frame t0: B_t = [<op_start>, <moan>, <dur_24>, <int_5>]
Frame t1: B_t = [<op_hold>,  <moan>, <dur_23>, <int_5>]
...
Frame t23: B_t = [<op_end>, <moan>, <dur_1>, <int_4>]
```

### 3.2 UCLM 出力契約 (ONNX Contract 追記対象)

- Acoustic head: `A_t' [B,8]` (8-way categorical x 1024)
- Control head: `B_t' [B,4]` (4-way categorical x 64)
- 1フレーム(10ms)ごとに同時生成
- `B_t'` が `<op_none>` の場合、音響のみ更新

### 3.3 推論時マージ

1. `A_t'` から RVQ latent を再構成
2. `B_t'` を `ControlEncoder` で埋め込み化
3. Decoder block ごとに `VoiceStateFiLM + Control bias` を適用
4. 音声を因果的に 10ms 生成

---

## 4. Temporal Dynamics

### 4.1 時間変化のモデリング

Voice State はフレームごとに変化可能:

```
Frame 0-99:    voice_state = [悲壮感, 徐々に強く]
Frame 100-199: voice_state = [悲壮感, ピーク]
Frame 200-299: voice_state = [悲壮感, 徐々に弱く]
```

### 4.2 実装

```python
class TemporalVoiceStateEncoder(nn.Module):
    """Encode temporal dynamics of voice_state.

    Input: voice_state [B, T, 8]
    Output: temporal_features [B, T, d_model]
    """

    def __init__(self, d_voice_state: int = 8, d_model: int = 512):
        super().__init__()
        self.conv1d = CausalConv1d(d_voice_state, d_model, kernel_size=5)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, voice_state: torch.Tensor) -> torch.Tensor:
        x = voice_state.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        return self.norm(x)
```

### 4.3 Sensual Continuity (官能的連続性)

官能的な表現は「感情ラベル」ではなく、微細パラメータの連続変化で成立する。  
そのため、生成時は毎フレームで直近コンテキストを必ず参照する。

- 入力履歴: `A_{t-k:t-1}, B_{t-k:t-1}` を rolling cache で保持 (推奨 1-2 秒)
- 条件: テキスト条件 + 話者条件 + `voice_state_t` + `delta_voice_state_t`
- 出力: `A_t, B_t` を同時予測し、笑い/嗚咽/息の余韻を次フレームへ伝搬

`delta_voice_state_t = voice_state_t - voice_state_{t-1}` を条件に含めることで、状態の絶対値だけでなく変化速度を学習させる。

### 4.4 Event Hysteresis

Non-verbal イベントは start/hold/end の離散遷移だけでなく、減衰履歴を持つ。

- `event_trace_t = alpha * event_trace_{t-1} + event_embed(B_t)` (`0 < alpha < 1`)
- Decoder では `VoiceStateFiLM + event_trace_t` を併用
- 目的: 笑い終わりの残響、泣き声の詰まり、湿った息の尾を連続的に再現

---

## 5. 学習戦略

### 5.1 Stage 1: Acoustic Codec Pretrain

```
データ: VCTK + JVS + LibriTTS-R
最適化対象:
  - Multi-scale STFT loss
  - Adversarial loss
  - RVQ commitment/codebook losses
出力:
  - Stream A (Acoustic RVQ) を安定化
```

### 5.2 Stage 2: Control Stream 学習

```
データ: Expresso + JVNV + Custom (感情/Non-verbal ラベル)
最適化対象:
  - Control token CE loss (op/type/duration/intensity)
  - Voice state reconstruction loss
  - Emotion classification loss
出力:
  - Stream B (Control tuple) の予測精度向上
```

### 5.3 Stage 3: 分離拘束 (Disentanglement)

```
目的:
  - A_t から感情情報を漏らし過ぎない
  - B_t から音素/話者情報を漏らし過ぎない

追加損失:
  - GRL-based adversarial loss (A_t -> emotion classifier)
  - GRL-based adversarial loss (B_t -> phoneme/speaker classifier)
  - Orthogonality loss (acoustic/control projection)
```

### 5.4 Stage 4: Sensual Continuity 最適化

```
目的:
  - フレーム境界での不自然な断絶を抑える
  - 息・粗さ・無声音の連続遷移を保持する

追加損失:
  - Transition smoothness loss (A_t の急峻ジャンプ抑制)
  - Breath-energy coupling loss (B_t の breath/intensity と高域エネルギーの整合)
  - Delta-state consistency loss (delta_voice_state と生成変化量の整合)
```

### 5.5 Stage 5: Non-verbal 長尺安定化

```
データ: 非言語発声コーパス (笑い、すすり泣き、ため息、息継ぎ)
最適化対象:
  - Long-event consistency loss (start/hold/end の整合)
  - Duration calibration loss (予測 dur_bin と実 dur の一致)
```

---

## 6. 定数 (Token Spec v2)

| Constant | Value | Description |
|---|---|---|
| `FRAME_SIZE` | 240 | 10ms @ 24kHz |
| `N_CODEBOOKS` | 8 | Acoustic RVQ codebooks |
| `RVQ_VOCAB_SIZE` | 1024 | Per-codebook IDs (`0..1023`) |
| `CONTROL_SLOTS` | 4 | `[op, type, dur, int]` |
| `CONTROL_VOCAB_SIZE` | 64 | Control token vocabulary |
| `DURATION_BINS` | 40 | 50ms x 1..40 (=50..2000ms) |
| `INTENSITY_BINS` | 8 | Event intensity bins |
| `NONVERBAL_TYPES` | 6 | laugh/sob/sigh/breath/moan/silence |
| `CODEBOOK_DIM` | 128 | Dimension per codebook |
| `LATENT_DIM` | 512 | Encoder output dimension |
| `D_VOICE_STATE` | 8 | Voice state parameters |

---

## 7. 実装ファイル

```
tmrvc-train/src/tmrvc_train/models/
  ├── emotion_codec.py          # Dual-stream codec core (EmotionAwareCodec, Encoder, Decoder)
  ├── voice_state_film.py       # VoiceStateFiLM conditioning
  ├── voice_state_encoder.py    # VoiceStateEncoder (explicit + ssl + delta → state_cond)
  ├── ssl_extractor.py          # WavLM SSL extractor (WavLMSSLExtractor, StreamingSSLExtractor)
  ├── control_encoder.py        # ControlEncoder for decoder-side
  ├── control_tokenizer.py      # ControlTokenizer, EventTrace, event<->tuple
  ├── disentangle_losses.py     # GRL, orthogonality, transition, breath losses
  ├── uclm.py                   # VCEncoder, VectorQuantizer
  ├── uclm_transformer.py       # CodecTransformer (dual-head)
  └── uclm_model.py             # DisentangledUCLM (unified model)

tmrvc-engine-rs/src/
  └── constants.rs              # Rust constants (TOKEN_SPEC_V2)
```

---

## 8. 設計整合性チェックリスト

- [x] RVQ token ID は全 codebook で `0..1023` のみ
- [x] Non-verbal は Control Stream (`B_t`) でのみ表現
- [x] `B_t = [op, type, dur, int]` の 4 スロット契約が UCLM と一致
- [x] Decoder が `RVQ latent + ControlEncoder + VoiceStateFiLM` を受ける
- [x] Control vocab size が 64 で Python/Rust/ONNX で一致
- [x] `DURATION_BINS=40` と duration 仕様 (50..2000ms) が一致
- [x] 分離拘束 loss が学習設定に含まれる
- [x] `delta_voice_state` と rolling context (1-2秒) が推論契約に含まれる
- [x] Transition/Breath coupling loss が学習設定に含まれる
- [x] Rust 側 constants.rs に `RVQ_VOCAB_SIZE` と `CONTROL_VOCAB_SIZE` を追加
- [x] `ssl_state` [128] が VoiceStateEncoder で処理される (WavLM → ssl_extractor.py)
- [x] `.tmrvc_speaker` に `ssl_state` を保存可能 (metadata 経由)
- [x] Rust 側で `ssl_state` を speaker ファイルから読み込み

---

## 9. Migration Notes (v1 -> v2)

1. `VOCAB_SIZE=1032` を廃止し、`RVQ_VOCAB_SIZE=1024` + `CONTROL_VOCAB_SIZE=64` に分離
2. 旧 special ID (`1024..1031`) は非推奨。Control tokens へ移行
3. UCLM head を single-head から dual-head (`acoustic_head`, `control_head`) に変更
4. ONNX contract を更新し、`control_tokens` 入出力を追加
5. `ssl_state` [128] を VoiceStateEncoder に追加 (WavLM から抽出)
6. `.tmrvc_speaker` に `ssl_state` を保存 (話者登録時に WavLM で抽出)
