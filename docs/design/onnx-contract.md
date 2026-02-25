# TMRVC ONNX Model I/O Contract

Kojiro Tanaka — ONNX contract
Created: 2026-02-16 (Asia/Tokyo)

> **Purpose:** Python (tmrvc-export) と Rust (tmrvc-engine-rs) の間のインターフェース仕様。
> 5つの ONNX モデルの入出力テンソル形状、state 管理、数値パリティ基準を定義する。

---

## 1. 共有定数

### 1.1 Source of Truth: `configs/constants.yaml`

```yaml
# Audio parameters
sample_rate: 24000
n_fft: 1024
hop_length: 240
window_length: 960
n_mels: 80
mel_fmin: 0
mel_fmax: 12000
n_freq_bins: 513          # n_fft / 2 + 1

# Model dimensions
d_content: 256            # Content encoder output dimension
d_speaker: 192            # Speaker embedding dimension
n_ir_params: 24           # IR estimator output: 8 subbands × 3 (RT60, DRR, tilt)
n_voice_source_params: 8  # Voice source params (breathiness, tension, jitter, shimmer, formant_shift, roughness)
n_acoustic_params: 32     # = n_ir_params + n_voice_source_params
d_converter_hidden: 384   # Converter hidden dimension
d_vocoder_features: 513   # Vocoder input: STFT magnitude bins (= n_freq_bins)

# Inference parameters
student_steps: 1          # Converter diffusion steps (distilled to 1)
ir_update_interval: 10    # IR estimator runs every N frames

# LoRA parameters
lora_rank: 4              # LoRA low-rank dimension
lora_alpha: 8             # LoRA scaling factor
n_lora_layers: 4          # Number of cross-attention layers with LoRA
```

### 1.2 自動生成

`scripts/generate_constants.py` により以下を自動生成:

| 出力 | パス | 用途 |
|---|---|---|
| Python module | `tmrvc-core/src/tmrvc_core/constants.py` | 学習・エクスポート |
| Rust constants | `tmrvc-engine-rs/src/constants.rs` | エンジン・VST |

```rust
// Auto-generated from configs/constants.yaml — DO NOT EDIT
pub const SAMPLE_RATE: usize = 24000;
pub const N_FFT: usize = 1024;
pub const HOP_LENGTH: usize = 240;
pub const WINDOW_LENGTH: usize = 960;
pub const N_MELS: usize = 80;
pub const N_FREQ_BINS: usize = 513;
pub const D_CONTENT: usize = 256;
pub const D_SPEAKER: usize = 192;
pub const N_IR_PARAMS: usize = 24;
pub const N_VOICE_SOURCE_PARAMS: usize = 8;
pub const N_ACOUSTIC_PARAMS: usize = 32;
pub const D_CONVERTER_HIDDEN: usize = 384;
pub const D_VOCODER_FEATURES: usize = 513;
pub const IR_UPDATE_INTERVAL: usize = 10;
pub const LORA_RANK: usize = 4;
pub const LORA_DELTA_SIZE: usize = 15872;
```

---

## 2. 5 モデルの I/O 仕様 (Streaming 版)

### 2.1 一覧表

| Model | File | Inputs | Outputs | Execution |
|---|---|---|---|---|
| **content_encoder** | `content_encoder.onnx` | mel_frame, f0, state_in | content, state_out | Per-frame (10ms) |
| **ir_estimator** | `ir_estimator.onnx` | mel_chunk, state_in | acoustic_params, state_out | Every ~10 frames (~100ms) |
| **speaker_encoder** | `speaker_encoder.onnx` | mel_ref | spk_embed, lora_delta | Offline only |
| **converter** | `converter.onnx` | content, spk_embed, acoustic_params, lora_delta, state_in | pred_features, state_out | Per-frame (10ms), 1-step |
| **converter_hq** | `converter_hq.onnx` | content, spk_embed, acoustic_params, lora_delta, state_in | pred_features, state_out | Per-frame (10ms), HQ mode, optional |
| **vocoder** | `vocoder.onnx` | features, state_in | stft_mag, stft_phase, state_out | Per-frame (10ms) |

### 2.2 content_encoder

音声の内容情報 (音素・韻律) を抽出する。ContentVec/WavLM teacher から蒸留された軽量 causal CNN。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `mel_frame` | `[1, 80, 1]` | float32 | 現在フレームの log-mel spectrogram |
| `f0` | `[1, 1, 1]` | float32 | 現在フレームの log-F0 (Hz → log scale, unvoiced = 0) |
| `state_in` | `[1, 256, 28]` | float32 | Causal conv の hidden state (§3 参照) |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `content` | `[1, 256, 1]` | float32 | Content feature vector |
| `state_out` | `[1, 256, 28]` | float32 | Updated hidden state |

### 2.3 ir_estimator

入力音声の音響環境 (残響、マイク特性) と声質特性 (息成分、緊張度等) を推定する。amortized 実行 (10 フレームに 1 回)。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `mel_chunk` | `[1, 80, N]` | float32 | 蓄積された mel frames (N = ir_update_interval = 10) |
| `state_in` | `[1, 128, 6]` | float32 | Causal conv の hidden state |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `acoustic_params` | `[1, 32]` | float32 | Acoustic conditioning params (24 IR + 8 voice source) |
| `state_out` | `[1, 128, 6]` | float32 | Updated hidden state |

**acoustic_params の内訳:**

| Index | Parameter | Subband | Range |
|---|---|---|---|
| 0-7 | RT60 (sec) | 8 subbands (0-375, 375-750, ..., 9375-12000 Hz) | [0.05, 3.0] |
| 8-15 | DRR (dB) | 同上 | [-10, 30] |
| 16-23 | Spectral tilt (dB/oct) | 同上 | [-6, 6] |
| 24-25 | Breathiness (low/high) | 2 subbands (<3kHz, ≥3kHz) | [0, 1] |
| 26-27 | Tension (low/high) | 同上 | [-1, 1] |
| 28 | Jitter | — | [0, 0.1] |
| 29 | Shimmer | — | [0, 0.1] |
| 30 | Formant shift | — | [-1, 1] |
| 31 | Roughness | — | [0, 1] |

### 2.4 speaker_encoder

話者の声質特徴を抽出し、LoRA delta を生成する。Offline のみ (enrollment 時に 1 回実行)。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `mel_ref` | `[1, 80, T_ref]` | float32 | 参照音声の mel spectrogram (T_ref は可変長、推奨: 300-1500 frames = 3-15 sec) |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `spk_embed` | `[1, 192]` | float32 | Speaker embedding vector (L2 normalized) |
| `lora_delta` | `[1, R]` | float32 | LoRA weight delta (flattened). R = n_lora_layers × (d_cond × rank + rank × d_model_x2) |

**lora_delta サイズ計算:**

```
Per layer (FiLM projection LoRA):
  d_cond = d_speaker + n_acoustic_params = 192 + 32 = 224
  d_model_x2 = d_converter_hidden × 2 = 384 × 2 = 768
  lora_A: d_cond × lora_rank = 224 × 4 = 896
  lora_B: lora_rank × d_model_x2 = 4 × 768 = 3,072
  Per layer total: 3,968

Total: n_lora_layers × 3,968 = 4 × 3,968 = 15,872 floats
R = 15,872
lora_delta shape: [1, 15872]
Memory: 15,872 × 4 bytes ≈ 62 KB
```

### 2.5 converter

Content features を target speaker の音響特徴に変換する。1-step denoiser (蒸留済み)。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `content` | `[1, 256, 1]` | float32 | Content encoder 出力 |
| `spk_embed` | `[1, 192]` | float32 | Speaker embedding (cached) |
| `acoustic_params` | `[1, 32]` | float32 | Acoustic conditioning params (cached) |
| `lora_delta` | `[1, 15872]` | float32 | Speaker LoRA weight delta (cached) |
| `state_in` | `[1, 384, 52]` | float32 | Causal conv の hidden state |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `pred_features` | `[1, 513, 1]` | float32 | Predicted STFT features for vocoder |
| `state_out` | `[1, 384, 52]` | float32 | Updated hidden state |

> **Note:** LoRA delta は runtime input として毎フレーム渡される。
> 話者切替時は lora_delta テンソルの差し替えのみで対応可能（ONNX モデルの再ロード不要）。

### 2.5b converter_hq (optional)

HQ mode 用の semi-causal converter。Content Encoder で T=1 ずつ生成した content vector を
7 フレーム分バッファリングし、一括入力する。出力は T=1。

`converter_hq.onnx` が存在しない場合、エンジンは Live mode (causal converter) のみで動作する。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `content` | `[1, 256, 7]` | float32 | 7 フレーム分の content features (1 current + 6 lookahead) |
| `spk_embed` | `[1, 192]` | float32 | Speaker embedding (cached) |
| `acoustic_params` | `[1, 32]` | float32 | Acoustic conditioning params (cached) |
| `lora_delta` | `[1, 15872]` | float32 | Speaker LoRA weight delta (cached) |
| `state_in` | `[1, 384, 46]` | float32 | Semi-causal conv の hidden state |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `pred_features` | `[1, 513, 1]` | float32 | Predicted STFT features for vocoder |
| `state_out` | `[1, 384, 46]` | float32 | Updated hidden state |

> **Note:** Input/output names are identical to the live converter for pipeline compatibility.
> State size is 46 frames (vs 52 for causal) because semi-causal blocks use right context
> instead of left context for some blocks.

### 2.6 vocoder

STFT 特徴量から magnitude と phase を予測する。iSTFT で波形復元。

**Inputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `features` | `[1, 513, 1]` | float32 | Converter 出力 (STFT features) |
| `state_in` | `[1, 256, 14]` | float32 | Causal conv の hidden state |

**Outputs:**

| Name | Shape | Type | Description |
|---|---|---|---|
| `stft_mag` | `[1, 513, 1]` | float32 | Predicted STFT magnitude (linear scale, ≥ 0) |
| `stft_phase` | `[1, 513, 1]` | float32 | Predicted STFT phase (radians, [-π, π]) |
| `state_out` | `[1, 256, 14]` | float32 | Updated hidden state |

> **Output interpretation:**
> `stft_complex = stft_mag * exp(j * stft_phase)`
> → iSTFT (n_fft=1024) → 窓付き信号 (960 samples) → Overlap-Add

---

## 3. State Tensor 仕様

### 3.1 Hidden State Shapes 一覧

| Model | State Shape | Elements | Memory (float32) | Description |
|---|---|---|---|---|
| content_encoder | `[1, 256, 28]` | 7,168 | 28 KB | 4-6 causal conv layers × kernel-1 |
| ir_estimator | `[1, 128, 6]` | 768 | 3 KB | 2-3 causal conv layers |
| converter | `[1, 384, 52]` | 19,968 | 78 KB | 6-8 causal conv layers × kernel-1 |
| converter_hq | `[1, 384, 46]` | 17,664 | 69 KB | 8 semi-causal layers (left_ctx only) |
| vocoder | `[1, 256, 14]` | 3,584 | 14 KB | 3-4 causal conv layers |
| **合計 (Live)** | | **31,488** | **~123 KB** | converter_hq 除く |
| **合計 (HQ)** | | **29,184** | **~114 KB** | converter を converter_hq に置換 |

### 3.2 State Tensor の構造

各 state tensor は causal convolution layers の受容野バッファを保持する。

```
state shape = [1, channels, total_context]

total_context = Σ (kernel_size_i - 1) for each causal conv layer

Example: content_encoder
  Layer 1: channels=256, kernel=7 → context = 6
  Layer 2: channels=256, kernel=7, dilation=1 → context = 6
  Layer 3: channels=256, kernel=7, dilation=2 → context = 12 (= (7-1)*2)
  Layer 4: channels=256, kernel=3 → context = 2
  Layer 5: channels=256, kernel=3 → context = 2
  Total context: 6 + 6 + 12 + 2 + 2 = 28
  State shape: [1, 256, 28]
```

### 3.3 初期化

すべての state tensor は **ゼロ初期化** (silence 入力に対応)。

```cpp
// ランタイム側の初期化 (擬似コード)
memset(contentEncoderState, 0, sizeof(float) * 1 * 256 * 28);
memset(irEstimatorState, 0, sizeof(float) * 1 * 128 * 6);
memset(converterState, 0, sizeof(float) * 1 * 384 * 52);
memset(vocoderState, 0, sizeof(float) * 1 * 256 * 14);
```

### 3.4 Hidden State Ping-Pong (Double Buffering)

各 streaming model は state tensor を 2 つ保持し、フレームごとに交互に使用する。

```
Frame N:   state_A → model → state_B
Frame N+1: state_B → model → state_A
Frame N+2: state_A → model → state_B
...
```

**目的:**
- In-place 更新を避け、state_in と state_out が同じメモリを指す問題を防止
- ONNX Runtime の IO Binding と併用して zero-copy 推論を実現

```cpp
struct PingPongState {
    float* bufferA;  // Pre-allocated
    float* bufferB;  // Pre-allocated
    int current = 0; // 0 = A is input, 1 = B is input

    float* getInput()  const { return current == 0 ? bufferA : bufferB; }
    float* getOutput() const { return current == 0 ? bufferB : bufferA; }
    void swap() { current ^= 1; }
};
```

---

## 4. 数値パリティ基準

### 4.1 Python vs Rust パリティ

| 基準 | 値 | 対象 |
|---|---|---|
| **Max absolute difference** | < 1e-5 | 全モデルの全出力テンソル |
| **Mean absolute difference** | < 1e-6 | 全モデルの全出力テンソル |
| **Relative error (non-zero elements)** | < 1e-4 | 全モデルの全出力テンソル |

### 4.2 パリティ検証手順

```python
# tmrvc-export/src/tmrvc_export/verify_parity.py

def verify_parity(model_name: str, test_inputs: dict, rtol=1e-4, atol=1e-5):
    """
    1. PyTorch model でテスト入力を推論
    2. 同じテスト入力を ONNX Runtime (Python) で推論
    3. 同じテスト入力を Rust engine で推論 (subprocess call)
    4. 3つの出力を比較
    """
    pytorch_out = run_pytorch(model_name, test_inputs)
    onnx_out = run_onnxruntime_python(model_name, test_inputs)
    rust_out = run_rust_engine(model_name, test_inputs)

    # PyTorch vs ONNX
    assert_close(pytorch_out, onnx_out, atol=atol, rtol=rtol)
    # ONNX vs Rust
    assert_close(onnx_out, rust_out, atol=atol, rtol=rtol)
```

### 4.3 テストケース

| テストケース | 内容 | 目的 |
|---|---|---|
| Zero input | 全入力ゼロ + 初期 state | 初期化の一致確認 |
| Single frame | 1 フレームのランダム入力 | 基本動作 |
| 10 frames sequential | 10 フレーム逐次処理 | State 伝搬の一致 |
| Known audio | 既知の音声ファイル (5 sec) | E2E の音質一致 |

---

## 5. 量子化

### 5.1 INT8 Dynamic Quantization

| 対象 | 量子化方式 | 期待される効果 |
|---|---|---|
| content_encoder | INT8 dynamic | 2-3x speedup, ~0.25x size |
| converter | INT8 dynamic | 2-3x speedup, ~0.25x size |
| vocoder | INT8 dynamic | 2-3x speedup, ~0.25x size |
| ir_estimator | INT8 dynamic | 2-3x speedup, ~0.25x size |
| speaker_encoder | 量子化しない | Offline 実行のため速度不要 |

### 5.2 量子化手順

```python
# tmrvc-export/src/tmrvc_export/quantize.py
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(input_path: str, output_path: str):
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        # State tensors は FP32 のまま (精度劣化防止)
        nodes_to_exclude=["state_in", "state_out"],
    )
```

### 5.3 量子化品質基準

| 指標 | 許容値 | 測定方法 |
|---|---|---|
| FP32 vs INT8 max abs diff | < 0.01 | 100 フレームの出力差 |
| Speaker similarity degradation | < 0.02 | ECAPA cosine on test set |
| UTMOS degradation | < 0.1 | 評価用テストセット |

---

## 6. `.tmrvc_speaker` ファイルフォーマット (v2)

### 6.1 バイナリレイアウト

```
Offset   Size (bytes)    Field
──────   ────────────    ──────────────────────────
0x0000   4               Magic: "TMSP" (0x544D5350)
0x0004   4               Version: uint32_le = 2
0x0008   4               embed_size: uint32_le = 192
0x000C   4               lora_size: uint32_le = 15872
0x0010   4               metadata_size: uint32_le (JSON UTF-8 byte count)
0x0014   4               thumbnail_size: uint32_le (常に 0: サムネイルは metadata JSON 内に base64 格納)
0x0018   768             spk_embed: float32_le[192]
0x0318   63488           lora_delta: float32_le[15872]
0x10118  metadata_size   metadata_json: UTF-8 JSON
         32              checksum: SHA-256 of all preceding bytes
```

Header = 24 bytes. サムネイルはメタデータ JSON 内に `thumbnail_b64` として base64 エンコードで格納。

### 6.2 メタデータ JSON スキーマ

```json
{
  "profile_name": "My Voice",
  "author_name": "Author Name",
  "co_author_name": "",
  "licence_url": "",
  "thumbnail_b64": "iVBORw0KGgo...",
  "created_at": "2026-02-18T12:00:00Z",
  "description": "",
  "source_audio_files": ["ref1.wav", "ref2.wav"],
  "source_sample_count": 480000,
  "training_mode": "embedding",
  "checkpoint_name": ""
}
```

- `profile_name`: プロファイル表示名
- `author_name`: 作成者名
- `co_author_name`: 共同作成者名（任意、空文字 = なし）
- `licence_url`: ライセンス確認 URL（任意、空文字 = なし）
- `thumbnail_b64`: 100×100px RGB PNG を base64 エンコードした文字列（空文字 = なし）
- `training_mode`: `"embedding"` | `"finetune"`
- `checkpoint_name`: fine-tune 時のみ非空
- `voice_source_preset`: 8 floats (voice source params) or `null` — ブレンド用プリセット
- `voice_source_param_names`: パラメータ名リスト（自己文書化用）
- 文字列フィールドは空文字許容。将来拡張時は不明キーを無視する。

### 6.3 サムネイル

- mel スペクトログラムのヒートマップ画像
- サイズ: 100×100 px、RGB
- 値域を min/max 正規化 → inferno 風カラーマップ、バイリニア補間
- メタデータ JSON の `thumbnail_b64` フィールドに base64 エンコードで格納
- `thumbnail_b64` が空文字の場合はサムネイルなし

### 6.4 読み込み検証

```
1. ファイルサイズ ≥ HEADER(24) + embed(768) + lora(98304) + checksum(32)
2. Magic number 検証 ("TMSP")
3. Version 検証 (== 2)
4. header の metadata_size / thumbnail_size から total size を計算・検証
5. SHA-256 checksum 検証
6. spk_embed, lora_delta, metadata を解析
7. metadata.thumbnail_b64 から base64 デコードでサムネイル取得
```

---

## 7. Model Directory Structure

```
models/
├── fp32/
│   ├── content_encoder.onnx     (~6 MB, ~1.5M params)
│   ├── ir_estimator.onnx        (~4-12 MB, ~1-3M params)
│   ├── speaker_encoder.onnx     (~20-40 MB, ~5-10M params, offline)
│   ├── converter.onnx           (~12-20 MB, ~3-5M params)
│   ├── converter_hq.onnx       (~12-20 MB, optional, HQ mode)
│   └── vocoder.onnx             (~1.3-20 MB, ~0.33-5M params)
│
├── int8/
│   ├── content_encoder_int8.onnx
│   ├── ir_estimator_int8.onnx
│   ├── converter_int8.onnx
│   └── vocoder_int8.onnx
│
├── constants.yaml               # Copy of configs/constants.yaml
└── metadata.json                # Model version, training info, etc.
```

### metadata.json

```json
{
    "version": "1.0.0",
    "created_at": "2026-xx-xx",
    "training_config": "train_student.yaml",
    "teacher_checkpoint": "teacher_v1_step800K",
    "constants_hash": "sha256:...",
    "models": {
        "content_encoder": {
            "params": 1500000,
            "opset": 17,
            "quantized": true
        },
        "converter": {
            "params": 4000000,
            "opset": 17,
            "quantized": true
        },
        "vocoder": {
            "params": 330000,
            "opset": 17,
            "quantized": true
        },
        "ir_estimator": {
            "params": 2000000,
            "opset": 17,
            "quantized": true
        },
        "speaker_encoder": {
            "params": 7000000,
            "opset": 17,
            "quantized": false
        }
    }
}
```

---

## 8. IO Binding (Zero-Copy Inference)

### 8.1 概要

ONNX Runtime の IO Binding を使用して、TensorPool の pre-allocated メモリ上で直接推論を行う。
（現行 Rust 実装では `ort` crate を介して同等の zero-copy 指向を維持する）。

### 8.2 呼び出しフロー

```cpp
// 1. Pre-allocated buffers from TensorPool
float* melFrame   = tensorPool_.getMelFrame();        // [1, 80, 1]
float* stateIn    = contentEncState_.getInput();      // [1, 256, 28]
float* content    = tensorPool_.getContent();          // [1, 256, 1]
float* stateOut   = contentEncState_.getOutput();      // [1, 256, 28]

// 2. Create OrtValue wrappers (no allocation)
OrtValue* inputMel   = CreateTensorWithDataAsOrtValue(melFrame, ...);
OrtValue* inputState = CreateTensorWithDataAsOrtValue(stateIn, ...);
OrtValue* outContent = CreateTensorWithDataAsOrtValue(content, ...);
OrtValue* outState   = CreateTensorWithDataAsOrtValue(stateOut, ...);

// 3. Bind inputs and outputs
OrtBindInput(binding, "mel_frame", inputMel);
OrtBindInput(binding, "state_in", inputState);
OrtBindOutput(binding, "content", outContent);
OrtBindOutput(binding, "state_out", outState);

// 4. Run (zero-copy: reads from melFrame, writes to content/stateOut)
OrtRunWithBinding(session, binding);

// 5. Ping-pong swap
contentEncState_.swap();
```

---

## 9. 整合性チェックリスト

- [x] 全モデルの input/output shapes が model-architecture.md と整合
- [x] State tensor shapes が各モデルの causal conv 構成と整合
- [x] constants.yaml の値が streaming-design.md のパラメータと一致
- [x] lora_delta サイズが model-architecture.md の LoRA 設計と一致
- [x] IO Binding 方針が `tmrvc-engine-rs` の TensorPool 実装と整合
- [x] .tmrvc_speaker format が architecture.md の enrollment フローと整合
- [x] 量子化対象が実行頻度 (per-frame models) と整合
