# TMRVC ONNX Model I/O Contract (UCLM Token Spec v2)

Kojiro Tanaka — ONNX contract
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-03-01 — UCLM dual-stream (`A_t` / `B_t`) 契約に更新

> **Purpose:** Python (`tmrvc-export`) と Rust (`tmrvc-engine-rs`) 間のインターフェース仕様。
> Disentangled UCLM (SOTAアーキテクチャ) で使用する ONNX モデル I/O、状態テンソル、パリティ基準を定義する。

---

## 1. Shared Constants

### 1.1 Source of Truth: `configs/constants.yaml`

```yaml
# Audio
sample_rate: 24000
frame_size: 240              # 10ms
frame_rate: 100

# Acoustic stream (A_t)
n_codebooks: 8
rvq_vocab_size: 1024         # valid IDs: 0..1023
codebook_dim: 128
latent_dim: 512

# Control stream (B_t)
control_slots: 4             # [op, type, dur, int]
control_vocab_size: 64

# Conditions
d_speaker: 192
d_voice_state_explicit: 8
d_voice_state_ssl: 128

# UCLM core
d_model: 512
n_layers: 12
n_heads: 8
context_frames: 200          # 2.0 sec at 10ms/frame
```

### 1.2 Runtime Invariants

- `A_t` は各 codebook ごとに `0..1023` のみを使用
- `B_t` は `[op, type, dur, int]` の 4 スロットを常に維持
- `frame_size=240` (10ms @24kHz) を推論の最小単位とする
- 生成時は rolling context (`A_{t-k:t-1}`, `B_{t-k:t-1}`) を常時参照
- `delta_voice_state = voice_state_t - voice_state_{t-1}` を UCLM 条件に含める

---

## 2. ONNX Model Set

### 2.1 Model List

| Model | File | Inputs | Outputs | Execution |
|---|---|---|---|---|
| **codec_encoder** | `codec_encoder.onnx` | `audio_frame`, `state_in` | `acoustic_tokens`, `state_out` | per-frame (10ms) |
| **vc_encoder** | `vc_encoder.onnx` | `source_A_t` | `vq_content_features` | VC時のみ |
| **voice_state_enc**| `voice_state_enc.onnx`| `explicit_state`, `ssl_state`, `delta_state` | `state_cond` | per-frame (10ms) |
| **uclm_core** | `uclm_core.onnx` | `content_features`, `b_ctx`, `spk_embed`, `state_cond`, `cfg_scale`, `kv_cache_in` | `logits_a`, `logits_b`, `kv_cache_out` | per-frame (10ms) |
| **codec_decoder** | `codec_decoder.onnx` | `acoustic_tokens`, `control_tokens`, `voice_state`, `event_trace_in`, `state_in` | `audio_frame`, `event_trace_out`, `state_out` | per-frame (10ms) |
| **speaker_encoder** | `speaker_encoder.onnx` | `mel_ref` | `spk_embed` | offline |

---

## 3. Per-Model I/O Contract

### 3.1 `codec_encoder.onnx`

Causal codec encoder + RVQ quantizer.

**Inputs**

| Name | Shape | Type | Description |
|---|---|---|---|
| `audio_frame` | `[1, 1, 240]` | `float32` | 10ms frame @24kHz |
| `state_in` | `[1, ENC_STATE_DIM, ENC_STATE_FRAMES]` | `float32` | encoder causal state |

**Outputs**

| Name | Shape | Type | Description |
|---|---|---|---|
| `acoustic_tokens` | `[1, 8]` | `int64` | `A_t` (8 codebooks, each `0..1023`) |
| `state_out` | `[1, ENC_STATE_DIM, ENC_STATE_FRAMES]` | `float32` | updated state |

---

### 3.2 `vc_encoder.onnx` (VC Mode Only)

Information Bottleneck (VQ) to remove speaker/style information from source tokens.

**Inputs**
| Name | Shape | Type | Description |
|---|---|---|---|
| `source_A_t` | `[1, 8, L]` | `int64` | source acoustic tokens |

**Outputs**
| Name | Shape | Type | Description |
|---|---|---|---|
| `vq_content_features` | `[1, d_model, L]` | `float32` | pure content representation |

---

### 3.3 `voice_state_enc.onnx`

Combines 8-dim explicit parameters, WavLM SSL features, and delta state into a single condition vector.

**Inputs**
| Name | Shape | Type | Description |
|---|---|---|---|
| `explicit_state` | `[1, 8]` | `float32` | heuristic parameters (breathiness, etc.) |
| `ssl_state` | `[1, 128]` | `float32` | latent style space from WavLM |
| `delta_state` | `[1, 8]` | `float32` | `voice_state_t - voice_state_{t-1}` for temporal dynamics |

**Outputs**
| Name | Shape | Type | Description |
|---|---|---|---|
| `state_cond` | `[1, d_model]` | `float32` | fused style condition |

---

### 3.4 `uclm_core.onnx`

Dual-stream token predictor for TTS/VC.

**Inputs**

| Name | Shape | Type | Description |
|---|---|---|---|
| `content_features` | `[1, d_model, L]` | `float32` | VQ bottlenecked features (VC) or text features (TTS) |
| `b_ctx` | `[1, 4, L]` | `int64` | control context (`B_{t-L:t-1}`) |
| `spk_embed` | `[1, 192]` | `float32` | speaker embedding |
| `state_cond` | `[1, d_model]` | `float32` | explicit + ssl state combined (from voice_state_enc) |
| `cfg_scale` | `[1]` | `float32` | CFG amplification scale (e.g., 1.5) |
| `kv_cache_in` | `[N_CACHE]` | `float32` | flattened KV cache |

`L` is context frames (default 200 = 2 sec).

**Outputs**

| Name | Shape | Type | Description |
|---|---|---|---|
| `logits_a` | `[1, 8, 1024]` | `float32` | next `A_t` distribution |
| `logits_b` | `[1, 4, 64]` | `float32` | next `B_t` distribution |
| `kv_cache_out` | `[N_CACHE]` | `float32` | updated KV cache |

**Sampling contract (Rust side):**

```rust
// Acoustic stream: 8 independent categorical draws
let next_a: [i64; 8] = sample_per_head(logits_a, temperature_a, top_k_a);

// Control stream: 4 slots [op, type, dur, int]
let next_b: [i64; 4] = sample_per_slot(logits_b, temperature_b, top_k_b);
```

---

### 3.5 `codec_decoder.onnx`

RVQ dequantization + causal decoder with control conditioning.

> **Note:** Current implementation uses simplified Linear projection instead of ConvTranspose1d backbone.
> Parity verification is skipped for this model. Full parity will be verified with trained checkpoints.

**Inputs**

| Name | Shape | Type | Description |
|---|---|---|---|
| `acoustic_tokens` | `[1, 8]` | `int64` | `A_t` |
| `control_tokens` | `[1, 4]` | `int64` | `B_t = [op,type,dur,int]` |
| `voice_state` | `[1, 8]` | `float32` | frame condition |
| `event_trace_in` | `[1, D_EVENT_TRACE]` | `float32` | hysteresis trace for non-verbal tails |
| `state_in` | `[1, DEC_STATE_DIM, DEC_STATE_FRAMES]` | `float32` | decoder causal state |

**Outputs**

| Name | Shape | Type | Description |
|---|---|---|---|
| `audio_frame` | `[1, 1, 240]` | `float32` | decoded 10ms audio |
| `event_trace_out` | `[1, D_EVENT_TRACE]` | `float32` | updated event trace |
| `state_out` | `[1, DEC_STATE_DIM, DEC_STATE_FRAMES]` | `float32` | updated decoder state |

---

### 3.6 `speaker_encoder.onnx` (Offline)

| Name | Shape | Type | Description |
|---|---|---|---|
| `mel_ref` | `[1, 80, T]` | `float32` | reference log-mel |
| `spk_embed` | `[1, 192]` | `float32` | normalized speaker embedding |

---

## 4. Streaming State Management

### 4.1 Persistent State Components

| State | Owner | Type |
|---|---|---|
| `enc_state` | codec_encoder | conv causal state |
| `kv_cache` | uclm_core | transformer KV cache |
| `dec_state` | codec_decoder | conv causal state |
| `event_trace` | codec_decoder | non-verbal hysteresis trace |
| `ctx_A` | engine | circular buffer for `A_t` |
| `ctx_B` | engine | circular buffer for `B_t` |
| `prev_voice_state` | engine | delta computation |

### 4.2 Per-Frame Runtime Sequence

1. `audio_frame(240)` -> `codec_encoder` -> `A_src_t`
2. update `ctx_A/ctx_B`
3. `vc_encoder(ctx_A)` -> `content_features`
4. extract/get `explicit_state` and `ssl_state`
5. `voice_state_enc(explicit_state, ssl_state)` -> `state_cond`
6. `uclm_core(content_features, ctx_B, speaker, state_cond, cfg_scale, kv_cache)` -> `logits_a`, `logits_b`
7. sample -> `A_t`, `B_t`
8. `codec_decoder(A_t, B_t, voice_state_t, event_trace, dec_state)` -> `audio_out_t`
7. swap ping-pong states

All buffers must be pre-allocated (RT-safe: no malloc/free/mutex on audio thread).

---

## 5. `.tmrvc_speaker` Contract (UCLM v2 related)

```json
{
  "version": 3,
  "spk_embed": [192 floats],
  "f0_mean": 220.0,
  "reference_A_tokens": [[8 ints, ...], "optional"],
  "reference_B_tokens": [[4 ints, ...], "optional"],
  "voice_source_preset": [8 floats, "optional"],
  "ssl_state": [128 floats, "optional - WavLM default SSL state"],
  "metadata": {"created_at": "2026-03-01T00:00:00Z"}
}
```

- `reference_tokens` を使う場合は `A/B` 両ストリームで保存する
- 旧 `reference_tokens[T,4]` は互換読み込みのみ (deprecated)
- `ssl_state` は話者登録時に WavLM から抽出した 128 次元の潜在スタイル表現
  - Rust 側では `ssl_state: Option<Vec<f32>>` として扱い、None の場合はゼロベクトルを使用

---

## 6. Numerical Parity

| Tensor type | Metric | Threshold |
|---|---|---|
| Float outputs | L_inf | `< 1e-4` |
| Float outputs | cosine similarity | `> 0.999` |
| Int outputs | exact match | `100%` |

Verification example:

```bash
uv run python -m tmrvc_export.verify_parity \
  --codec-encoder models/fp32/codec_encoder.onnx \
  --uclm-core models/fp32/uclm_core.onnx \
  --codec-decoder models/fp32/codec_decoder.onnx
```

---

## 7. Consistency Checklist

- [x] `frame_size=240` (10ms) が Python/Rust/ONNX で一致
- [x] `A_t`: `[1,8]` / id range `0..1023`
- [x] `B_t`: `[1,4]` / vocab `64` (`[op,type,dur,int]`)
- [x] `uclm_core` が `logits_a[1,8,1024]` と `logits_b[1,4,64]` を出力
- [x] `delta_voice_state` が推論入力に含まれる (voice_state_enc に `delta_state` 入力として追加)
- [x] `event_trace` が decoder の入出力で維持される
- [x] rolling context (`ctx_A`, `ctx_B`) が 1-2秒保持される (CONTEXT_FRAMES=200)
- [x] 全 state が pre-allocated で RT-safe
- [x] `uclm_core` が CFG (Classifier-Free Guidance) 対応 (`cfg_scale` 入力追加)
- [x] CFG 公式: `output = uncond + cfg_scale * (cond - uncond)` を常に両パス計算で実装
- [x] KV Cache が `uclm_core` の入出力で維持される (`kv_cache_in` / `kv_cache_out`)
- [x] 全6モデルの ONNX エクスポートが成功 (vc_encoder, voice_state_enc, uclm_core, codec_encoder, codec_decoder, speaker_encoder)
- [x] パリティ検証 L_inf < 1e-4 (codec_decoder は簡易実装のため検証スキップ)
