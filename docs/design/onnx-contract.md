# TMRVC ONNX Model I/O Contract

この文書は、`tmrvc-export`、`tmrvc-serve`、`tmrvc-engine-rs` が共有する現行 ONNX 契約を定義する。前提は `UCLM v3 mainline` であり、TTS は `pointer-based causal progression`、VC は `causal semantic conditioning` を使う。

## 1. 共有不変条件

- sample rate: `24,000 Hz`
- frame step: `240 samples` (`10 ms`)
- acoustic stream `A_t`: `8` codebooks, each `0..1023`
- control stream `B_t`: `4` slots on the same frame clock
- inference core is causal
- runtime state is explicit and pre-allocated

数値定数の単一正本は `configs/constants.yaml` とする。

## 2. モデル集合

| Model | File | 役割 |
|---|---|---|
| `codec_encoder` | `codec_encoder.onnx` | waveform -> `A_t` |
| `semantic_encoder` | `semantic_encoder.onnx` | VC 用 causal semantic features |
| `voice_state_encoder` | `voice_state_encoder.onnx` | explicit / ssl / prosody 条件の融合 |
| `uclm_core` | `uclm_core.onnx` | `A_t / B_t / pointer` 予測 |
| `codec_decoder` | `codec_decoder.onnx` | `A_t / B_t` -> waveform |
| `speaker_encoder` | `speaker_encoder.onnx` | reference audio -> speaker embedding |

## 3. runtime state 契約

### 3.1 Shared engine state

| State | Shape | 説明 |
|---|---|---|
| `kv_cache` | implementation-defined | UCLM attention cache |
| `ctx_a` | `[1, 8, K]` | acoustic token context |
| `ctx_b` | `[1, 4, K]` | control token context |
| `decoder_state` | implementation-defined | codec decoder causal state |
| `encoder_state` | implementation-defined | codec/semantic encoder causal state |

### 3.2 Pointer state

| Field | Shape | 説明 |
|---|---|---|
| `text_index` | `[1] int64` | 現在参照中の text unit index |
| `progress` | `[1] float32` | current unit 内の連続進行量 |
| `finished` | `[1] bool/int64` | EOS 到達フラグ |

TTS runtime は `target_length` の事前確定に依存せず、この pointer state を frame ごとに更新する。

## 4. モデル別 I/O

### 4.1 `codec_encoder.onnx`

Inputs:

| Name | Shape | Type |
|---|---|---|
| `audio_frame` | `[1, 1, 240]` | `float32` |
| `state_in` | implementation-defined | `float32` |

Outputs:

| Name | Shape | Type |
|---|---|---|
| `acoustic_tokens` | `[1, 8]` | `int64` |
| `state_out` | implementation-defined | `float32` |

### 4.2 `semantic_encoder.onnx`

VC のみで使う。

Inputs:

| Name | Shape | Type |
|---|---|---|
| `source_a_ctx` | `[1, 8, K]` | `int64` |
| `state_in` | implementation-defined | `float32` |

Outputs:

| Name | Shape | Type |
|---|---|---|
| `semantic_features` | `[1, K, d_model]` | `float32` |
| `state_out` | implementation-defined | `float32` |

### 4.3 `voice_state_encoder.onnx`

Inputs:

| Name | Shape | Type |
|---|---|---|
| `explicit_state` | `[1, 1, 8]` | `float32` |
| `ssl_state` | `[1, 1, d_ssl]` | `float32` |
| `prosody_latent` | `[1, 1, d_prosody]` | `float32` |
| `delta_state` | `[1, 1, 8]` | `float32` |

Outputs:

| Name | Shape | Type |
|---|---|---|
| `state_cond` | `[1, 1, d_model]` | `float32` |

### 4.4 `uclm_core.onnx`

TTS / VC 共通コア。

Inputs:

| Name | Shape | Type | 説明 |
|---|---|---|---|
| `content_features` | `[1, K, d_model]` | `float32` | text-aligned features or semantic features |
| `ctx_a` | `[1, 8, K]` | `int64` | acoustic history |
| `ctx_b` | `[1, 4, K]` | `int64` | control history |
| `speaker_embed` | `[1, d_speaker]` | `float32` | target speaker |
| `state_cond` | `[1, 1, d_model]` | `float32` | fused state condition |
| `pointer_state` | `[1, 3]` or structured equivalent | `float32/int64` | `text_index`, `progress`, `finished` |
| `pace` | `[1]` | `float32` | pacing multiplier |
| `hold_bias` | `[1]` | `float32` | bias toward hold |
| `boundary_bias` | `[1]` | `float32` | bias toward boundary advance |
| `cfg_scale` | `[1]` | `float32` | guidance scale |
| `kv_cache_in` | implementation-defined | `float32` | transformer cache |

Outputs:

| Name | Shape | Type | 説明 |
|---|---|---|---|
| `logits_a` | `[1, 8, vocab_a]` | `float32` | next acoustic logits |
| `logits_b` | `[1, 4, vocab_b]` | `float32` | next control logits |
| `advance_logit` | `[1, 1]` | `float32` | advance vs hold |
| `progress_delta` | `[1, 1]` | `float32` | pointer progress update |
| `next_pointer_state` | structured equivalent | `float32/int64` | updated pointer state |
| `kv_cache_out` | implementation-defined | `float32` | updated cache |

### 4.5 `codec_decoder.onnx`

Inputs:

| Name | Shape | Type |
|---|---|---|
| `acoustic_tokens` | `[1, 8]` | `int64` |
| `control_tokens` | `[1, 4]` | `int64` |
| `voice_state` | `[1, 1, 8]` | `float32` |
| `decoder_state_in` | implementation-defined | `float32` |

Outputs:

| Name | Shape | Type |
|---|---|---|
| `audio_frame` | `[1, 1, 240]` | `float32` |
| `decoder_state_out` | implementation-defined | `float32` |

### 4.6 `speaker_encoder.onnx`

Inputs:

| Name | Shape | Type |
|---|---|---|
| `mel_ref` | `[1, 80, T]` | `float32` |

Outputs:

| Name | Shape | Type |
|---|---|---|
| `speaker_embed` | `[1, d_speaker]` | `float32` |

## 5. TTS runtime sequence

1. text frontend が text units を生成する
2. pointer state を初期化する
3. `voice_state_encoder` で frame 条件を作る
4. `uclm_core` で `A_t / B_t / advance_logit / progress_delta` を生成する
5. engine が pointer state を更新する
6. `codec_decoder` が 240 samples を復元する
7. `finished` または EOS 条件で終了する

## 6. VC runtime sequence

1. `codec_encoder` が source frame を `A_t` に変換する
2. `semantic_encoder` が causal semantic features を生成する
3. `voice_state_encoder` が target style 条件を生成する
4. `uclm_core` が target `A_t / B_t` を生成する
5. `codec_decoder` が output frame を復元する

## 7. `.tmrvc_speaker` 契約

```json
{
  "version": 3,
  "speaker_embed": [192 floats],
  "ssl_state": [128 floats],
  "voice_state_preset": [8 floats],
  "metadata": {
    "created_at": "2026-03-07T00:00:00Z",
    "language": "ja"
  }
}
```

話者ファイルは pointer state や duration 情報を持たない。

### 7.1 `.tmrvc_speaker` v3 拡張

v3 では `prompt_kv_cache` を話者ファイルに含めることができる。事前に `encode_speaker_prompt` で生成した KV cache を永続化し、推論時のリファレンス再エンコードを省略する。

```json
{
  "version": 3,
  "speaker_embed": [192 floats],
  "ssl_state": [128 floats],
  "voice_state_preset": [8 floats],
  "prompt_kv_cache": "base64-encoded tensor or external .bin reference",
  "metadata": {
    "created_at": "2026-03-07T00:00:00Z",
    "language": "ja"
  }
}
```

## 8. parity 基準

| 項目 | 基準 |
|---|---|
| float tensor | `L_inf < 1e-4` |
| token ids | exact match |
| pointer state update | exact match or deterministic tolerance-defined match |

## 9. 禁止事項

- `MFA` 由来境界を ONNX runtime 必須入力にすること
- duration 展開済み全文フレーム列を TTS runtime 契約に戻すこと
- hidden mutable state を契約外で持つこと

## 10. v3 ONNX 拡張

### 10.1 `encode_speaker_prompt.onnx`

Speaker Prompt Encoder を独立した ONNX グラフとしてエクスポートする。推論時にリファレンス音声から speaker embedding と prompt KV cache を生成する。

Inputs:

| Name | Shape | Type | 説明 |
|---|---|---|---|
| `prompt_codec_tokens` | `[1, T_prompt, n_codebooks]` | `int64` | リファレンス音声の codec tokens |
| `speaker_embed` (optional) | `[1, d_speaker]` | `float32` | 外部 speaker embedding (融合用) |

Outputs:

| Name | Shape | Type | 説明 |
|---|---|---|---|
| `refined_speaker_embed` | `[1, d_model]` | `float32` | timbre bottleneck 経由の話者埋め込み |
| `prompt_kv_cache` | `[1, n_layers, 2, n_heads, T_prompt, d_head]` | `float32` | 再利用可能な KV cache |

### 10.2 `uclm_core.onnx` v3 追加入力

既存の `uclm_core.onnx` に以下の入力が追加される。

| Name | Shape | Type | 説明 |
|---|---|---|---|
| `prompt_kv_cache` | `[1, n_layers, 2, n_heads, T_prompt, d_head]` | `float32` | `encode_speaker_prompt` で生成した cached tensor。未使用時はゼロテンソル |
| `cfg_scale` | `[1]` | `float32` | classifier-free guidance scale (既存定義の再掲、v3 で必須化) |
| `voice_state` | `[1, 1, 8]` | `float32` | 明示的な voice state (従来の `state_cond` への融合ではなく、独立入力として扱う) |
| `delta_voice_state` | `[1, 1, 8]` | `float32` | voice state の frame 間差分 (独立入力) |

**`state_cond` との関係**: v3 では `voice_state` と `delta_voice_state` を独立入力として受け取る。`voice_state_encoder` による融合済み `state_cond` は引き続きサポートするが、v3 runtime では独立入力を優先する。これにより engine 側で voice state の直接制御が可能になる。

### 10.3 Updated TTS Runtime Sequence (v3)

1. text frontend が text units を生成する
2. `encode_speaker_prompt` でリファレンス音声から `refined_speaker_embed` と `prompt_kv_cache` を生成する (初回のみ、以降はキャッシュを再利用)
3. pointer state を初期化する
4. `voice_state_encoder` で frame 条件を作る (または `voice_state` / `delta_voice_state` を直接渡す)
5. `uclm_core` に `prompt_kv_cache` を含む全入力を渡し、`A_t / B_t / advance_logit / progress_delta` を生成する
6. engine が pointer state を更新する
7. `codec_decoder` が 240 samples を復元する
8. `finished` または EOS 条件で終了する
