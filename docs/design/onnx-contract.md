# TMRVC ONNX Model I/O Contract

この文書は、`tmrvc-export`、`tmrvc-serve`、`tmrvc-engine-rs`、`tmrvc-vst` が共有する
`UCLM v3 mainline` の ONNX 契約を定義する。正本は `plan/worker_01_architecture.md` と
`plan/worker_04_serving.md` であり、この文書はその runtime/export 版である。

## 1. Shared Invariants

- sample rate: `24,000 Hz`
- hop length: `240 samples`
- frame step: `10 ms`
- acoustic stream `A_t`: `8` codebooks
- control stream `B_t`: `4` slots
- inference core is causal
- pointer state is explicit and serializable
- mainline TTS does not require `MFA`, `TextGrid`, or `durations.npy`

数値定数の単一正本は `configs/constants.yaml` とする。

frame-index を伴う全 artifact / telemetry は以下に従う。

- `sample_rate = 24000`
- `hop_length = 240`
- `start_frame` inclusive
- `end_frame` exclusive
- `T = ceil(num_samples / 240)`

## 2. Model Set

| Model | File | Role |
|---|---|---|
| `codec_encoder` | `codec_encoder.onnx` | waveform -> `A_t` |
| `semantic_encoder` | `semantic_encoder.onnx` | VC 用 causal semantic features |
| `encode_speaker_prompt` | `encode_speaker_prompt.onnx` | reference evidence -> `speaker_embed`, `prompt_kv_cache` |
| `uclm_core` | `uclm_core.onnx` | `A_t / B_t / advance_logit / progress_delta / boundary_confidence` |
| `codec_decoder` | `codec_decoder.onnx` | `A_t / B_t` -> waveform |
| `speaker_encoder` | `speaker_encoder.onnx` | reference audio -> speaker embedding |

`voice_state_encoder.onnx` や `state_cond` は mainline public contract ではない。必要なら
adapter/export convenience graph として追加してよいが、`uclm_core.onnx` の canonical
I/O を置き換えてはならない。

## 3. Runtime State Contract

### 3.1 Shared engine state

| State | Shape | Description |
|---|---|---|
| `kv_cache` | implementation-defined | UCLM attention cache |
| `ctx_a` | `[1, 8, K]` | acoustic token history |
| `ctx_b` | `[1, 4, K]` | control token history |
| `decoder_state` | implementation-defined | codec decoder causal state |
| `encoder_state` | implementation-defined | codec/semantic encoder causal state |
| `prompt_kv_cache` | implementation-defined | few-shot prompt cache, speaker change まで保持可能 |

### 3.2 Pointer state

| Field | Shape / Type | Description |
|---|---|---|
| `text_index` | `[1] int64` | 現在参照中の canonical text-unit index |
| `progress_value` | `[1] float32` | current unit 内の連続進行量 |
| `boundary_confidence` | `[1] float32` | optional boundary trust scalar |
| `stall_frames` | `[1] int64` | 連続 non-advance frame count |
| `finished` | `[1] bool` or `[1] int64` | EOS 到達フラグ |

TTS runtime は `target_length` の事前確定に依存せず、この pointer state を frame ごとに更新する。

## 4. Model I/O

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

VC only.

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

### 4.3 `encode_speaker_prompt.onnx`

Inputs:

| Name | Shape | Type | Description |
|---|---|---|---|
| `prompt_codec_tokens` | `[1, T_prompt, n_codebooks]` | `int64` | reference audio 由来の codec tokens |
| `speaker_embed` | `[1, d_speaker]` | `float32` | optional external anchor |

Outputs:

| Name | Shape | Type | Description |
|---|---|---|---|
| `refined_speaker_embed` | `[1, d_speaker]` | `float32` | runtime が使う speaker anchor |
| `prompt_kv_cache` | implementation-defined | `float32` | reusable prompt KV cache |

### 4.4 `uclm_core.onnx`

TTS / VC 共通コアの canonical public boundary。

Inputs:

| Name | Shape | Type | Description |
|---|---|---|---|
| `content_features` | `[1, K, d_model]` | `float32` | text-aligned features or VC semantic features |
| `language_ids` | `[1, L_lang]` or `[1, 1]` | `int64` | utterance-level or span-conditioned language contract |
| `ctx_a` | `[1, 8, K]` | `int64` | acoustic history |
| `ctx_b` | `[1, 4, K]` | `int64` | control history |
| `speaker_embed` | `[1, d_speaker]` | `float32` | target speaker embedding |
| `prompt_kv_cache` | implementation-defined | `float32` | few-shot prompt cache |
| `explicit_voice_state` | `[1, 1, 8]` | `float32` | current 8-D physical control |
| `delta_voice_state` | `[1, 1, 8]` | `float32` | frame-to-frame physical delta |
| `ssl_voice_state` | `[1, 1, d_ssl]` | `float32` | optional SSL-derived state evidence |
| `dialogue_context` | `[1, C_ctx, d_model]` or `[1, d_model]` | `float32` | bounded multi-turn text context |
| `acting_intent` | `[1, d_act]` | `float32` | utterance-level acting intent |
| `local_prosody_latent` | `[1, d_prosody]` or `[1, 1, d_prosody]` | `float32` | local prosody control latent |
| `pointer_state` | structured equivalent | `float32/int64` | `text_index`, `progress_value`, `boundary_confidence`, `stall_frames`, `finished` |
| `pace` | `[1]` | `float32` | pacing multiplier |
| `hold_bias` | `[1]` | `float32` | bias toward hold |
| `boundary_bias` | `[1]` | `float32` | bias toward boundary advance |
| `cfg_scale` | `[1]` | `float32` | guidance scale |
| `kv_cache_in` | implementation-defined | `float32` | transformer cache |

Outputs:

| Name | Shape | Type | Description |
|---|---|---|---|
| `logits_a` | `[1, 8, vocab_a]` | `float32` | next acoustic logits |
| `logits_b` | `[1, 4, vocab_b]` | `float32` | next control logits |
| `advance_logit` | `[1, 1]` | `float32` | advance vs hold |
| `progress_delta` | `[1, 1]` | `float32` | pointer progress update |
| `boundary_confidence` | `[1, 1]` | `float32` | model-predicted boundary trust |
| `next_pointer_state` | structured equivalent | `float32/int64` | updated pointer state |
| `kv_cache_out` | implementation-defined | `float32` | updated cache |

`state_cond` は internal fused representation として export wrapper の内部でのみ許可される。
`uclm_core.onnx` の canonical public boundary では `state_cond` を mainline input 名として使わない。

### 4.5 `codec_decoder.onnx`

Inputs:

| Name | Shape | Type |
|---|---|---|
| `acoustic_tokens` | `[1, 8]` | `int64` |
| `control_tokens` | `[1, 4]` | `int64` |
| `explicit_voice_state` | `[1, 1, 8]` | `float32` |
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

## 5. Canonical CFG Contract

unconditional pass は runtime wrapper が以下を drop / zero した入力セットで再実行する。

- `explicit_voice_state`
- `delta_voice_state`
- `ssl_voice_state`
- `speaker_embed`
- `prompt_codec_tokens` or `prompt_kv_cache`
- `dialogue_context`
- `acting_intent`
- `local_prosody_latent`

以下は保持する。

- text / language inputs
- causal `acoustic_history`
- pointer state

この mask 集合は training, PyTorch inference, ONNX export, Rust runtime, VST runtime で一致しなければならない。

## 6. Runtime Sequences

### 6.1 TTS

1. text frontend が canonical text units を生成する
2. `SpeakerProfile` または on-the-fly reference から `speaker_embed` と `prompt_kv_cache` を得る
3. pointer state を初期化する
4. `uclm_core` に raw conditioning と cache を渡す
5. engine が `advance_logit`, `progress_delta`, `boundary_confidence` を用いて pointer state を更新する
6. `codec_decoder` が 240 samples を復元する
7. `finished` または EOS 条件で終了する

### 6.2 VC

1. `codec_encoder` が source frame を `A_t` に変換する
2. `semantic_encoder` が causal semantic features を生成する
3. `uclm_core` が target `A_t / B_t` を生成する
4. `codec_decoder` が output frame を復元する

VC が pointer state を bypass しても、public runtime schema は同じ field set を維持する。

## 7. SpeakerProfile Handoff

few-shot speaker prompting の永続契約は `docs/design/speaker-profile-spec.md` を正本とする。
ONNX runtime/export 側では最低限以下を扱えること。

- `speaker_profile_id`
- `speaker_embed`
- `prompt_codec_tokens`
- optional `prompt_kv_cache` or cache blob reference
- `prompt_encoder_fingerprint`

`prompt_encoder_fingerprint` が active runtime と一致しない場合、cache は stale と見なし、
再エンコードする。

## 8. Parity Criteria

| Item | Criterion |
|---|---|
| float tensor | `L_inf < 1e-4` unless stricter test freezes it |
| token ids | exact match |
| pointer state update | exact match or deterministic tolerance-defined match |
| CFG mask application | exact same field set across runtimes |

## 9. Forbidden

- `MFA` 由来境界を ONNX runtime 必須入力にすること
- duration 展開済み全文フレーム列を TTS runtime 契約に戻すこと
- `state_cond` を canonical public API として再昇格させること
- hidden mutable state を契約外で持つこと
- unconditional CFG に undocumented conditioning leak を残すこと
