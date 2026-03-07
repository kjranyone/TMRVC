# TMRVC Rust Engine Design

この文書は `tmrvc-engine-rs` と `tmrvc-vst` が従う mainline runtime 設計を定義する。前提は `24 kHz / 10 ms`、`dual-stream token generation`、`pointer-based TTS`、`causal VC` である。

## 1. スコープ

- `tmrvc-engine-rs`: RT-safe streaming inference core
- `tmrvc-vst`: VST3 wrapper

## 2. コア責務

### 2.1 TTS

- pointer state を保持しながら text を frame ごとに消費する
- `uclm_core` の `advance_logit / progress_delta` を使って進行を更新する
- `target_length` の事前一括展開をしない

### 2.2 VC

- source audio を `codec_encoder` で token 化する
- `semantic_encoder` で causal semantic context を作る
- target speaker / state 条件で `A_t / B_t` を再生成する

## 3. engine state

| State | 用途 |
|---|---|
| `enc_state` | codec / semantic encoder causal state |
| `kv_cache` | uclm core attention cache |
| `dec_state` | codec decoder causal state |
| `ctx_a` | recent acoustic tokens |
| `ctx_b` | recent control tokens |
| `pointer_state` | TTS の text progression |
| `voice_state_prev` | delta 計算 |

全 state は固定サイズで事前確保する。

## 4. frame 処理

### 4.1 TTS

1. current pointer state を読む
2. `voice_state_encoder` を実行する
3. `uclm_core` に text-conditioned features と state を渡す
4. `A_t / B_t / advance_logit / progress_delta` を得る
5. pointer state を更新する
6. `codec_decoder` で 240 samples を出す

### 4.2 VC

1. input frame を `codec_encoder` に渡す
2. `semantic_encoder` を回す
3. `voice_state_encoder` を回す
4. `uclm_core` を回す
5. `codec_decoder` で出力 frame を得る

## 5. RT-safe 要件

- audio thread で allocation しない
- lock を持たない
- file I/O をしない
- state swap は固定バッファで行う

## 6. 外部制御

engine は少なくとも以下を受け取れること。

- `pace`
- `hold_bias`
- `boundary_bias`
- speaker preset
- voice state preset

## 7. 禁止事項

- duration predictor に依存した mainline TTS runtime
- `MFA` artifact を engine 入力に要求すること
- 非因果な lookahead を core path に入れること

## 8. v3 Rust Engine 拡張

### 8.1 SpeakerPromptEncoder Support

Rust runtime は `encode_speaker_prompt.onnx` を呼び出し、リファレンス音声から `refined_speaker_embed` と `prompt_kv_cache` を生成する。

- 初回発話時に `encode_speaker_prompt` を実行する
- 生成された `prompt_kv_cache` を engine state に保持する
- 同一話者の後続発話では `encode_speaker_prompt` の再実行を省略し、キャッシュ済み KV を `uclm_core` に直接渡す

### 8.2 prompt_kv_cache Persistence

| State | Shape | Lifetime |
|---|---|---|
| `prompt_kv_cache` | `[1, n_layers, 2, n_heads, T_prompt, d_head]` | 話者変更まで永続 |

- `prompt_kv_cache` は固定サイズバッファとして事前確保する
- 話者切り替え時にのみ再計算する
- multi-turn 対話では frame 処理ループの外側で保持し、各 frame で `uclm_core` への入力として供給する

### 8.3 Stall Detection

pointer が進行しないまま frame が蓄積される状況 (stall) を検出し、安全に生成を終了する機構。

| Parameter | Type | Description |
|---|---|---|
| `stall_frames` | `u32` | pointer が advance しなかった連続 frame 数のカウンタ |
| `max_stall` | `u32` | stall 許容閾値 (この frame 数を超えたら強制終了) |

**動作**:
1. 各 frame で `advance_logit` を評価する
2. advance が発生しなかった場合、`stall_frames` をインクリメントする
3. advance が発生した場合、`stall_frames` を 0 にリセットする
4. `stall_frames > max_stall` となった場合、生成を安全に終了する (EOS として扱う)

### 8.4 voice_state / delta_voice_state as Explicit Inputs

v3 では `voice_state` と `delta_voice_state` を `voice_state_encoder` で融合した `state_cond` ではなく、独立した制御入力として `uclm_core` に渡す。

- `voice_state`: `[1, 1, 8]` float32 — 現在の voice state (8 次元物理パラメータ)
- `delta_voice_state`: `[1, 1, 8]` float32 — 前 frame からの voice state 差分

engine は毎 frame、外部制御 (GUI スライダー、MIDI CC、オートメーション等) から `voice_state` を受け取り、前 frame との差分を `delta_voice_state` として計算する。これにより、推論中のリアルタイム voice state 操作が可能になる。

engine state テーブルへの追記:

| State | 用途 |
|---|---|
| `prompt_kv_cache` | speaker prompt の KV cache (話者変更まで永続) |
| `stall_frames` | stall 検出用連続非進行フレームカウンタ |

### 8.5 Pointer Fallback Policy in Rust

Python serve の `step_pointer()` と完全に一致する pointer fallback policy を Rust runtime でも実装する。

| Parameter | Type | Description |
|---|---|---|
| `max_frames_per_unit` | `u32` | configurable engine parameter。1 text unit あたりの最大フレーム数。超過時に forced advance を発動する |

**要件**:
- **Forced-advance behavior**: `max_frames_per_unit` 超過時の forced advance は Python の `step_pointer()` と完全に一致すること
- **Skip-protection behavior**: `boundary_confidence` が `skip_protection_threshold` を下回る場合の advance ブロックは Python と完全に一致すること
- **Telemetry reporting**: `forced_advance_count` および `skip_protection_count` を runtime telemetry に報告すること
- **Observability**: すべての fallback 判定は observable でなければならない (silent な fallback は禁止)
