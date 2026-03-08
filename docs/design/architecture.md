# TMRVC Unified Architecture

TMRVC は、`10 ms` 因果クロックで動作する unified codec language model により、TTS と VC を単一系で扱う音声生成システムである。現行 mainline は `UCLM v3` であり、外部 forced alignment を中核に置かず、内部アライメント学習と pointer-based text progression を採用する。

## 1. 設計原則

### 1.1 Unified

- TTS と VC を別モデルに分断しない
- 共通の token contract と backbone を使う
- 相違は入力条件と補助損失に閉じ込める

### 1.2 Causal 10 ms Core

- 1 frame = `240 samples @ 24 kHz`
- 推論コアは未来参照を持たない
- streaming state は固定サイズで保持する

### 1.3 Internal Alignment

- TTS の text progression は内部で学習する
- mainline は `MFA`, `TextGrid`, `durations.npy` を必須前提にしない
- 進行制御は `pointer state` と `advance / hold` 判断で扱う

### 1.4 Dual-Stream Token Sync

- acoustic stream `A_t`: codec RVQ token
- control stream `B_t`: event / pacing / local prosody control
- 両者は常に同じ frame clock 上で同期する

### 1.5 Physical-First Control

- 話者性と内容を分離した上で、8 次元 voice state と local prosody latent を条件に使う
- 抽象ラベルより、連続的で編集可能な制御量を優先する
- 8 次元 voice state は UI 上のつまみではなく、data -> export -> train -> runtime を貫く supervision contract として扱う

### 1.6 Release Tiers

- Tier 1 (release-critical):
  - pointer-based causal TTS
  - shared runtime/schema contract
  - 8-D physical control with masks/confidences/provenance
  - bounded dialogue context
  - few-shot speaker prompting under a reproducible contract
- Tier 2 (research-track):
  - backbone modernization bundles
  - advanced prosody predictors
  - CFG acceleration variants
  - second-stage acoustic refinement

Tier 2 は Tier 1 の契約を壊してはならず、rollback path を持つ。

## 2. システム構成

```text
TTS path:
  text
    -> normalizer / g2p / grapheme backend
    -> text units
    -> TextEncoder
    -> UCLM Core
       -> A_t head
       -> B_t head
       -> pointer head
    -> Codec Decoder
    -> waveform

VC path:
  source waveform
    -> Codec Encoder
    -> causal semantic encoder
    -> UCLM Core
       -> A_t head
       -> B_t head
    -> Codec Decoder
    -> waveform

Shared conditions:
  speaker embedding
  explicit voice state
  ssl voice-state / local prosody latent
  pacing controls
```

## 3. 主要コンポーネント

### 3.1 Text Frontend

- monolingual dataset を default とする
- multilingual / code-switch dataset も許可するが、`language_id` と `language_spans` を明示する
- text normalization と text-unit 化を担当する
- 出力は phoneme または grapheme based text units

### 3.2 Codec

- waveform と `A_t / B_t` の相互変換器
- `A_t` は高周波を含む音響内容を保持する
- `B_t` はイベントと phrasing の補助制御を保持する

### 3.3 UCLM Core

- TTS / VC 共通の causal backbone
- 入力条件に応じて `A_t / B_t` を毎 frame 予測する
- TTS では pointer head により text consumption を進める

### 3.4 Pointer Head

- 出力: `advance_logit`, `progress_delta`, optional local prosody signal
- 役割: 現在の text unit を維持するか次へ進むかを 10 ms ごとに決める
- duration 展開ではなく online progression を担う

### 3.5 Voice State / Prosody

- `explicit_voice_state`: 8 次元の物理パラメータ
- `ssl_voice_state`: frame-level の潜在状態
- `local_prosody_latent`: 局所的な間、勢い、語尾処理の自由度

## 4. タスク別動作

### 4.1 TTS

- 入力は text units
- pointer state を通じて text を段階的に消費する
- 生成長は duration predictor ではなく pointer policy が決める

### 4.2 VC

- 入力は source audio 由来の causal semantic representation
- source speaker / target speaker / style を分離条件として扱う
- streaming 時は pseudo future context で過度な平板化を抑える

## 5. 主系統で禁止するもの

- `MFA` を mainline TTS の必須依存にすること
- `durations.npy` 欠損を理由に TTS 全体を不能扱いすること
- 非因果な future lookahead をコア推論に混入させること
- TTS と VC で別々の token contract を持つこと

## 6. legacy 互換

旧 duration ベース経路は比較実験と checkpoint 互換のために残してよいが、以下の条件を満たすこと。

- mainline 設計の説明に混ぜない
- `legacy` と明示する
- quality / regression 比較用途に限定する

## v3 Tensor Contracts

本セクションは UCLM v3 の各コンポーネント間で交わされるテンソル契約を定義する。`B` はバッチサイズ、`T` は時間フレーム数を表す。

### Pointer State

| Field | Shape / Type | Description |
|---|---|---|
| `text_index` | `[B]` int | 現在のフォネームインデックス (monotonic non-decreasing) |
| `progress_value` | `[B]` float | 現在フォネーム内の進行度 (0-1), clamp/reset behavior は明示的 |
| `advance_logit` | `[B]` float | advance/hold 判定ロジット |
| `boundary_confidence` | `[B]` float (optional) | モデル予測による境界信頼度 (dummy zero 禁止) |
| `stall_frames` | `[B]` int (optional) | 同一 text unit での連続 hold フレーム数 (deadlock 検出用) |

EOS / stop behavior は最終 text unit で明示的に定義される。`stall_frames` が `max_frames_per_unit` を超過した場合は force-advance が発動する。

### Pointer Outputs

| Key | Shape | Description |
|---|---|---|
| `logits_a` | `[B, n_codebooks, T, vocab_a]` | acoustic token logits |
| `logits_b` | `[B, n_slots, T, vocab_b]` | control token logits |
| `advance_logit` | `[B, T, 1]` | advance/hold logit (canonical key; `pointer_logits` は使用禁止) |
| `progress_delta` | `[B, T, 1]` | phoneme 内進行度の更新量 (sigmoid, 0-1) |
| `boundary_confidence` | `[B, T, 1]` | モデル予測による境界信頼度 (dummy zero 禁止) |
| `hidden_states` | `[B, T, d_model]` | 診断・diversity loss 用の中間表現 |
| `next_pointer_state` | dict | 推論時に生成される次ステップの pointer state |

### Physical Control Supervision

| Tensor | Shape | Description |
|---|---|---|
| `voice_state_targets` | `[B, T, 8]` | curated or direct-labeled physical targets |
| `voice_state_observed_mask` | `[B, T, 8]` | 次元ごとの観測可否。unknown を 0 とみなさないための mask |
| `voice_state_confidence` | `[B, T, 8]` or `[B, T, 1]` | pseudo-label reliability |
| `voice_state_target_source` | serializable enum / record | direct / pseudo-labeled / absent の provenance |

unknown や低信頼な次元は loss から除外し、dense zero で埋めて neutral state と誤解させてはならない。

### Dialogue Context

| Tensor | Shape | Description |
|---|---|---|
| `dialogue_context` (canonical) | `[B, C_ctx, d_model]` | multi-turn テキストコンテキスト (turn-role markers 付き encoded text embeddings) |
| `dialogue_context` (pooled) | `[B, d_model]` | convenience shorthand (内部 projector が pooling) |

canonical API 境界では 3D `[B, C_ctx, d_model]` 形式を使用する。2D `[B, d_model]` は pooled convenience shorthand として受け入れ可能。raw audio は初期 v3 の dialogue-context modality ではない。

### Acting Intent

| Tensor | Shape | Description |
|---|---|---|
| `acting_intent` | `[B, D_act]` | utterance-level acting intent |

### Local Prosody Latent

| Tensor | Shape | Description |
|---|---|---|
| `local_prosody_latent` (frozen initial v3) | `[B, d_prosody]` | utterance-global prosody latent (initial v3 mainline policy) |
| `local_prosody_latent` (future extension) | `[B, T_plan, d_prosody]` | time-local prosody planning (将来拡張) |

初期 v3 では utterance-global `[B, d_prosody]` を frozen policy とする。DialogueContextProjector は 2D 入力を `unsqueeze(1)` で T 方向にブロードキャストし、将来の 3D 入力もコード変更なしで受け入れ可能とする。

### Acoustic History (ONNX runtime)

| Tensor | Shape | Dtype | Description |
|---|---|---|---|
| `ctx_a` | `[1, 8, K]` | int64 | acoustic token context buffer |
| `ctx_b` | `[1, 4, K]` | int64 | control token context buffer |

### Speaker Embed

| Tensor | Shape | Description |
|---|---|---|
| `speaker_embed` | `[B, D_spk]` | speaker embedding vector |

## v3 Components

本セクションは UCLM v3 で追加・拡張された主要コンポーネントの設計を定義する。

### Speaker Prompt Encoder (Few-Shot Voice Cloning)

**Purpose**: 3-10 秒のリファレンス音声を speaker embedding と prompt KV features にエンコードする。few-shot voice cloning の基盤となるモジュール。

**Architecture**:

```text
prompt_codec_tokens [B, T_prompt, n_codebooks]
  -> codec token embedding
  -> 2-layer transformer encoder
  -> timbre bottleneck
  -> refined_speaker_embed [B, d_model]
     prompt_features [B, T_prompt, d_model]
```

**Timbre Bottleneck**: prosody がプロンプトからリークすることを防ぐ。話者の音色情報のみを抽出し、韻律は ProsodyPredictor や dialogue context から独立に制御する。

**Inputs**:

| Tensor | Shape | Description |
|---|---|---|
| `prompt_codec_tokens` | `[B, T_prompt, n_codebooks]` | リファレンス音声の codec tokens |
| `speaker_embed` (optional) | `[B, d_speaker]` | 外部 speaker embedding (指定時は timbre bottleneck 出力と融合) |

**Outputs**:

| Tensor | Shape | Description |
|---|---|---|
| `refined_speaker_embed` | `[B, d_model]` | timbre bottleneck を経た話者埋め込み |
| `prompt_features` | `[B, T_prompt, d_model]` | prompt 由来の KV features |

**prompt_kv_cache**: `prompt_features` から生成される KV cache は、会話ターンをまたいで永続化できる。同一話者で複数発話を生成する場合、リファレンス音声の再エンコードを省略し推論コストを削減する。

### Prosody Predictor (Flow-matching)

**Purpose**: テキストとコンテキストから局所的な prosody latent を予測する。推論時に明示的な韻律制御を不要にしつつ、訓練時は多様な韻律パターンを学習する。

**Why Flow-matching**: VAE に比べて高い多様性と決定論的マッピングを両立し、Diffusion よりも 1-step 推論で効率的である。

**Architecture**:

```text
phoneme_features [B, L, d_model]
  -> text pooling
  -> optional context/speaker fusion
  -> Flow-matching network
  -> local_prosody_latent [B, d_prosody]
```

**Training**: リファレンス音声から Reference Encoder 経由で target latent を抽出し、Flow-matching の学習ターゲットとして使用する。条件付き flow field を学習し、ノイズから target latent への ODE trajectory を回帰する。

**Inference**: 単一 ODE step (または高品質化のために N-step) で latent を生成する。Manual override やサンプリングによる多様性制御もオプションとして可能。

**Inputs**:

| Tensor | Shape | Description |
|---|---|---|
| `phoneme_features` | `[B, L, d_model]` | TextEncoder 出力 |
| `dialogue_context` (optional) | `[B, C_ctx, d_model]` or `[B, d_model]` | 対話コンテキスト埋め込み (3D multi-turn or 2D pooled) |
| `speaker_embed` (optional) | `[B, d_speaker]` | 話者埋め込み |

**Outputs**:

| Tensor | Shape | Description |
|---|---|---|
| `local_prosody_latent` | `[B, d_prosody]` | 局所韻律潜在変数 (utterance-global, frozen initial v3 policy) |

### Dialogue Context Projector

**Purpose**: dialogue context, acting intent, local prosody latent を model conditioning として content features に統合する。

**動作**:
- 各入力は独立に `d_model` へ線形射影される
- 射影結果は content features に加算される
- 入力の任意のサブセットが `None` であってよい (partial conditioning)

| Input | Projection | Broadcast |
|---|---|---|
| `dialogue_context` `[B, C_ctx, d_model]` or `[B, d_model]` | `dialogue_proj` -> `[B, (C_ctx,) d_model]` | 2D 時は `unsqueeze(1)` で T 方向にブロードキャスト; 3D 時は pooling 後にブロードキャスト |
| `acting_intent` `[B, D_act]` | `acting_proj` -> `[B, d_model]` | `unsqueeze(1)` で T 方向にブロードキャスト |
| `local_prosody_latent` `[B, d_prosody]` (initial v3) or `[B, T, d_prosody]` (future) | `prosody_proj` -> `[B, (T,) d_model]` | 2D 時は `unsqueeze(1)` で T 方向にブロードキャスト; 3D 時は T が異なる場合 `F.interpolate(mode="nearest")` |

任意の入力が `None` の場合、対応する加算はスキップされる (寄与ゼロ)。

### CFG-Compatible Conditioning

**Purpose**: classifier-free guidance (CFG) による推論時の条件制御強化。

**Training**: condition dropout を適用する。訓練中に一定確率で条件入力をマスクし、unconditional な生成パスも同時に学習する。

**Inference**: `cfg_scale` パラメータで条件付き/無条件の出力を補間する。

```text
output = uncond + cfg_scale * (cond - uncond)
```

**Canonical unconditional mask set**:

- drop or zero:
  - `explicit_voice_state`
  - `delta_voice_state`
  - `ssl_voice_state`
  - `speaker_embed`
  - `prompt_codec_tokens` / `prompt_kv_cache`
  - `dialogue_context`
  - `acting_intent`
  - `local_prosody_latent`
- preserve:
  - text / language inputs
  - causal `acoustic_history`
  - pointer state

この drop 集合は PyTorch, ONNX, Rust, UI-facing runtime で一致しなければならない。

**Safe bounds**: `cfg_scale` の範囲に安全制限を設ける。過大な `cfg_scale` は pointer/EOS の判定を不安定化させるため、実用上の上限を設定し、pointer logits と EOS 判定には CFG 補正を制限的に適用する。

### Anti-Collapse Diagnostics

v3 の条件制御が実際に生成結果に影響を与えていることを検証するための診断指標群。

| Metric | Definition | Purpose |
|---|---|---|
| `context_separation_score` | 異なるコンテキスト間での prosody latent の pairwise distance | コンテキストが韻律に反映されているか |
| `prosody_collapse_score` | `between_context_variance / total_variance` | 韻律が条件に関わらず均一化していないか |
| `control_response_score` | control sweep と出力メトリクスの monotonic correlation | 制御入力が出力に単調に反映されているか |

これらの指標が閾値を下回る場合、条件制御の崩壊 (collapse) が発生している可能性がある。訓練中のモニタリングと品質ゲートに使用する。

### Pointer Failure-Handling Policy

pointer-based text progression が stall や premature skip を起こした場合の安全策を定義する。Python serve、Rust runtime、ONNX export のすべてで同一セマンティクスを保証する。

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_frames_per_unit` | int | 50 | 1 text unit あたりの最大フレーム数。超過時に forced advance を発動する |
| `skip_protection_threshold` | float | 0.3 | advance を許可する最低 `boundary_confidence`。これを下回る場合、pointer advance をブロックして premature skip を防止する |

| Telemetry Counter | Description |
|---|---|
| `forced_advance_count` | forced advance が発動された回数 |
| `skip_protection_count` | skip protection によって advance がブロックされた回数 |

**`step_pointer()` canonical state transition**: pointer の advance / hold / force-advance / skip-protect の判定を行う単一の正規関数。すべてのランタイム (Python serve, Rust runtime, ONNX export) はこの関数のセマンティクスに従う。

### Speaker Embed vs Prompt Codec Tokens Priority

speaker identity の再現において `speaker_embed` と `prompt_codec_tokens` が異なる信号を示した場合の優先順位を定義する。

| Source | Role | Priority |
|---|---|---|
| `speaker_embed` | global timbre anchor — 話者の同一性を決定する | Identity について最優先 |
| `prompt_codec_tokens` | local texture reference — breathiness や color など timbre envelope 内の質感を補正する | Identity が一致する範囲で適用 |

**競合時の解決ルール**: `speaker_embed` が speaker identity について常に優先する。`prompt_codec_tokens` は timbre envelope の範囲内で質感の refinement を担うが、identity 判定を覆すことはできない。

**Injection points**:
- `speaker_embed`: early global conditioning / FiLM を通じてモデル全体に注入する
- `prompt_codec_tokens`: bounded cross-attention / prefix memory を通じて局所的に参照する

## VC and Pointer State Interaction Policy

TTS と VC は同一の UCLM Core backbone を共有するが、text progression の機構は明確に異なる。

- **TTS**: pointer head が primary progression mechanism として機能する。text units 上の読み上げ位置を毎 frame 追跡し、advance / hold の判断を自律的に行う。pointer state (`text_index`, `progress`, `finished`) は TTS runtime の必須状態である。
- **VC**: pointer による text progression を必要としない。VC は source audio の frame-synchronous conversion を行い、source と target の frame clock が 1:1 で対応する。content features は source codec tokens から直接生成されるため、text alignment の概念自体が適用されない。
- **Explicit non-goal**: VC を TTS の pointer-based progression path に強制することは設計上の非目標である。VC に pointer state を持たせることは、不要な複雑性を導入し、frame-synchronous conversion の利点を損なう。両パスの相違は入力条件と pointer head の有無に閉じ込める。

## Checkpoint Compatibility Policy

v3 モデルは v2 checkpoint との後方互換性を維持する。

- **`strict=False` ロード**: v3 の `DisentangledUCLM` は v2 checkpoint を `strict=False` でロードする。v2 に存在しない新規キーは初期値で初期化され、v2 の既存キーはそのまま復元される。
- **v3 新規キー**: 以下のモジュールキーは v3 で新規追加されたものであり、v2 checkpoint には存在しない:
  - `pointer_head.*` -- pointer-based text progression
  - `speaker_prompt_encoder.*` -- few-shot voice cloning
  - `prosody_predictor.*` -- VAE-style prosody prediction
  - `context_projector.*` -- dialogue / acting / prosody conditioning fusion
- **v2 キーの保全**: 既存の v2 キー (`text_encoder`, `uclm_core`, `voice_state_enc`, `vc_encoder`, `codec_encoder`, `codec_decoder` 等) は変更されない。weight の shape と意味は v2 と完全に同一である。
- **Append-only policy**: 新規モジュールの追加は常に append-only で行う。既存キーの rename、shape 変更、semantics 変更は禁止する。これにより v2 checkpoint の互換性を破壊せずに v3 の機能拡張を行える。

## Quantization Policy

UCLM v3 のモデルグラフは量子化フレンドリーな設計を採用している。

- **FP8 互換性**: SwiGLU activation と RMSNorm を採用しており、FP8 推論と互換性がある。BatchNorm や LayerNorm に比べて RMSNorm は統計量の蓄積が不要であり、低精度演算での数値安定性が高い。
- **INT8 SmoothQuant / weight-only quantization**: Rust engine (`tmrvc-engine-rs`) は INT8 SmoothQuant および weight-only quantization をサポートする。SwiGLU の smooth な activation 分布が SmoothQuant の channel-wise scaling と好相性である。
- **RMSNorm と SwiGLU の選定理由**: これらの選定には推論性能だけでなく量子化親和性も考慮されている。SwiGLU は ReLU 系に比べて出力分布が滑らかであり、quantization error が小さい。RMSNorm は mean subtraction を行わないため、量子化後の bias shift が発生しにくい。
- **量子化対応訓練 (QAT) は初期 v3 では不要**: 初期 v3 リリースでは post-training quantization (PTQ) のみをサポートする。QAT は将来の最適化オプションとして保留し、PTQ で十分な品質が得られることを前提とする。
