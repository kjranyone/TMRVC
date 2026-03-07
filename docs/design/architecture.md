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
  ssl state / prosody latent
  pacing controls
```

## 3. 主要コンポーネント

### 3.1 Text Frontend

- dataset 単位で単一言語を前提にする
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

- `explicit_state`: 8 次元の物理パラメータ
- `ssl_state`: frame-level の潜在状態
- `prosody_latent`: 局所的な間、勢い、語尾処理の自由度

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
| `text_index` | `[B]` int | 現在のフォネームインデックス |
| `progress` | `[B]` float | 現在フォネーム内の進行度 (0-1) |
| `finished` | `bool` | テキスト全体の消費が完了したか |

### Pointer Outputs

| Key | Shape | Description |
|---|---|---|
| `logits_a` | `[B, n_codebooks, T, vocab_a]` | acoustic token logits |
| `logits_b` | `[B, n_slots, T, vocab_b]` | control token logits |
| `pointer_logits` | `[B, T, 1]` | advance/hold logit |
| `progress_delta` | `[B, T, 1]` | phoneme 内進行度の更新量 (sigmoid, 0-1) |

### Dialogue Context

| Tensor | Shape | Description |
|---|---|---|
| `dialogue_context` | `[B, D_ctx]` | scene/dialogue embedding |

### Acting Intent

| Tensor | Shape | Description |
|---|---|---|
| `acting_intent` | `[B, D_act]` | utterance-level acting intent |

### Local Prosody Latent

| Tensor | Shape | Description |
|---|---|---|
| `prosody_latent` | `[B, T, D_pro]` | local prosody planning signal, interpolated to match content feature length |

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

### Prosody Predictor (VAE-style)

**Purpose**: テキストとコンテキストから局所的な prosody latent を予測する。推論時に明示的な韻律制御を不要にしつつ、訓練時は多様な韻律パターンを学習する。

**Architecture**:

```text
phoneme_features [B, L, d_model]
  -> text pooling
  -> optional context/speaker fusion
  -> MLP
  -> mu [B, d_prosody], log_var [B, d_prosody]
```

**Training**: reparameterization trick を使用する (`z = mu + eps * std`, `eps ~ N(0, 1)`)。KL divergence loss により潜在空間の正則化を行う。

**Inference**: `mu` をそのまま返す (deterministic)。確率的サンプリングは行わない。

**Inputs**:

| Tensor | Shape | Description |
|---|---|---|
| `phoneme_features` | `[B, L, d_model]` | TextEncoder 出力 |
| `dialogue_context` (optional) | `[B, D_ctx]` | 対話コンテキスト埋め込み |
| `speaker_embed` (optional) | `[B, d_speaker]` | 話者埋め込み |

**Outputs**:

| Tensor | Shape | Description |
|---|---|---|
| `prosody_latent` | `[B, d_prosody]` | 局所韻律潜在変数 |

### Dialogue Context Projector

**Purpose**: dialogue context, acting intent, prosody latent を model conditioning として content features に統合する。

**動作**:
- 各入力は独立に `d_model` へ線形射影される
- 射影結果は content features に加算される
- 入力の任意のサブセットが `None` であってよい (partial conditioning)

| Input | Projection | Broadcast |
|---|---|---|
| `dialogue_context` `[B, D_ctx]` | `dialogue_proj` -> `[B, d_model]` | `unsqueeze(1)` で T 方向にブロードキャスト |
| `acting_intent` `[B, D_act]` | `acting_proj` -> `[B, d_model]` | `unsqueeze(1)` で T 方向にブロードキャスト |
| `prosody_latent` `[B, T, D_pro]` | `prosody_proj` -> `[B, T, d_model]` | T が異なる場合 `F.interpolate(mode="nearest")` |

任意の入力が `None` の場合、対応する加算はスキップされる (寄与ゼロ)。

### CFG-Compatible Conditioning

**Purpose**: classifier-free guidance (CFG) による推論時の条件制御強化。

**Training**: condition dropout を適用する。訓練中に一定確率で条件入力をマスクし、unconditional な生成パスも同時に学習する。

**Inference**: `cfg_scale` パラメータで条件付き/無条件の出力を補間する。

```text
output = uncond + cfg_scale * (cond - uncond)
```

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
