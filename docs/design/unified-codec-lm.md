# Unified Codec Language Model

この文書は、TMRVC mainline の `UCLM v3` を定義する。目的は、`A_t / B_t` dual-stream contract を維持したまま、TTS を `internal alignment + causal pointer` に移行し、VC を低遅延 causal semantic conditioning で統合することである。

## 1. 生成問題の定式化

各 frame `t` に対して、UCLM は次を推定する。

```text
P(A_t, B_t, G_t | C_t, S, V_t, H_t)
```

- `A_t`: acoustic tokens
- `B_t`: control tokens
- `G_t`: pointer-related outputs
- `C_t`: 条件コンテキスト
  - TTS: text encoder features + pointer state
  - VC: causal semantic features
- `S`: speaker embedding
- `V_t`: explicit / ssl / prosody conditions
- `H_t`: causal history and KV cache

## 2. 入力表現

### 2.1 TTS 条件

- normalized text
- text units (`phoneme_ids` または grapheme ids)
- pointer state
  - `text_index`
  - `progress_value`
  - optional `boundary_confidence`
  - optional `stall_frames`

### 2.2 VC 条件

- source codec tokens
- causal semantic encoder output
- target speaker embedding
- target style / voice state

### 2.3 共通条件

- `speaker_embed`
- `explicit_voice_state`
- `delta_voice_state`
- `ssl_voice_state`
- `local_prosody_latent`
- `prompt_codec_tokens` or `prompt_kv_cache`
- `dialogue_context`
- external controls: `pace`, `hold_bias`, `boundary_bias`

## 3. 出力

### 3.1 Token heads

- `logits_a`: acoustic token logits
- `logits_b`: control token logits

### 3.2 Pointer head

- `advance_logit`: 次の text unit に進む確率
- `progress_delta`: 現在 unit 内進行量の更新
- optional `prosody_delta`: 局所的 phrasing 補助

### 3.3 互換出力

legacy 比較のために duration branch を残す場合でも、mainline の一次出力は pointer head である。

## 4. TTS 推論

TTS は duration 展開を事前に完了させず、1 frame ごとに pointer を更新する。

```text
for each 10 ms step:
  1. current text unit を pointer state から参照
  2. TextEncoder feature と条件埋め込みを UCLM に入力
  3. A_t / B_t / advance_logit / progress_delta を得る
  4. advance or hold を決定し pointer state を更新
  5. Codec Decoder で waveform を復元
```

この設計により、文脈に応じた間、食い気味、溜め、語尾処理を online に変化させられる。

## 5. 学習

### 5.1 共通損失

- `loss_a`: acoustic token CE
- `loss_b`: control token CE
- `loss_state`: 必要に応じた state reconstruction / consistency loss

### 5.2 TTS 損失

- `loss_pointer`: advance / hold の分類損失
- `loss_progress`: progress 回帰
- `loss_alignment`: MAS / CTC 系内部アライメント損失

`durations.npy` を中心教師にしない。legacy ablation でだけ duration loss を残してよい。

### 5.3 VC 損失

- token reconstruction
- speaker preservation
- content preservation
- low-latency consistency

## 6. データ契約

TTS サンプルで必要なのは:

- text
- text units
- audio-side frame artifacts
- optional `voice_state` supervision artifacts
- optional `bootstrap_alignment.json`

必要ではないもの:

- MFA-generated TextGrid
- phoneme duration table

## 7. 実装境界

### 7.1 mainline

- pointer-based TTS
- internal alignment
- causal serving
- no hard MFA dependency

### 7.2 legacy

- duration predictor
- TextGrid injection
- MFA batch utilities

legacy は mainline 仕様に干渉してはならない。

## 7.3 Release-Critical Contract

mainline 正本として固定するもの:

- pointer-based causal progression
- `advance_logit` を主キーとする pointer 出力
- `voice_state_targets` / mask / confidence / provenance
- `SpeakerProfile` による few-shot prompt contract
- `sample_rate = 24000`, `hop_length = 240`, `T = ceil(num_samples / 240)` の frame 規約

v3 リリースに含まれるが、段階的に統合するもの:

- modern transformer backbone（RoPE, GQA, SwiGLU, RMSNorm, FlashAttention2）— 実装済み
- CFG 全モード（off, full, lazy, distilled）
- second-stage acoustic refinement（v3.1 upgrade path）

## v3 Pointer Contract

本セクションは `forward_tts_pointer()` の正式な入出力仕様と、関連モジュールの動作を定義する。

### `forward_tts_pointer()` シグネチャ

```python
def forward_tts_pointer(
    self,
    phoneme_ids: torch.Tensor,        # [B, L] phoneme token ids
    language_ids: torch.Tensor,        # [B, L] language token ids
    pointer_state: PointerState | None,# streaming 推論用 pointer state
    acoustic_history: torch.Tensor,    # [B, n_codebooks, T_hist] or embedded equivalent
    speaker_embed: torch.Tensor,       # [B, d_speaker]
    explicit_voice_state: torch.Tensor | None = None,   # [B, T, 8] or [B, 8]
    delta_voice_state: torch.Tensor | None = None,      # [B, T, 8] or [B, 8]
    ssl_voice_state: torch.Tensor | None = None,        # [B, T, d_ssl]
    target_b: torch.Tensor | None = None,               # teacher forcing only
    target_length: int | None = None,                   # training/eval only
    cfg_scale: float = 1.0,
    dialogue_context: torch.Tensor | None = None,  # [B, C_ctx, d_model] or [B, d_model]
    acting_intent: torch.Tensor | None = None,     # [B, D_act]
    local_prosody_latent: torch.Tensor | None = None,   # [B, d_prosody] or [B, T, d_prosody]
    prompt_kv_cache: torch.Tensor | None = None,
) -> dict
```

### 出力 dict

| Key | Shape | Description |
|---|---|---|
| `logits_a` | `[B, n_codebooks, T, vocab_a]` | acoustic token logits |
| `logits_b` | `[B, n_slots, T, vocab_b]` | control token logits |
| `advance_logit` | `[B, T, 1]` | advance/hold logit (canonical key) |
| `progress_delta` | `[B, T, 1]` | phoneme 内進行度 (sigmoid, 0-1) |
| `boundary_confidence` | `[B, T, 1]` | boundary trust score |
| `hidden_states` | `[B, T, d_model]` | transformer hidden states |

### DialogueContextProjector

`DialogueContextProjector` は `dialogue_context`, `acting_intent`, `local_prosody_latent` を受け取り、それぞれを `d_model` 次元に線形射影して content features に加算する。

- `dialogue_context` (`[B, D_ctx]`): `dialogue_proj` で `[B, d_model]` に射影し、`unsqueeze(1)` で `[B, 1, d_model]` として T 方向にブロードキャスト加算。
- `acting_intent` (`[B, D_act]`): `acting_proj` で同様に `[B, 1, d_model]` にブロードキャスト加算。
- `local_prosody_latent` (`[B, T, D_pro]`): `prosody_proj` で `[B, T, d_model]` に射影して加算。T が content features と異なる場合は `F.interpolate(mode="nearest")` でリサイズ。

入力が `None` の場合は寄与ゼロ (加算をスキップ)。

### PointerHead

`PointerHead` は transformer hidden states (`[B, T, d_model]`) を受け取り、dual projection で 2 つの主出力と optional diagnostics を生成する。

- **`advance_proj`**: `Linear(d_model, d_model//4) -> GELU -> Linear(d_model//4, 1)` — advance/hold の生ロジット (`advance_logit`) を出力。
- **`progress_proj`**: `Linear(d_model, d_model//4) -> GELU -> Linear(d_model//4, 1) -> Sigmoid` — 現在 phoneme 内の進行度を `[0, 1]` で出力。
- optional `boundary_proj`: 境界信頼度を出力。

## Few-Shot Speaker Adaptation Contract

本セクションは v3 の few-shot voice cloning に関する契約を定義する。

### `encode_speaker_prompt()` API

```python
def encode_speaker_prompt(
    prompt_audio: torch.Tensor,   # reference waveform
    prompt_text: str | None,      # optional transcript for alignment hint
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        speaker_embed: [B, d_model] — refined speaker embedding
        prompt_kv_cache: [B, n_layers, 2, n_heads, T_prompt, d_head] — reusable KV cache
    """
```

### prompt_kv_cache Reuse

`prompt_kv_cache` は会話ターンをまたいで再利用できる。同一話者による連続発話では、リファレンス音声の再エンコードを省略し、キャッシュされた KV を `forward_tts_pointer()` に直接渡す。これにより multi-turn 対話の推論コストを大幅に削減する。cache の有効性は `SpeakerProfile` の encoder/tokenizer fingerprint に従って判定する。

### Timbre-Prosody Disentanglement

speaker prompt は timbre のみを提供する。韻律は以下から独立に制御される:

- `ProsodyPredictor`: テキストとコンテキストから予測
- `dialogue_context` / `acting_intent`: DialogueContextProjector 経由で conditioning
- `explicit_voice_state`: 8 次元の物理パラメータによる直接制御

この分離により、同一話者で異なる感情・演技スタイルの音声を生成できる。

### Cross-Lingual Few-Shot

prompt の言語と target text の言語が異なっていてよい。Speaker Prompt Encoder の timbre bottleneck は言語非依存な音色表現を抽出するため、日本語リファレンスから英語音声を生成する (またはその逆の) cross-lingual few-shot が可能である。

## Updated `forward_tts_pointer()` Signature

v3 で追加されたパラメータを含む完全なシグネチャ:

```python
def forward_tts_pointer(
    self,
    phoneme_ids: torch.Tensor,        # [B, L] phoneme token ids
    language_ids: torch.Tensor,        # [B, L] language token ids
    pointer_state: PointerState | None,# streaming 推論用 pointer state
    speaker_embed: torch.Tensor,       # [B, d_speaker]
    acoustic_history: torch.Tensor,    # [B, n_codebooks, T_hist]
    explicit_voice_state: torch.Tensor | None = None,
    delta_voice_state: torch.Tensor | None = None,
    ssl_voice_state: torch.Tensor | None = None,
    target_b: torch.Tensor | None = None,
    target_length: int | None = None,
    cfg_scale: float = 1.0,
    dialogue_context: torch.Tensor | None = None,  # [B, C_ctx, d_model] or [B, d_model]
    acting_intent: torch.Tensor | None = None,     # [B, D_act]
    local_prosody_latent: torch.Tensor | None = None,   # [B, d_prosody] or [B, T, d_prosody]
    # --- v3 additions ---
    prompt_kv_cache: torch.Tensor | None = None,   # [B, n_layers, 2, n_heads, T_prompt, d_head]
    token_language_ids: torch.Tensor | None = None, # [B, L] per-token language ids
) -> dict
```

### Updated v3 Output Dict

| Key | Shape | Description |
|---|---|---|
| `logits_a` | `[B, n_codebooks, T, vocab_a]` | acoustic token logits |
| `logits_b` | `[B, n_slots, T, vocab_b]` | control token logits |
| `advance_logit` | `[B, T, 1]` | advance/hold logit |
| `progress_delta` | `[B, T, 1]` | phoneme 内進行度 (sigmoid, 0-1) |
| `boundary_confidence` | `[B, T, 1]` | phoneme 境界の信頼度スコア |
| `hidden_states` | `[B, T, d_model]` | transformer hidden states |
| `next_pointer_state` | `PointerState` | 更新後の pointer state |

## CFG Unconditional Contract

unconditional pass では以下を drop / zero する:

- `explicit_voice_state`
- `delta_voice_state`
- `ssl_voice_state`
- `speaker_embed`
- `prompt_codec_tokens` / `prompt_kv_cache`
- `dialogue_context`
- `acting_intent`
- `local_prosody_latent`

以下は保持する:

- text / language inputs
- pointer state
- causal `acoustic_history`

この契約は training, PyTorch inference, ONNX, Rust で一致しなければならない。
