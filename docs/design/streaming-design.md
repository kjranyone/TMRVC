# TMRVC Streaming Design & Latency Budget

TMRVC runtime は `24 kHz / 10 ms` を固定クロックとする causal streaming system である。mainline TTS は duration 展開ではなく pointer progression で text を消費し、VC は causal semantic context で source を変換する。

## 1. 基本パラメータ

| パラメータ | 値 |
|---|---|
| 内部 sample rate | `24,000 Hz` |
| hop length | `240 samples` |
| frame rate | `100 Hz` |
| acoustic tokens per frame | `8` |
| control tokens per frame | `4` |

## 2. streaming pipeline

### 2.1 TTS

```text
text units + pointer state
  -> UCLM Core
     -> A_t logits
     -> B_t logits
     -> advance / hold decision
  -> update pointer state
  -> Codec Decoder
  -> 240-sample audio chunk
```

### 2.2 VC

```text
input audio frame
  -> Codec Encoder
  -> causal semantic encoder
  -> UCLM Core
  -> Codec Decoder
  -> 240-sample audio chunk
```

## 3. TTS runtime state

```text
pointer_state:
  unit_index
  progress
  finished

shared_state:
  kv_cache
  speaker embedding
  voice state / prosody controls
```

`target_length` を事前に固定生成しない。終了は pointer state と EOS 条件で決める。

## 4. レイテンシバジェット

| 構成要素 | 目標 |
|---|---|
| Codec Encoder | `~5 ms` |
| UCLM Core | `~8-10 ms` |
| Codec Decoder | `~5 ms` |
| Algorithmic latency | `10 ms` |
| End-to-end nominal | `~35-50 ms` |

## 5. 外部制御

runtime は次の pacing control を受け付ける。

- `pace`
- `hold_bias`
- `boundary_bias`

これらは pointer の advance policy に影響を与えるが、非因果な future lookahead は導入しない。

## 6. RT 制約

- audio thread で `malloc/free` をしない
- blocking I/O をしない
- mutex を持ち込まない
- state は固定サイズで事前確保する

## 7. 禁止事項

- duration predictor による全文長の事前展開を mainline runtime に戻すこと
- `MFA` 由来境界を runtime 必須入力にすること
- future context 依存の non-causal TTS を core path に入れること
