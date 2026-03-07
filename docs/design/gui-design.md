# TMRVC GUI Design

この文書は、`TMRVC Research Studio` と realtime UI が mainline の `pointer-based UCLM` をどう可視化し、どう操作するかを定義する。旧 duration-centric UI は標準仕様に含めない。

## 1. UI 原則

- mainline の観測対象は `A_t`, `B_t`, `voice_state`, `pointer_state`
- pacing 制御は `pace`, `hold_bias`, `boundary_bias`
- legacy 機能は分離表示し、既定画面に混ぜない
- 学習・推論・export の状態を同じ語彙で表す

## 2. Research Studio

### 2.1 ページ構成

| Page | 目的 |
|---|---|
| `Datasets` | dataset 登録、言語確認、text coverage 確認 |
| `UCLM Train` | v3 pointer 学習の制御 |
| `Codec Train` | codec 学習 |
| `Serve / TTS` | TTS サーバー制御と応答確認 |
| `VC Demo` | realtime VC 確認 |
| `Export` | ONNX export / parity |
| `Validation` | quality gate と smoke result |
| `Legacy` | v2 / MFA 互換機能 |

### 2.2 UCLM Train 画面

必須項目:

- dataset selection
- cache dir / output dir
- `tts_mode = pointer`
- batch size / steps / device
- quality gate summary
- text coverage summary

表示メトリクス:

- `loss_a`
- `loss_b`
- `loss_pointer`
- `loss_alignment`
- valid TTS sample ratio
- token range anomalies

### 2.3 Serve / TTS 画面

入力:

- text
- character
- language
- style preset
- `pace`
- `hold_bias`
- `boundary_bias`

表示:

- waveform preview
- generated duration
- pointer timeline
  - current text index
  - progress
  - advance events
- voice state trace
- control token trace

### 2.4 VC Demo 画面

表示:

- input/output level
- per-frame inference time
- overruns
- semantic encoder health
- speaker embedding loaded state

## 3. pointer 可視化

TTS mainline UI は以下を可視化する。

| 項目 | 内容 |
|---|---|
| `text_index` | 現在の text unit |
| `progress` | unit 内進行量 |
| `advance_logit` | 次 unit へ進む傾向 |
| `pace` | 全体ペース倍率 |
| `hold_bias` | 間を維持する傾向 |
| `boundary_bias` | 境界進行の傾向 |

duration bar は mainline の主表示にしない。

## 4. エラー表示

学習系:

- G2P backend missing
- text coverage low
- token range anomaly
- quality gate failure

推論系:

- missing checkpoints
- character not found
- pointer state divergence
- ONNX parity mismatch

## 5. realtime UI

realtime UI は次を優先する。

- current latency
- engine ready state
- speaker profile loaded
- dry/wet
- output gain
- runtime pacing controls

TTS 用 realtime UI を持つ場合も、mainline は duration slider ではなく pacing controls を出す。

## 6. Legacy 扱い

`MFA`, `TextGrid`, `duration injection` は `Legacy` ページへ隔離する。既定 onboarding や main dashboard からは導かない。

## 7. Human Roles

| Role | Responsibility |
|---|---|
| `annotator` | トランスクリプトの修正、言語スパンの修正、話者アサインの修正 |
| `auditor` | AI ラベルのレビュー、データセット適格性の判定 |
| `director` | 演技品質の判定、制御可能性の評価、キャスティング適合性の判断 |
| `rater` | ブラインド主観評価の実施 |
| `admin` | モデル管理、データセット管理、合法性管理、エクスポート管理 |

## 8. Human Workflow Contract

UI は完全な HITL (Human-in-the-Loop) ループをカバーする。

1. **Ingest and legality assignment** — データ取り込みと合法性の割り当て
2. **Annotation correction and speaker/language cleanup** — アノテーション修正と話者・言語のクリーンアップ
3. **Audit and promotion/rejection review** — 監査と昇格・却下のレビュー
4. **Holdout/train split confirmation** — ホールドアウト・学習分割の確認
5. **Model audition and dramatic tuning** — モデルオーディションとドラマチックチューニング
6. **Blinded subjective evaluation** — ブラインド主観評価
7. **Final export approval** — 最終エクスポート承認

各ステップは以下を記録する:

| Field | Description |
|---|---|
| `actor_role` | 操作者の役割 |
| `actor_identity` | 操作者の識別子 |
| `timestamp` | 操作のタイムスタンプ |
| `before_state` | 操作前の状態 |
| `after_state` | 操作後の状態 |
| `reason_note` | 理由・メモ |

## 9. Casting Gallery

- 話者プロファイルの保存・読み込み・エクスポートが可能
- セッションをまたいで永続化する
- `.tmrvc_speaker` ファイルとしてエクスポート可能

## 10. Session Persistence

- ワークショップ状態 (スライダー位置、コンテキスト、アクター、比較履歴) の保存・読み込みが可能
- セッション間で作業状態を維持する
