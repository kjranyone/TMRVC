# TMRVC GUI Design

この文書は、`plan/worker_12_gradio_control_plane.md` を正本として、`TMRVC Control Plane` が mainline の `pointer-based UCLM` をどう可視化し、どう操作するかを定義する。旧 duration-centric UI は標準仕様に含めない。

## 1. UI 原則

- mainline の観測対象は `A_t`, `B_t`, `voice_state`, `pointer_state`
- pacing 制御は `pace`, `hold_bias`, `boundary_bias`
- legacy 機能は分離表示し、既定画面に混ぜない
- 学習・推論・export の状態を同じ語彙で表す

## 2. Research Studio

### 2.1 ページ構成

| Page | 目的 |
|---|---|
| `Drama Workshop` | TTS 試聴、pacing / `voice_state` 制御、take 管理 |
| `Realtime VC` | realtime VC 確認 |
| `Curation Auditor` | transcript / speaker / language 修正、promote/reject |
| `Dataset Manager` | ingest、合法性、health dashboard、curation orchestration |
| `Evaluation Arena` | blind A/B / MOS |
| `Speaker Enrollment` | reference audio から `SpeakerProfile` 作成 |
| `Training Monitor` | 学習進行と quality gate の確認 |
| `Batch Script` | batch generation / workshop 補助 |
| `ONNX Export` | export / parity |
| `Server Control` | serve 状態と操作 |
| `System Admin` | auth / health / approval policy / telemetry |

### 2.2 Drama Workshop 画面

必須項目:

- text
- `speaker_profile_id` または on-the-fly reference
- `pace`
- `hold_bias`
- `boundary_bias`
- explicit 8-D `voice_state`
- context injection
- compare / take management

表示メトリクス:

- waveform preview
- pointer timeline
- `voice_state` trace
- control token trace
- take lineage / ranking / notes

### 2.3 Curation / Dataset 画面

必須項目:

- dataset upload / registration
- legality / provenance assignment
- manifest browser
- transcript / speaker / language correction
- promote / reject / review action
- export trigger
- post-v3.0 personal-voice training を残す場合は、raw audio upload をそのまま学習に流さず、`VAD -> ASR/transcript check -> G2P -> boundary/alignment refinement` の lightweight preparation job を明示的に踏ませる
- low-confidence transcript/G2P/alignment 項目は review queue に出し、未解決のまま fine-tune を開始させない

表示:

- quality score
- provider confidence
- approval history
- current `metadata_version`
- blocking reasons
- export artifact state

### 2.4 Evaluation / Admin 画面

表示:

- blind A/B assignment
- rater QC
- latency / VRAM / model health
- active runtime contract
- audit-critical actions

### 2.5 Realtime VC 画面

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

推論系:

- missing checkpoints
- character not found
- pointer state divergence
- ONNX parity mismatch

運用系:

- stale `metadata_version`
- policy forbidden
- already submitted
- missing legality / split gate

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

`MFA`, `TextGrid`, `duration injection` は mainline UI の既定画面に出さない。必要なら dev-only / legacy-only 経路へ隔離する。

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
5. **Export / package creation** — export と package 作成
6. **Model audition and dramatic tuning** — モデルオーディションとドラマチックチューニング
7. **Blinded subjective evaluation** — ブラインド主観評価
8. **Final export approval** — 最終エクスポート承認

各ステップは以下を記録する:

| Field | Description |
|---|---|
| `actor_role` | 操作者の役割 |
| `actor_identity` | 操作者の識別子 |
| `timestamp` | 操作のタイムスタンプ |
| `metadata_version` | optimistic locking 用のレコード版数 |
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

## 11. Personal Voice Training Boundary

post-v3.0 に Personal Voice Training を導入する場合でも、UI は "upload wav -> LoRA train" の単純化された幻想を見せてはならない。

- training job の前に canonical trainable artifacts を生成する preparation stage が必須
- preparation stage は Worker 07-owned curation/orchestration path を通す
- transcript/G2P/bootstrap preparation に失敗したデータは block か visible downgrade にする
- raw uploaded audio を直接 trainer に渡す GUI-only shortcut は非準拠
