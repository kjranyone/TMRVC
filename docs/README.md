# TMRVC Documentation

Updated: 2026-03-08

このディレクトリは、`plan/` を正本とする `UCLM v3` mainline 契約を、設計仕様・運用仕様・評価仕様として整理した正本です。現行 docs の前提は次のとおりです。

- `MFA` は mainline の設計前提にしない
- TTS は `internal alignment + causal pointer` を使う
- `A_t / B_t` の dual-stream token contract は維持する
- 未完了タスクの正本は `plan/`
- `docs/design/` は `plan` から凍結された契約を記述する
- `docs/training/` は監査・是正・補助資料を置く

## オペレーター向け

| File | 役割 |
|---|---|
| `user-manual.md` | クイックリファレンス |
| `operator-guide.md` | WebUI 中心の運用手順書（データ登録、キュレーション、Drama Workshop、評価、voice_state 監督） |

## まず読む

| File | 役割 |
|---|---|
| `design/architecture.md` | システム全体の設計入口 |
| `design/unified-codec-lm.md` | UCLM コアのモデル仕様 |
| `design/emotion-aware-codec.md` | codec / token contract |
| `design/onnx-contract.md` | export / serve / rust 間の I/O 契約 |
| `design/streaming-design.md` | 10 ms streaming runtime 設計 |
| `design/dataset-preparation-flow.md` | dataset / cache の標準仕様 |
| `design/curation-contract.md` | curation manifest / promotion / export 契約 |
| `design/gui-design.md` | WebUI / HITL control plane 契約 |
| `design/auth-spec.md` | auth / audit / optimistic locking 契約 |
| `training/README.md` | 学習文書の入口 |

## 実装計画

現行の active backlog は `plan/` が正本です。

| File | 役割 |
|---|---|
| `../plan/README.md` | 全体計画と依存関係 |
| `../plan/worker_01_architecture.md` | モデル設計 |
| `../plan/worker_04_serving.md` | serving / runtime |
| `../plan/worker_06_validation.md` | validation / acceptance |
| `../plan/worker_12_gradio_control_plane.md` | Gradio/WebUI control plane |
| `../plan/dramatic_acting_requirements.md` | drama-grade acting requirement |
| `../plan/fish_audio_s2_competitive_strategy.md` | 競争戦略 |
| `../plan/repo_remaining_inventory_2026_03.md` | repo 全体の pending-only inventory |

## 学習文書

| File | 内容 |
|---|---|
| `training/README.md` | 学習資料の入口 |
| `training/integrity-audit-2026-03-04.md` | 整合性監査 |
| `training/integrity-remediation-2026-03-05.md` | 是正結果 |

## 運用ルール

1. 実装計画・依存関係・進捗判定は `plan/` を正本とする
2. `docs/design/` は mainline 契約と release gate を記述する
3. legacy 互換機能は必要最小限の注記に留める
4. 数値定数は `configs/constants.yaml` を単一正本とする
