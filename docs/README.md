# TMRVC Documentation

Updated: 2026-03-07

このディレクトリは、`UCLM v3` mainline の設計と運用仕様をまとめた正本です。現行 docs の前提は次のとおりです。

- `MFA` は mainline の設計前提にしない
- TTS は `internal alignment + causal pointer` を使う
- `A_t / B_t` の dual-stream token contract は維持する
- docs は completed `AS IS` を記述し、旧仕様の運用手順は主文書に残さない

## まず読む

| File | 役割 |
|---|---|
| `design/architecture.md` | システム全体の設計入口 |
| `design/unified-codec-lm.md` | UCLM コアのモデル仕様 |
| `design/emotion-aware-codec.md` | codec / token contract |
| `design/onnx-contract.md` | export / serve / rust 間の I/O 契約 |
| `design/streaming-design.md` | 10 ms streaming runtime 設計 |
| `design/dataset-preparation-flow.md` | dataset / cache の標準仕様 |
| `training/README.md` | 学習文書の入口 |

## 実装計画

現行の実装分解は `plan/` が正本です。

| File | 役割 |
|---|---|
| `../plan/README.md` | 全体計画と依存関係 |
| `../plan/worker_01_architecture.md` | モデル設計 |
| `../plan/worker_02_training.md` | 学習系 |
| `../plan/worker_03_dataset_alignment.md` | dataset / text supervision |
| `../plan/worker_04_serving.md` | serving / runtime |
| `../plan/worker_05_devops_docs.md` | dev tooling / docs |
| `../plan/worker_06_validation.md` | validation / acceptance |

## 学習文書

| File | 内容 |
|---|---|
| `training/README.md` | 学習資料の入口 |
| `training/integrity-audit-2026-03-04.md` | 整合性監査 |
| `training/integrity-remediation-2026-03-05.md` | 是正結果 |

## 運用ルール

1. docs は mainline の completed state を記述する
2. legacy 互換機能は必要最小限の注記に留める
3. 数値定数は `configs/constants.yaml` を単一正本とする
4. 実装計画は `plan/`、設計仕様は `docs/design/` に置く
