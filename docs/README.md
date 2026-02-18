# TMRVC Documentation

## docs/design/ — 現行設計 (正本)

Frame-by-frame causal streaming アーキテクチャに基づく設計資料群。
CPU-only で end-to-end 50ms 以下のリアルタイム Voice Conversion を目指す。

| File | Description |
|---|---|
| `architecture.md` | システム全体アーキテクチャ、モジュール構成、データフロー、主要設計判断 |
| `streaming-design.md` | ストリーミング設計、レイテンシバジェット、Audio Thread パイプライン、スレッドモデル |
| `onnx-contract.md` | 5 ONNX モデルの I/O 仕様、State tensor、共有定数、数値パリティ基準 |
| `model-architecture.md` | Content Encoder / Converter / Vocoder / IR Estimator / Speaker Encoder / Teacher の詳細設計 |
| `cpp-engine-design.md` | C++ エンジン (tmrvc-engine) と VST3 プラグイン (tmrvc-plugin) の設計 |
| `training-plan.md` | Teacher 学習計画: コーパス構成、段階的学習フェーズ、コスト見積もり、蒸留への接続 |

## docs/reference/ — 参考資料

現行設計の意思決定に影響を与えた資料。直接の実装仕様としては使用しないが、
品質目標、蒸留手法の選択肢、先行研究の知見として参照する。

| File | Description |
|---|---|
| `system_design.md` | 旧設計 (chunk-based, 15M Student, OT-CFM DiT Teacher)。品質目標 (SECS>=0.92)、3段階蒸留 (Shortcut FM→DMD→ADCD)、データセット計画 (Emilia 101Kh) が参考になる |
| `concept.md` | IR-aware 先行研究マップ。BUDDy, Gencho 等の IR 推定手法、RIR データセット、実装ガイドライン |
