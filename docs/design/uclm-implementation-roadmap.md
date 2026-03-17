# UCLM Implementation Map

この文書は、mainline 設計をどのモジュールに実装するかを示す現行マップである。工程管理の正本は `plan/` に置くため、ここでは設計とコード境界だけを定義する。

## 1. 実装の正本

- 全体計画: `../../plan/README.md`
- active worker files: `../../plan/worker_01_architecture.md`, `worker_04_serving.md`, `worker_06_validation.md`, `worker_12_gradio_control_plane.md`

このファイルに段階別ロードマップや旧前提の詳細は持たない。

## 2. モジュール境界

| 領域 | 主担当モジュール |
|---|---|
| 共通定数 / audio base | `tmrvc-core` |
| dataset / cache / text supervision | `tmrvc-data` |
| UCLM / codec 学習 | `tmrvc-train` |
| export contract | `tmrvc-export` |
| API serving | `tmrvc-serve` |
| streaming runtime | `tmrvc-engine-rs` |
| 開発メニュー / オペレーション | `dev.py` |

## 3. mainline 実装テーマ

### 3.1 Architecture

- pointer-based TTS head
- internal alignment learning
- dual-stream token generation
- causal semantic VC path

### 3.2 Training

- pointer loss
- alignment loss
- no hard dependency on `durations.npy`
- legacy duration branch の明示的分離

### 3.3 Data

- monolingual default with explicit multilingual / code-switch metadata
- text units without forced alignment
- cache quality metrics for text coverage and `voice_state` supervision coverage

### 3.4 Serving

- pointer-state driven TTS runtime
- pacing control API
- no duration-expansion main path

## 4. 受け入れ条件

- `dev.py` の主経路で `MFA` なし学習が完結する
- TTS 学習が `phoneme_ids` のみで成立する
- serve が pointer-state TTS を動かせる
- `docs/design/acceptance-thresholds.md` の release gates と矛盾しない
- legacy 機能は mainline と明確に分離される
