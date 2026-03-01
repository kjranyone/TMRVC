# TMRVC Documentation

Updated: 2026-03-01 (UCLM v2)

## プロジェクト概要

TMRVC は **Disentangled UCLM** を中核にした Voice Conversion / TTS 統合プロジェクト。
SOTAレベルの官能的表現と完全な分離(Disentanglement)を達成します。

- 内部処理: 24kHz / 10ms フレーム
- 生成表現: `A_t` (Acoustic RVQ, `[8]`) + `B_t` (Control tuple, `[4]`)
- スタイル表現: explicit(8-dim) + ssl(WavLM 128-dim)
- 推論: `codec_encoder -> (vc_encoder) -> voice_state_enc -> uclm_core -> codec_decoder`

---

## 主要ドキュメント (まず読む)

| File | 役割 |
|---|---|
| `design/architecture.md` | 全体設計の入口 (Disentangled UCLM) |
| `design/unified-codec-lm.md` | UCLM アーキテクチャ詳細 |
| `design/emotion-aware-codec.md` | Token Spec v2 (`A_t/B_t`) の正本 |
| `design/onnx-contract.md` | Python/Rust/ONNX の I/O 契約 |
| `design/streaming-design.md` | 10ms ストリーミング / レイテンシ設計 |
| `design/rust-engine-design.md` | Rust 推論エンジン設計 |
| `design/uclm-implementation-roadmap.md` | 実装ロードマップ |

---

## 学習ドキュメント

| File | 内容 |
|---|---|
| `training/README.md` | 学習パイプライン統合ガイド |
| `training/uclm-training-plan.md` | UCLM 学習計画 (Phase 1-4) |

---

## 参照資料 (Historical)

`reference/` は研究メモ・旧設計の背景資料。
実装契約としては使用せず、判断根拠の参照に限定する。

---

## 実装モジュール対応

| Module | 役割 |
|---|---|
| `tmrvc-core` | 共有定数・型 |
| `tmrvc-data` | 前処理・キャッシュ生成 |
| `tmrvc-train` | UCLM 学習 |
| `tmrvc-export` | ONNX 出力/検証 |
| `tmrvc-engine-rs` | RT 推論エンジン |
| `tmrvc-rt` / `tmrvc-vst` | ユーザー向け実行系 |
| `tmrvc-serve` | TTS サーバー |

---

## docs 更新ルール

1. Token 仕様変更時は以下を同時更新する:
   - `design/emotion-aware-codec.md`
   - `design/onnx-contract.md`
   - `design/unified-codec-lm.md`
   - `training/README.md`
2. 旧仕様を残す場合は `Legacy Note` を明示する。
3. 数値定数 (`frame_size`, vocab など) は `configs/constants.yaml` と一致させる。
