# 引き継ぎタスク B: Rust エンジン (tmrvc-engine-rs) への SOTA UCLM 統合

## 1. 前提条件と現在のアーキテクチャ

このプロジェクトは「リアルタイム音声変換（VC）および音声合成（TTS）」を目的とした **TMRVC** プラットフォームです。
現在、オーディオスレッド（C++/Rust）で 50ms 以下のレイテンシでストリーミング推論を行うためのエンジン改修フェーズに入っています。

これまでに Python 側で **「Disentangled UCLM (Unified Codec Language Model)」** のアーキテクチャが実装され、以下の仕様が確定しています。
*   **フレーム単位**: 10ms (24kHz で 240 samples) ごとに推論ループが回ります。
*   **トークン仕様 (Token Spec v2)**:
    *   Acoustic Stream (`A_t`): `[1, 8]` の整数テンソル（値域 `0..1023`）。
    *   Control Stream (`B_t`): `[1, 4]` の整数テンソル（スロット: `[op, type, dur, int]`）。
*   **設計書**:
    *   エンジン全体のアーキテクチャとメモリレイアウトは `docs/design/rust-engine-design.md` に記載されています（旧 cpp-engine-design.md から移行済み）。
    *   各 ONNX モデルの入出力仕様は `docs/design/onnx-contract.md` に記載されています。

## 2. ゴール
Rust の推論エンジン (`tmrvc-engine-rs`) をアップデートし、新しい Disentangled UCLM の ONNX モデル群をロードして、リアルタイムストリーミング処理（10ms フレーム）を実行できるようにすること。

## 3. 作業リスト (Task B)

1.  **ONNX セッション管理 (`OrtBundle`) のアップデート**
    *   `tmrvc-engine-rs/src/engine/ort_bundle.rs` (新規作成または既存修正) にて、以下の新しい ONNX セッションを読み込めるように更新してください。
        *   `codec_encoder` (既存のままか微修正)
        *   `vc_encoder` (新規)
        *   `voice_state_enc` (新規)
        *   `uclm_core` (新規)
        *   `codec_decoder` (既存のままか微修正)
2.  **リングバッファ (コンテキスト履歴) の実装**
    *   推論には過去のトークン履歴（`ctx_A`, `ctx_B`）が必要です。
    *   `TensorPool` または専用のバッファ構造体を用いて、直近数秒間（例: L=200フレーム）の `A_t` と `B_t` を保持し、毎フレーム更新する RT-safe な（メモリ確保を行わない）リングバッファを実装してください。
3.  **推論パイプライン (`process_one_frame`) の実装**
    *   `StreamingEngine::process_one_frame` メソッド内に、以下の推論シーケンスを実装してください。
        1. 入力波形 (240 samples) -> `codec_encoder` -> `source_A_t`
        2. `source_A_t` -> `vc_encoder` -> `content_features` (VQ bottlenecked)
        3. 明示的パラメータとSSL状態 -> `voice_state_enc` -> `state_cond`
        4. `content_features` + `ctx_B` + `state_cond` + `cfg_scale` -> `uclm_core` -> `logits_a`, `logits_b`
        5. ロジットからトークンをサンプリング (Categorical)
        6. `A_t`, `B_t`, `state_cond` -> `codec_decoder` -> 出力波形 (240 samples)
4.  **RT-safe 制約の徹底**
    *   オーディオスレッド（`process_one_frame` 内）で `Vec::new()` 等の動的メモリ確保（ヒープアロケーション）が一切発生しないように、全ての入出力用テンソルを初期化時に事前確保（Pre-allocate）する設計を徹底してください。
