# 引き継ぎタスク A: SOTA UCLM の ONNX エクスポート実装

## ステータス: ✅ 完了 (2026-03-01)

---

## 完了内容

### 実装済み (3モデル)

1. **vc_encoder.onnx**
   - `source_A_t [B,8,L]` → `vq_content_features [B,512,L]`
   - VQ Information Bottleneck で speaker/style 情報を除去

2. **voice_state_enc.onnx**
   - `explicit_state [B,8]`, `ssl_state [B,128]` → `state_cond [B,512]`
   - WavLM + explicit voice parameters の融合

3. **uclm_core.onnx**
   - `content_features [B,512,L]`, `b_ctx [B,4,L]`, `spk_embed [B,192]`, `state_cond [B,512]`, `cfg_scale [1]`
   - → `logits_a [B,8,1024]`, `logits_b [B,4,64]`
   - Dual-stream token prediction (A_t / B_t)

### パリティ検証

| テンソル | L_inf | 状態 |
|---|---|---|
| vc_encoder | 5.96e-08 | ✅ |
| voice_state_enc | 2.24e-08 | ✅ |
| uclm_core (logits_a) | 1.91e-06 | ✅ |
| uclm_core (logits_b) | 1.46e-06 | ✅ |

### 追加変更

- `configs/constants.yaml`: d_model を 512 に変更
- `tmrvc-export/src/tmrvc_export/export_uclm.py`: エクスポートスクリプト
- CLI: `tmrvc-export-uclm` コマンド追加

---

## 残課題 (Task B 以降で対応)

### High Priority

| 課題 | 説明 | 対応 |
|---|---|---|
| codec_encoder | 因果 EnCodec エンコーダ | 新規実装必要 |
| codec_decoder | RVQ デコーダ + ControlEncoder | 新規実装必要 |
| speaker_encoder | オフライン話者埋め込み | 新規実装必要 |
| delta_voice_state | ONNX入力に追加 | 要対応 |
| KV Cache | ONNX入出力に追加 | 要対応 |

### Medium Priority

| 課題 | 説明 |
|---|---|
| CFG Logic | cfg_scale に応じた条件分岐 |
| event_trace | codec_decoder での減衰履歴 |

---

## 元の仕様 (アーカイブ)

### 1. 前提条件と現在のアーキテクチャ

このプロジェクトは「リアルタイム音声変換（VC）および音声合成（TTS）」を目的とした **TMRVC** プラットフォームです。
現在、SOTA（State-of-the-Art）レベルの表現力（特に官能的な息遣いや感情）を達成するために、モデルアーキテクチャを **「Disentangled UCLM (Unified Codec Language Model)」** に刷新しました。

これまでに以下の Python 実装が完了しています。
*   **データパイプライン**: `tmrvc-data` 内に、EnCodecトークン、WavLM（128次元潜在スタイル: `ssl_state`）、8次元物理パラメータ（`explicit_state`）を抽出するスクリプトを実装済み。
*   **モデルアーキテクチャ**: `tmrvc-train/models` 内に以下のモジュールを実装済み。
    *   `VCEncoder`: 情報ボトルネック (Vector Quantizer) を用い、ソース音声から話者/スタイル情報を削ぎ落とす。
    *   `VoiceStateEncoder`: 8次元パラメータとWavLM特徴量を融合し、純粋なスタイル条件 (`state_cond`) を抽出。
    *   `CodecTransformer` (UCLM Core): 音響トークン(`A_t`: 8 codebooks)と制御トークン(`B_t`: 4 slots)のデュアルストリームを予測。
    *   推論時の表現増幅（CFG: Classifier-Free Guidance）対応。
*   **設計書**: 最新のONNX I/O 仕様は `docs/design/onnx-contract.md` に記載されています。

### 2. ゴール
`tmrvc-export` パッケージ内に、PyTorchで学習した `DisentangledUCLM` を、推論エンジン (Rust) が読み込める形式で ONNX エクスポートするスクリプトを作成すること。

### 3. 作業リスト (Task A) - ✅ 全て完了

1.  **エクスポート用ラッパークラスの実装** ✅
    *   `tmrvc-export/src/tmrvc_export/export_uclm.py` を作成
    *   `VCEncoderExportWrapper`, `VoiceStateEncExportWrapper`, `UCLMCoreExportWrapper` 実装
2.  **ONNX エクスポート処理の実装** ✅
    *   `torch.onnx.export` で3モデルを出力
    *   `dynamic_axes` 設定済み
3.  **パリティ検証の実装** ✅
    *   ONNX Runtime で L_inf < 1e-4 を検証
4.  **CLI エントリーポイントの作成** ✅
    *   `tmrvc-export-uclm` コマンド追加
