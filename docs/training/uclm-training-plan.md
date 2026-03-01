# Disentangled UCLM Training Plan

Kojiro Tanaka
Created: 2026-03-01

> **Overview:** UCLM (Unified Codec Language Model) は TTS と VC を単一のモデルで学習します。
> 以前の「TTS用」「VC用」に分かれていた学習プランは本ドキュメントに統合されました。

## 1. 目的

SOTAレベルの表現力（官能的表現、息遣い、感情の揺らぎ）と完全な Disentanglement（分離）を備えた UCLM を学習する。

## 2. フェーズ構成

| Phase | タスク | データセット | 目標 |
|---|---|---|---|
| **Phase 1** | Acoustic Pretrain (EnCodec Decoder Fine-tuning) | Expresso, Intimate | 高周波・息の音の解像度向上 |
| **Phase 2** | UCLM Base (TTS/VC Joint) | LibriTTS-R, VCTK | 基礎的な言語理解と音響生成 |
| **Phase 3** | Disentanglement Training | + JVS | GRL/VQ を用いた話者・内容の分離 |
| **Phase 4** | Emotional & Sensual Fine-tuning | JVNV, Expresso, Custom | 非言語・官能表現の増幅 (CFG学習) |

## 3. 損失関数と正則化

1.  **Reconstruction (CE Loss):** `A_t` (RVQ) と `B_t` (Control) の交差エントロピー。
2.  **Information Bottleneck (VQ Loss):** VC Encoder における Commitment Loss。
3.  **Adversarial Loss (GRL):** `voice_state` からの話者/テキスト分類エラー。

## 4. データセットの準備状況

-   データセットの前処理 (WavLM 特徴量の抽出など) は `docs/design/dataset-preparation-flow.md` を参照。
