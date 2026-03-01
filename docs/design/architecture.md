# TMRVC Unified Architecture (UCLM v2)

TMRVC は、Unified Codec Language Model (UCLM) v2 を核とした、リアルタイム TTS および高精度 VC を実現する統合音声生成システムである。

## 1. コア・コンセプト

すべての音声生成タスクを「デュアルストリーム・トークン予測」として定義する。

- **Unified (統合)**: TTS（テキストからの生成）と VC（音声をソースとした変換）を同一のトランスフォーマー・バックボーンで処理。
- **Dual-Stream Token Spec v2**:
    - `A_t` (Acoustic Stream): 音響特徴（RVQ トークン）。
    - `B_t` (Control Stream): 非言語イベント（呼吸、強弱、タイミング）。
- **10ms Native Core**: 10ms フレーム単位の処理により、極低遅延なストリーミングを実現。

## 2. システム構成

### 2.1 推論パイプライン

1.  **入力層**:
    - TTS Mode: テキスト → G2P (音素) → TextEncoder
    - VC Mode: ソース音声 → CodecEncoder → VCEncoder (VQ Bottleneck)
2.  **制御層**:
    - 8-dim Voice State: 物理パラメータ（息漏れ、緊張度等）+ SSL Latent
    - Speaker Embedding: LoRA によるパーソナライズ
3.  **生成層 (UCLM Core)**:
    - Dual-head Transformer による `A_t`, `B_t` トークンの同時予測
4.  **出力層**:
    - Emotion-Aware Codec Decoder による音声復元

## 3. コンポーネント定義

- **UCLM (Unified Codec LM)**: トークン予測のメインエンジン。
- **Emotion-Aware Codec**: 高圧縮・高音質な音声トークナイザー。
- **ContextStylePredictor**: LLM による文脈に応じた 8次元物理パラメータの予測。
- **DurationPredictor**: TTS モードにおける正確な時間アライメント。
