# Unified Codec Language Model (UCLM) v2

## 1. Abstract

**Unified Codec Language Model (UCLM)** は、テキスト読み上げ (TTS) と音声変換 (VC) を単一のニューラル・アーキテクチャで実現する統合音声生成モデルである。従来の分離されたパイプラインとは異なり、UCLM はすべての音声生成タスクを「条件付けされたコーデック・トークン変換」として定義し、極低遅延なリアルタイム生成を可能にする。

Key Features:
1. **Unified Architecture**: 同一のトランスフォーマー・バックボーンで TTS と VC を実行。
2. **Dual-Stream Token Spec v2**: 音響トークン (`A_t`) と制御トークン (`B_t`) の同時生成。
3. **8-dim Physical Voice State**: 息漏れ、緊張度などの物理パラメータによる直接的な表現制御。
4. **Real-time Core**: 10ms フレーム単位の因果的（Causal）推論。

---

## 2. アーキテクチャ

### 2.1 統一定式化

TTS と VC は、以下の確率分布のサンプリングとして統合される：

```
P(A_t, B_t | input, speaker, voice_state_t, past_context)
```

- **Acoustic stream `A_t`**: `[B, 8]`, RVQ IDs `0..1023`
- **Control stream `B_t`**: `[B, 4]=[op, type, dur, int]`
- **Frame unit**: 10ms (`240 samples @ 24kHz`)

### 2.2 構成要素

1.  **Modality Encoders**:
    - `TextEncoder`: 音素 ID を連続的な特徴量に変換。
    - `VCEncoder`: ソース音声トークンを VQ Bottleneck を通じて抽象化。
2.  **VoiceStateEncoder**:
    - 8次元の物理パラメータと SSL (WavLM) 潜在空間を統合。
    - GRL (Gradient Reversal Layer) により、内容や話者情報を排除。
3.  **Codec Transformer**:
    - 因果的アテンションを用いたデュアルヘッド・トランスフォーマー。
    - `A_t` と `B_t` をパラレルまたはインターリーブで予測。
4.  **Emotion-Aware Codec Decoder**:
    - トークン列から高品質な 24kHz 波形を復元。

---

## 3. 入出力仕様

### 3.1 制御パラメータ (8-dim Voice State)

| Dim | Name | Range | Description |
|---|---|---|---|
| 0 | Breathiness | [0, 1] | 息漏れの多さ |
| 1 | Tension | [0, 1] | 声帯の緊張度 |
| 2 | Arousal | [0, 1] | 覚醒度・エネルギー |
| 3 | Valence | [-1, 1] | 感情の正負（快・不快） |
| 4 | Roughness | [0, 1] | 声の掠れ・ざらつき |
| 5 | Voicing | [0, 1] | 有声性の強さ |
| 6 | Energy | [0, 1] | 全体的な音量感 |
| 7 | Speech Rate | [0.5, 2.0] | 発話速度 |

---

## 4. 学習戦略

### 4.1 マルチタスク学習

TTS と VC のタスクをランダムにサンプリングして同時学習する。

- **TTS Task**: `target_A, target_B` をテキストから予測。
- **VC Task**: `target_A, target_B` をソース音声のトークンから予測。
- **Adversarial Loss**: GRL を用いて話者情報の分離を強化。
- **CFG Dropout**: 15% の確率で条件をドロップし、推論時の Classifier-Free Guidance を可能にする。

---

## 5. 推論プロトコル

### 5.1 リアルタイム・ストリーミング

1.  10ms 単位で入力を受け取り、KV キャッシュを更新。
2.  `A_t`, `B_t` をサンプリング（Greedy または Top-p）。
3.  直ちにデコーダへ渡し、240 サンプルの音声を出力。
4.  理論上のアルゴリズム遅延：10ms。
