# TMRVC System Architecture

Kojiro Tanaka — architecture design
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-03-01 — UCLM Token Spec v2 を反映

> **Goal:** End-to-end 50ms 以下 (DAW バッファ込み)、CPU-only リアルタイム Voice Conversion + TTS での究極の官能的表現の実現。
> **Current Paradigm:** Disentangled UCLM — 単一モデルでTTSとVCを統合。Adversarial Disentanglement (分離) と WavLM ベースの Latent Voice State を導入した SOTA アーキテクチャ。
> **Token Spec v2:** `A_t` (Acoustic RVQ, `[B,8]`, `0..1023`) + `B_t` (Control tuple, `[B,4]`, vocab=64) の dual-stream。

---

## 1. システム概要

```
┌────────────────────────────────────────────────────────────────────────┐
│                           DAW (Host)                                   │
│                                                                        │
│   Audio In ──▶ ┌─────────────────────────────────┐ ──▶ Audio Out      │
│                │     TMRVCProcessor (VST3)       │                    │
│                │  ┌───────────────────────────┐  │                    │
│                │  │    StreamingEngine         │  │                    │
│                │  │  ┌─────────────────────┐  │  │                    │
│                │  │  │  UCLM (ONNX)       │  │  │                    │
│                │  │  │  ┌───────────────┐  │  │  │                    │
│                │  │  │  │CodecEncoder   │  │  │  │                    │
│                │  │  │  │UCLM Core      │  │  │  │                    │
│                │  │  │  │CodecDecoder   │  │  │  │                    │
│                │  │  │  └───────────────┘  │  │  │                    │
│                │  │  └─────────────────────┘  │  │                    │
│                │  └───────────────────────────┘  │                    │
│                └─────────────────────────────────┘                    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. パラダイムの歴史

TMRVC は 3 つの設計パラダイムを経て、現在は **UCLM** に統合されている。

### 2.1 パラダイム比較

| パラダイム | モデル数 | TTS | VC | 設計資料 |
|---|---|---|---|---|
| **Teacher-Student** | 5+ | ✗ | ✓ | (旧設計、非公開) |
| **Codec-Latent** | 3 | ✗ | ✓ | `codec-latent-design.md` |
| **UCLM** | 2 | ✓ | ✓ | `unified-codec-lm.md` |

### 2.2 各パラダイムの概要

#### Teacher-Student (Phase 0-1)

```
Source Audio → ContentEncoder → Content → Converter → Vocoder → Target Audio
                                       ↑
                                 SpeakerEmbed + AcousticParams
```

- **特長:** 従来の VC パイプライン
- **課題:** パイプライン複雑、TTS 非対応

#### Codec-Latent (Phase 2)

```
Source Audio → CodecEncoder → Tokens → TokenModel (Mamba) → Tokens → Decoder → Audio
```

- **特長:** トークン予測による VC
- **課題:** TTS 非対応、学習複雑

#### UCLM (Current)

```
┌─────────────────────────────────────────────────────────────────┐
│                      Disentangled UCLM                           │
│                                                                  │
│  TTS Mode: text + speaker + (voice_state + SSL) → UCLM → A_t/B_t │
│  VC Mode:  source_A_t (VQ-bottlenecked) + speaker +              │
│            (voice_state + SSL) → UCLM → A_t/B_t                  │
│                                                                  │
│  Components:                                                     │
│    - Fine-tuned EnCodec (Encoder + Decoder)                      │
│    - VC Encoder w/ Information Bottleneck (VQ)                   │
│    - VoiceStateEncoder w/ WavLM SSL & GRL (Adversarial)          │
│    - UCLM Core (Transformer) + CFG (Classifier-Free Guidance)    │
│    - Control Encoder (B_t -> decoder conditioning)               │
│    - SpeakerEncoder                                              │
└─────────────────────────────────────────────────────────────────┘
```

- **特長:** TTS + VC 統合、パイプライン簡素化
- **状態:** 現在実装中

---

## 3. 現在のアーキテクチャ (UCLM)

### 3.1 モデル構成

| モデル | 入力 | 出力 | 用途 |
|---|---|---|---|
| **EnCodec** | waveform | `A_t` tokens [8, T] | 符号化/復号 |
| **Control Tokenizer** | labels / estimated events | `B_t` tokens [4, T] | Non-verbal 制御トークン化 |
| **UCLM Core** | (text \| source_A_t) + explicit_state + ssl_state + spk_embed + past(A,B) | `A_t', B_t'` | dual-stream 生成 (推論時 CFG 適用) |
| **VCEncoder (VQ)** | source_A_t | bottlenecked_features | 音声のスタイル/話者情報削ぎ落とし |
| **VoiceStateEncoder** | explicit_state [T, 8], ssl_state [T, 128] | features [T, d] | GRLによる分離・純粋スタイル抽出 |
| **SpeakerEncoder** | audio | embed [192] | 話者埋め込み |

### 3.2 フレームレート

| コンポーネント | サンプルレート | フレームレート |
|---|---|---|
| DAW Audio | 48kHz | — |
| 内部 Audio | 24kHz | — |
| Mel / Voice State | 24kHz | 100 fps (10ms) |
| Acoustic Tokens (`A_t`) | 24kHz | 100 fps (10ms) |
| Control Tokens (`B_t`) | 24kHz | 100 fps (10ms) |

### 3.3 推論フロー (VC Mode)

```
1. DAW Audio (48kHz) → Resample → 24kHz
2. 24kHz Audio → EnCodec Encoder → source_A_t [8, T]
3. source_A_t + spk_embed + voice_state + past(A,B) → UCLM Core → target_A_t + target_B_t
4. target_A_t + target_B_t → Codec Decoder (+Control Encoder/FiLM) → 24kHz Audio
5. 24kHz Audio → Resample → 48kHz → DAW Output
```

### 3.4 推論フロー (TTS Mode)

```
1. Text → TextEncoder → text_features
2. text_features + voice_state + spk_embed + past(A,B) → UCLM Core → A_t + B_t
3. A_t + B_t → Codec Decoder → 24kHz Audio
```

---

## 4. 共通コンポーネント

### 4.1 SpeakerEncoder

- **アーキテクチャ:** ECAPA-TDNN
- **出力:** 192 次元埋め込み
- **学習:** 事前学習済み (VoxCeleb)
- **ファイル:** `.tmrvc_speaker`

### 4.2 Voice State Parameters

| Index | Name | Range | Description |
|---|---|---|---|
| 0 | breathiness | [0, 1] | 気息成分 |
| 1 | tension | [0, 1] | 声帯緊張 |
| 2 | arousal | [0, 1] | 感情覚醒度 |
| 3 | valence | [-1, 1] | 感情価 |
| 4 | roughness | [0, 1] | 声の粗さ |
| 5 | voicing | [0, 1] | 有声/無声 |
| 6 | energy | [0, 1] | エネルギー |
| 7 | rate | [0.5, 2.0] | 話速 |

### 4.3 EnCodec

- **モデル:** `facebook/encodec_24khz`
- **Codebooks:** 8
- **Vocabulary:** 1024
- **Bandwidth:** 6 kbps
- **運用:** special token を持たず `0..1023` のみ (special/control は `B_t` 側)

---

## 5. データパイプライン

```
Raw Audio (data/raw/)
    │
    ├──▶ scripts/data/prepare_dataset.py
    │      ├── Normalize (24kHz, loudness)
    │      ├── Annotate (Whisper, emotion)
    │      ├── Extract (mel, f0, spk_embed)
    │      └── Save (data/cache/)
    │
    ├──▶ scripts/annotate/add_codec_to_cache.py
    │      ├── EnCodec → acoustic_tokens (A_t)
    │      ├── Control tokenizer → control_tokens (B_t)
    │      └── SSLVoiceStateEstimator → explicit_state + ssl_state
    │
    └──▶ UCLMDataset
           └── DataLoader → Training
```

---

## 6. 学習パイプライン

### 6.1 Phase 構成

| Phase | データ | モデル | 目標 |
|---|---|---|---|
| **A** | LibriTTS-R | UCLM (TTS only) | 基本 TTS |
| **B** | VCTK + JVS | UCLM (TTS + VC) | VC 統合 |
| **C** | moe_multispeaker | UCLM | 多話者・感情 |
| **D** | Expresso + Custom | UCLM | NV・濡れ場 |

### 6.2 CLI

```bash
# データ前処理
uv run python scripts/data/parallel_prepare.py \
    --input data/moe_multispeaker_voices \
    --name moe_multispeaker \
    --n-jobs 4

# codec_tokens 追加 (speaker 単位)
uv run python scripts/annotate/add_codec_to_cache.py \
    --cache-dir data/cache/moe_multispeaker/train \
    --raw-dir data/raw/moe_multispeaker_voices \
    --speaker moe_spk_01 \
    --device cuda

# UCLM 学習
uv run tmrvc-train-uclm \
    --cache-dir data/cache \
    --datasets vctk moe_multispeaker \
    --device cuda
```

---

## 7. 用語集

| 用語 | 定義 |
|---|---|
| **UCLM** | Unified Codec Language Model。TTS/VC 統合モデル |
| **Codec-Latent** | EnCodec トークンを潜在表現として扱うパラダイム |
| **Voice State** | breathiness, tension 等 8 次元の音響パラメータ |
| **NV** | Non-Verbal Vocalization（笑い、泣き、息 etc.） |
| **StreamingEngine** | Rust によるリアルタイム推論エンジン |
| **.tmrvc_speaker** | 話者埋め込みファイル形式 |

---

## 8. 設計資料一覧

### Current (有効)

| ファイル | 内容 |
|---|---|
| `unified-codec-lm.md` | UCLM アーキテクチャ詳細 |
| `uclm-implementation-roadmap.md` | 実装ロードマップ |
| `streaming-design.md` | ストリーミング設計 |
| `dataset-preparation-flow.md` | データ準備フロー |

### Archive (参考: docs/design/archive/ に移動済)

| ファイル | 内容 | 状態 |
|---|---|---|
| `model-architecture.md` | Legacy (旧 Codec-Latent 定義) | アーカイブ済 |
| `codec-latent-design.md` | Codec-Latent (Mamba案) | アーカイブ済 |
| `UCLM.md` | UCLM 概要 (旧版) | アーカイブ済 |

### Legacy & Spec

| ファイル | 状態 |
|---|---|
| `acoustic-condition-pathway.md` | 参考資料 (補助条件経路) |
| `pitch-control-design.md` | 参考資料 (ピッチ制御) |
| `onnx-contract.md` | UCLM ONNX 仕様 (v2) |
| `rust-engine-design.md` | Rust エンジン設計 (有効) |

---

## 9. Consistency Checklist

- [x] UCLM は 10ms (100fps) で `A_t/B_t` を同時生成
- [x] constants.yaml に UCLM 定数追加済み
- [x] `RVQ_VOCAB_SIZE=1024` と `CONTROL_VOCAB_SIZE=64` が全実装で一致
- [x] Rust エンジンは UCLM 対応済み (processor.rs)
- [x] ONNX エクスポート実装済み (export_uclm.py)
