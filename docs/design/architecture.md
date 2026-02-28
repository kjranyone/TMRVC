# TMRVC System Architecture

Kojiro Tanaka — architecture design
Created: 2026-02-16 (Asia/Tokyo)
Updated: 2026-02-28 — UCLM パラダイムに統合

> **Goal:** End-to-end 50ms 以下 (DAW バッファ込み)、CPU-only リアルタイム Voice Conversion + TTS。
> **Current Paradigm:** UCLM (Unified Codec Language Model) — 単一モデルで TTS と VC を統合。

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
│                      UCLM (Unified Codec LM)                     │
│                                                                  │
│  TTS Mode:  text + voice_state + speaker → CodecLM → Audio      │
│  VC Mode:   source_tokens + speaker + voice_state → Audio       │
│                                                                  │
│  Components:                                                     │
│    - EnCodec (Encoder + Decoder)                                │
│    - UCLM Core (Transformer)                                    │
│    - VoiceStateEncoder                                          │
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
| **EnCodec** | waveform | tokens [n_cb, T] | 符号化/復号 |
| **UCLM Core** | (text \| source_tokens) + voice_state + spk_embed | tokens | トークン生成 |
| **VoiceStateEncoder** | voice_state [T, 8] | features [T, d] | 条件付け |
| **SpeakerEncoder** | audio | embed [192] | 話者埋め込み |

### 3.2 フレームレート

| コンポーネント | サンプルレート | フレームレート |
|---|---|---|
| DAW Audio | 48kHz | — |
| 内部 Audio | 24kHz | — |
| Mel / Voice State | 24kHz | 100 fps (10ms) |
| EnCodec Tokens | 24kHz | 75 fps (~13.3ms) |

### 3.3 推論フロー (VC Mode)

```
1. DAW Audio (48kHz) → Resample → 24kHz
2. 24kHz Audio → EnCodec Encoder → tokens [8, T]
3. tokens + spk_embed + voice_state → UCLM Core → target_tokens
4. target_tokens → EnCodec Decoder → 24kHz Audio
5. 24kHz Audio → Resample → 48kHz → DAW Output
```

### 3.4 推論フロー (TTS Mode)

```
1. Text → TextEncoder → text_features
2. text_features + voice_state + spk_embed → UCLM Core → tokens
3. tokens → EnCodec Decoder → 24kHz Audio
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

---

## 5. データパイプライン

```
Raw Audio (data/raw/)
    │
    ├──▶ prepare_dataset.py
    │      ├── Normalize (24kHz, loudness)
    │      ├── Annotate (Whisper, emotion)
    │      ├── Extract (mel, f0, spk_embed)
    │      └── Save (data/cache/)
    │
    ├──▶ add_codec_to_cache.py
    │      ├── EnCodec → codec_tokens
    │      └── VoiceStateEstimator → voice_state
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
uv run python scripts/parallel_prepare.py \
    --input data/moe_multispeaker_voices \
    --name moe_multispeaker \
    --n-jobs 4

# codec_tokens 追加
uv run python scripts/add_codec_to_cache.py --speaker moe_spk_01 --device cuda

# UCLM 学習
uv run tmrvc-train-uclm \
    --cache-dir data/cache \
    --datasets vctk,moe_multispeaker \
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

### Archive (参考)

| ファイル | 内容 | 状態 |
|---|---|---|
| `expressive-tts-design-VSF.md` | Voice Source Flow (パイプライン案) | 古い |
| `unified-generator-tts-UCG.md` | UCG (初期案) | 古い |
| `codec-latent-design.md` | Codec-Latent (Mamba案) | 参考 |

### Legacy (更新予定)

| ファイル | 状態 |
|---|---|
| `model-architecture.md` | UCLM に更新必要 |
| `acoustic-condition-pathway.md` | 参考資料 (UCLM では使用しない) |
| `onnx-contract.md` | UCLM ONNX 仕様に更新必要 |
| `cpp-engine-design.md` | Rust エンジン設計 (有効) |

---

## 9. Consistency Checklist

- [ ] UCLM のフレームレート: EnCodec 75fps vs Voice State 100fps → リサンプリング済み
- [ ] constants.yaml に UCLM 定数追加済み
- [ ] Rust エンジンは UCLM 対応予定
- [ ] ONNX エクスポート未実装
