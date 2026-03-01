# TMRVC ユーザーマニュアル

TMRVC は CPU のみでリアルタイム Voice Conversion を実現するソフトウェアです。
自分の声をリアルタイムで別の話者の声に変換できます。
Updated: 2026-03-01 (UCLM v2)

---

## 目次

1. [動作環境](#1-動作環境)
2. [ファイル構成](#2-ファイル構成)
3. [ボイスプロファイルの作成](#3-ボイスプロファイルの作成)
   - [音声の準備](#31-音声の準備)
   - [簡易モード（Embedding のみ）](#32-簡易モードembedding-のみ)
   - [高品質モード（Few-shot Fine-tuning）](#33-高品質モードfew-shot-fine-tuning)
   - [GUI から作成](#34-gui-から作成)
4. [リアルタイム変換（tmrvc-rt）](#4-リアルタイム変換tmrvc-rt)
   - [起動と初期設定](#41-起動と初期設定)
   - [パラメータの調整](#42-パラメータの調整)
   - [品質モードの切り替え](#43-品質モードの切り替え)
5. [VST3 プラグインとしての利用](#5-vst3-プラグインとしての利用)
6. [ボイスプロファイルの管理](#6-ボイスプロファイルの管理)
7. [トラブルシューティング](#7-トラブルシューティング)

---

## 1. 動作環境

| 項目 | 要件 |
|------|------|
| OS | Windows 10/11 (64-bit) |
| CPU | Intel / AMD x86-64（SSE4.2 以上） |
| メモリ | 512 MB 以上（モデルロード用） |
| ストレージ | ONNX モデル一式で約 50 MB |
| オーディオ | ASIO / WASAPI 対応デバイス |

**ボイスプロファイル作成時の追加要件:**

| 項目 | 要件 |
|------|------|
| Python | 3.11 以上 |
| パッケージマネージャ | uv |
| GPU（推奨） | Intel Arc / NVIDIA CUDA（CPU でも動作可） |

---

## 2. ファイル構成

リアルタイム変換に必要なファイルは以下の通りです。

```
TMRVC/
├── tmrvc-rt.exe                         # リアルタイム変換 GUI
├── models/
│   └── fp32/
│       ├── codec_encoder.onnx           # 必須
│       ├── uclm_core.onnx               # 必須
│       ├── codec_decoder.onnx           # 必須
│       └── speaker_encoder.onnx         # 任意（プロファイル作成時）
└── speakers/
    ├── my_voice.tmrvc_speaker           # ボイスプロファイル
    └── ...
```

> **Note:** ONNX モデルは学習・蒸留・エクスポートの工程で生成されます。
> 配布パッケージには同梱済みです。

---

## 3. ボイスプロファイルの作成

TMRVC では、変換先の声を `.tmrvc_speaker` ファイルとして保存します。
一度作成すれば、何度でも再利用できます（ファイルサイズは約 64 KB）。

作成方法は 2 つあります。

| モード | 必要な音声 | 所要時間 | 品質 |
|--------|-----------|---------|------|
| 簡易（Embedding） | 3〜10 秒 | 数秒 | 普通 |
| 高品質（Fine-tuning） | 10〜60 秒 | 1〜5 分 | 高い |

### 3.1 音声の準備

変換先にしたい話者の音声ファイルを用意します。

**推奨条件:**

- フォーマット: WAV, FLAC, MP3, OGG（WAV 推奨）
- サンプルレート: 24 kHz 以上（自動リサンプルされます）
- チャンネル: モノラル推奨（ステレオの場合はミックスダウンされます）
- 内容: 通常の会話音声（歌声・囁き声は非推奨）
- 環境: なるべく静かな環境での録音
- ファイル数: 1 ファイルでも可、複数ファイルでも可

**音声の長さの目安:**

| 長さ | 品質への影響 |
|------|-------------|
| 3〜5 秒 | 簡易モードで最低限動作。声質の再現はやや粗い |
| 10〜30 秒 | Fine-tuning で十分な品質。推奨 |
| 30〜60 秒 | 話者の特徴をより正確に捉える |
| 60 秒以上 | 品質向上は緩やかになる |

### 3.2 簡易モード（Embedding のみ）

話者の声の特徴ベクトル（192 次元）のみを抽出します。
LoRA 重み調整は行わないため高速ですが、声質の再現度は劣ります。

**tmrvc-rt GUI から:**

1. tmrvc-rt を起動
2. 「Voice Profile」パネルを開く
3. 音声ファイルをドラッグ＆ドロップ
4. 「Embedding Only」を選択
5. プロファイル名・作者名を入力
6. 「Create」をクリック

**CLI から:**

```bash
uv run tmrvc-enroll \
  --audio-dir ./reference_audio/ \
  --output models/target_voice.tmrvc_speaker \
  --level light \
  --name "ターゲットボイス" \
  --device xpu
```

> `--level light` は埋め込みのみを生成します。

### 3.3 標準モード（Embedding + Style + Reference Tokens）

話者埋め込みに加えて style embedding と reference token を保存します。
現行の安定運用では `--level standard` を推奨します。

**CLI から:**

```bash
uv run tmrvc-enroll \
  --audio-dir ./reference_audio/ \
  --output models/target_voice.tmrvc_speaker \
  --level standard \
  --codec-checkpoint checkpoints/codec/best.pt \
  --name "ターゲットボイス" \
  --device xpu
```

**主要オプション:**

| オプション | 既定値 | 説明 |
|-----------|--------|------|
| `--audio` | — | 単一音声ファイル |
| `--audio-dir` | — | 音声ファイルのディレクトリ |
| `--output` | — | 出力 `.tmrvc_speaker` パス（必須） |
| `--level` | `standard` | `light` / `standard` / `full` |
| `--codec-checkpoint` | — | `standard/full` で参照 token 抽出に使用 |
| `--token-model` | — | `full` で LoRA fine-tune に使用 |
| `--finetune-steps` | 200 | `full` の fine-tune ステップ数 |
| `--max-ref-frames` | 150 | 保存する参照 token の最大フレーム数 |
| `--name` | `Speaker` | プロファイル表示名 |
| `--device` | `cpu` | 使用デバイス（`xpu` / `cuda` / `cpu`） |

**出力例:**

```
[INFO] Processing 5 audio file(s)
[INFO] Extracting speaker embedding...
[INFO] Extracting style embedding...
[INFO] Extracting reference tokens (max 150 frames)...
[INFO] Created speaker profile: models/target_voice.tmrvc_speaker
[INFO] Level: standard
```

> `--level full` は実験的で未実装項目を含むため、通常運用では `light` か `standard` を使用します。

### 3.4 GUI から作成

tmrvc-rt の GUI からもプロファイルを作成できます。

1. 「Voice Profile」パネルを開く
2. 参照音声ファイルをドラッグ＆ドロップ（複数可）
3. 作成モードを選択:
   - **Embedding Only** — 高速、品質は普通
   - **Fine-tune** — 1〜5 分、高品質（Python 環境が必要）
4. プロファイル名と出力先を指定
5. 「Create」をクリック
6. 進捗バーが 100% になればプロファイル完成

> Fine-tune モードは内部で `tmrvc-enroll --level full` を呼び出します。
> 現時点では `full` は未実装項目を含むため、`standard` を推奨します。

---

## 4. リアルタイム変換（tmrvc-rt）

### 4.1 起動と初期設定

```bash
# Rust GUI を起動
cargo run -p tmrvc-rt --release
```

または配布版の `tmrvc-rt.exe` を実行します。

**初回セットアップ:**

1. **オーディオデバイス設定**
   - Input Device: マイクを選択
   - Output Device: スピーカー / ヘッドホンを選択
   - Buffer Size: 256 samples 推奨（低レイテンシなら 128）

2. **モデル読み込み**
   - Model Directory: `models/fp32` を指定
   - 「Load Models」をクリック
   - `codec_encoder / uclm_core / codec_decoder` が読み込まれます

3. **ボイスプロファイル読み込み**
   - Speaker File: 作成済みの `.tmrvc_speaker` を選択
   - 「Load Speaker」をクリック

4. **変換開始**
   - 「Start」をクリック
   - マイクに向かって話すと、リアルタイムで変換された声が出力されます

### 4.2 パラメータの調整

リアルタイム変換中に以下のパラメータをスライダーで調整できます。

#### 基本パラメータ

| パラメータ | 範囲 | 説明 |
|-----------|------|------|
| **Dry/Wet** | 0〜100% | 原音と変換音のミックス比率。100% で完全変換 |
| **Output Gain** | -∞〜+12 dB | 出力音量の調整 |

#### 声質パラメータ

| パラメータ | 範囲 | 説明 |
|-----------|------|------|
| **α Timbre** | 0.0〜1.0 | 声色の変換強度。0 = 自分の声、1 = ターゲットの声 |
| **β Prosody** | 0.0〜1.0 | 抑揚の変換強度。スタイルファイル読み込み時のみ有効 |
| **γ Articulation** | 0.0〜1.0 | 滑舌の変換強度。スタイルファイル読み込み時のみ有効 |

> **Tip:** まずは α Timbre = 0.8、Dry/Wet = 100% から始めて調整するのがおすすめです。
> α = 1.0 にすると声質は最も近くなりますが、不自然になることがあります。

#### レイテンシ・品質パラメータ

| パラメータ | 範囲 | 説明 |
|-----------|------|------|
| **Quality (q)** | 0.0〜1.0 | レイテンシと品質のトレードオフ |

詳細は [4.3 品質モードの切り替え](#43-品質モードの切り替え) を参照してください。

### 4.3 品質モードの切り替え

Quality スライダー（`q`）でレイテンシと品質のバランスを調整できます。

```
q = 0.0 ─────── q = 0.3 ──────── q = 1.0
│                 │                 │
│  Live モード     │  HQ モード       │
│  レイテンシ ~35-45ms │  レイテンシ ~60-90ms │
│  品質: 標準      │  品質: 高い       │
```

| モード | q の範囲 | レイテンシ | 動作 |
|--------|---------|-----------|------|
| **Live** | 0.0〜0.3 | 約 35〜45 ms | 因果的処理、低遅延 |
| **HQ** | 0.3〜1.0 | 約 60〜90 ms | 品質優先、追加文脈を使用 |

**HQ モードの前提条件:**
- 実装バージョンにより未対応の場合があります
- 未対応時は q の値に関わらず Live 相当で動作します

**モード切り替え時の挙動:**
- 100 ms のクロスフェードで滑らかに切り替わります
- CPU 過負荷を検知すると自動的に Live モードに切り替わります（適応劣化機能）

> **配信・通話用途:** q = 0.0（Live モード）推奨。遅延が最小です。
> **録音・後処理用途:** q = 0.8〜1.0（HQ モード）推奨。品質を優先できます。

---

## 5. VST3 プラグインとしての利用

TMRVC は VST3 プラグインとして DAW から利用できます。

### セットアップ

1. `tmrvc.vst3` を DAW の VST3 プラグインフォルダにコピー
   - Windows: `C:\Program Files\Common Files\VST3\`
2. DAW を起動し、トラックにプラグインを挿入
3. プラグイン UI でモデルとボイスプロファイルを読み込む

### DAW での推奨設定

| 設定 | 推奨値 | 理由 |
|------|--------|------|
| バッファサイズ | 256 samples | レイテンシと安定性のバランス |
| サンプルレート | 48 kHz | 内部で 24 kHz に自動変換 |
| プラグイン位置 | インサート最後 | 他のエフェクトの後に配置 |

### オートメーション可能パラメータ

以下のパラメータは DAW のオートメーションで制御できます。

| パラメータ | ID | 範囲 |
|-----------|-----|------|
| Dry/Wet | `dry_wet` | 0.0〜1.0 |
| Output Gain | `output_gain` | 0.0〜2.0 |
| Quality | `latency_quality` | 0.0〜1.0 |
| α Timbre | `alpha_timbre` | 0.0〜1.0 |
| β Prosody | `beta_prosody` | 0.0〜1.0 |
| γ Articulation | `gamma_articulation` | 0.0〜1.0 |
| Voice Source Blend | `voice_source_alpha` | 0.0〜1.0 |

> **Note:** ボイスプロファイルの切り替えはオートメーション非対応です。
> プロファイルはプラグイン UI から手動で読み込んでください。

### レイテンシ補正

DAW が自動レイテンシ補正に対応している場合、TMRVC は処理遅延を報告します。

| Quality 設定 | 報告レイテンシ |
|-------------|--------------|
| Live (q ≤ 0.3) | 10 ms 相当 (例: 480 samples @48kHz) |
| HQ (q > 0.3) | 実装依存（GUI 表示値を優先） |

---

## 6. ボイスプロファイルの管理

### ファイル形式

`.tmrvc_speaker` は独自のバイナリ形式（v3）です。

```
┌──────────────────────────────────────────┐
│ ヘッダ (24 bytes)                         │
│  Magic: "TMSP"   Version: 3              │
├──────────────────────────────────────────┤
│ 話者 Embedding (768 bytes)               │
│  192 次元 × float32                      │
├──────────────────────────────────────────┤
│ Optional blocks                           │
│  style_embed / reference_A/B_tokens /     │
│  lora_delta / voice_source_preset         │
├──────────────────────────────────────────┤
│ メタデータ (可変長, JSON)                  │
│  プロファイル名, 作者名, サムネイル等       │
├──────────────────────────────────────────┤
│ SHA-256 チェックサム (32 bytes)            │
└──────────────────────────────────────────┘
```

**ファイルサイズ:** 可変（オプション項目の有無で変動）

### メタデータの内容

| フィールド | 説明 |
|-----------|------|
| `version` | フォーマットバージョン (`3`) |
| `spk_embed` | 192 次元の話者埋め込み |
| `f0_mean` | 話者平均 F0 |
| `style_embed` | 任意のスタイル埋め込み |
| `reference_A_tokens` | 任意の in-context acoustic token |
| `reference_B_tokens` | 任意の in-context control token |
| `lora_delta` | 任意の適応パラメータ |
| `voice_source_preset` | 任意の 8 次元プリセット |
| `metadata` | 表示名・作者・作成日時など |

### 話者の切り替え

リアルタイム変換中でもプロファイルを切り替えられます。

1. 新しい `.tmrvc_speaker` ファイルを選択
2. 「Load Speaker」をクリック
3. 即座に切り替わります（ONNX モデルの再読み込みは不要）

> 切り替え時に一瞬音声が途切れることがあります。

---

## 7. トラブルシューティング

### 音が出ない

| 確認項目 | 対処 |
|---------|------|
| オーディオデバイスが正しく選択されているか | Input / Output Device を確認 |
| モデルが読み込まれているか | 「Load Models」が成功したか確認 |
| ボイスプロファイルが読み込まれているか | 「Load Speaker」が成功したか確認 |
| Dry/Wet が 0% になっていないか | スライダーを確認 |
| Output Gain が 0 になっていないか | スライダーを確認 |

### 音が途切れる・ノイズが入る

| 症状 | 原因と対処 |
|------|-----------|
| プチプチ音 | バッファサイズを 256 → 512 に増やす |
| 定期的な途切れ | CPU 負荷が高い。Quality を下げる (q → 0.0) |
| ノイズ混入 | マイクのゲインを下げる。ノイズゲートを併用する |

### HQ モードに切り替わらない

- 実行バイナリが HQ 機能を持つバージョンか確認してください
- 未対応バイナリでは Quality スライダーを上げても Live 相当で動作します

### 話者プロファイル作成が失敗する

| エラー | 対処 |
|--------|------|
| `checkpoint not found` | `--codec-checkpoint` / `--token-model` のパスを確認 |
| `no audio files found` | `--audio-dir` に対応形式のファイルがあるか確認 |
| `CUDA out of memory` | `--device cpu` を指定するか、音声を短くする |
| `XPU device error` | CPU フォールバックせず、バッチサイズ削減や再起動で対処 |

### CPU 使用率が高い

- Quality を下げる (q = 0.0)
- バッファサイズを 512 以上にする
- 他のアプリケーションを閉じる
- INT8 量子化モデル（`models/int8/`）がある場合はそちらを使用する

---

## 付録

### A. オフライン変換

リアルタイムではなく、ファイル単位で変換することもできます。

```bash
cargo run -p tmrvc-engine-rs --release --bin offline_convert -- \
  --input input.wav \
  --output output.wav \
  --model-dir models/fp32 \
  --speaker speakers/target_voice.tmrvc_speaker \
  --dry-wet 1.0 \
  --latency-q 1.0
```

| オプション | 説明 |
|-----------|------|
| `--input` | 入力音声ファイル |
| `--output` | 出力先パス |
| `--model-dir` | ONNX モデルのディレクトリ |
| `--speaker` | ボイスプロファイル |
| `--dry-wet` | 原音/変換音の比率 (0.0〜1.0) |
| `--output-gain` | 出力ゲイン |
| `--latency-q` | 品質設定（オフラインでは 1.0 推奨） |

### B. 環境変数

| 変数 | 既定値 | 説明 |
|------|--------|------|
| `TMRVC_MODEL_DIR` | `models/fp32` | ONNX モデルディレクトリ |
| `TMRVC_SPEAKER_PATH` | — | デフォルトのボイスプロファイル |
| `TMRVC_STYLE_PATH` | — | スタイルファイル（抑揚・滑舌の目標値） |
| `TMRVC_ONNX_DIR` | `models/fp32` | GUI 用モデルパス |

### C. スタイルファイル（.tmrvc_style）

ボイスプロファイルとは別に、抑揚（ピッチ）と滑舌の目標値を記録するファイルです。

- β Prosody / γ Articulation スライダーと組み合わせて使用
- ボイスプロファイルなしでも、スタイルファイル単独で使用可能
- tmrvc-rt の GUI またはオフラインツールで参照音声から生成

### D. 用語集

| 用語 | 説明 |
|------|------|
| **ボイスプロファイル** | 変換先の声の特徴を記録した `.tmrvc_speaker` ファイル |
| **Embedding** | 話者の声を 192 次元のベクトルで表現したもの |
| **LoRA Delta** | 適応時に保存される追加パラメータ（任意） |
| **Few-shot** | 少量の音声データ（数秒〜数十秒）でモデルを適応させる手法 |
| **Live モード** | 低遅延（概ね 35〜45ms）の因果的処理。リアルタイム向き |
| **HQ モード** | 品質優先モード（実装依存）。録音向き |
| **Dry/Wet** | 原音（Dry）と変換音（Wet）の混合比率 |
| **α Timbre** | 声色の変換強度 |
| **UCLM Core** | `A_t/B_t` を生成する中核 ONNX モデル |
| **Control Tokens (`B_t`)** | 非言語イベントと制御情報を持つ 4 スロット token |

---

## 開発者向け: 学習パイプライン

モデルの学習・蒸留・TTS 拡張については [docs/training/README.md](training/README.md) を参照してください。
