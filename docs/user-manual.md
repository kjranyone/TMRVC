# TMRVC ユーザーマニュアル

TMRVC は CPU のみでリアルタイム Voice Conversion を実現するソフトウェアです。
自分の声をリアルタイムで別の話者の声に変換できます。

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
│       ├── content_encoder.onnx         # 必須
│       ├── converter.onnx               # 必須
│       ├── converter_hq.onnx            # 任意（HQ モード用）
│       ├── vocoder.onnx                 # 必須
│       ├── ir_estimator.onnx            # 必須
│       └── speaker_encoder.onnx         # 任意（簡易プロファイル作成用）
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
uv run tmrvc-finetune \
  --audio-dir ./reference_audio/ \
  --checkpoint checkpoints/distill/best.pt \
  --output speakers/target_voice.tmrvc_speaker \
  --steps 0 \
  --profile-name "ターゲットボイス" \
  --author-name "作者名"
```

> `--steps 0` を指定すると Embedding のみで LoRA Fine-tuning をスキップします。

### 3.3 高品質モード（Few-shot Fine-tuning）

話者の特徴ベクトルに加え、Converter 内部の LoRA 重み（15,872 パラメータ）を
少量の音声データで最適化します。200 ステップ程度の学習で声質の再現度が大きく向上します。

**CLI から:**

```bash
uv run tmrvc-finetune \
  --audio-dir ./reference_audio/ \
  --checkpoint checkpoints/distill/best.pt \
  --output speakers/target_voice.tmrvc_speaker \
  --profile-name "ターゲットボイス" \
  --author-name "作者名" \
  --device auto
```

**主要オプション:**

| オプション | 既定値 | 説明 |
|-----------|--------|------|
| `--audio-dir` | — | 音声ファイルのディレクトリ（`--audio-files` と排他） |
| `--audio-files` | — | 音声ファイルを直接指定（スペース区切り） |
| `--checkpoint` | — | 蒸留チェックポイントのパス（必須） |
| `--output` | — | 出力 `.tmrvc_speaker` のパス（必須） |
| `--steps` | 200 | Fine-tuning のステップ数 |
| `--lr` | 1e-3 | 学習率 |
| `--device` | auto | 使用デバイス（auto / xpu / cuda / cpu） |
| `--profile-name` | ファイル名 | プロファイルの表示名 |
| `--author-name` | — | 作者名（メタデータに記録） |
| `--licence-url` | — | ライセンス URL |
| `--config` | — | 追加設定の YAML ファイル |

**出力例:**

```
[INFO] Loading checkpoint from checkpoints/distill/best.pt
[INFO] Extracting speaker embedding from 5 audio files...
[INFO] Speaker embedding: [192] (L2-normalized)
[INFO] Preparing fine-tuning data...
[INFO] Starting LoRA fine-tuning (200 steps, lr=1e-3)
[INFO] Step  50/200: loss=0.0842
[INFO] Step 100/200: loss=0.0634
[INFO] Step 150/200: loss=0.0571
[INFO] Step 200/200: loss=0.0548
[INFO] Saved speaker profile to speakers/target_voice.tmrvc_speaker
[INFO] File size: 64,856 bytes
```

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

> Fine-tune モードは内部で `tmrvc-finetune` CLI を呼び出します。
> Python 環境と蒸留チェックポイントが必要です。

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
   - 5 つのモデルが順に読み込まれます

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
│  レイテンシ ~20ms │  レイテンシ ~80ms  │
│  品質: 標準      │  品質: 高い       │
```

| モード | q の範囲 | レイテンシ | 動作 |
|--------|---------|-----------|------|
| **Live** | 0.0〜0.3 | 約 20 ms | 現在のフレームのみで変換（因果的） |
| **HQ** | 0.3〜1.0 | 約 80 ms | 7 フレーム先読みで変換（半因果的） |

**HQ モードの前提条件:**
- `converter_hq.onnx` が `models/fp32/` に存在すること
- 存在しない場合は q の値に関わらず Live モードで動作します

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
| Live (q ≤ 0.3) | 20 ms 相当 (例: 960 samples @48kHz) |
| HQ (q > 0.3) | 80 ms 相当 (例: 3,840 samples @48kHz) |

---

## 6. ボイスプロファイルの管理

### ファイル形式

`.tmrvc_speaker` は独自のバイナリ形式（v2）です。

```
┌──────────────────────────────────────────┐
│ ヘッダ (24 bytes)                         │
│  Magic: "TMSP"   Version: 2              │
├──────────────────────────────────────────┤
│ 話者 Embedding (768 bytes)               │
│  192 次元 × float32                      │
├──────────────────────────────────────────┤
│ LoRA Delta (63,488 bytes)                │
│  15,872 パラメータ × float32              │
│  （簡易モードでは全て 0）                  │
├──────────────────────────────────────────┤
│ メタデータ (可変長, JSON)                  │
│  プロファイル名, 作者名, サムネイル等       │
├──────────────────────────────────────────┤
│ SHA-256 チェックサム (32 bytes)            │
└──────────────────────────────────────────┘
```

**ファイルサイズ:** 約 64 KB（メタデータの量で多少変動）

### メタデータの内容

| フィールド | 説明 |
|-----------|------|
| `profile_name` | プロファイルの表示名 |
| `author_name` | 作者名 |
| `co_author_name` | 共同作者名（任意） |
| `licence_url` | ライセンス URL（任意） |
| `created_at` | 作成日時 |
| `training_mode` | `"embedding"` または `"finetune"` |
| `source_audio_files` | 使用した音声ファイル名のリスト |
| `thumbnail_b64` | サムネイル画像（mel スペクトログラムの PNG, base64） |

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

- `models/fp32/converter_hq.onnx` が存在するか確認してください
- ファイルがない場合、Quality スライダーに関わらず Live モードで動作します

### Fine-tuning が失敗する

| エラー | 対処 |
|--------|------|
| `checkpoint not found` | `--checkpoint` のパスを確認 |
| `no audio files found` | `--audio-dir` に対応形式のファイルがあるか確認 |
| `CUDA out of memory` | `--device cpu` を指定するか、音声を短くする |
| `XPU fallback warning` | 正常動作。一部演算が CPU にフォールバックしているだけ |

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
| **LoRA Delta** | Converter モデルの微調整パラメータ（15,872 個） |
| **Few-shot** | 少量の音声データ（数秒〜数十秒）でモデルを適応させる手法 |
| **Live モード** | 低遅延（20 ms）の因果的処理。リアルタイム向き |
| **HQ モード** | 高品質（80 ms 遅延）の半因果的処理。録音向き |
| **Dry/Wet** | 原音（Dry）と変換音（Wet）の混合比率 |
| **α Timbre** | 声色の変換強度 |
| **Converter** | 話者の声質を変換する ONNX モデル |
| **IR Estimator** | 音響環境（残響等）を推定する ONNX モデル |

---

## 開発者向け: 学習パイプライン

モデルの学習・蒸留・TTS 拡張については [docs/training/README.md](training/README.md) を参照してください。
