# TMRVC Training Plan (Merged)

Last updated: 2026-02-25

This file is the single source of truth for training operations.
It merges the former training process dictionary into `training-plan.md`.

## Scope

- VC: Teacher -> Distillation -> ONNX export -> speaker assets
- TTS: frontend training + script/evaluation
- Style: emotion preprocessing + style training + pseudo labeling
- Research: publication track for novelty (SSL + BPEH + LCD)

## Policy

- No backward compatibility before release.
- Legacy checkpoints are rejected.
- No migration guide is maintained for pre-release internals.

## Phase Dictionary

| ID | Domain | Step | Purpose |
|---|---|---|---|
| D00 | Data | Environment setup | Ensure reproducible runtime and GPU visibility |
| D10 | Data | Raw dataset acquisition | Collect corpora for VC/TTS/Style |
| D20 | Data | Feature cache build | Generate `mel/content/f0/spk_embed` |
| D30 | Data | Cache verification | Detect missing/corrupted entries |
| D40 | Data | Text metadata enrichment | Fill `meta.json` with `text/language/language_id` |
| D50 | Data | Forced alignment | Generate `token_ids(or phoneme_ids)/durations` |
| V10 | VC | Teacher training | Train Phase 0/1a/1b/2/reflow |
| V20 | VC | Distillation training | Train Phase A/B/B2/C |
| V30 | VC | ONNX export | Produce deployable runtime models |
| V40 | VC | Speaker/character assets | Create `.tmrvc_speaker` / `.tmrvc_character` |
| T10 | TTS | Frontend training | Train TextEncoder/Duration/F0/ContentSynthesizer |
| T20 | TTS | Frontend evaluation | Compare tokenizer vs phoneme and generate WAVs |
| S10 | Style | Emotion preprocessing | Build `mel.npy` + `emotion.json` cache |
| S20 | Style | StyleEncoder training | Train emotion/VAD/prosody heads |
| S30 | Style | Pseudo labeling | Expand emotion labels to unlabeled datasets |
| R10 | Research | Baseline lock | Freeze baseline checkpoints and eval list |
| R20 | Research | SSL/BPEH implementation | Add Scene State Latent + Breath/Pause Event Head |
| R30 | Research | LCD implementation | Add latency-conditioned distillation and monotonic loss |
| R40 | Research | Ablation suite | Run B0/B1/B2/B3/B4 + ablations |
| R50 | Research | Paper package | Finalize tables, stats, and reproducibility scripts |
| Q10 | QA | E2E smoke | Validate minimal full pipeline |

## Detailed Steps

### D00 Environment setup

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Done when GPU server reports CUDA device availability.

### D10 Raw dataset acquisition

Representative commands:

```bash
uv run python scripts/download_datasets.py --dataset jsut --output-dir data/raw
uv run python scripts/download_datasets.py --dataset ljspeech --output-dir data/raw
# VCTK/JVS/LibriTTS-R/Expresso/JVNV/EmoV-DB/RAVDESS are acquired per each license source.
```

### D20 Feature cache build

```bash
uv run tmrvc-preprocess --dataset vctk --raw-dir data/raw/VCTK-Corpus --cache-dir data/cache --device cuda --skip-existing -v
uv run tmrvc-preprocess --dataset jvs --raw-dir data/raw/jvs_corpus --cache-dir data/cache --device cuda --skip-existing -v
uv run tmrvc-preprocess --dataset libritts_r --raw-dir data/raw/libritts_r --cache-dir data/cache --device cuda --skip-existing -v
uv run tmrvc-preprocess --dataset tsukuyomi --raw-dir data/raw/tsukuyomi --cache-dir data/cache --device cuda --skip-existing -v
```

Output per utterance:

- `mel.npy`
- `content.npy`
- `f0.npy`
- `spk_embed.npy`
- `meta.json`

### D30 Cache verification

```bash
uv run tmrvc-verify-cache --cache-dir data/cache --dataset vctk
```

Done when invalid count is zero.

### D40 Text metadata enrichment

```bash
uv run python scripts/enrich_cache_text_metadata.py \
  --cache-dir data/cache \
  --raw-dir data/raw \
  --datasets jvs vctk tsukuyomi
```

Done when `meta.json` has `text/language/language_id`.

### D50 Forced alignment

```bash
uv run python scripts/run_forced_alignment.py \
  --cache-dir data/cache \
  --dataset jvs \
  --language ja \
  --frontend tokenizer \
  --overwrite
```

Done when each TTS utterance has:

- `token_ids.npy` (or `phoneme_ids.npy`)
- `durations.npy`

### V10 Teacher training

```bash
uv run tmrvc-train-teacher --config configs/train_teacher.yaml --cache-dir data/cache --phase 0 --dataset vctk,jvs --device cuda
uv run tmrvc-train-teacher --config configs/train_teacher.yaml --cache-dir data/cache --phase 1a --dataset vctk,jvs,libritts_r --resume <phase0_ckpt> --device cuda
uv run tmrvc-train-teacher --config configs/train_teacher.yaml --cache-dir data/cache --phase 1b --resume <phase1a_ckpt> --device cuda
uv run tmrvc-train-teacher --config configs/train_teacher.yaml --cache-dir data/cache --phase 2 --resume <phase1b_ckpt> --device cuda
```

Optional reflow:

```bash
uv run python scripts/generate_reflow_pairs.py --teacher-ckpt <teacher_ckpt> --cache-dir data/cache --dataset vctk --output-dir data/reflow_pairs --device cuda
uv run tmrvc-train-teacher --config configs/train_teacher.yaml --cache-dir data/cache --phase reflow --resume <phase2_ckpt> --device cuda
```

### V20 Distillation training

```bash
uv run tmrvc-distill --config configs/train_student.yaml --cache-dir data/cache --dataset vctk,jvs,libritts_r --teacher-ckpt <teacher_ckpt> --phase A --device cuda
uv run tmrvc-distill --config configs/train_student.yaml --cache-dir data/cache --dataset vctk,jvs,libritts_r --teacher-ckpt <teacher_ckpt> --phase B --resume <phaseA_ckpt> --device cuda
uv run tmrvc-distill --config configs/train_student.yaml --cache-dir data/cache --dataset vctk,jvs,libritts_r --teacher-ckpt <teacher_ckpt> --phase B2 --resume <phaseB_ckpt> --device cuda
uv run tmrvc-distill --config configs/train_student.yaml --cache-dir data/cache --dataset vctk,jvs,libritts_r --teacher-ckpt <teacher_ckpt> --phase C --resume <phaseB2_ckpt> --device cuda
```

### V30 ONNX export

```bash
uv run tmrvc-export --checkpoint <distill_ckpt> --output-dir models --verify
```

Done when `models/fp32/*.onnx` exists and parity verify passes.

### V40 Speaker/character assets

From reference audio:

```bash
uv run python scripts/generate_speaker_file.py --audio ref1.wav ref2.wav --name actor --output models/actor.tmrvc_speaker
```

Few-shot adaptation:

```bash
uv run tmrvc-finetune --audio-dir data/sample_voice/target --checkpoint <distill_ckpt> --output models/target.tmrvc_speaker --device cuda
```

Character creation:

```bash
uv run tmrvc-create-character models/target.tmrvc_speaker -o models/target.tmrvc_character --name target --language ja
```

### T10 TTS frontend training

```bash
uv run tmrvc-train-tts \
  --cache-dir data/cache \
  --dataset jvs,tsukuyomi \
  --text-frontend tokenizer \
  --device cuda \
  --max-steps 200000 \
  --checkpoint-dir checkpoints/tts
```

Done when `checkpoints/tts/tts_step*.pt` is produced and losses stabilize.

### T20 TTS evaluation

Frontend A/B evaluation:

```bash
uv run python scripts/evaluate_tts_frontends.py examples/batch_script_generation/yuri_conversation.yaml \
  --tts-checkpoint <tts_ckpt> \
  --vc-checkpoint <vc_ckpt> \
  --device cuda \
  --frontends tokenizer phoneme \
  --create-ab \
  --output-dir eval/quick_check_ab
```

Script batch WAV generation:

```bash
uv run python examples/batch_script_generation/generate.py \
  --input examples/batch_script_generation/yuri_conversation.yaml \
  --output-dir outputs/yuri_wav \
  --tts-checkpoint <tts_ckpt> \
  --vc-checkpoint <vc_ckpt> \
  --device cuda
```

### S10 Emotion preprocessing

```bash
uv run python scripts/preprocess_emotion.py --dataset expresso --raw-dir data/raw/expresso --cache-dir data/cache -v
uv run python scripts/preprocess_emotion.py --dataset jvnv --raw-dir data/raw/jvnv --cache-dir data/cache -v
uv run python scripts/preprocess_emotion.py --dataset emov_db --raw-dir data/raw/EmoV-DB --cache-dir data/cache -v
uv run python scripts/preprocess_emotion.py --dataset ravdess --raw-dir data/raw/ravdess --cache-dir data/cache -v
```

### S20 Style training

```bash
uv run tmrvc-train-style --cache-dir data/cache --dataset expresso,jvnv,emov_db,ravdess --device cuda --max-steps 50000
```

### S30 Pseudo labeling

Train classifier:

```bash
uv run python scripts/apply_pseudo_labels.py train \
  --cache-dir data/cache \
  --datasets expresso,jvnv,emov_db,ravdess \
  --output checkpoints/emotion_cls.pt \
  --device cuda
```

Apply labels:

```bash
uv run python scripts/apply_pseudo_labels.py label \
  --cache-dir data/cache \
  --classifier checkpoints/emotion_cls.pt \
  --datasets vctk,jvs \
  --confidence 0.8 \
  --device cuda
```

## Execution Order (Production)

1. D00 -> D10 -> D20 -> D30
2. D40 -> D50
3. V10 -> V20 -> V30
4. V40
5. T10 -> T20
6. S10 -> S20 -> S30
7. Q10 smoke and report

## Execution Order (Research Novelty Track)

1. R10 baseline lock
2. R20 SSL/BPEH implementation and training
3. R30 LCD implementation and training
4. R40 ablations and statistical validation
5. R50 paper package freeze

Reference design:

- `docs/research/research-novelty-plan.md`

## Artifact Dictionary

| Artifact | Produced in | Used by |
|---|---|---|
| `data/cache/<dataset>/.../mel.npy` | D20 | VC/TTS/Style training |
| `data/cache/<dataset>/.../content.npy` | D20 | VC/TTS training |
| `data/cache/<dataset>/.../f0.npy` | D20 | VC/TTS training |
| `data/cache/<dataset>/.../spk_embed.npy` | D20 | VC/TTS training |
| `data/cache/<dataset>/.../meta.json` | D20/D40 | D50/TTS training |
| `token_ids.npy` / `phoneme_ids.npy` | D50 | TTS training |
| `durations.npy` | D50 | TTS training |
| `checkpoints/teacher_step*.pt` | V10 | V20 |
| `checkpoints/distill/distill_step*.pt` | V20 | V30, TTS backend |
| `checkpoints/distill/*.voice_source_stats.json` | V20 | voice source preset analysis |
| `checkpoints/tts/tts_step*.pt` | T10 | T20, serve |
| `checkpoints/style/style_step*.pt` | S20 | style inference |
| `models/fp32/*.onnx` | V30 | Rust engine / VST / RT |
| `*.tmrvc_speaker` | V40 | serve/gui |
| `*.tmrvc_character` | V40 | script/gui |
| `eval/*/summary.json` | T20 | evaluation report |

## Dataset Specification

### Datasets per Phase

| Phase | Datasets | Role | Hours | Lang |
|---|---|---|---|---|
| V10 Phase 0 | vctk, jvs | Architecture verification | ~74h | en, ja |
| V10 Phase 1a+ | vctk, jvs, libritts_r | Full VC training | ~659h | en, ja |
| V20 | vctk, jvs, libritts_r | Distillation | ~659h | en, ja |
| T10 | jsut, ljspeech, vctk, jvs | TTS frontend | ~108h | en, ja |
| S10/S20 | expresso, jvnv, emov_db, ravdess | Emotion/style | ~53h | en, ja |
| S30 | vctk, jvs (target) | Pseudo labeling | — | en, ja |

### Built-in Dataset Adapters

| Adapter name | Type | Lang | Speakers | Hours | Raw directory layout |
|---|---|---|---|---|---|
| `vctk` | VC | en | 109 | 44 | `VCTK-Corpus/wav48_silence_trimmed/{speaker}/{utt}_mic1.flac` |
| `jvs` | VC | ja | 100 | 30 | `jvs_corpus/jvs_ver1/{speaker}/{subset}/wav24kHz16bit/{utt}.wav` |
| `libritts_r` | VC | en | 2456 | 585 | `libritts_r/train-clean-*/.../{speaker}/{chapter}/{utt}.wav` |
| `tsukuyomi` | VC/custom | any | any | any | Flat or nested: `root/**/*.{wav,flac,ogg}` |
| `expresso` | emotion | en | 4 | 40 | `expresso/{read,improvised}/{speaker}_{style}_{id}.wav` |
| `jvnv` | emotion | ja | 4 | 4 | `jvnv/jvnv_ver1/{speaker}/{emotion}/{utt}.wav` |
| `emov_db` | emotion | en | 4 | 7 | `EmoV-DB/{speaker}/{emotion}/{utt}.wav` |
| `ravdess` | emotion | en | 24 | 2 | `ravdess/Actor_{nn}/{coded_filename}.wav` |

### Supported Emotion Categories (12-class)

| ID | Name | Example datasets |
|---|---|---|
| 0 | happy | expresso, jvnv, ravdess |
| 1 | sad | expresso, jvnv, ravdess |
| 2 | angry | expresso, jvnv, emov_db, ravdess |
| 3 | fearful | jvnv, ravdess |
| 4 | surprised | jvnv, ravdess |
| 5 | disgusted | jvnv, emov_db, ravdess |
| 6 | neutral | all |
| 7 | bored | — |
| 8 | excited | expresso |
| 9 | tender | — |
| 10 | sarcastic | — |
| 11 | whisper | expresso |

## Custom Dataset Preparation

### VC / TTS 用カスタムデータ

`tsukuyomi` アダプタが汎用ローダーとして動作する。
任意のディレクトリ構造の音声ファイルを受け付ける。

#### 必要なディレクトリ構造

```
data/raw/my_dataset/
├── speaker_a/          # フォルダ名 = 話者ID
│   ├── line_001.wav
│   ├── line_002.wav
│   └── ...
├── speaker_b/
│   └── ...
└── transcripts.txt     # TTS用 (任意)
```

単一話者の場合はフラット構造でもよい:

```
data/raw/my_dataset/
├── 001.wav
├── 002.wav
└── ...
```

#### 音声ファイル要件

| 項目 | 要件 |
|---|---|
| フォーマット | WAV, FLAC, OGG (16bit推奨) |
| サンプルレート | 任意 (内部で 24kHz にリサンプル) |
| チャンネル | モノラル推奨 (ステレオは左チャンネル使用) |
| 発話長 | 2-15秒 推奨 (長い場合は自動セグメント) |
| 品質 | 残響・ノイズが少ないほど良い |

#### 前処理コマンド

```bash
uv run tmrvc-preprocess \
  --dataset tsukuyomi \
  --raw-dir data/raw/my_dataset \
  --cache-dir data/cache \
  --content-teacher wavlm \
  --device cuda \
  --skip-existing -v
```

キャッシュ上のデータセット名は `tsukuyomi` になる。
複数のカスタムデータセットを使う場合は `--cache-dir` を分けるか、
`dataset_adapters.py` の `ADAPTERS` dict にエントリを追加する。

#### TTS 用テキスト付与

カスタムデータに TTS 学習用テキストを付与するには、
各発話の `meta.json` に `text` と `language_id` を追加する。

方法 1: `inject_text_to_cache.py` を拡張

方法 2: 手動スクリプト

```python
import json
from pathlib import Path

transcripts = {}  # {utterance_id: text}
with open("data/raw/my_dataset/transcripts.txt") as f:
    for line in f:
        utt_id, text = line.strip().split("|", 1)
        transcripts[utt_id] = text

cache_root = Path("data/cache/tsukuyomi/train")
for spk_dir in cache_root.iterdir():
    for utt_dir in spk_dir.iterdir():
        meta_path = utt_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        utt_id = meta["utterance_id"]
        if utt_id in transcripts:
            meta["text"] = transcripts[utt_id]
            meta["language_id"] = 0  # 0=ja, 1=en
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
```

テキスト付与後に forced alignment を実行:

```bash
uv run python scripts/run_forced_alignment.py \
  --cache-dir data/cache --dataset tsukuyomi --language ja --frontend tokenizer
```

### 演劇・ゲーム台本音声の準備

演技音声（エロゲ収録音声、ドラマCD等）をスタイル学習に使う場合。

#### ディレクトリ構造

```
data/raw/drama_voice/
├── actor_sakura/
│   ├── happy/
│   │   ├── sakura_happy_001.wav
│   │   └── ...
│   ├── angry/
│   │   └── ...
│   ├── sad/
│   │   └── ...
│   ├── whisper/
│   │   └── ...
│   └── neutral/
│       └── ...
└── actor_yuki/
    └── (同構造)
```

サブフォルダ名が感情ラベルとして使われる。
JVNV 形式に合わせるとそのまま `preprocess_emotion.py` が使える。

#### 感情ラベルなしの場合

フォルダ分けせず全部 `neutral/` に入れて VC/TTS 学習に使い、
S30 の擬似ラベリングで自動分類する方法もある。

```
data/raw/drama_voice/
└── actor_sakura/
    └── neutral/
        ├── line_001.wav   # 実際は怒りだが、ラベルなし
        ├── line_002.wav
        └── ...
```

→ S30 で `--confidence 0.8` で自動分類

#### emotion.json フォーマット (手動作成する場合)

```json
{
  "emotion_id": 2,
  "emotion": "angry",
  "vad": [0.1, 0.8, 0.7],
  "prosody": [0.6, 0.8, 0.7]
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `emotion_id` | int | 0-11 (上記 12-class 表を参照) |
| `emotion` | str | カテゴリ名 |
| `vad` | [float, float, float] | Valence, Arousal, Dominance (各 -1〜1) |
| `prosody` | [float, float, float] | 話速, エネルギー, ピッチレンジ (各 0〜1) |

VAD 参考値:

| 感情 | Valence | Arousal | Dominance |
|---|---|---|---|
| happy | 0.7 | 0.5 | 0.3 |
| sad | -0.7 | -0.3 | -0.5 |
| angry | -0.3 | 0.8 | 0.7 |
| fearful | -0.5 | 0.6 | -0.6 |
| neutral | 0.0 | 0.0 | 0.0 |
| whisper | 0.0 | -0.6 | -0.3 |
| excited | 0.6 | 0.8 | 0.4 |
| tender | 0.5 | -0.2 | 0.1 |

### 大量 WAV ファイルの取り込みワークフロー

数万ファイル・100GB 規模の音声データを効率よくパイプラインに乗せる手順。

#### Step 1: スキャン・レポート (データの全体像を把握)

```bash
python scripts/prepare_bulk_voice.py \
  --input /path/to/eroge_voice_collection \
  --report -v
```

出力例:
```
Speaker              Total     Kept     Skip    Hours
------------------------------------------------------------
sakura                4200     3100     1100      3.2
yuki                  3800     2900      900      2.8
miko                  5100     3800     1300      4.1
...
------------------------------------------------------------
TOTAL                38000    28000    10000     32.5

Skip reasons:
  too_short                    8200    (< 1.0s: 相槌・SE)
  near_silence                 1500    (rms < 0.005)
  too_long                      300    (> 30s: BGM混入等)
```

→ `_bulk_report.json` にレポートが保存される。
話者の重複・極端に少ない話者の確認に使う。

#### Step 2: フィルタリング + コピー

```bash
# WAV のままコピー (ローカル作業)
python scripts/prepare_bulk_voice.py \
  --input /path/to/eroge_voice_collection \
  --output data/raw/eroge_clean

# FLAC 圧縮 (サーバー転送用、WAV → FLAC で ~50% 削減)
python scripts/prepare_bulk_voice.py \
  --input /path/to/eroge_voice_collection \
  --output data/raw/eroge_clean \
  --flac
```

フィルタ閾値の調整:
```bash
# 相槌 (「うん」「ああ」) も含めたい場合
--min-duration 0.5

# 長いモノローグも含めたい場合
--max-duration 60
```

#### Step 3: (任意) Whisper 自動書き起こし

テキストなしデータに TTS 学習用テキストを付与する。

```bash
python scripts/prepare_bulk_voice.py \
  --input /path/to/eroge_voice_collection \
  --output data/raw/eroge_clean \
  --transcribe \
  --whisper-model large-v3 \
  --language ja \
  --device cuda
```

各話者フォルダに `transcripts.txt` が生成される:
```
ev001_a_01|ねぇ、もっとこっち来て……
ev001_a_03|えっ……そんな急に……っ
ev001_a_05|嫌いになったりしないから……
```

**注意:** Whisper は喘ぎ声や非言語発声を hallucinate しやすい。
結果の手動確認を推奨。

#### Step 4: 前処理 (tmrvc-preprocess)

```bash
uv run tmrvc-preprocess \
  --dataset tsukuyomi \
  --raw-dir data/raw/eroge_clean \
  --cache-dir data/cache \
  --content-teacher wavlm \
  --device cuda \
  --skip-existing -v
```

#### Step 5: (任意) テキスト注入

Whisper の書き起こしをキャッシュに注入:

```python
import json
from pathlib import Path

cache_root = Path("data/cache/tsukuyomi/train")
for spk_dir in cache_root.iterdir():
    transcript_path = Path("data/raw/eroge_clean") / spk_dir.name / "transcripts.txt"
    if not transcript_path.exists():
        continue
    # Load transcripts
    transcripts = {}
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if "|" not in line:
            continue
        stem, text = line.split("|", 1)
        transcripts[stem.strip()] = text.strip()

    for utt_dir in spk_dir.iterdir():
        meta_path = utt_dir / "meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # tsukuyomi adapter prefixes: tsukuyomi_{stem}
        raw_stem = meta["utterance_id"].removeprefix("tsukuyomi_")
        if raw_stem in transcripts:
            meta["text"] = transcripts[raw_stem]
            meta["language_id"] = 0  # ja
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
```

#### Step 6: サーバーへの転送

```bash
# FLAC 圧縮済みなら ~50GB
rsync -avP data/raw/eroge_clean/ server:TMRVC/data/raw/eroge_clean/

# または tar で一発
tar cf - data/raw/eroge_clean/ | ssh server "cd TMRVC && tar xf -"
```

#### 話者の整理に関する注意

| 問題 | 対処 |
|---|---|
| 同名キャラ・別声優 | フォルダ名を `sakura_game1`, `sakura_game2` に分ける |
| 別名キャラ・同声優 | speaker embedding の類似度で自動検出 → 同一話者IDに統合 |
| ファイル数が極端に少ない話者 (< 50) | VC 学習には十分だが TTS/Style には足りない。他の話者と混ぜるか除外 |
| SE・BGM 混入 | `prepare_bulk_voice.py` の RMS フィルタで大部分除去 |

### R18 演劇音声の準備

エロゲ収録音声・ASMR・ドラマCD 等のアダルトコンテンツ向け音声データ。
通常の感情カテゴリでは表現しきれない声質パターンが多い。

#### 既存 12 クラスとの対応

| R18 での呼称 | 最も近い既存カテゴリ | VAD | prosody | 備考 |
|---|---|---|---|---|
| 普通の台詞 | neutral | 0, 0, 0 | 0.5, 0.5, 0.5 | そのまま |
| 怒り | angry | -0.3, 0.8, 0.7 | 0.6, 0.8, 0.7 | そのまま |
| 泣き | sad | -0.7, 0.3, -0.5 | 0.3, 0.4, 0.4 | arousal 高め (嗚咽) |
| 喜び | happy | 0.7, 0.5, 0.3 | 0.6, 0.6, 0.7 | そのまま |
| 囁き | whisper | 0.2, -0.5, -0.3 | 0.3, 0.1, 0.2 | そのまま |
| 吐息混じり | whisper 亜種 | 0.3, 0.2, -0.2 | 0.3, 0.2, 0.3 | arousal を whisper より高め |
| 照れ | tender 亜種 | 0.4, 0.3, -0.4 | 0.4, 0.3, 0.4 | dominance 低め |
| 挑発 | sarcastic 亜種 | 0.2, 0.5, 0.6 | 0.5, 0.6, 0.6 | dominance 高め |
| 恍惚 | excited 亜種 | 0.6, 0.7, -0.3 | 0.4, 0.5, 0.8 | dominance 低め、pitch 高め |
| 喘ぎ (弱) | excited 亜種 | 0.3, 0.5, -0.5 | 0.3, 0.3, 0.6 | 既存カテゴリで近似 |
| 喘ぎ (強) | excited 亜種 | 0.4, 0.9, -0.6 | 0.2, 0.7, 0.9 | arousal 最大域 |
| 絶頂 | (該当なし) | 0.5, 1.0, -0.8 | 0.1, 1.0, 1.0 | 破綻覚悟の極端値 |

**設計方針:** カテゴリを増やすのではなく VAD + prosody の連続値で表現する。
12 クラスのカテゴリはあくまで学習の足場。推論時は `StyleParams` の連続値を直接指定して、
カテゴリ間の中間表現（「照れながら怒る」等）を自然に生成する。

#### ディレクトリ構造 (推奨)

感情フォルダ名を既存カテゴリに寄せる:

```
data/raw/eroge_voice/
├── actor_sakura/
│   ├── neutral/          → emotion_id=6
│   │   ├── sakura_neutral_001.wav
│   │   └── ...
│   ├── happy/            → emotion_id=0
│   ├── sad/              → emotion_id=1
│   ├── angry/            → emotion_id=2
│   ├── whisper/          → emotion_id=11
│   ├── tender/           → emotion_id=9  (照れ・吐息混じりはここ)
│   ├── excited/          → emotion_id=8  (恍惚・喘ぎ系はここ)
│   └── sarcastic/        → emotion_id=10 (挑発・S系はここ)
└── actor_yuki/
    └── (同構造)
```

`preprocess_emotion.py` の JVNV パーサーがそのまま使えるように、
フォルダ名を上記の英語カテゴリ名に合わせる。

#### 感情ラベルが付いていない場合

収録音声にラベルがない場合、2段階で対処:

1. **まず全部 VC/TTS データとして取り込む** (感情ラベル不要)

```bash
uv run tmrvc-preprocess --dataset tsukuyomi \
  --raw-dir data/raw/eroge_voice --cache-dir data/cache \
  --device cuda --skip-existing -v
```

2. **S30 擬似ラベリングで自動分類**

```bash
# 学習済み感情分類器で自動ラベル付け
uv run python scripts/apply_pseudo_labels.py label \
  --cache-dir data/cache \
  --classifier checkpoints/emotion_cls.pt \
  --datasets tsukuyomi \
  --confidence 0.8 \
  --device cuda
```

confidence < 0.8 の発話はスキップされる。
R18 特有の声質は既存分類器の精度が低い可能性があるため、
手動で `emotion.json` を修正するか confidence 閾値を下げる。

#### emotion.json の手動作成 (高精度)

台本のト書きから感情を抽出してラベリングする場合:

```python
# 台本から emotion.json を一括生成するスクリプト例
import json
from pathlib import Path

# ト書き → StyleParams マッピング (プロジェクト固有)
DIRECTION_MAP = {
    "普通に": {"emotion": "neutral", "vad": [0, 0, 0]},
    "怒って": {"emotion": "angry", "vad": [-0.3, 0.8, 0.7]},
    "囁いて": {"emotion": "whisper", "vad": [0.2, -0.5, -0.3]},
    "照れながら": {"emotion": "tender", "vad": [0.4, 0.3, -0.4]},
    "息を荒くして": {"emotion": "excited", "vad": [0.3, 0.7, -0.5]},
    "泣きながら": {"emotion": "sad", "vad": [-0.7, 0.3, -0.5]},
    "挑発的に": {"emotion": "sarcastic", "vad": [0.2, 0.5, 0.6]},
}

# 台本ファイル: "ファイル名|ト書き|テキスト" 形式
with open("script_directions.txt", encoding="utf-8") as f:
    for line in f:
        fname, direction, text = line.strip().split("|", 2)
        style = DIRECTION_MAP.get(direction, DIRECTION_MAP["普通に"])
        # cache の対応ディレクトリに emotion.json を書き出す
        # ...
```

#### 台本ファイルとの統合 (T20)

学習済みモデルでエロゲ台本を読ませるには Script 形式 YAML を作る:

```yaml
title: "Scene 1"
situation: "二人きりの部屋で"
characters:
  sakura:
    personality: "甘えん坊で感情的"
    voice_description: "高めの声、やや息混じり"
    speaker_file: models/sakura.tmrvc_speaker
    language: ja
dialogue:
  - speaker: sakura
    text: "ねぇ、もっとこっち来て……"
    hint: "tender"
    speed: 0.85
  - speaker: sakura
    text: "えっ……そんな急に……っ"
    hint: "surprised"
    speed: 0.9
```

`hint` に感情カテゴリ名を書くと StyleParams に変換される。
ContextStylePredictor (Phase 4) が有効なら hint なしでも
situation + 会話履歴から自動推定する。

### カスタムアダプタ追加

`tsukuyomi` アダプタでは対応できないディレクトリ構造の場合:

```python
# tmrvc-data/src/tmrvc_data/dataset_adapters.py に追加

class DramaAdapter(DatasetAdapter):
    name = "drama"

    def iter_utterances(self, root: Path, split: str = "train") -> Iterator[Utterance]:
        for actor_dir in sorted(root.iterdir()):
            if not actor_dir.is_dir():
                continue
            for wav in sorted(actor_dir.rglob("*.wav")):
                yield Utterance(
                    utterance_id=f"drama_{wav.stem}",
                    speaker_id=f"drama_{actor_dir.name}",
                    dataset="drama",
                    audio_path=wav,
                    language="ja",
                )

ADAPTERS["drama"] = DramaAdapter
```

`tmrvc-preprocess --dataset drama --raw-dir data/raw/drama_voice ...` で使用可能になる。

## Practical Notes

- `scripts/run_pipeline.py` is for small smoke checks, not production-scale training.
- `tmrvc-preprocess` supported datasets are currently `vctk`, `jvs`, `libritts_r`, `tsukuyomi`.
- `zh/ko` frontend and alignment args are available; production quality requires dedicated training corpora.
- The `tsukuyomi` adapter is the recommended catch-all for custom audio directories.
