# TMRVC v4 Master Plan

この文書は、TMRVC を `v3` から段階移行せず、`v4` として一度に切り替えるための正本計画である。
本計画は、既存 `v3` の 8-D `voice_state` 契約との後方互換を前提にしない。
また、学習データの入口を「話者分離済み・転写済みデータセット」ではなく、
「話者未分離・転写未整備でもよい大量音声ファイル群」に置く。

## 1. 目的

`v4` の目的は次の 2 つを同時に満たすことである。

- Fish Audio S2 級の open-ended な表現自由度に近づくこと
- TMRVC の強みである編集可能な physical-first control を維持すること

`v4` は、自然言語指示のみで表現を寄せるシステムではない。
`v4` は、`LLM-driven intent control` と `editable physical control` を両立する
acting-centric architecture である。

## 2. 非目標

- `v3` checkpoint の互換利用
- `v3` の API / GUI / ONNX 契約の延命
- Whisper + LLM のみを教師信号として物理制御を成立させること
- 話者未分離の raw audio を直接 train loader に流すこと

## 3. v4 の基本方針

### 3.1 Single-Cutover

- `v4` は別系統のメジャーアーキテクチャとする
- `v3` 契約を保つための暫定 compatibility layer は作らない
- Core / Data / Train / Export / Serve / Rust / GUI を同一 release 単位で切り替える

### 3.2 Data-First

- 入口は raw corpus とする
- 学習の前に必ず bootstrap / curation / annotation を通す
- train-ready cache は raw corpus から再生成する

### 3.3 Physical + Latent Hybrid

- 演技表現は `explicit physical controls` と `acting texture latent` に分ける
- physical 側は解釈可能で編集可能でなければならない
- latent 側は physical で説明しきれない残差を吸収する
- latent を physical slider と同じ public control surface に混ぜない

### 3.4 Causal-First

- 10 ms causal core を維持する
- 非因果な smoothing や future-dependent post-filter を mainline に入れない
- biological constraints は causal prior と遷移正則化として実装する

## 4. 外部比較の位置づけ

Fish Audio S2 は `v4` の競合比較対象として残す。
正本の baseline freeze は [external-baseline-registry.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/external-baseline-registry.md) に従う。
ただし `v4` の設計方針は Fish の完全模倣ではない。

`v4` が取り入れるもの:

- 大規模 raw audio corpus 運用
- 自然言語ベースの acting / instruction interface
- broad expressivity を志向した高次表現制御

`v4` が維持するもの:

- physical parameter の解釈可能性
- 明示編集可能な control surface
- deterministic trajectory editing

## 5. v4 全体像

`v4` は以下の 6 本柱で構成する。

1. Survey
2. Data Bootstrap
3. Model and Training
4. Runtime and Export
5. GUI and User Control
6. Evaluation and Release Gates

以後の各節は、この順序に沿って実装の正本仕様を定める。

## 6. Survey

### 6.1 目的

- Fish Audio S2 を含む主要 competitor の入力形式、制御面、few-shot 条件、streaming 性能を比較する
- `v4` の public claim に必要な head-to-head axis を固定する
- benchmark protocol を学習設計と切り離さずに先に定義する

### 6.2 成果物

- competitor summary
- prompt/control taxonomy
- evaluation subset mapping
- `Fish S2 に勝つ` という文言を許容する条件

Survey の詳細比較対象と freeze 情報は
[external-baseline-registry.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/external-baseline-registry.md)
および
[evaluation-protocol.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/evaluation-protocol.md)
に従う。

## 7. Data Bootstrap

### 7.1 出発点

入力は、話者分離されていない大量の音声ファイル群である。
転写や整備済み metadata を前提としない。

raw corpus の例:

```text
data/raw_corpus/<corpus_id>/**/*.wav
data/raw_corpus/<corpus_id>/**/*.flac
data/raw_corpus/<corpus_id>/**/*.mp3
```

### 7.2 Bootstrap Pipeline

raw corpus から train-ready utterance cache を作る標準パイプラインは次の通りとする。

1. ingest
2. audio normalization
3. VAD segmentation
4. overlap / music / noise rejection
5. diarization or speaker clustering
6. pseudo speaker assignment
7. speaker embedding extraction
8. Whisper transcription
9. text normalization and G2P
10. DSP / SSL physical feature extraction
11. LLM semantic / acting annotation
12. confidence scoring and artifact masking
13. train-ready cache export

### 7.3 Bootstrap の原則

- raw corpus を直接 train loader に流してはならない
- `speaker_id` は pseudo でもよいが、utterance 単位で安定していなければならない
- diarization / clustering が低信頼なセグメントは学習から除外または低重み化する
- overlap speaker 区間は原則除外する
- BGM が大きい区間は speaker embedding と physical extraction の両方を汚染するため除外対象とする

### 7.4 train-ready cache の最小契約

`v4` の train-ready cache は、既存 `v3` と別契約である。
最低限、以下を含む。

- `acoustic tokens`
- `control tokens`
- `pseudo speaker_id`
- `speaker_embed`
- `text transcript`
- `phoneme_ids`
- `language metadata`
- `physical control targets`
- `acting semantic annotations`
- `quality/confidence metadata`

cache 詳細は
[dataset-preparation-flow.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/dataset-preparation-flow.md)
と整合する別紙 `v4 dataset contract` で固定する。

### 7.5 Supervision Tier

`v4` では全サンプルを同じ品質とみなさない。
各 utterance は次のいずれかに分類される。

- Tier A: speaker / transcript / physical labels / semantic labels が高信頼
- Tier B: transcript と speaker は高信頼、physical または semantic の一部が pseudo-label
- Tier C: transcript と基本 speaker anchor はあるが、physical supervision が疎
- Tier D: reference-only または auxiliary-only。mainline loss の一部にのみ使う

### 7.6 LLM と DSP/SSL の役割分担

Whisper + LLM は以下を担当する。

- transcript
- punctuation recovery
- scene summary
- dialogue intent
- emotion description
- acting hint

DSP / SSL / audio-derived estimator は以下を担当する。

- physical voice control targets
- confidence
- observed mask
- speaker timbre anchor

Whisper + LLM のみで physical supervision を置換してはならない。

## 8. Model and Training

### 8.1 v4 の conditioning 分解

`v4` の acting conditioning は少なくとも次の 4 系統に分ける。

1. `speaker identity`
2. `explicit physical controls`
3. `acting texture latent`
4. `dialogue / semantic intent`

この分解を保ったまま causal core に投影する。

### 8.2 Physical Controls

`v4` の explicit physical control は 8-D 固定ではなく拡張する。
初期提案は `12-D` から `16-D` の範囲とする。

候補例:

- pitch level
- pitch range
- energy level
- pressedness
- spectral tilt
- breathiness
- voice irregularity
- openness
- aperiodicity
- formant shift
- vocal effort
- creak or subharmonicity

設計上の注意:

- `HNR` は既存 `breathiness` と意味重複しやすい
- `tension` は `pressedness` と `spectral_tilt` の両方に跨りやすい
- 新次元は「名前が違う」だけでなく、相関が高すぎないことを確認する

### 8.3 Acting Texture Latent

physical で説明しきれない残差を `acting texture latent` として別経路に持つ。
初期提案は `16-D` から `32-D` とする。

要件:

- public physical control と別 tensor であること
- UI 上で raw latent 全軸を直接 expose しないこと
- reference audio または semantic prompt から推定可能であること
- collapse を避けるため、独立正則化または residual usage penalty を持つこと

### 8.4 Intent Compiler

LLM は free-form style generator ではなく `Intent Compiler` として扱う。
出力は少なくとも以下を含む。

- `physical targets`
- `acting texture latent prior`
- `pacing controls`
- optional `dialogue state`

### 8.5 Biological Constraints

高次意図から各パラメータへの関係は独立同分布としない。
`v4` は共起と遷移の prior を学習する。

実装方針:

- low-rank covariance prior
- intent-conditioned parameter prior
- frame-to-frame transition prior
- physically implausible combination への penalty

禁止事項:

- future frame を見た smoothing
- offline 専用の non-causal consistency pass を mainline に入れること

### 8.6 Loss 構成

`v4` の主要 loss は次を含む。

- codec token prediction loss
- control token prediction loss
- pointer progression loss
- explicit physical supervision loss
- acting latent regularization loss
- disentanglement loss
- speaker consistency loss
- prosody prediction loss
- semantic alignment loss

低信頼 pseudo-label は mask と confidence で重み付けする。
unknown dimension を dense zero として教師信号にしてはならない。

## 9. Runtime and Export

### 9.1 v4 Contract

`v4` は `v3` の 8-D `voice_state` 契約を延命しない。
Core / Export / Serve / Rust は新契約へ統一する。

`v4` runtime contract の主要項目:

- physical control tensor
- acting texture latent tensor
- pacing controls
- pointer state
- speaker identity anchor
- trajectory record

### 9.2 Export

ONNX export は `v4` の conditioning 分解をそのまま保持する。
少なくとも次の境界を持つ。

- physical encoder
- acting latent encoder or predictor
- speaker encoder
- UCLM core
- codec path

### 9.3 Serve API

Serve API は `explicit_voice_state [8]` を前提としない。
新しい request surface は次の 3 層を持つ。

- simple mode
- physical advanced mode
- prompt / instruction mode

### 9.4 Rust Runtime

Rust runtime は `v4` conditioning を causal に処理できなければならない。
physical path と latent path は別バッファとして扱う。
streaming numerical parity は release blocker とする。

## 10. GUI and User Control

### 10.1 設計原則

- raw latent 全軸をユーザーに見せない
- 主要な physical control は直接編集可能にする
- open-ended acting は AI 補助面に寄せる

### 10.2 UI 構成

`v4` GUI は少なくとも以下の面を持つ。

- Basic physical panel
- Advanced physical panel
- Prompt / acting panel
- Reference-driven panel
- Trajectory editing panel

### 10.3 Basic Panel

Basic panel に出すのは、使用頻度が高く意味が明快な物理量に限る。
目安は 6 個から 8 個である。

### 10.4 Advanced Panel

Advanced panel は `12-D` から `16-D` の explicit physical control を扱う。
ただし default は closed とする。

### 10.5 Acting Panel

Acting latent は以下のような抽象操作面で扱う。

- intensity
- instability
- tenderness
- tension
- spontaneity
- reference mix

これらは latent の macro control であり、生ベクトルそのものではない。

## 11. Evaluation and Release Gates

### 11.1 Dataset QC

必須評価:

- diarization purity
- speaker cluster consistency
- overlap rejection precision
- transcript WER or CER proxy
- physical label coverage
- physical label confidence calibration
- language coverage

### 11.2 Model QC

必須評価:

- controllability
- edit reproducibility
- speaker similarity
- semantic alignment
- naturalness
- prompt following
- physical parameter calibration

### 11.3 Runtime QC

必須評価:

- Python vs ONNX parity
- Python vs Rust parity
- batch vs streaming numerical parity
- latency and RTF
- memory ceiling

### 11.4 External Comparison

Fish Audio S2 比較は release blocker ではなく、competitor-facing claim blocker とする。
Fish に勝つと主張するなら、対象タスクと protocol を固定した head-to-head report を必須とする。

### 11.5 Release Gate

`v4` は次を満たさなければ出荷しない。

- raw corpus から train-ready cache を再生成できる
- pseudo speaker bootstrap の品質が acceptance threshold を満たす
- physical editing が定量的に再現可能である
- acting prompt が latent path に反映される
- streaming parity が成立する
- GUI / Serve / Rust が同一 conditioning 契約を共有する

## 12. 実装順序

`v4` は一括切替だが、実装の依存順はある。
正しい順序は次の通りとする。

1. Survey freeze
2. v4 dataset contract freeze
3. raw-audio bootstrap pipeline 実装
4. v4 model contract freeze
5. train / export / serve / rust / gui 実装
6. end-to-end validation
7. v4 checkpoint training
8. release sign-off

この順序を逆転して、未定義の data contract の上に runtime を先に固めてはならない。

## 13. 既存文書との関係

この文書は、`v4` の single-cutover 計画の正本である。
詳細仕様は以下の設計文書へ分解して追記または改訂する。

- [dataset-preparation-flow.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/dataset-preparation-flow.md)
- [curation-contract.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/curation-contract.md)
- [architecture.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/architecture.md)
- [onnx-contract.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/onnx-contract.md)
- [streaming-design.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/streaming-design.md)
- [gui-design.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/gui-design.md)
- [evaluation-protocol.md](/home/kojirotanaka/kjranyone/TMRVC/docs/design/evaluation-protocol.md)

これらの個別文書が本書と矛盾する場合、`v4` に関しては本書を優先する。

## 14. 決定事項

- `v4` は incremental migration ではなく single-cutover で進める
- データ入口は raw unlabeled audio corpus とする
- pseudo speaker bootstrap は必須工程である
- physical controls と acting texture latent は別契約にする
- Whisper + LLM のみで physical supervision を置換しない
- Fish Audio S2 survey は competitor baseline として維持する
