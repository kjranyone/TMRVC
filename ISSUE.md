# TMRVC 技術課題と次世代アーキテクチャ研究ノート

## 0. 前提条件

この文書では、`MFA を学習・推論・データ準備の前提にしてはならない` ことを禁則事項として扱う。

意味:

- `phoneme_ids + durations` を作るために MFA を必須としない
- `DurationPredictor` の教師として MFA durations を要求しない
- TTS 品質改善の前提として MFA の高精度化を置かない
- MFA は残すとしても、比較実験・可視化・デバッグ専用に限定する

この前提により、TMRVC v3 は `MFA 依存アーキテクチャの改良版` ではなく、`MFA 非依存を前提とした新設計` として考える。

## 1. 問題設定

現在の TMRVC v2 は、TTS 側で `phoneme_ids + durations` を教師として使う設計であり、実運用上は MFA (Montreal Forced Aligner) に強く依存している。これは実装上は分かりやすいが、次の点で長期的な制約になる。

- 自然さの上限が低い。MFA は平均的な境界を返すため、溜め、食い気味、ささやき、泣き、叫びのような非定常なタイミングを学習しにくい。
- 感情音声に弱い。ISSUE の通り、TMRVC が狙う expressive speech ほど強制アラインメントの誤差が増えやすい。
- データスケールに不利。前処理パイプラインが重く、失敗点も増える。
- 研究潮流から遅れる。近年の高品質 TTS は、MAS/flow matching/infilling/CTC 系の内部アライメント学習へ寄っている。

ただし、MFA を単に外せば良いわけではない。TMRVC は以下を同時に満たす必要がある。

- 10 ms 単位の因果推論
- TTS / VC の単一コア維持
- 外部からのタイミング介入
- 低遅延 VC の品質維持


## 2. 文献から見えたこと

### 2.1 MFA 非依存 TTS は既に主流

- `Glow-TTS` は MAS により外部アライナ不要の単調アライメント学習を成立させた。
- `VITS` 系は MAS を含む単段 end-to-end 化で品質を引き上げた。
- `Matcha-TTS` は flow matching で高速かつ高品質、かつ外部アライメント不要を示した。
- `E2 TTS` と `F5-TTS` は duration model や G2P すら必須としない極端に単純な枠組みでも非常に高い自然さに到達できることを示した。

結論として、`MFA を使わないと自然さが出ない` は既に成立しない。むしろ高自然さ系は、外部 duration supervision を減らす方向に進んでいる。

### 2.2 ただし、そのまま TMRVC に入れると低遅延を壊す

- `E2 TTS`, `F5-TTS`, `Voicebox`, `NaturalSpeech 2` は自然さは強いが、基本的に非因果・非ストリーミング寄り。
- これらは TMRVC の `10 ms causal O(1)` 目標にそのまま適合しない。
- 一方で `MoChA` や `RNN-T` 系の研究は、単調でオンラインなポインタ進行を学習できることを示している。

結論として、TMRVC が取るべき方向は `完全オフライン SOTA の丸写し` ではなく、`内部アライメント学習 + オンライン単調ポインタ` である。

### 2.3 低遅延 VC は「ASR 依存を薄めた streaming semantic modeling」が強い

- `DualVC 2` は chunk 内未来文脈を活用して 186.4 ms で品質を上げた。
- `DualVC 3` は ASR 依存を外し、semantic token と pseudo context により 50 ms 級まで短縮した。
- `StreamVoice` は look-ahead なしの streaming zero-shot VC を示した。
- `StreamVoice+` は end-to-end 化で ASR 依存をさらに下げた。

結論として、VC 側は `固定 duration 付き TTS の延長` ではなく、`因果 semantic encoder + future context 補償` が有力である。


## 3. ISSUE にある 3 案の評価

### A. Emission Token 方式

`<NEXT_PHONEME>` あるいは `ADVANCE / HOLD` のような制御を B stream に持たせる案は、有望である。理由は次の通り。

- `MoChA` や `RNN-T` と思想的に整合する。
- 推論時に因果的に 1 step ずつポインタを進められる。
- 10 ms ごとの外部制御に落とし込みやすい。
- MFA を学習の必須依存から外しやすい。

懸念:

- 離散境界のみだと、滑らかな加減速や歌唱表現でギクシャクしやすい。
- `advance` の誤爆が起きると復帰が難しい。

評価:

- `v3 の主軸候補`
- ただし単独ではなく、`soft progress` か `boundary confidence` を併設した方が良い

### B. Continuous Flow Alignment (進捗カーブ)

進捗を `0.0 -> 1.0` の連続量として扱う発想は、表現的には非常に良い。特に slow speech, emotional timing, singing 的な滑らかさには理にかなっている。

ただし、現時点で TMRVC のような 10 ms causal codec LM にそのまま適用できる成熟した参照実装・標準系は乏しい。

懸念:

- 学習安定性の設計コストが高い
- 進捗値と codec token 生成の整合条件を自前で作り込む必要がある
- 失敗時のデバッグが最も難しい

評価:

- `研究枝としては面白い`
- `v3 本線にはまだ早い`

### C. Causal MAS / CTC 統合

`MAS/CTC を内部損失として使い、推論時は因果ポインタで回す` という構成は、文献的にも最も筋が良い。

- MAS は外部アライナ不要 TTS で実績がある
- CTC / transducer 系は online monotonic alignment と相性が良い
- 2026-02 の `CTC-TTS` は streaming TTS 文脈で、MFA ベースより CTC ベースが有利であることを直接主張している

注意:

- `Causal MAS` そのものは確立した名前付き標準手法ではない
- 実際には `offline alignment loss + online pointer policy` のハイブリッドとして設計する方が安全

評価:

- `v3 の学習戦略として最有力`
- ただし推論本体は MAS ではなく `pointer / emission head` で構成すべき


## 4. 設計原則

v3 では次を原則とする。

### 4.1 禁止するもの

- `MFA 由来 durations` を主教師にすること
- `DurationPredictor` を TTS 本流の中心に置くこと
- 非因果な全文 infilling を標準推論経路にすること
- 未来文全体への依存を前提にすること

### 4.2 必須要件

- 10 ms ごとの因果更新
- text consumption のオンライン決定
- 外部からのタイミング制御
- TTS / VC の単一コア維持
- 学習時の内部アライメント獲得

### 4.3 評価基準

- 自然さ
- 応答テンポの文脈依存性
- 感情発話での破綻率
- streaming latency
- 制御可能性


## 5. 推奨する UCLM v3 方針

### 5.1 結論

TMRVC v3 の本線は、次の組み合わせが最も現実的である。

1. `MFA 廃止`
2. `学習時は MAS/CTC 系の内部アライメント損失`
3. `推論時は causal pointer head (ADVANCE/HOLD or boundary probability)`
4. `表情・間の自然さのために soft progress / prosody latent を併設`
5. `VC 側は streaming semantic encoder + pseudo future context 補償`

これは `自然さ`, `制御性`, `低遅延`, `実装可能性` のバランスが最も良い。

### 5.2 TTS 側の具体像

#### Core idea

- Text encoder 出力に対して、各 10 ms step で `現在の text index` を参照する
- B stream は少なくとも以下を出す
  - `advance_prob`: 次の音素/文字へ進む確率
  - `progress_delta`: 現在単位の内部進捗
  - `prosody_latent`: 溜め、勢い、脱力、句境界感などを表す局所潜在

#### Training

- `alignment teacher` は MFA ではなく MAS または CTC から得る
- loss は 3 本立てが良い
  - `token loss`: codec / control token の通常損失
  - `alignment loss`: MAS または CTC
  - `pointer supervision`: teacher alignment から導く advance/halt 教師

#### Inference

- 完全因果
- 各 10 ms step で pointer を更新
- 外部制御は `advance bias` または `target pace curve` で注入

この設計なら、MFA を使わずに自然さを上げながら、既存の `10 ms token clock` を温存できる。

### 5.3 VC 側の具体像

VC は TTS と同じ pointer 問題ではない。ここで重要なのは `将来文脈不足をどう補うか` である。

- `DualVC 3` が示すように、semantic token 化と pseudo context は有効
- `StreamVoice+` が示すように、ASR 依存を薄めた end-to-end streaming 化が有効

TMRVC では次の構成が合う。

- source speech -> causal semantic encoder
- semantic stream -> lightweight future-context predictor
- predictor 出力を UCLM core の補助条件として使う
- speaker / voice state / semantic を分離したまま codec token を出す

これにより VC 側だけのために大きな look-ahead を持ち込まずに済む。


## 6. v2 から v3 への置換方針

### 6.1 本流から外すもの

- `DurationPredictor`
- `durations.npy` 必須前提の dataset 構造
- `phoneme_ids + durations` が無いと TTS 学習できない前提
- `TextGrid -> durations` を品質の中核とみなす運用

### 6.2 暫定互換として残してよいもの

- `DurationPredictor` を ablation 用の分岐として残す
- `durations.npy` を旧モデル比較のためだけに読む
- MFA をデバッグ時の boundary 可視化に使う

### 6.3 新しく本流に入れるもの

- `pointer head`
- `advance / hold` 制御
- `soft progress latent`
- MAS/CTC ベースの alignment loss
- VC 用 causal semantic context predictor


## 7. アーキテクチャ提案

### 提案名

`UCLM v3: Causal Pointer Codec Language Model`

### 新しい B stream 仕様

- `B_t[0]`: advance gate
- `B_t[1]`: progress delta
- `B_t[2:K]`: local prosody / phrasing control

### Text conditioning

- `text_index_t` は因果ポインタで更新
- `text_context_t` は `text_index_t` 周辺の局所 window のみ参照
- 未来単語全体を使わない optional mode を標準にする

### 外部制御

- `pace curve`
- `phrase boundary bias`
- `hold/advance override`

これなら DAW から 10 ms グリッドで直接編集できる。


## 8. 実験優先順位

### Phase 0: 現行品質の切り分け

- `PHONE2ID` と現行 phone set の不整合を解消する
- 現行 UCLM v2 の TTS 品質ボトルネックが alignment なのか phone coverage なのかを切り分ける
- dataset ごとの unknown rate を定量化する

理由:

現時点の TMRVC は、MFA 依存そのものよりも `phone vocabulary mismatch` の方が直近品質に効いている可能性がある。ここを放置すると v3 の評価も歪む。

### Phase 1: 最小リスク研究プロトタイプ

- 既存 `DurationPredictor` を互換枝へ退避し、別枝で `advance gate head` を追加
- teacher は既存 durations ではなく、codec 時系列と text 長の単調対応から近似生成する
- 推論では `duration mode` を主経路にしない

狙い:

- 既存資産を壊さず、pointer 制御の収束性だけ先に見られる
- ただしこの段階でも「MFA 必須」には戻さない

### Phase 2: MFA 依存除去

- durations 教師を完全にやめ、MAS または CTC ベースの alignment loss に置換
- pointer head を主経路へ昇格
- MFA は optional evaluation / debugging tool へ格下げ

### Phase 3: VC の streaming quality 強化

- causal semantic encoder を導入
- pseudo future context predictor を追加
- chunk latency と MOS / SIM / CER の Pareto を評価


## 9. 採用判断

### 採用する

- `internal alignment learning`
- `causal pointer`
- `soft prosody / progress conditioning`
- `streaming semantic context prediction for VC`

### 採用しない

- `MFA を長期的な学習前提として残す`
- `E2 TTS / F5-TTS 型の完全非因果 infilling をそのまま中核に据える`
- `Continuous Flow Alignment をいきなり本線採用する`

### 研究枝として残す

- continuous progress curve
- differentiable monotonic attention variants
- recent CTC-based streaming TTS の取り込み


## 10. 実装タスクへの翻訳

### 10.1 tmrvc-train

- `duration_predictor.py` を本流依存から切り離す
- `uclm_model.py` に pointer head を追加する
- `trainer.py` に pointer loss と alignment loss を追加する
- `dataset/uclm_dataset.py` から `durations.npy` 必須前提を外す

### 10.2 tmrvc-serve

- `uclm_engine.py` の TTS 経路から duration 展開依存を外す
- 因果 pointer state を engine state に追加する
- 外部制御 API として `pace`, `hold`, `boundary_bias` を追加する

### 10.3 dev.py / 運用

- `13) MFA実行 + TTSアライメント注入` は v2 legacy 扱いへ下げる
- v3 系では `MFA 無しで学習可能` を標準フローにする
- 実験メニューを `v2 legacy` と `v3 pointer` に分ける


## 11. 最終提言

TMRVC が目指すべきなのは、`MFA を外した F5-TTS 風モデル` ではない。そうすると低遅延性と制御性を失う。

目標は次の一点に絞るべきである。

`外部アライナを不要にしつつ、10 ms 因果ポインタで text consumption を制御する codec LM`

研究として最も成功確率が高いのは、

- `学習`: MAS / CTC で内部アライメント学習
- `推論`: emission / pointer head
- `表現`: local prosody latent
- `VC`: pseudo future context 補償

という構成である。

これが、自然で有機的な TTS と、低遅延で高品質な VC を同時に狙ううえで、現時点で最も筋が良い。


## 12. 参考文献

- Glow-TTS: A Generative Flow for Text-to-Speech via Monotonic Alignment Search
  - https://arxiv.org/abs/2005.11129
- Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech (VITS)
  - https://arxiv.org/abs/2106.06103
- Matcha-TTS: A fast TTS architecture with conditional flow matching
  - https://arxiv.org/abs/2309.03199
- E2 TTS: Embarrassingly Easy Fully Non-Autoregressive Zero-Shot TTS
  - https://arxiv.org/abs/2406.18009
- F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching
  - https://arxiv.org/abs/2410.06885
- StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models
  - https://arxiv.org/abs/2306.07691
- NaturalSpeech 2: Latent Diffusion Models are Natural and Zero-Shot Speech and Singing Synthesizers
  - https://arxiv.org/abs/2304.09116
- Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale
  - https://arxiv.org/abs/2306.15687
- Monotonic Chunkwise Attention
  - https://arxiv.org/abs/1712.05382
- Sequence Transduction with Recurrent Neural Networks
  - https://arxiv.org/abs/1211.3711
- DualVC 2: Dynamic Masked Convolution for Unified Streaming and Non-Streaming Voice Conversion
  - https://arxiv.org/abs/2309.15496
- DualVC 3: Leveraging Language Model Generated Pseudo Context for End-to-end Low Latency Streaming Voice Conversion
  - https://arxiv.org/abs/2406.07846
- StreamVoice: Streamable Context-Aware Language Modeling for Real-time Zero-Shot Voice Conversion
  - https://arxiv.org/abs/2401.11053
- StreamVoice+: Evolving into End-to-end Streaming Zero-shot Voice Conversion
  - https://arxiv.org/abs/2408.02178
- CTC-TTS: LLM-based dual-streaming text-to-speech with CTC alignment
  - https://arxiv.org/abs/2602.19574
