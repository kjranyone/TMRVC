# Track: Codec Strategy And Generation Design

## Status: Active (2026-03-18)

## Decision

方針C（ハイブリッド）を採用する。

- 製品本線: EnCodec / single-codebook — 現行UCLM ARを温存
- 研究本線: Mimi — AR+NAR分割で生成設計を変える
- 両者を同じ評価軸で比較し、v4.0 release に使う codec を決定する

## Background

Mimi (Kyutai, 2024) を UCLM の全8CB同時ARに差したところ、
生成される音声が意味をなさなかった。

原因分析:

- Mimi の各 codebook は EnCodec CB0 より局所時間相関がかなり低い
  (Mimi CB0: 0.189, EnCodec CB0: 0.483)
- 単純な next-token AR で利用できる局所予測性が弱い
- ただし Mimi が AR-LM で「全く使えない」わけではない
  (Moshi, Sesame CSM, LFM2-Audio 等で採用実績あり)
- 問題は「Mimi が悪い」のではなく「UCLMの生成設計がMimiに合っていない」

## 核心の問い

> UCLM の競争優位は、既存 AR コアを活かした高速実装にあるのか、
> それとも physical control を活かすために生成分解まで変えてよいのか。

## 実験計画

### 比較4条件

| ID | Codec | 生成方式 | 目的 |
|----|-------|---------|------|
| A  | EnCodec 24kHz | 全CB同時AR (現行UCLM) | 現状設計の上限確認 |
| B  | Mimi | CB0 AR + CB1-7 NAR | Mimiの正しい使い方での比較 |
| C  | Mimi | delay pattern AR | delay pattern の有効性確認 |
| D  | WavTokenizer or X-Codec 2 | 1本AR | single-codebook の実力確認 |

条件Aが最優先。条件Bが対抗。条件Dが保険兼有力案。
条件Cは latency トレードオフが大きいため第3候補。

### 評価指標 (5軸 + 補助)

Primary:

1. naturalness (再構成品質、MOS proxy)
2. speaker similarity (話者類似性)
3. physical control の追従性 (monotonicity, calibration)
4. patch / replay の決定論性 (replay fidelity, edit locality)
5. streaming latency / first-audio latency

Auxiliary:

- token perplexity
- free-running 崩壊率
- codebook ごとの予測精度

### 実験手順

Step 1: 条件A (EnCodec + 現行AR) でベースラインを確立
- EnCodec encode で cache 再生成
- 現行 UCLM そのままで学習 → 音声生成
- 5軸で測定 → ベースラインスコア確定

Step 2: 条件D (single-codebook) で比較
- WavTokenizer or X-Codec 2 の encode で cache 生成
- 1本AR のまま学習 → 生成
- ベースラインとの差分を測定

Step 3: 条件B (Mimi + AR/NAR) で比較
- UCLM に NAR refinement head を追加
- CB0 のみ AR、CB1-7 は NAR で生成
- ベースラインとの差分を測定

Step 4: 判断
- 3条件の5軸スコアを比較
- v4.0 release codec を決定

## Constraints

- 条件A は現行 UCLM を壊さない
- 条件B は UCLM 再設計を伴う (AR+NAR 分割)
- 条件D は vocab_size / frame_rate の変更を伴う
- いずれの条件でも、physical control / trajectory / pointer の設計は維持する
- 10ms causal streaming は条件B/Dでも守る (codec decode 側で担保)

## 一番避けるべきこと

「Mimi を現行 8CB 同時 AR にそのまま差して、やっぱダメだった」で終わること。
それは codec 比較ではなく、設計ミスマッチの再確認にしかならない。

## Timeline

- Step 1 (EnCodec baseline): 即時実行
- Step 2 (single-codebook): Step 1 と並行可能
- Step 3 (Mimi AR/NAR): Step 1 完了後
- Step 4 (判断): 全条件完了後

## Out Of Scope

- Mimi を全CB同時ARで再試行 (既に失敗、設計ミスマッチ確認済み)
- codec 自体の fine-tuning (frozen pre-trained を使う)
- 5条件以上の比較実験 (議論が増えて結論が出ない)
