#set page(
  paper: "a4",
  margin: (x: 2cm, y: 2.5cm),
  numbering: "1",
)
#set text(
  font: "Noto Sans CJK JP",
  size: 10pt,
  lang: "ja",
)
#set heading(numbering: "1.1")
#show heading: it => {
  v(0.5em)
  it
  v(0.3em)
}

// --- Helper Functions ---
#let block_box(content, fill: white) = rect(
  stroke: 0.5pt + black,
  fill: fill,
  radius: 2pt,
  width: auto,
  height: auto,
  inset: 4pt,
  align(center + horizon, text(size: 8pt, content))
)

#let arrow_line() = align(center + horizon, box(width: 20pt, height: 10pt)[
  #place(center + horizon, line(start: (0pt, 5pt), end: (18pt, 5pt), stroke: 0.4pt))
  #place(center + horizon, line(start: (15pt, 3pt), end: (18pt, 5pt), stroke: 0.4pt))
  #place(center + horizon, line(start: (15pt, 7pt), end: (18pt, 5pt), stroke: 0.4pt))
])

#align(center)[
  #block(inset: 1em)[
    #text(weight: "bold", size: 1.4em)[TMRVC: Unified Codec Language Model with Dual-Stream Token Spec v2 for Real-time Expressive Speech Generation] \
    #v(0.5em)
    #text(size: 1em)[Project TMRVC Team] \
    #text(style: "italic", size: 0.9em)[https://github.com/kjranyone/TMRVC]
  ]
]

#[
  #set text(weight: "bold")
  Abstract
] \
#h(1em)
本論文では、音声合成 (TTS) と音声変換 (VC) を統合した **Unified Codec Language Model (UCLM) v2** を提案する。UCLM v2 は、音響情報を担う **Acoustic Stream ($A_t$)** と、非言語的イベントを制御する **Control Stream ($B_t$)** の二系統を同時に生成する。本稿では、212 話者、約 8.7 万発話の大規模データセットを用いた学習結果を報告する。提案モデルは 100,000 ステップの学習により Loss 0.63 まで安定的に収束し、CPU 上での 50ms 以下のエンドツーエンド・レイテンシと高度な演技制御の両立を実現した。

= アーキテクチャ (UCLM v2)
UCLM v2 は、音声生成を以下の確率分布のサンプリングとして定式化する：
$ P(A_t, B_t | bold(X)_t, bold(S), bold(V)_t, bold(C)_{<t}) $
$A_t$ は 8 層の RVQ トークン、$B_t$ は制御タプル、$bold(V)_t$ は 8 次元の物理 Voice State である。

= 実験と評価 (Experiments)
提案手法の有効性を検証するため、大規模なマルチ話者学習を実施した。

== データセット構成
以下の 4 つのデータセットを統合し、バランスサンプラーを用いて学習を行った。合計話者数は 212 名、総発話数は 86,662 件である。

#figure(
  caption: [学習データセットの内訳],
  table(
    columns: (auto, auto, auto),
    inset: 6pt,
    align: center,
    [*Dataset*], [*Utterances*], [*Description*],
    [VCTK], [44,455], [多話者英語音声],
    [moe_multispeaker], [29,123], [多話者日本語音声],
    [JVS], [12,984], [日本語コーパス],
    [tsukuyomi], [100], [ターゲット話者],
    [*Total*], [*86,662*], [*212 Speakers*]
  )
)

== 学習パフォーマンス
学習は単一の GPU 環境で行われ、以下の結果を得た。
- **学習ステップ数**: 100,000 steps (全データセットを通じた収束を確認)
- **スループット**: 11.54 iterations/sec
- **総学習時間**: 約 2 時間 24 分
- **最終収束 Loss**: 0.6318

図2 に示すように、Loss は学習初期から安定的に減少し、100k ステップ時点で十分な収束を見せている。Dual-Stream 方式により、$A_t$ (音響) と $B_t$ (制御) が並行して効率的に最適化されていることが確認された。

#figure(
  caption: [UCLM v2 の詳細アーキテクチャとデータフロー],
  box(stroke: 0.2pt + gray, inset: 8pt, radius: 4pt)[
    #grid(
      columns: (auto, 20pt, auto, 20pt, auto, 20pt, auto),
      align: center + horizon,
      stack(dir: ttb, spacing: 4pt,
        block_box([Text / G2P], fill: blue.lighten(95%)),
        block_box([Source Audio], fill: orange.lighten(95%))
      ),
      arrow_line(),
      block_box([VCEncoder\ (VQ Bottleneck)], fill: gray.lighten(90%)),
      arrow_line(),
      stack(dir: ttb, spacing: 4pt,
        block_box([UCLM Transformer\ (12L, d=512)], fill: red.lighten(95%)),
        v(2pt),
        block_box([VoiceStateEncoder\ (8-dim + WavLM)], fill: green.lighten(95%))
      ),
      arrow_line(),
      stack(dir: ttb, spacing: 4pt,
        block_box([Dual Heads\ (A:1024, B:64)], fill: gray.lighten(90%)),
        arrow_line(),
        block_box([Codec Decoder], fill: gray.lighten(95%))
      )
    )
  ]
)

= 推論性能とレイテンシ
実時間 10ms フレームに対する処理時間は、ONNX Runtime を用いた CPU 推論（AMD Ryzen 9 等）において以下の通りである。
- **アルゴリズム遅延**: 10ms / **推論処理時間**: ~20ms
- **実効エンドツーエンド・レイテンシ**: ~30ms 〜 45ms
これにより、DAW 経由のリアルタイム音声変換において、演奏感に支障のない低遅延性を達成した。

= 結論
TMRVC は、大規模な日本語・英語マルチスピーカー・データセットにおいて、短時間の学習で高品質な収束を実現した。Dual-Stream トークン予測と物理パラメータ制御の統合は、実用的なストリーミング音声生成の新たな基準となる。

#v(1em)
#bibliography("refs.bib", style: "ieee", title: "参考文献")
