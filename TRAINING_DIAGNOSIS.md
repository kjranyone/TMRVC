# UCLM 学習停滞診断レポート

更新日: 2026-04-01  
対象: `/home/kojirotanaka/kjranyone/TMRVC` の現行コードベースに対する静的コードレビュー

## 結論

現在の学習停滞は、単なる「データ多様性が高すぎる」「262M では足りない」といった統計的難しさだけでは説明しきれない。

現行の学習経路には、以下のような **実装上の不整合** が複数残っている。

1. 実験条件として前提にされている `grad_accum=8` と scheduler が、現行の training path には入っていない
2. TTS 学習で text supervision を持たないサンプルが混入しうる
3. train 用 `collate_fn` が重要 supervision を落としており、pointer / expressive / voice-state の教師信号が有効化されていない
4. `Condition A/B/C/D` を切り替えて検証するための導線が壊れており、実際には Condition A に固定されている
5. voice-state conditioning に非因果畳み込みが残っており、train / streaming の整合性を壊している
6. `f0_condition` が二重注入されている

したがって、現状は「学習が遅い」の前に、**想定したアルゴリズムがそのまま訓練されていない** と判断する。

## レビュー前提

- 本レポートはコードの静的レビューに基づく
- 実機での再学習・再現実験はまだ行っていない
- `torch` 実行環境がこの場には入っておらず、簡易実験や pytest による裏取りは未実施
- ただし、下記の指摘はすべて現行コードの制御フローから直接読み取れる

## 重大所見

### 1. `grad_accum=8` と scheduler が実装されていない

最重要の問題。診断仮説では「gradient accumulation を入れた」「warmup + cosine decay を入れた」という前提になっているが、現行の学習コードからはそれを確認できない。

#### 根拠

- `UCLMTrainer.train_step()` は呼び出しのたびに先頭で `optimizer.zero_grad(set_to_none=True)` を実行している  
  `tmrvc-train/src/tmrvc_train/trainer.py:443-446`
- その後、`accumulate=False` のとき即座に `optimizer.step()` している  
  `tmrvc-train/src/tmrvc_train/trainer.py:761-766`
- CLI 側は常に `trainer.train_step(batch)` をそのまま呼んでおり、`accumulate=True` を使っていない  
  `tmrvc-train/src/tmrvc_train/cli/train_uclm.py:236-240`
- optimizer 作成後に LR scheduler を作っている箇所がない  
  `tmrvc-train/src/tmrvc_train/cli/train_uclm.py:212-232`

#### 含意

- 実効 batch は依然として `batch_size` そのもの
- 「小 batch で勾配分散が大きい」問題は、まだ解消されていない可能性が高い
- loss 曲線の解釈が、実際の学習条件とズレている

#### 判定

**仮説ではなく事実上のコード不整合**

---

### 2. TTS 学習に text supervision を持たないサンプルが混入しうる

これも重大。実キャッシュに `phoneme_ids.npy` を持たない utterance が混ざっている場合、TTS 学習バッチにゼロ埋め phoneme 列が混入する。

#### 根拠

- dataset の既定値は `require_tts_supervision=False`  
  `tmrvc-train/src/tmrvc_train/cli/train_uclm.py:145-168`
- dataset 側も、その場合は `phoneme_ids.npy` がなくてもサンプルを除外しない  
  `tmrvc-train/src/tmrvc_train/dataset/uclm_dataset.py:85-89`
- train 用 `collate_fn` は、phoneme を持たないサンプルに対してゼロ埋め placeholder を入れる  
  `tmrvc-train/src/tmrvc_train/cli/train_uclm.py:123-134`
- trainer は `batch["phoneme_ids"] is not None` だけで TTS 実行可能と判定する  
  `tmrvc-train/src/tmrvc_train/trainer.py:526-531`

#### 何が起きるか

- バッチ中に 1 件でも phoneme 付きサンプルがいれば、batch 全体が TTS 候補になる
- phoneme を持たないサンプルも、全ゼロ phoneme 列として text encoder に入る
- その結果、TTS loss が無意味な teacher forcing を一部サンプルに対して学習する

#### 条件

これは **実データ側に non-TTS sample が混在している場合に発火する条件付きの不具合** である。  
ただし、現在の training path はそれを防ぐ設計になっていない。

#### 判定

**実データ構成次第で致命的**

---

### 3. train 用 `collate_fn` が supervision artifact を大量に落としている

これにより、設計上は存在する pointer / expressive / voice-state 教師信号が学習で実際には使われていない。

#### dataset が返しているもの

`DisentangledUCLMDataset.__getitem__()` は少なくとも以下を返す。

- `dialogue_context`
- `acting_intent`
- `prosody_targets`
- `text_suprasegmentals`
- `voice_state_targets`
- `voice_state_observed_mask`
- `voice_state_confidence`
- `bootstrap_alignment`

参照:

- `tmrvc-train/src/tmrvc_train/dataset/uclm_dataset.py:255-330`

#### しかし train 側 `collate_fn` は通していない

train 用 `collate_fn` が stack しているのは主に以下だけ。

- `target_a`
- `target_b`
- `source_a_t`
- `explicit_state`
- `ssl_state`
- `speaker_embed`
- `speaker_id`
- `f0_condition`
- `phoneme_ids`
- `phoneme_lens`
- `language_id`

参照:

- `tmrvc-train/src/tmrvc_train/cli/train_uclm.py:87-142`

#### 直接の影響

1. `bootstrap_alignment` が trainer に届かない  
   `tmrvc-train/src/tmrvc_train/trainer.py:919-928`

2. `voice_state_targets` が trainer に届かない  
   `tmrvc-train/src/tmrvc_train/trainer.py:1017-1036`

3. `text_suprasegmentals` が届かない  
   `tmrvc-train/src/tmrvc_train/trainer.py:611-613`

4. `dialogue_context` / `prosody_targets` / `acting_intent` が届かない  
   `tmrvc-train/src/tmrvc_train/trainer.py:595-610`

#### 補足

data 側には、これらを正しく保持する canonical collate 実装がすでに存在する。

- `tmrvc-data/src/tmrvc_data/uclm_dataset.py:509-651`

つまり、学習停滞の一部は「理論が悪い」のではなく、**train 側が dataset contract を破っている** ことが原因である可能性が高い。

#### 判定

**事実上の配線ミス**

---

### 4. Condition A の妥当性を疑う以前に、現行 training path は Condition A に固定されている

これはアルゴリズム仮説の検証不能性の問題。

#### 根拠

- `uclm_loss()` の default は `codec_condition="A"`  
  `tmrvc-train/src/tmrvc_train/models/uclm_loss.py:384-420`
- `DisentangledUCLM` の default も `codec_condition="A"`  
  `tmrvc-train/src/tmrvc_train/models/uclm_model.py:447-465`
- `train_uclm()` では model / trainer 生成時に `codec_condition` を渡していない  
  `tmrvc-train/src/tmrvc_train/cli/train_uclm.py:204-232`
- pipeline 側は `--codec-condition` を渡そうとしているが  
  `tmrvc-train/src/tmrvc_train/pipeline.py:372-375`
- CLI 側にその引数が存在しない  
  `tmrvc-train/src/tmrvc_train/cli/train_uclm.py:260-371`

#### 含意

- 「Condition A が悪いのでは」という仮説を、公式 training path では比較実験できない
- Condition B/C/D を試したつもりでも、実際には A で回っている危険がある

#### 判定

**実験設計上の重大な障害**

---

### 5. voice-state encoder が非因果で、10ms causal 制約に違反している

#### 根拠

- docstring は temporal conv を causal と説明している  
  `tmrvc-train/src/tmrvc_train/models/voice_state_encoder.py:57-63`
- 実装は `Conv1d(..., kernel_size=5, padding=2)` の対称畳み込み  
  `tmrvc-train/src/tmrvc_train/models/voice_state_encoder.py:104-107`
- full sequence をそのまま畳み込んでいる  
  `tmrvc-train/src/tmrvc_train/models/voice_state_encoder.py:158-161`
- streaming 用 encoder は別実装であり、train と一致していない  
  `tmrvc-train/src/tmrvc_train/models/voice_state_encoder.py:176-220`

#### 含意

- 学習時には未来フレームを見た state condition が core に入る
- 推論時 streaming path ではその未来情報が消える
- これは `Issue C: streaming numerical parity` より前段の、より本質的な train / infer mismatch

#### 判定

**設計制約違反**

---

### 6. `f0_condition` が二重注入されている

#### 根拠

- model 外側で `content_features` に `self.f0_proj(f0_condition)` を加算  
  TTS: `tmrvc-train/src/tmrvc_train/models/uclm_model.py:868-870`  
  VC: `tmrvc-train/src/tmrvc_train/models/uclm_model.py:646-647`
- そのまま core に `f0_condition` を渡す  
  TTS: `tmrvc-train/src/tmrvc_train/models/uclm_model.py:938-950`
- core 側でも別の `f0_proj` で再加算  
  `tmrvc-train/src/tmrvc_train/models/uclm_transformer.py:578-586`

#### 含意

- F0 conditioning が意図より過大になる
- conditioning balance を崩し、他条件の寄与を相対的に弱める
- 特に early stage で acoustic token CE の最適化を不安定化する可能性がある

#### 判定

**明確な実装バグ候補**

---

### 7. `pointer_target_source` / `alignment_loss_type` は多くが dead knob

#### 根拠

- config validation は存在する  
  `tmrvc-train/src/tmrvc_train/trainer.py:241-274`
- しかし `_tts_pointer_step()` の実際の分岐は
  - hard bootstrap
  - MAS
  - soft transition
  にほぼ固定されている  
  `tmrvc-train/src/tmrvc_train/trainer.py:912-983`

#### 含意

- CLI で knob を変えても、想定した実験になっていない可能性がある
- 仮説検証の解釈が汚れる

#### 判定

**診断を難しくする設計不良**

---

### 8. `loss_a only` 実験でも、実装によっては完全に `loss_a only` になっていない疑いがある

#### 根拠

- `uclm_loss()` は常に `loss_b` を total loss に加える  
  `tmrvc-train/src/tmrvc_train/models/uclm_loss.py:483-498`
- dataset は `control_tokens.npy` がない場合、`target_b` をゼロで捏造する  
  `tmrvc-train/src/tmrvc_train/dataset/uclm_dataset.py:237-242`

#### 含意

- 「loss_a only」と言っても、呼び出し側で本当に `loss_b` を無効化していない限り、shared trunk には control loss の勾配が残る
- 実験ノートとコードパスの整合確認が必要

#### 判定

**要再確認**

## 仮説資料に対する再評価

元の仮説のうち、以下は依然として検討価値がある。

- Condition A が RVQ 残差構造に対して不利ではないか
- 実効 batch が小さすぎないか
- multilingual / multi-speaker / multi-corpus の curriculum が必要ではないか
- multi-loss 最適化で codec CE が弱まっていないか

ただし、これらを議論する前に、以下を是正しないと議論が汚染される。

1. 実際の training path に grad accumulation と scheduler を入れる
2. TTS batch に non-text sample を入れない
3. train collate を dataset contract と一致させる
4. `codec_condition` を CLI から end-to-end で通す
5. voice-state encoder を causal にする
6. F0 二重注入を解消する

## 優先修正順

### Priority 0: 実験条件をコードと一致させる

- gradient accumulation を実装する
- warmup + cosine scheduler を実装する
- 実際にログへ `effective_batch_size` と `lr` を出す

### Priority 1: データ経路の汚染を止める

- `require_tts_supervision=True` を既定にするか、TTS 用 sampler を別建てにする
- text を持たない sample は TTS branch に絶対に入れない
- train 側 `collate_fn` を廃止し、data 側 canonical collate に寄せる

### Priority 2: 実験可能性を回復する

- `--codec-condition` を CLI に追加し、model / trainer / loss に通す
- `alignment_loss_type` と `pointer_target_source` の dead path を整理する

### Priority 3: 因果性と conditioning を正す

- `VoiceStateEncoder` を真に causal な実装へ置き換える
- `f0_condition` の注入点を 1 箇所に統一する

### Priority 4: その後にアルゴリズム比較

- Condition A vs B を比較
- batch / curriculum / corpus mixture を比較
- その段階で初めて「モデル容量不足」や「データ多様性過大」を論じる

## 現時点の総括

このコードベースで最初に疑うべきは scaling law ではない。  
まず疑うべきは、**学習実験の前提そのものがコード上で成立しているか** である。

現時点の判断としては、学習停滞の主因候補は以下の順で強い。

1. 学習条件の不一致  
   grad accumulation / scheduler 不在
2. TTS データ経路の汚染  
   non-text sample 混入
3. train collate の supervision 欠落
4. Condition A 固定化による検証不能
5. 非因果 voice-state conditioning
6. F0 二重注入

これらを潰した後でなお `loss_a` が落ちないなら、その時点で初めて「Condition A の本質的難しさ」や「curriculum 必須性」が主犯候補になる。
