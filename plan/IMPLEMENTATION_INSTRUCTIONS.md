# v4 Implementation Instructions

本文書は plan/ の全 track から導出した実装指示の正本である。
critical path 順に並べ、各指示は単独で実行可能な単位にしてある。

凡例:
- `[NEW]` 新規ファイル作成
- `[MOD]` 既存ファイル改修
- `[DEL]` 削除
- `[TEST]` テスト追加

---

## Phase 0: Survey Freeze (track_survey)

実装ではなく文書作業。他の Phase の前提条件。

### 0-1. Competitor summary の凍結

```
対象: docs/design/external-baseline-registry.md [MOD]
```

- Fish Audio S2 のエントリに以下を追記:
  - rich-transcription ASR によるインライン演技タグ注入
  - inline instruction following（外部 LLM 不使用）
  - RL fine-tuning（rich-transcription ASR を reward に再利用）
  - frozen version / date / public artifact reference
- CosyVoice 3, Qwen3-TTS のエントリも同形式で凍結
- 各エントリに `frozen: true` と `freeze_date: YYYY-MM-DD` を付与

完了条件: 3 competitor × frozen version pin が存在する

### 0-2. Prompt/control taxonomy matrix の作成

```
対象: docs/design/evaluation-protocol.md [MOD]
```

- 制御方式を分類: prompt-only / physical-only / hybrid / reference-driven
- 各 competitor + TMRVC v4 をマッピング
- v4 固有の軸（inline tag + physical + latent の 3 層）を明示

完了条件: taxonomy matrix が docs に存在し、v4 の位置が一意に定義されている

### 0-3. Fish S2 勝利条件の凍結

```
対象: docs/design/evaluation-protocol.md [MOD]
```

- 勝利軸: acting editability, trajectory replay fidelity, edit locality
- guardrail 軸: first-take naturalness, few-shot speaker similarity, latency
- narrow claim ルール: editability で勝つが naturalness で負ける場合、claim は editability のみ

完了条件: 勝利条件が docs に frozen として存在する

---

## Phase 1: Data Bootstrap Pipeline (track_data_bootstrap)

### 1-1. Bootstrap パイプライン orchestrator 実装

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/pipeline.py [NEW]
依存: tmrvc-data/src/tmrvc_data/bootstrap/contracts.py (既存)
```

- `BootstrapPipeline` クラスを実装
- 13 stage を順次実行する orchestrator
- 各 stage は `BootstrapUtterance` を受け取り、フィールドを埋めて返す
- stage 間の中間状態は disk に永続化（再開可能）
- 各 stage の成功/失敗/スキップをログ出力

```python
class BootstrapPipeline:
    def run(self, config: BootstrapConfig) -> BootstrapResult: ...
    def run_stage(self, stage: BootstrapStage, utterances: list[BootstrapUtterance]) -> list[BootstrapUtterance]: ...
    def resume(self, checkpoint_path: Path) -> BootstrapResult: ...
```

完了条件: `python -m tmrvc_data.cli.bootstrap --corpus-dir data/raw_corpus/test` が 13 stage を通過する

### 1-2. Stage 1-3: Ingest / Normalization / VAD

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/ingest.py [NEW]
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/normalize.py [NEW]
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/vad.py [NEW]
```

- ingest: glob で wav/flac/mp3 を列挙、フォーマット検証、メタデータ (sample rate, duration) 抽出
- normalize: ラウドネス正規化 (target -23 LUFS)、DC 除去、24kHz リサンプル
- VAD: silero-vad または pyannote-vad で発話区間検出、無音で分割

完了条件: 生の wav ディレクトリから正規化済み utterance segment のリストが出力される

### 1-3. Stage 4-6: Rejection / Diarization / Pseudo Speaker

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/rejection.py [NEW]
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/diarization.py [NEW]
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/pseudo_speaker.py [NEW]
依存: tmrvc-data/src/tmrvc_data/curation/providers/diarization.py (既存)
依存: tmrvc-data/src/tmrvc_data/curation/providers/speaker_clustering.py (既存)
```

- rejection: overlap / music / noise 判定、BGM 大の区間を除外
- diarization: PyAnnoteDiarizationProvider を呼び出し、話者クラスタを推定
- pseudo_speaker: CrossFileSpeakerClustering で cross-file クラスタリング → pseudo speaker_id 付与

完了条件: 各 utterance に pseudo_speaker_id が付与され、低信頼区間は `rejected=True` フラグ付き

### 1-4. Stage 7-9: Speaker Embedding / Transcription / G2P

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/speaker_embed.py [NEW]
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/transcription.py [NEW]
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/text_normalize.py [NEW]
依存: tmrvc-data/src/tmrvc_data/curation/providers/asr.py (既存: Qwen3ASRProvider)
依存: tmrvc-data/src/tmrvc_data/g2p.py (既存)
```

- speaker_embed: ECAPA-TDNN で 192-dim speaker embedding 抽出
- transcription: Qwen3-ASR-1.7B で転写、word timestamp 付き
- text_normalize: テキスト正規化 + G2P で phoneme_ids 生成

完了条件: 各 utterance に speaker_embed, transcript, phoneme_ids, language が付与される

### 1-5. Stage 10: DSP/SSL Physical Feature Extraction

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/physical_extraction.py [NEW]
依存: tmrvc-data/src/tmrvc_data/wavlm_extractor.py (既存)
依存: tmrvc-data/src/tmrvc_data/curation/providers/voice_state.py (既存: VoiceStateEstimator)
```

- 12-D physical control targets を frame 単位で抽出:
  - pitch_level, pitch_range, energy_level, pressedness, spectral_tilt, breathiness,
    voice_irregularity, openness, aperiodicity, formant_shift, vocal_effort, creak
- WavLM で 128-dim SSL 特徴量を抽出
- 各次元に confidence score を付与
- observed_mask: 抽出成功した次元を True、失敗を False

完了条件: 各 utterance に `physical_targets: [T, 12]`, `physical_confidence: [T, 12]`, `ssl_features: [T, 128]` が付与される

### 1-6. Stage 11: LLM Semantic / Acting Annotation

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/semantic_annotation.py [NEW]
モデル: Qwen/Qwen3.5-9B (batch offline)
```

- 各 utterance について Qwen3.5-9B に以下を生成させる:
  - scene_summary: 1 文の場面要約
  - dialogue_intent: 発話意図 (inform / request / comfort / scold / etc.)
  - emotion_description: 感情記述 (free-form)
  - acting_hint: 演技指示 (free-form)
- 出力は JSON 形式で parse、失敗時は空文字列 + confidence=0

完了条件: 各 utterance に acting_annotations dict が付与される

### 1-7. Stage 11b: Enriched Transcript 生成

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/enriched_transcript.py [NEW]
モデル: Qwen/Qwen3.5-9B (batch offline)
依存: 1-4 の transcription 出力 + 1-5 の physical extraction 出力
```

- plain transcript + 検出済み audio events (laugh, inhale, etc.) + physical targets を入力として
  Qwen3.5-9B にインラインタグ付き enriched transcript を生成させる
- プロンプト例:
  ```
  以下のテキストに、音声から検出されたイベントと演技指示をインラインタグとして挿入してください。
  タグ形式: [tag_name]
  使用可能タグ: [inhale], [exhale], [laugh], [sigh], [emphasis], [pause], [angry], [whisper], ...
  元テキスト: 本当にありがとう
  検出イベント: laugh at 1.2s, inhale at 0.1s
  物理特徴: breathiness=0.7 (high), energy=0.3 (low)
  ```
- タグ位置を word/phoneme boundary にアライン
- free-form タグは canonical surface form に正規化

完了条件: 各 utterance に `enriched_transcript` フィールドが付与される

### 1-8. Stage 12-13: Confidence Scoring / Cache Export

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/confidence.py [NEW]
対象: tmrvc-data/src/tmrvc_data/bootstrap/stages/cache_export.py [NEW]
依存: tmrvc-core/src/tmrvc_core/types.py::V4BootstrapCacheEntry
```

- confidence: 全フィールドの信頼度を集約、SupervisionTier (A/B/C/D) を分類
  - Tier A: transcript_confidence > 0.9 AND speaker_confidence > 0.9 AND physical_coverage > 0.8
  - Tier B: transcript/speaker は高信頼、physical/semantic の一部が pseudo
  - Tier C: transcript あり、physical supervision 疎
  - Tier D: それ以外
- cache_export: V4BootstrapCacheEntry 形式で msgpack/arrow に書き出し

完了条件: `data/v4_cache/<corpus_id>/` に V4BootstrapCacheEntry が出力され、tier 分布レポートが表示される

### 1-9. Bootstrap quality gate 計測コード

```
対象: tmrvc-data/src/tmrvc_data/bootstrap/quality_gates.py [NEW]
テスト: tests/data/test_bootstrap_quality.py [MOD]
```

- 7 メトリクスを計測する関数群:
  - diarization_purity()
  - speaker_cluster_consistency()
  - overlap_rejection_precision()
  - transcript_quality_proxy() (WER/CER)
  - physical_label_coverage()
  - physical_label_confidence_calibration()
  - language_coverage()
- 各メトリクスは float を返す。閾値は track_validation が定義（ここでは計測のみ）

完了条件: `python -m tmrvc_data.cli.bootstrap --quality-report` で 7 メトリクスが出力される

---

## Phase 2: Architecture Contract Freeze (track_architecture)

### 2-1. Text encoder の acting tag 拡張

```
対象: tmrvc-train/src/tmrvc_train/models/text_encoder.py [MOD]
対象: tmrvc-core/src/tmrvc_core/acting_tags.py [NEW]
定数: configs/constants.yaml::n_acting_tags=35, extended_vocab_size=237
```

- acting_tags.py に frozen tag vocabulary を定義:
  ```python
  ACTING_TAG_VOCAB = {
      # vocal events
      "[inhale]": 200, "[exhale]": 201, "[laugh]": 202, "[sigh]": 203,
      "[cough]": 204, "[click]": 205, "[sob]": 206,
      # prosodic markers
      "[emphasis]": 207, "[prolonged]": 208, "[pause]": 209,
      # acting directives
      "[angry]": 210, "[whisper]": 211, "[calm]": 212, "[excited]": 213,
      "[tender]": 214, "[professional]": 215, "[sad]": 216, "[fearful]": 217,
      "[sarcastic]": 218, "[bored]": 219,
      # bracket start/end for free-form
      "[act_start]": 220, "[act_end]": 221,
      # reserved
      ...  # up to 234 (35 tags total)
  }
  ```
- text_encoder.py の embedding layer を `PHONEME_VOCAB_SIZE (200)` → `EXTENDED_VOCAB_SIZE (237)` に拡張
- acting tag token は phoneme token と同じ hidden dimension で embedding
- pointer attention mechanism で acting tag も consumed text unit として扱う

```python
# text_encoder.py の変更箇所
class TextEncoder(nn.Module):
    def __init__(self, ...):
        # before: self.embed = nn.Embedding(PHONEME_VOCAB_SIZE, d_model)
        # after:
        self.embed = nn.Embedding(EXTENDED_VOCAB_SIZE, d_model)
```

テスト: tests/train/test_text_encoder_acting_tags.py [NEW]
- tag token が embedding を返すこと
- pointer が tag token を consumed unit として扱うこと
- tag あり/なしで forward pass の output shape が同じであること

完了条件: text encoder が enriched transcript (phoneme + acting tag 混在) を受け取れる

---

## Phase 3: Training Pipeline (track_training)

### 3-1. Supervision tier-aware loss weighting

```
対象: tmrvc-train/src/tmrvc_train/trainer.py [MOD]
対象: tmrvc-train/src/tmrvc_train/dataset/uclm_dataset.py [MOD]
```

- dataset: V4BootstrapCacheEntry から supervision_tier フィールドを読み込み、batch に含める
- trainer: loss 計算時に tier-based weight を適用:
  ```python
  TIER_WEIGHTS = {"A": 1.0, "B": 0.7, "C": 0.3, "D": 0.1}
  sample_weight = TIER_WEIGHTS[batch.supervision_tier]
  loss = loss * sample_weight
  ```
- unknown dimension (physical target が NaN or masked) を dense zero として扱わない:
  ```python
  physical_loss = F.mse_loss(pred, target, reduction='none')
  physical_loss = physical_loss * batch.physical_mask  # only supervised dims
  ```

テスト: tests/train/test_supervision_tier.py [NEW]
- Tier D sample の loss 寄与が全体の 10% 未満であること
- masked dimension が loss に寄与しないこと

完了条件: trainer.py が tier-aware weighting で動作し、テスト通過

### 3-2. Biological constraint regularization

```
対象: tmrvc-train/src/tmrvc_train/losses/bio_constraints.py [NEW]
対象: tmrvc-train/src/tmrvc_train/trainer.py [MOD]
定数: BIO_COVARIANCE_RANK=8, BIO_TRANSITION_PENALTY_WEIGHT=0.1
```

- low-rank covariance prior:
  ```python
  class CovariancePrior(nn.Module):
      """12-D physical controls の共起構造を学習する low-rank prior."""
      def __init__(self, d_physical=12, rank=8):
          self.L = nn.Parameter(torch.randn(d_physical, rank) * 0.01)
      def log_prob(self, x):  # x: [B, T, 12]
          cov = self.L @ self.L.T + 1e-4 * torch.eye(12)
          return MultivariateNormal(torch.zeros(12), cov).log_prob(x)
  ```
- frame-to-frame transition prior:
  ```python
  def transition_penalty(physical_trajectory):  # [B, T, 12]
      delta = physical_trajectory[:, 1:] - physical_trajectory[:, :-1]
      return (delta ** 2).mean() * BIO_TRANSITION_PENALTY_WEIGHT
  ```
- physically implausible combination penalty:
  ```python
  def implausible_penalty(physical):
      # 例: breathiness > 0.8 AND energy > 0.8 は物理的に非現実的
      breath = physical[..., 5]  # breathiness
      energy = physical[..., 2]  # energy_level
      violation = F.relu(breath - 0.7) * F.relu(energy - 0.7)
      return violation.mean()
  ```
- trainer.py の total loss に 3 項を追加

テスト: tests/train/test_bio_constraints.py [NEW]
- covariance prior が non-zero gradient を出すこと
- transition penalty が急激な変化にペナルティを与えること
- implausible combination でペナルティが上昇すること

完了条件: bio constraint loss が active で、implausible combination が 50% 以上減少する smoke test 通過

### 3-3. Full v4 loss composition の有効化

```
対象: tmrvc-train/src/tmrvc_train/trainer.py [MOD]
```

以下の 9 loss が全て active であることを確認・有効化:

1. codec token prediction loss — 既存
2. control token prediction loss — 既存
3. pointer progression loss — 既存
4. explicit physical supervision loss (12-D) — 既存だが 12-D 確認
5. acting latent regularization loss — 既存 (acting_latent.py)
6. disentanglement loss — 確認・有効化
7. speaker consistency loss — 確認・有効化
8. prosody prediction loss — 確認・有効化
9. semantic alignment loss — 確認・有効化

テスト: tests/train/test_loss_composition.py [NEW]
- 9 個全ての loss term が non-zero gradient を返すこと

完了条件: smoke test で 9 loss 全てが active

### 3-4. Enriched transcript training path

```
対象: tmrvc-train/src/tmrvc_train/dataset/uclm_dataset.py [MOD]
対象: tmrvc-train/src/tmrvc_train/trainer.py [MOD]
依存: Phase 2-1 (text encoder acting tag 拡張)
```

- dataset: V4BootstrapCacheEntry から enriched_transcript を読み込み、
  acting tag を token_ids に変換して phoneme_ids と混在させる
- training 時にランダムで切替:
  ```python
  if random.random() < 0.5:
      token_ids = self.tokenize_enriched(entry.enriched_transcript)
  else:
      token_ids = entry.phoneme_ids  # plain, no tags
  ```
- acting tag の有無で codec token prediction loss に差が出ることを期待

テスト: tests/train/test_enriched_transcript.py [NEW]
- 同じテキストで tag あり/なしの forward pass output が異なること
- tag なし学習でも TTS 品質が維持されること（regression test）

完了条件: A/B divergence test 通過

### 3-5. RL fine-tuning phase

```
対象: tmrvc-train/src/tmrvc_train/rl/ [NEW directory]
  - tmrvc-train/src/tmrvc_train/rl/reward.py [NEW]
  - tmrvc-train/src/tmrvc_train/rl/trainer_rl.py [NEW]
  - tmrvc-train/src/tmrvc_train/rl/config.py [NEW]
依存: Qwen3-ASR-1.7B (reward model として再利用)
依存: bootstrap の DSP/SSL extractors (physical compliance 計測)
```

- reward.py:
  ```python
  class InstructionFollowingReward:
      """生成音声を rich-transcription ASR で再転写し、inline tag の compliance を計測。"""
      def compute(self, generated_audio, input_enriched_transcript) -> float: ...

  class PhysicalComplianceReward:
      """生成音声の physical features を DSP/SSL で計測し、target との乖離を計算。"""
      def compute(self, generated_audio, physical_targets) -> float: ...

  class IntelligibilityReward:
      """plain transcript の WER/CER。"""
      def compute(self, generated_audio, reference_transcript) -> float: ...

  class NaturalnessGuard:
      """silence, noise, repetition 検出。"""
      def compute(self, generated_audio) -> float: ...
  ```
- trainer_rl.py:
  ```python
  class RLTrainer:
      """PPO or REINFORCE で UCLM codec token policy を fine-tune。"""
      def __init__(self, base_model, reward_fns, config): ...
      def train_step(self, batch) -> dict: ...  # returns reward breakdown
  ```
- supervised training 収束後に実行（concurrent ではない）
- RL 中に plain-text TTS quality が 5% 以上劣化したら early stop

テスト: tests/train/test_rl_phase.py [NEW]
- instruction-following score が RL 前より 20% 以上改善すること
- physical control monotonicity が RL 後も 0.8 以上であること
- plain-text naturalness の劣化が 5% 以内であること

完了条件: RL phase が動作し、3 つの guard condition を満たす

### 3-6. dev.py の v4 書き換え

```
対象: dev.py [MOD] (1080 行 → 全面書き換え、v3 互換不要)
```

v4 メニュー構成:

```
=== TMRVC v4 Development Menu ===
1. Bootstrap: raw corpus → train-ready cache
2. Training: v4 supervised training (12-D + 24-D + enriched transcript)
3. RL Fine-tuning: instruction-following RL phase
4. Dataset Management: corpus listing, tier summary, cache regeneration
5. Curation: v4 pipeline (ingest → score → export → validate)
6. Finalize: checkpoint promotion with v4 quality gates
7. Character Management: few-shot enrollment via backend API
8. Serve: v4 inference server startup
9. Integrity Check: v4 contract validation across core/train/export/serve/rust
```

主な変更:
- v3 の `REQUIRED_TRAINING_FIELDS` を v4 用に更新
  (voice_state_loss_weight → physical_12d_loss_weight + acting_latent_loss_weight)
- `tts_mode: pointer` は維持、`legacy_duration` を削除
- dataset 選択 UI を v4 cache ベースに変更
- supervision tier summary 表示を追加
- enriched transcript プレビュー機能を追加
- RL phase の開始/resume/status 確認メニューを追加
- legacy MFA path を全削除

完了条件: `uv run python dev.py` で v4 メニューが表示され、各メニュー項目が適切なモジュールに委譲される

---

## Phase 4: Serving Cutover (track_serving)

### 4-1. Bootstrap 出力の serving 統合

```
対象: tmrvc-serve/src/tmrvc_serve/schemas.py [MOD]
対象: tmrvc-serve/src/tmrvc_serve/uclm_engine.py [MOD]
```

- schemas.py: リクエストに pseudo_speaker_id, confidence-bearing controls を受け入れ
- uclm_engine.py: bootstrap 由来の speaker profile を few-shot enrollment として消費

完了条件: bootstrap で生成された pseudo speaker profile で inference が動作する

---

## Phase 5: Validation Gates (track_validation)

### 5-1. Bootstrap QC 閾値定義

```
対象: docs/design/acceptance-thresholds.md [MOD]
```

- 7 メトリクスに pass/fail 閾値を定義（Phase 1-9 の計測コードと対応）
- 例:
  - diarization_purity >= 0.85
  - speaker_cluster_consistency >= 0.80
  - overlap_rejection_precision >= 0.90
  - transcript_wer <= 0.15
  - physical_label_coverage >= 0.70
  - physical_confidence_calibration_error <= 0.20
  - language_coverage: 全対象言語で utterance_count >= 100

完了条件: acceptance-thresholds.md に v4 bootstrap 閾値が frozen として存在

### 5-2. Controllability metric harness

```
対象: tests/train/test_controllability.py [MOD]
対象: scripts/eval/measure_controllability.py [NEW]
```

- physical control response monotonicity: 各次元を 0→1 に sweep し、
  対応する acoustic feature の相関係数を計測 (> 0.8)
- physical calibration error: target vs measured の RMSE (< 0.15)
- edit locality: 1 フレーム範囲のみ変更し、他フレームへの影響を計測
- inline tag instruction-following rate: tag compliance percentage
- RL reward compliance: 4 reward の加重平均

完了条件: `python scripts/eval/measure_controllability.py` で 5 メトリクスが出力される

### 5-3. Fish S2 head-to-head protocol 実装

```
対象: scripts/eval/fish_s2_comparison.py [NEW]
依存: docs/design/evaluation-protocol.md (frozen protocol)
```

- 勝利軸 3 つ + guardrail 軸 3 つの自動評価スクリプト
- Fish S2 の public artifact から reference 音声を取得（手動 or API）
- TMRVC v4 で同一 prompt/text から生成
- 両者を blind 比較するための出力フォーマット

完了条件: 比較レポートのテンプレートが実行可能

---

## Phase 6: GUI Cutover (track_gui)

### 6-1. v4 Workshop パネル実装

```
対象: tmrvc-gui/src/tmrvc_gui/gradio_app.py [MOD]
```

- Basic physical panel: 6 sliders (pitch_level, energy_level, breathiness, pressedness, spectral_tilt, vocal_effort)
- Advanced physical panel: 12 sliders (全 12-D、default は closed)
- Acting macro panel: 4 sliders (intensity, instability, tenderness, tension)
- Acting prompt panel: テキストエリア（inline tag 入力可能）
- Reference-driven panel: reference audio upload → acting latent 推定
- Trajectory panel: trajectory_id 表示、inspect / patch / replay / transfer ボタン

各生成結果に provenance ラベルを表示:
- `[fresh compile]` / `[deterministic replay]` / `[cross-speaker transfer]` / `[patched replay]`

完了条件: Workshop タブに 6 パネルが存在し、provenance ラベルが全生成結果に表示される

---

## Dependency Summary

```
Phase 0 (survey) ─── 文書作業、blocking なし
  │
Phase 1 (bootstrap) ─── 新規 pipeline 実装 (最重量)
  │
Phase 2 (architecture) ─── text encoder 拡張
  │
Phase 3 (training) ─── loss / tier / bio / enriched / RL / dev.py
  │
Phase 4 (serving) ─── bootstrap 出力統合
  │
Phase 5 (validation) ─── 閾値定義 / harness 実装 / Fish S2 比較
  │
Phase 6 (gui) ─── パネル実装
```

Phase 1 が最も重く、全体の blocker。Phase 3 は Phase 1 と Phase 2 の完了後。
Phase 4-6 は Phase 3 と並行可能な部分がある。
