# TMRVC オペレーターガイド

このガイドは、TMRVC v3 の日常運用を WebUI 中心で行うオペレーター向けの手順書である。
技術仕様の詳細は `docs/design/` 配下の各設計文書を参照すること。

## 前提

- `tmrvc-serve` と `tmrvc-gui` が起動済みであること（`docs/user-manual.md` § 2.1 参照）
- チェックポイントが `checkpoints/uclm/uclm_latest.pt` と `checkpoints/codec/codec_latest.pt` に配置済みであること

---

## 1. データセットの登録とインジェスト

### 1.1 WebUI から登録する

1. `Dataset Manager` ページを開く
2. **新規データセット** ボタンから dataset を作成する
3. 音声ファイル（WAV / FLAC）をアップロードする
4. **合法性 (Legality)** と **Provenance** を設定する
   - 合法性が未設定のデータは後続のキュレーションや学習に進めない
5. 言語を設定する（`ja`, `en`, `zh`, `ko` など）

### 1.2 CLI から登録する場合（開発者向け）

```bash
# dev.py メニュー 13: インジェスト
python dev.py  # → 13 を選択

# または直接
uv run tmrvc-curate ingest --dataset-dir data/raw/my_dataset
```

---

## 2. キュレーション

キュレーションは 4 段階で進行する: インジェスト → スコアリング → エクスポート → 検証。

### 2.1 スコアリングと昇格判定

1. `Dataset Manager` ページで対象 dataset を選択する
2. **キュレーション開始** を押す（バックエンドで `tmrvc-curate score` が実行される）
3. 進捗は `Dataset Manager` の health dashboard で確認できる

### 2.2 監査と修正（Curation Auditor）

1. `Curation Auditor` ページを開く
2. 各アイテムの transcript / speaker / language を確認・修正する
3. **Promote** または **Reject** を選択する
   - Promote: 学習対象に昇格
   - Reject: 品質不足として除外
4. 修正内容は `metadata_version` による楽観ロックで競合管理される

### 2.3 エクスポート

1. `Dataset Manager` ページで **エクスポート** を押す
2. promoted subset が学習用 cache に materialize される
3. エクスポート結果は **ダウンロード** ボタンから取得可能（`/artifacts/` 経由）

### 2.4 検証レポート

1. `Dataset Manager` の **検証レポート** ボタンで、エクスポート後の整合性を確認する
2. レポートには以下が含まれる:
   - coverage（カバレッジ率）
   - voice_state observed ratio
   - confidence summary
   - suprasegmental feature alignment

### 2.5 CLI での代替操作

```bash
python dev.py  # → 14: スコアリング, 15: エクスポート, 16: 検証, 17: サマリー
```

---

## 3. Drama Workshop（TTS 試聴）

### 3.1 基本操作

1. `Drama Workshop` ページを開く
2. テキストを入力する
3. 話者を選択する:
   - **Casting Gallery** から既存の `SpeakerProfile` を選ぶ、または
   - **Reference Audio** をアップロードして on-the-fly で話者を指定する
4. **生成** を押して音声を試聴する

### 3.2 制御パラメータ

| パラメータ | 説明 | 範囲 |
|---|---|---|
| `pace` | 発話速度 | 0.5 – 3.0 |
| `hold_bias` | 現在の音素を保持する傾向 | -1.0 – 1.0 |
| `boundary_bias` | 音素境界を強調する傾向 | -1.0 – 1.0 |
| `voice_state` | 8次元の物理的音声状態 | 各次元 float |
| `cfg_scale` | Classifier-Free Guidance 強度 | 0.5 – 5.0 |
| `cfg_mode` | CFG モード (`off` / `full` / `lazy` / `distilled`) | — |
| `emotion` | 感情オーバーライド | テキスト |
| `hint` | 演技ヒント（ソフトガイダンス） | テキスト |

### 3.3 Take 管理

- 生成結果は **take** として保存される
- take 間で比較・ランキング・ノート付けが可能
- 最良の take を **採用** として選択できる

### 3.4 表示メトリクス

- Waveform preview
- Pointer timeline（音素ポインタの進行）
- Voice state trace
- Control token trace
- RTF / 生成時間

---

## 4. SpeakerProfile と Casting Gallery

### 4.1 プロファイル作成

1. `Speaker Enrollment` ページを開く
2. 3〜10 秒のリファレンス音声をアップロードする
3. （任意）リファレンスのテキスト書き起こしを入力する
4. **プロファイル作成** を押す
5. `speaker_profile_id` が発行される

### 4.2 プロファイルの利用

- `Drama Workshop` の **Casting Gallery** からプロファイルを選択して TTS に使う
- API では `speaker_profile_id` フィールドで指定する
- プロファイルには speaker embedding と prompt codec tokens が含まれる

### 4.3 キャッシュ無効化

- prompt encoder のモデルが更新された場合、既存プロファイルの `prompt_codec_tokens` は無効になる
- `Speaker Enrollment` ページから再エンコードを実行する

仕様の詳細は `docs/design/speaker-profile-spec.md` を参照する。

---

## 5. Evaluation Arena（ブラインド評価）

### 5.1 評価セッションの設定

1. `Evaluation Arena` ページを開く
2. 評価タイプを選択する:
   - **Blind A/B**: 2 システム間のペア比較
   - **MOS**: 5 段階の絶対評価
3. 評価対象のシステムを設定する:
   - TMRVC v3（現行）
   - v2 legacy（内部回帰ガード用）
   - 外部ベースライン（SOTA 主張用、`docs/design/external-baseline-registry.md` で固定）

### 5.2 外部ベースラインとの比較

外部ベースラインは評価開始前に固定（freeze）されている必要がある。

- **Primary**: `Qwen3-TTS-12Hz-1.7B-Base` @ `fd4b254`
- **Secondary**: `Fun-CosyVoice3-0.5B-2512` @ `29e01c4`

ベースラインの変更は `docs/design/external-baseline-registry.md` に新エントリを追加し、evaluation protocol version を bump する必要がある。「or newer successor」は禁止。

### 5.3 評価の実行

1. 評価プロンプトセット（`tmrvc_eval_public_v1` 等）を選択する
2. 評価者（rater）を割り当てる
3. QC 設定（重複率、最小評価数）を確認する
4. **評価開始** を押す
5. 各評価者はブラインドで音声ペアを聴き、優劣を選択する

### 5.4 結果の確認

- 信頼区間付きの統計結果が表示される
- sign-off 基準: win rate ≥ 0.55 または p < 0.05

---

## 6. voice_state 監督フロー

`voice_state` は 8 次元の物理的音声状態ベクトル（息遣い、声帯の緊張度など）であり、TTS の制御性を向上させる。

### 6.1 疑似ラベル生成（キュレーション時）

1. キュレーションのスコアリング段階（§ 2.1）で、音声から `voice_state` 疑似ラベルが自動抽出される
2. 各疑似ラベルには **confidence** スコアが付与される
3. 抽出結果は `voice_state.npy` / `voice_state_mask.npy` / `voice_state_confidence.npy` として保存される

### 6.2 エクスポート（Worker 10）

エクスポート時に以下のファイルが学習用 cache に出力される:

| ファイル | 内容 |
|---|---|
| `voice_state.npy` | `[T, 8]` 疑似ラベル値 |
| `voice_state_mask.npy` | `[T]` 有効フレームマスク |
| `voice_state_confidence.npy` | `[T]` confidence スコア |
| `voice_state_provenance.json` | 抽出手法・バージョン情報 |

### 6.3 学習での消費（Worker 02）

- Trainer は `voice_state_loss_weight > 0` の場合に voice_state 監督を有効にする
- mask された部分のみ loss を計算する（partial supervision）
- confidence が低いフレームは自動的に重み付けが軽減される

```bash
uv run tmrvc-train-uclm \
  --tts-mode pointer \
  --voice-state-loss-weight 0.3
```

### 6.4 制御性の検証（Worker 06 / Worker 11）

- `voice_state_label_utility_score`: ラベルあり vs なし（ablation）の制御性差分
- `voice_state_calibration_error`: 疑似ラベルの confidence と実際の誤差の整合性
- キュレーション検証レポート（§ 2.4）で coverage と confidence summary を確認する

### 6.5 Drama Workshop での利用

- `Drama Workshop` で 8 次元の `voice_state` スライダーを操作して音声を生成する
- API では `explicit_voice_state` フィールド（8 要素の float リスト）で指定する
- `delta_voice_state` で差分指定も可能

---

## 7. 学習の監視

### 7.1 Training Monitor

1. `Training Monitor` ページを開く
2. 以下を確認する:
   - 損失推移（total, TTS, VC, pointer, voice_state, cfg_distillation）
   - カリキュラムステージ（Stage 1 → 2 → 3）
   - Quality gate の通過状況

### 7.2 カリキュラムステージ

| ステージ | 内容 |
|---|---|
| Stage 1 | Base LM（基本コーデック予測） |
| Stage 2 | Alignment & Pointer（ポインタ学習） |
| Stage 3 | Drama & Dialogue（CFG、演技的特徴） |

Stage 3 では **anti-forgetting replay** により Stage 1/2 のデータが混合される（`stage3_replay_mix_ratio` で制御、default 0.2）。

---

## 8. ONNX エクスポートとデプロイ

1. `ONNX Export` ページを開く
2. 対象モデルを選択してエクスポートする
3. Python / Rust / ONNX 間の parity テストを確認する
4. エクスポート後のアーティファクトは `Server Control` からデプロイ可能

---

## 参照文書

| 文書 | 内容 |
|---|---|
| `docs/user-manual.md` | クイックリファレンス |
| `TRAIN_GUIDE.md` | 学習フロー詳細 |
| `docs/design/gui-design.md` | UI 設計仕様 |
| `docs/design/speaker-profile-spec.md` | SpeakerProfile 仕様 |
| `docs/design/curation-contract.md` | キュレーション契約 |
| `docs/design/external-baseline-registry.md` | 外部ベースライン登録簿 |
| `docs/design/evaluation-protocol.md` | 評価プロトコル |
| `docs/design/dataset-preparation-flow.md` | データセット作成フロー |
