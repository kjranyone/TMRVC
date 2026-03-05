# TMRVC 整合性監査レポート (2026-03-04)

## Update (2026-03-05)
- 本レポートの Critical-1〜4 / High-1〜2 は是正済み。
- Medium-1（学習デフォルト設定未配置）も是正済み。
- 是正内容:
  - `tmrvc-serve/style_resolver.py` の構文崩壊を復旧
  - `tmrvc-data/features.py` の不正 import を修正
  - `UCLMEngine` / `app.py` / `VCEnginePool` / `vc_streaming` 契約の互換化
  - `ws_chat` の欠損依存 (`tts_engine`) を除去し互換経路を実装
  - `/tts` の `vars(metrics)` 実行時例外を解消
  - `configs/train_uclm.yaml.example` を追加し、`train_pipeline` に `.example` フォールバックを追加
- 再検証:
  - `python -m compileall -q tmrvc-*/src` → 全パッケージ OK
  - `uv run tmrvc-serve --port 8010` (5秒スモーク) → 起動/停止 OK
  - `uv run pytest tests/serve/test_serve.py -q` → 3 passed

以下の指摘一覧は「監査実施時点（2026-03-04）」の記録として保持する。

## 1. 監査対象
- データセット設定 (`configs/datasets.yaml`) と実体データの一致
- 学習フロー (`dev.py` / `tmrvc-train-pipeline` / `tmrvc-train-uclm`) の整合性
- 推論サーバー (`tmrvc-serve`) の起動可能性と API 接続整合性
- パッケージ単位の構文健全性（`compileall`）

## 2. 実施コマンド（抜粋）
- `uv run python -m compileall -q tmrvc-*/src`
- `uv run tmrvc-train-pipeline --help`
- `uv run tmrvc-train-uclm --help`
- `uv run tmrvc-serve --port 8010`
- `uv run python - <<'PY' ... get_adapter(...).iter_utterances(...) ... PY`

## 3. 結論サマリ
- 学習側（データセット選択、キャッシュ選定、学習データ混入防止、品質ゲート）は概ね整合している。
- 推論サーバー側は **現状そのままでは起動不能**（構文エラー + インターフェース不一致）。
- `tmrvc-data` にも構文エラーが1件あり、関連機能は使用不能。
- したがって「サーバーは存在するか？」への答えは **Yes（実装・CLIはある）が、現状は壊れており運用不可**。

## 4. 事実ベースの確認結果

### 4.1 データセット実体チェック（enabled のみ）
- `vctk`: 44,455 utterances / 110 speakers
- `jvs`: 12,997 utterances / 100 speakers
- `tsukuyomi`: 100 utterances / 1 speaker
- `moe_multispeaker_voices`: 29,589 utterances / 1 speaker

`configs/datasets.yaml` の `raw_dir` 指定と現物の走査結果は、現時点では一致。

### 4.2 コンパイル整合性
- `tmrvc-core/src`: OK
- `tmrvc-train/src`: OK
- `tmrvc-export/src`: OK
- `tmrvc-gui/src`: OK
- `tmrvc-data/src`: **NG**
- `tmrvc-serve/src`: **NG**

## 5. 指摘事項（重大度順）

### Critical-1: `tmrvc-serve` が起動時に構文エラーで停止
- 事象: `uv run tmrvc-serve --port 8010` が即時クラッシュ
- 原因: [`tmrvc-serve/src/tmrvc_serve/style_resolver.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/style_resolver.py:103) の `IndentationError`
- 影響: FastAPI アプリ import 自体が失敗し、全エンドポイント利用不可

### Critical-2: `tmrvc-data` に構文エラー
- 原因: [`tmrvc-data/src/tmrvc_data/features.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-data/src/tmrvc_data/features.py:12)
  - `from tmrvc_core.constants import D_MODEL, 1024, HOP_LENGTH, SAMPLE_RATE`
- 影響: `tmrvc_data.features` の import 不可

### Critical-3: サーバー内部 API が相互不一致
- [`app.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/app.py:53) は `UCLMEngine(uclm_checkpoint=..., codec_checkpoint=..., device=...)` を呼ぶ
- しかし [`uclm_engine.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/uclm_engine.py:45) の `__init__` は `(device, d_model)` のみ
- [`app.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/app.py:58) は `load_models()` 引数なし呼び出し
- しかし [`uclm_engine.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/uclm_engine.py:57) は `load_models(uclm_path, codec_path)` 必須
- 影響: 構文エラー解消後もエンジン初期化で失敗

### Critical-4: VC ルートと VC プールの契約不一致
- [`vc_streaming.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/routes/vc_streaming.py:46) は `VCEnginePool(..., codec_checkpoint=..., max_gpu_inference=..., session_timeout_sec=...)` を渡す
- しかし [`vc_engine_pool.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/vc_engine_pool.py:74) の `__init__` はそれらを受け取らない
- [`vc_engine_pool.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/vc_engine_pool.py:63) は `load_from_combined_checkpoint` を呼ぶが `UCLMEngine` に未実装
- 影響: VC streaming 経路は実行不能

### High-1: `/tts` 応答整形で例外の可能性
- [`tts.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/routes/tts.py:97) で `vars(metrics)` を呼ぶ
- [`uclm_engine.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/uclm_engine.py:185) の戻り値 `metrics` は `dict`
- 影響: 実行時 `TypeError` の可能性

### High-2: `ws_chat` ルートが存在しないモジュールへ依存
- [`ws_chat.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/routes/ws_chat.py:37) が `tmrvc_serve.tts_engine` を import
- しかし `tmrvc-serve/src/tmrvc_serve/tts_engine.py` は存在しない
- 同ファイルは `engine.synthesize_sentences(...)` も前提だが、現 `UCLMEngine` に当該メソッドなし

### Medium-1: 学習設定ファイルのデフォルトパスが未配置
- [`train_pipeline.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-train/src/tmrvc_train/cli/train_pipeline.py:169) は `configs/train_uclm.yaml` を既定値にする
- ただし現リポジトリには当該ファイルが存在しない
- 影響: 警告後にデフォルト値実行となり、意図しない学習条件で走るリスク

## 6. 学習系の整合性評価（今回修正分）
- dataset 別 `raw_dir` 解決、`type/adapter_type` 互換、`--datasets` による学習対象固定は有効
- `dev.py` の full 学習（メニュー1）は、enabled dataset スコープでキャッシュ管理できる
- 学習前品質ゲート（欠損率/話者数/トークン範囲）は実装済み

## 7. 推奨修正順（運用復旧優先）
1. `tmrvc-serve/style_resolver.py` の構文・参照整合を復旧（起動ブロッカー解除）
2. `tmrvc-data/features.py` の不正 import を修正
3. `UCLMEngine` / `app.py` / `vc_engine_pool.py` / `vc_streaming.py` のインターフェースを一本化
4. `ws_chat.py` の未実装依存（`tts_engine`, `synthesize_sentences`）を実装 or 経路停止
5. `configs/train_uclm.yaml` を配置し、`dev.py` 実行時設定を明示化
6. CI に最低限追加:
   - `python -m compileall -q tmrvc-*/src`
   - `uv run tmrvc-train-pipeline --help`
   - `uv run tmrvc-serve --port 0`（短時間 smoke）

## 8. 判定（監査時点: 2026-03-04）
- 学習 CLI 系: **実運用可能（改善済み）**
- サーバー系: **現状は非運用（要復旧）**
- 全体: **整合性は部分的に成立、Serve/Data に重大欠陥あり**
