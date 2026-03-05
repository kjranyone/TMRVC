# TMRVC 整合性是正レポート (2026-03-05)

## 対象
- 2026-03-04 監査で検出された Critical / High / Medium 指摘の是正

## 実装済み
1. `tmrvc-serve` 起動ブロッカー解消
- `style_resolver.py` を再実装し、構文エラーを解消
- `UCLMEngine` の初期化/ロード契約を `app.py` と互換化
- `/tts` の `vars(metrics)` 例外を解消
- `ws_chat` の欠損依存 (`tts_engine`) を除去し、互換経路を実装
- `VCEnginePool` を `vc_streaming` 契約に合わせて再実装

2. `tmrvc-data` 構文エラー解消
- `features.py` の不正 import を修正
- `output_dim` 定数参照を明示化（`D_CONTENT_VEC`, `D_WAVLM_LARGE`）

3. 学習デフォルト設定の明示
- `configs/train_uclm.yaml.example` を新規追加
- `tmrvc-train-pipeline` で `configs/train_uclm.yaml` がない場合の `.example` フォールバックを追加

## 検証結果
- `python -m compileall -q tmrvc-core/src tmrvc-data/src tmrvc-train/src tmrvc-export/src tmrvc-serve/src tmrvc-gui/src`  
  → OK
- `uv run pytest tests/serve/test_serve.py tests/data/test_wavlm_extractor.py -q`  
  → 9 passed
- `timeout 5s uv run tmrvc-serve --port 8010`  
  → 起動/停止を確認（チェックポイント未指定時は warning のみ）

## 変更ファイル
- [`tmrvc-data/src/tmrvc_data/features.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-data/src/tmrvc_data/features.py)
- [`tmrvc-serve/src/tmrvc_serve/style_resolver.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/style_resolver.py)
- [`tmrvc-serve/src/tmrvc_serve/uclm_engine.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/uclm_engine.py)
- [`tmrvc-serve/src/tmrvc_serve/vc_engine_pool.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/vc_engine_pool.py)
- [`tmrvc-serve/src/tmrvc_serve/routes/ws_chat.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/routes/ws_chat.py)
- [`tmrvc-serve/src/tmrvc_serve/routes/tts.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-serve/src/tmrvc_serve/routes/tts.py)
- [`tmrvc-train/src/tmrvc_train/cli/train_pipeline.py`](/home/kojirotanaka/kjranyone/TMRVC/tmrvc-train/src/tmrvc_train/cli/train_pipeline.py)
- [`configs/train_uclm.yaml.example`](/home/kojirotanaka/kjranyone/TMRVC/configs/train_uclm.yaml.example)
