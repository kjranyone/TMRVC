# Stream 1: GUI Worker Integration — DONE

## Goal

tmrvc-gui の 4 worker を実際の tmrvc-train / tmrvc-data / tmrvc-export に接続する。

## Status: COMPLETE

| Worker | File | Status |
|---|---|---|
| `train_worker.py` | `tmrvc-gui/src/tmrvc_gui/workers/train_worker.py` | Done (local mode wired, SSH placeholder) |
| `data_worker.py` | `tmrvc-gui/src/tmrvc_gui/workers/data_worker.py` | Done (real preprocessing pipeline) |
| `export_worker.py` | `tmrvc-gui/src/tmrvc_gui/workers/export_worker.py` | Done (FP32 + INT8 quantize) |
| `eval_worker.py` | `tmrvc-gui/src/tmrvc_gui/workers/eval_worker.py` | Done (SECS, UTMOS, F0 correlation) |

GUI Pages (DataPrepPage, TeacherTrainPage, OnnxExportPage) もワーカーに接続済み。

## Tasks

### 1.1 data_worker.py (最も簡単)

```
Current (stub):
  _run_step_stub(step_name) → time.sleep()

Target:
  from tmrvc_data.preprocessing import resample_audio, normalize_loudness, trim_silence
  from tmrvc_data.features import extract_content_vec, extract_f0

  Each step maps to actual tmrvc_data function call.
  Progress reporting via self.progress.emit(percent).
```

**Mapping:**
| Step | tmrvc_data function |
|---|---|
| resample | `preprocessing.resample_audio()` |
| normalize | `preprocessing.normalize_loudness()` |
| vad_trim | `preprocessing.trim_silence()` |
| segment | `preprocessing.segment_audio()` |
| features | `features.ContentVecExtractor` + `features.F0Extractor` |

### 1.2 export_worker.py

```
Current (stub):
  time.sleep(0.1)  # simulate FP32 export
  time.sleep(0.05) # simulate INT8 quantize

Target:
  from tmrvc_export.export_onnx import (
      export_content_encoder, export_converter,
      export_vocoder, export_ir_estimator, export_speaker_encoder,
  )
  from tmrvc_export.quantize import quantize_model

  1. Load checkpoint → instantiate model
  2. Call export_xxx(model, output_path)
  3. Optional: quantize_model(fp32_path, int8_path)
```

**Key consideration:** checkpoint → model instantiation ロジックが必要。
tmrvc_train に `load_student_models(checkpoint_path) -> dict[str, nn.Module]` ユーティリティを追加するか検討。

### 1.3 eval_worker.py

```
Current (stub):
  return {"utmos": 3.85, "speaker_sim": 0.91, ...}  # dummy

Target:
  from tmrvc_train.eval_metrics import compute_secs, f0_correlation, utmos_proxy

  1. Load model from checkpoint
  2. Generate converted mel for eval set
  3. Compute each metric
  4. Return real values
```

### 1.4 train_worker.py (Local mode)

```
Current (stub):
  for step in range(total_steps):
      time.sleep(0.01)  # simulate training
      self.metric.emit({"loss": random_value})

Target:
  from tmrvc_train.trainer import TeacherTrainer

  1. Build trainer from config
  2. for step, metrics in trainer.train_iter():
         self.progress.emit(step / total_steps * 100)
         self.metric.emit(metrics)
         if self.is_cancelled: break
  3. trainer.save_checkpoint()
```

**Note:** trainer.py に `train_iter()` ジェネレータを追加する必要がある可能性。
現状の `train_epoch()` はエポック単位なので、ステップ単位で yield するインターフェースが必要。

### 1.5 train_worker.py (SSH mode) — 後回し可

SSH 経由のリモート学習は複雑。優先度低。
- paramiko or subprocess + ssh で接続
- tmux/screen でセッション管理
- ログファイルの tail -f でメトリクス取得

## Implementation Order

```
1.1 data_worker    → 最も単純、既存関数の呼び出しのみ
1.2 export_worker  → checkpoint ロードのユーティリティ作成が必要
1.3 eval_worker    → export_worker と同じ checkpoint ロードを共有
1.4 train_worker   → trainer.py に train_iter() 追加が必要
1.5 SSH mode       → 後回し
```

## Acceptance Criteria

- [x] data_worker: 実際の前処理パイプラインが GUI から実行できる
- [x] export_worker: checkpoint → ONNX export → INT8 quantize が GUI から実行できる
- [x] eval_worker: 実際の評価メトリクスが表示される
- [x] train_worker: Teacher 学習が GUI から起動・停止できる
- [x] 各 worker のエラーハンドリング (例外 → error signal → GUI 表示)
- [x] 155 テスト全 pass (既存 106 + 新規 49)
- [x] GUI ページ (DataPrepPage, TeacherTrainPage, OnnxExportPage) がワーカーに接続済み

## Dependencies

- tmrvc-train (done)
- tmrvc-data (done)
- tmrvc-export (done)
- checkpoint ロードユーティリティ (新規作成)
