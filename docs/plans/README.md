# TMRVC Next Phase Plans

Created: 2026-02-17

## Current Status

| Package | Status | Notes |
|---|---|---|
| **tmrvc-core** | Done | Constants, mel, types. 44 tests pass |
| **tmrvc-data** | Done | Preprocessing, features, DataLoader |
| **tmrvc-train** | Done | Models, training loops, distillation, 62 new tests (106 total) |
| **tmrvc-export** | Done | ONNX export, quantize, parity verification |
| **tmrvc-gui** | Done | Pages + workers wired, 155 tests pass |
| **tmrvc-rt** | Done | Rust standalone app (reference implementation) |
| **tmrvc-engine** | Not Started | C++ streaming engine (JUCE-free) |
| **tmrvc-plugin** | Not Started | JUCE VST3 wrapper |

## Parallel Work Streams

以下は互いに独立して進められる。依存関係がある場合は明記。

| # | Stream | File | Status | Blocked By | Estimated Effort |
|---|---|---|---|---|---|
| 1 | GUI Worker Integration | [gui-integration.md](gui-integration.md) | DONE | None | 1-2 sessions |
| 2 | Constants Auto-generation | [constants-autogen.md](constants-autogen.md) | DONE | None | 1 session |
| 3 | Training Execution | [training-execution.md](training-execution.md) | Pending | Data download | Days (GPU) |
| 4 | C++ Engine | [cpp-engine.md](cpp-engine.md) | Pending | None (Rust ref exists) | 3-5 sessions |
| 5 | VST3 Plugin | [vst3-plugin.md](vst3-plugin.md) | Pending | Stream #4 | 2-3 sessions |

## Dependency Graph

```
Stream 1 (GUI)          ─── independent ───
Stream 2 (constants)    ─── independent ───
Stream 3 (training)     ─── independent (needs GPU + data) ───
Stream 4 (C++ engine)   ─── independent (tmrvc-rt is reference) ───
Stream 5 (VST3 plugin)  ─── depends on Stream #4 ───
```

## Priority Recommendation

**即座に着手可能:**
- Stream 1 (GUI integration): tmrvc-train/export が完成したので接続するだけ
- Stream 2 (constants autogen): YAML → Python/Rust/C++ の自動生成パイプライン

**GPU 確保後:**
- Stream 3 (training): データ準備 → Phase 0 → Phase 1 → Phase 2

**任意タイミング:**
- Stream 4 (C++ engine): tmrvc-rt が完全な参考実装として存在。VST3 が不要なら後回し可
- Stream 5 (VST3): tmrvc-engine 完成後
