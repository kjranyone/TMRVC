# Stream 2: Constants Auto-generation Pipeline

## Goal

`configs/constants.yaml` を Single Source of Truth として、Python / Rust / C++ の定数を自動生成する。
現在は Python (`constants.py`) のみ YAML から読み込んでいるが、Rust (`constants.rs`) と C++ (`constants.h`) は手動同期。

## Current State

| Language | File | Method | Status |
|---|---|---|---|
| Python | `tmrvc-core/src/tmrvc_core/constants.py` | Runtime YAML load | Done (自動) |
| Rust | `tmrvc-rt/src/engine/constants.rs` | Hard-coded | Manual sync required |
| C++ | `tmrvc-engine/include/tmrvc/constants.h` | Not yet created | Planned |

`scripts/generate_constants.py` は architecture.md で言及されているが未作成。

## Tasks

### 2.1 generate_constants.py 作成

```python
# scripts/generate_constants.py
# Usage: uv run python scripts/generate_constants.py

Input:  configs/constants.yaml
Output:
  - tmrvc-rt/src/engine/constants.rs     (Rust)
  - tmrvc-engine/include/tmrvc/constants.h (C++ header)
```

**Rust テンプレート:**
```rust
// AUTO-GENERATED from configs/constants.yaml — DO NOT EDIT
pub const SAMPLE_RATE: u32 = {{ sample_rate }};
pub const HOP_LENGTH: usize = {{ hop_length }};
pub const N_MELS: usize = {{ n_mels }};
// ...
```

**C++ テンプレート:**
```cpp
// AUTO-GENERATED from configs/constants.yaml — DO NOT EDIT
#pragma once
namespace tmrvc {
constexpr int kSampleRate = {{ sample_rate }};
constexpr int kHopLength = {{ hop_length }};
constexpr int kNMels = {{ n_mels }};
// ...
} // namespace tmrvc
```

### 2.2 Derived Constants

YAML にない派生定数も生成する:

```yaml
# constants.yaml にある値
hop_length: 240
n_fft: 1024
window_length: 960

# 派生値 (generate_constants.py で計算)
# causal_pad = window_length - hop_length = 720
# n_fft_bins = n_fft // 2 + 1 = 513
# lora_delta_total = d_converter_hidden * 64 = 24576
```

### 2.3 CI Validation

```bash
# pre-commit hook or CI check:
uv run python scripts/generate_constants.py --check
# → 生成結果が既存ファイルと一致しなければ exit 1
```

### 2.4 tmrvc-rt constants.rs の現状更新

現在の `tmrvc-rt/src/engine/constants.rs` を生成スクリプトの出力に置き換える。
手動メンテナンスのコメントを削除し、`AUTO-GENERATED` ヘッダを追加。

## Implementation Order

```
2.1 generate_constants.py 作成 (Rust + C++ 出力)
2.2 tmrvc-rt/constants.rs を生成出力に置換、動作確認
2.3 --check モード追加
2.4 (Optional) pre-commit hook 設定
```

## Acceptance Criteria

- [ ] `uv run python scripts/generate_constants.py` で Rust + C++ ファイルが生成される
- [ ] 生成された Rust constants が tmrvc-rt の既存値と一致
- [ ] `--check` モードで差分検出が動作
- [ ] constants.yaml を変更 → regenerate → Rust/C++ に反映される
