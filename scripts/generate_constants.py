#!/usr/bin/env python3
"""Generate Python, C++, and Rust constant files from configs/constants.yaml.

Usage:
    python scripts/generate_constants.py          # generate all
    python scripts/generate_constants.py --check   # verify files are up-to-date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "configs" / "constants.yaml"
PY_OUT = REPO_ROOT / "tmrvc-core" / "src" / "tmrvc_core" / "_generated_constants.py"
CPP_OUT = REPO_ROOT / "tmrvc-engine" / "include" / "tmrvc" / "constants.h"
RUST_OUT = REPO_ROOT / "tmrvc-engine-rs" / "src" / "constants.rs"

# Keys relevant to the Rust/C++ runtime engine (exclude training-only keys).
_RUNTIME_KEYS = {
    "sample_rate", "n_fft", "hop_length", "window_length",
    "n_mels", "mel_fmin", "mel_fmax", "n_freq_bins", "log_floor",
    "d_content", "d_speaker", "n_ir_params", "d_converter_hidden",
    "d_vocoder_features", "ir_update_interval",
    "lora_rank", "lora_delta_size",
    "content_encoder_state_frames", "ir_estimator_state_frames",
    "converter_state_frames", "vocoder_state_frames",
    "max_lookahead_hops", "converter_hq_state_frames",
    "hq_threshold_q", "crossfade_frames",
}

# Rust name overrides (YAML key → Rust const name) for backward compat.
_RUST_NAME_MAP: dict[str, str] = {
    "content_encoder_state_frames": "CONTENT_ENC_STATE_FRAMES",
    "ir_estimator_state_frames": "IR_EST_STATE_FRAMES",
}


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------

def _py_value(v: object) -> str:
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        inner = ", ".join(_py_value(x) for x in v)
        return f"[{inner}]"
    return str(v)


def generate_python(cfg: dict) -> str:
    lines = [
        '"""Auto-generated constants — DO NOT EDIT MANUALLY.',
        "",
        "Run: python scripts/generate_constants.py",
        '"""',
        "",
    ]
    for key, val in cfg.items():
        name = key.upper()
        lines.append(f"{name} = {_py_value(val)}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# C++
# ---------------------------------------------------------------------------

def _cpp_type(v: object) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    return "auto"


def _cpp_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v}f"
    if isinstance(v, list):
        inner = ", ".join(_cpp_value(x) for x in v)
        return f"{{{inner}}}"
    return str(v)


def generate_cpp(cfg: dict) -> str:
    lines = [
        "// Auto-generated constants — DO NOT EDIT MANUALLY.",
        "// Run: python scripts/generate_constants.py",
        "",
        "#pragma once",
        "",
        "#include <array>",
        "",
        "namespace tmrvc {",
        "",
    ]
    for key, val in cfg.items():
        name = key.upper()
        ctype = _cpp_type(val)
        cval = _cpp_value(val)
        if isinstance(val, list):
            lines.append(
                f"constexpr std::array<int, {len(val)}> {name} = {cval};"
            )
        else:
            lines.append(f"constexpr {ctype} {name} = {cval};")
    lines += [
        "",
        "}  // namespace tmrvc",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

def _rust_type(v: object) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, float):
        return "f32"
    if isinstance(v, int):
        return "usize"
    return "usize"


def _rust_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        inner = ", ".join(_rust_value(x) for x in v)
        return f"[{inner}]"
    return str(v)


def generate_rust(cfg: dict) -> str:
    lines = [
        "// Auto-generated from configs/constants.yaml — DO NOT EDIT MANUALLY.",
        "// Run: python scripts/generate_constants.py",
        "",
        "#![allow(dead_code)]",
        "",
        "// --- Audio parameters ---",
    ]

    sections = {
        "audio": [
            "sample_rate", "n_fft", "hop_length", "window_length",
            "n_mels", "mel_fmin", "mel_fmax", "n_freq_bins", "log_floor",
        ],
        "model": [
            "d_content", "d_speaker", "n_ir_params",
            "d_converter_hidden", "d_vocoder_features",
        ],
        "inference": ["ir_update_interval"],
        "lora": ["lora_rank", "lora_delta_size"],
        "state": [
            "content_encoder_state_frames", "ir_estimator_state_frames",
            "converter_state_frames", "vocoder_state_frames",
        ],
        "hq": [
            "max_lookahead_hops", "converter_hq_state_frames",
            "hq_threshold_q", "crossfade_frames",
        ],
    }

    section_headers = {
        "audio": "// --- Audio parameters ---",
        "model": "\n// --- Model dimensions ---",
        "inference": "\n// --- Inference parameters ---",
        "lora": "\n// --- LoRA parameters ---",
        "state": "\n// --- State tensor context lengths ---",
        "hq": "\n// --- Lookahead / HQ mode ---",
    }

    # Remove the initial audio header since we add it per section
    lines.pop()

    for section, keys in sections.items():
        lines.append(section_headers[section])
        for key in keys:
            if key not in cfg:
                continue
            val = cfg[key]
            rust_name = _RUST_NAME_MAP.get(key, key.upper())
            rtype = _rust_type(val)
            rval = _rust_value(val)
            if isinstance(val, list):
                lines.append(
                    f"pub const {rust_name}: [{_rust_type(val[0])}; {len(val)}] = {rval};"
                )
            else:
                lines.append(f"pub const {rust_name}: {rtype} = {rval};")

    # Derived constants
    lines.append("")
    lines.append("// --- Derived constants ---")
    lines.append("pub const RING_BUFFER_CAPACITY: usize = 4096;")
    lines.append("")
    lines.append(
        "// Past context for causal windowing: WINDOW_LENGTH - HOP_LENGTH"
    )
    lines.append(
        "pub const PAST_CONTEXT: usize = WINDOW_LENGTH - HOP_LENGTH;"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _write_if_changed(path: Path, content: str) -> bool:
    """Write content to path. Return True if file was changed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.write_text(content, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Verify generated files are up-to-date (exit 1 if stale).",
    )
    args = parser.parse_args()

    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)

    targets: list[tuple[Path, str]] = [
        (PY_OUT, generate_python(cfg)),
        (RUST_OUT, generate_rust(cfg)),
    ]

    # C++ only if directory exists
    if CPP_OUT.parent.exists():
        targets.append((CPP_OUT, generate_cpp(cfg)))

    if args.check:
        stale = []
        for path, expected in targets:
            if not path.exists():
                stale.append(f"  MISSING: {path}")
            elif path.read_text(encoding="utf-8") != expected:
                stale.append(f"  STALE:   {path}")
        if stale:
            print("Constants are out of date:")
            print("\n".join(stale))
            print("\nRun: python scripts/generate_constants.py")
            sys.exit(1)
        else:
            print("All generated constants are up-to-date.")
            sys.exit(0)

    for path, content in targets:
        changed = _write_if_changed(path, content)
        status = "Updated" if changed else "Up-to-date"
        print(f"{status}: {path}")


if __name__ == "__main__":
    main()
