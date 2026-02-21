#!/usr/bin/env bash
# Phase 1a preprocessing pipeline: ContentVec (768d) → WavLM (1024d)
#
# Usage:
#   bash scripts/preprocess_phase1a.sh [--device xpu] [--clean-old]
#
# This script:
# 1. Optionally removes old 768d caches
# 2. Re-preprocesses all available datasets with WavLM-large (1024d)
# 3. Verifies cache integrity
#
# Prerequisites:
#   - data/raw/VCTK-Corpus/     (required)
#   - data/raw/tsukuyomi/       (required)
#   - data/raw/jvs_corpus/      (optional, Phase 1a)
#   - data/raw/libritts_r/      (optional, Phase 1a)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CACHE_DIR="$REPO_ROOT/data/cache"

# Defaults
DEVICE="xpu"
CLEAN_OLD=false
CONTENT_TEACHER="wavlm"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)   DEVICE="$2"; shift 2 ;;
        --clean-old) CLEAN_OLD=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--device xpu|cpu] [--clean-old]"
            echo "  --device      Device for model inference (default: xpu)"
            echo "  --clean-old   Remove old 768d caches before preprocessing"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Phase 1a Preprocessing Pipeline ==="
echo "  Device:          $DEVICE"
echo "  Content teacher: $CONTENT_TEACHER"
echo "  Cache dir:       $CACHE_DIR"
echo "  Clean old:       $CLEAN_OLD"
echo ""

# Export env vars for XPU compatibility
export PYTORCH_ENABLE_XPU_FALLBACK=1
export PYTHONUNBUFFERED=1

# Step 0: Clean old 768d caches
if [ "$CLEAN_OLD" = true ]; then
    echo "--- Step 0: Cleaning old 768d caches ---"

    # tsukuyomi old speaker dirs (768d)
    for old_dir in "$CACHE_DIR/tsukuyomi/train/tsukuyomi_Vol_1_JVS" \
                   "$CACHE_DIR/tsukuyomi/train/tsukuyomi_imported"; do
        if [ -d "$old_dir" ]; then
            echo "  Removing old cache: $old_dir"
            rm -rf "$old_dir"
        fi
    done

    # VCTK old caches (768d) — check content_dim before removal
    if [ -d "$CACHE_DIR/vctk/train" ]; then
        first_meta=$(find "$CACHE_DIR/vctk/train" -name "meta.json" -print -quit 2>/dev/null || true)
        if [ -n "$first_meta" ]; then
            dim=$(python -c "import json; print(json.load(open('$first_meta'))['content_dim'])" 2>/dev/null || echo "unknown")
            if [ "$dim" = "768" ]; then
                echo "  Removing old VCTK cache (content_dim=768)"
                rm -rf "$CACHE_DIR/vctk/train"
            else
                echo "  VCTK cache already at content_dim=$dim, keeping"
            fi
        fi
    fi
    echo ""
fi

# Step 1: Preprocess each available dataset
preprocess_dataset() {
    local dataset="$1"
    local raw_dir="$2"

    if [ ! -d "$raw_dir" ]; then
        echo "  SKIP: $dataset — raw data not found at $raw_dir"
        return 0
    fi

    echo "--- Preprocessing: $dataset ($raw_dir) ---"
    local start_time=$(date +%s)

    uv run tmrvc-preprocess \
        --dataset "$dataset" \
        --raw-dir "$raw_dir" \
        --cache-dir "$CACHE_DIR" \
        --content-teacher "$CONTENT_TEACHER" \
        --device "$DEVICE" \
        -v

    local end_time=$(date +%s)
    local elapsed=$(( end_time - start_time ))
    echo "  $dataset completed in ${elapsed}s"
    echo ""
}

echo "=== Step 1: Feature extraction (WavLM 1024d) ==="
preprocess_dataset "tsukuyomi" "$REPO_ROOT/data/raw/tsukuyomi"
preprocess_dataset "vctk"      "$REPO_ROOT/data/raw/VCTK-Corpus/VCTK-Corpus"
preprocess_dataset "jvs"       "$REPO_ROOT/data/raw/jvs_corpus"
preprocess_dataset "libritts_r" "$REPO_ROOT/data/raw/libritts_r"

# Step 2: Verify cache integrity
echo "=== Step 2: Cache verification ==="
uv run python -c "
from tmrvc_data.cache import FeatureCache
import json

cache = FeatureCache('$CACHE_DIR')
datasets = ['tsukuyomi', 'vctk', 'jvs', 'libritts_r']

for ds in datasets:
    entries = cache.iter_entries(ds, 'train')
    if not entries:
        print(f'  {ds}: no entries')
        continue
    # Check content_dim
    from pathlib import Path
    first = entries[0]
    meta_path = Path('$CACHE_DIR') / ds / 'train' / first['speaker_id'] / first['utterance_id'] / 'meta.json'
    with open(meta_path) as f:
        meta = json.load(f)
    dim = meta['content_dim']
    result = cache.verify(ds, 'train')
    print(f'  {ds}: {result[\"valid\"]}/{result[\"total\"]} valid, content_dim={dim}')
"

echo ""
echo "=== Phase 1a preprocessing complete ==="
