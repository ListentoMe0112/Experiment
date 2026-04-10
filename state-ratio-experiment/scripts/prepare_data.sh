#!/usr/bin/env bash
# =============================================================================
# Data Preparation: Download and preprocess GSM8K and MATH datasets
# =============================================================================
set -xeuo pipefail

# Use HuggingFace mirror for mainland China
export HF_ENDPOINT=https://hf-mirror.com

# Project root (where pyproject.toml lives)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$HOME/data/gsm8k" "$HOME/data/math"

echo "Preparing GSM8K dataset..."
uv run --project "$PROJECT_ROOT" python3 "$SCRIPT_DIR/preprocess_gsm8k.py" \
    --local_save_dir "$HOME/data/gsm8k/"

echo "Preparing MATH dataset..."
uv run --project "$PROJECT_ROOT" python3 "$SCRIPT_DIR/preprocess_math.py" \
    --local_save_dir "$HOME/data/math/"

echo "Data preparation complete."
echo "  GSM8K: $HOME/data/gsm8k/{train,test}.parquet"
echo "  MATH:  $HOME/data/math/{train,test}.parquet"
