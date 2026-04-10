#!/usr/bin/env bash
# =============================================================================
# Data Preparation: Download and preprocess GSM8K and MATH datasets
# =============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$HOME/data/gsm8k" "$HOME/data/math"

echo "Preparing GSM8K dataset..."
python3 "$SCRIPT_DIR/preprocess_gsm8k.py" \
    --local_save_dir "$HOME/data/gsm8k/"

echo "Preparing MATH dataset..."
python3 "$SCRIPT_DIR/preprocess_math.py" \
    --local_save_dir "$HOME/data/math/"

echo "Data preparation complete."
