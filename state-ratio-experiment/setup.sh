#!/usr/bin/env bash
# =============================================================================
# One-click setup: install uv, sync project environment, prepare data
#
# Usage (after git clone):
#   bash setup.sh                # full setup: env + data
#   bash setup.sh --env-only     # only install environment, skip data
#
# Prerequisites:
#   - Linux with Python >= 3.10
#   - CUDA-capable GPUs (for training)
#   - Internet access (for downloading packages, models, datasets)
#
# This script will:
#   1. Install uv (if not already installed)
#   2. Create venv and install all dependencies via uv sync
#   3. Download and preprocess GSM8K + MATH datasets
# =============================================================================
set -euo pipefail

# Use HuggingFace mirror for mainland China
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  State-Ratio Experiment — Setup"
echo "=========================================="

# --- Step 1: Install uv if not found ---
if ! command -v uv &>/dev/null; then
    echo ">>> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    echo ">>> uv installed: $(uv --version)"
else
    echo ">>> uv already installed: $(uv --version)"
fi

# --- Step 2: Sync project environment ---
echo ">>> Syncing project environment..."
uv sync --project "$SCRIPT_DIR"
echo ">>> Environment ready."

# --- Step 3: Prepare data (unless --env-only) ---
if [[ "${1:-}" == "--env-only" ]]; then
    echo ">>> Skipping data preparation (--env-only)."
else
    echo ">>> Preparing datasets (GSM8K + MATH)..."
    bash "$SCRIPT_DIR/scripts/prepare_data.sh"
    echo ">>> Data preparation complete."
fi

echo ""
echo "=========================================="
echo "  Setup complete! To run experiments:"
echo ""
echo "  # Run all 3 experiments:"
echo "  bash run_all.sh"
echo ""
echo "  # Run individual experiments:"
echo "  bash run_all.sh grpo     # GRPO baseline"
echo "  bash run_all.sh gspo     # GSPO baseline"
echo "  bash run_all.sh sc       # State-Corrected GRPO"
echo "  bash run_all.sh ablation # k ablation sweep"
echo "=========================================="
