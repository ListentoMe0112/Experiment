#!/usr/bin/env bash
# =============================================================================
# One-click setup inside the Docker container
#
# Usage (inside container):
#   bash setup.sh                # full setup: download model + prepare data
#   bash setup.sh --data-only    # only prepare data, skip model download
#
# Prerequisites:
#   - Running inside verlai/verl:vllm018.dev1 Docker container
#   - GPU access via --gpus flag
#   - Model/data volumes mounted
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "  State-Ratio Experiment — Setup"
echo "=========================================="

# --- Step 1: Download model if not cached ---
if [[ "${1:-}" != "--data-only" ]]; then
    export MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen2.5-7B-Instruct}
    if [ ! -d "$MODEL_PATH" ] || [ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
        echo ">>> Downloading Qwen2.5-7B-Instruct to $MODEL_PATH ..."
        python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='$MODEL_PATH')"
    else
        echo ">>> Model already cached at $MODEL_PATH"
    fi
fi

# --- Step 2: Prepare data ---
echo ">>> Preparing datasets (GSM8K + MATH)..."
bash "$SCRIPT_DIR/scripts/prepare_data.sh"
echo ">>> Data preparation complete."

echo ""
echo "=========================================="
echo "  Setup complete! To run experiments:"
echo ""
echo "  bash run_all.sh           # run all 3 experiments"
echo "  bash run_all.sh grpo      # GRPO baseline"
echo "  bash run_all.sh gspo      # GSPO baseline"
echo "  bash run_all.sh sc        # State-Corrected GRPO"
echo "  bash run_all.sh ablation  # k ablation sweep"
echo "=========================================="
