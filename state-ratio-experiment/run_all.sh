#!/usr/bin/env bash
# =============================================================================
# Master script: Run all three experiments for the state ratio comparison
#
# Experiment Design:
#   1. GRPO baseline     — per-token clip, no state correction (k=0)
#   2. GSPO baseline     — sequence-level geometric mean ratio clip
#   3. SC-GRPO (ours)    — GRPO + truncated prefix state correction (k=5)
#
# All experiments share:
#   - Model: Qwen2.5-7B-Instruct
#   - Hardware: 8× GPU (single node)
#   - Data: GSM8K + MATH
#   - Advantage: GRPO (group-relative normalization)
#   - Epochs: 15
#
# Usage:
#   bash setup.sh                       # first-time setup (install env + data)
#   bash run_all.sh                     # run all 3 experiments sequentially
#   bash run_all.sh grpo                # run only GRPO
#   bash run_all.sh gspo                # run only GSPO
#   bash run_all.sh sc                  # run only State-Corrected GRPO
#   bash run_all.sh ablation            # run k ablation (k=0,3,5,10,-1)
# =============================================================================
set -xeuo pipefail

# Use HuggingFace mirror for mainland China
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Sync uv project environment
uv sync --project "$SCRIPT_DIR"

# Make all scripts executable
chmod +x "$SCRIPT_DIR"/scripts/*.sh

TARGET=${1:-all}

case $TARGET in
    grpo)
        echo ">>> Running GRPO baseline..."
        bash "$SCRIPT_DIR/scripts/run_grpo_baseline.sh"
        ;;
    gspo)
        echo ">>> Running GSPO baseline..."
        bash "$SCRIPT_DIR/scripts/run_gspo_baseline.sh"
        ;;
    sc)
        echo ">>> Running State-Corrected GRPO (k=5)..."
        bash "$SCRIPT_DIR/scripts/run_state_corrected.sh"
        ;;
    ablation)
        echo ">>> Running k ablation sweep..."
        bash "$SCRIPT_DIR/scripts/run_ablation_k.sh"
        ;;
    all)
        echo ">>> Step 1/3: GRPO baseline"
        bash "$SCRIPT_DIR/scripts/run_grpo_baseline.sh"

        echo ">>> Step 2/3: GSPO baseline"
        bash "$SCRIPT_DIR/scripts/run_gspo_baseline.sh"

        echo ">>> Step 3/3: State-Corrected GRPO (k=5)"
        bash "$SCRIPT_DIR/scripts/run_state_corrected.sh"

        echo ">>> All experiments complete. Check project: state_ratio_experiment"
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash run_all.sh [grpo|gspo|sc|ablation|all]"
        exit 1
        ;;
esac
