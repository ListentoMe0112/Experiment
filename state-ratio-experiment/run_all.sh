#!/usr/bin/env bash
# =============================================================================
# Master script: Run all experiments for the state ratio correction comparison
#
# Experiment Design (based on Outlook §8.1):
#
#   Baselines:
#     1. GRPO baseline     — per-token clip, no state correction
#     2. GSPO baseline     — sequence-level geometric mean ratio clip
#
#   State-Corrected GRPO (ours):
#     3. SC-GRPO (truncated_window, k=5)  — truncated prefix IS correction
#     4. SC-GRPO (min_prefix)             — MinPRO-style min prefix ratio
#     5. SC-GRPO (vtrace, c̄=1.0)         — V-trace truncated IS
#     6. SC-GRPO (log_ema, α=0.9)        — EMA in log-space
#     7. SC-GRPO (self_normalized)        — group-normalized IS
#
# All experiments share:
#   - Model: Qwen2.5-1.5B-Instruct
#   - Hardware: 8× H100 80GB (single node)
#   - Data: GSM8K + MATH
#   - Advantage: GRPO (group-relative normalization)
#   - Epochs: 15
#   - Loss aggregation: token-mean (DAPO-style)
#   - PPO clip: asymmetric (low=0.2, high=0.28)
#
# Key insight: old_log_prob = π_β (rollout policy). In verl's default mode,
# _compute_old_log_prob() runs BEFORE actor updates, so old_log_probs ≈ π_β.
# The state ratio d^{π_θ}(s_t)/d^{π_β}(s_t) = ∏_{j<t} π_θ(o_j)/π_β(o_j)
# is computed from log_prob - old_log_prob, which is exactly π_θ/π_β.
#
# Usage (inside Docker container):
#   bash setup.sh                       # first-time setup
#   bash run_all.sh                     # run all experiments
#   bash run_all.sh grpo                # run only GRPO baseline
#   bash run_all.sh gspo                # run only GSPO baseline
#   bash run_all.sh sc                  # run SC-GRPO (default: truncated_window k=5)
#   bash run_all.sh strategies          # run all strategy comparisons
#   bash run_all.sh ablation            # run full ablation sweep
# =============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ========================= Download model if not cached =======================
export MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen2.5-1.5B-Instruct}
if [ ! -d "$MODEL_PATH" ] || [ -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]; then
    echo ">>> Downloading Qwen2.5-1.5B-Instruct to $MODEL_PATH ..."
    huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir "$MODEL_PATH"
fi
export HF_HUB_OFFLINE=1

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
        echo ">>> Running State-Corrected GRPO (default: truncated_window k=5)..."
        bash "$SCRIPT_DIR/scripts/run_state_corrected.sh"
        ;;
    strategies)
        echo ">>> Running all strategy comparisons..."
        bash "$SCRIPT_DIR/scripts/run_ablation_k.sh" strategies
        ;;
    ablation)
        echo ">>> Running full ablation sweep..."
        bash "$SCRIPT_DIR/scripts/run_ablation_k.sh" all
        ;;
    all)
        echo ">>> Step 1/4: GRPO baseline"
        bash "$SCRIPT_DIR/scripts/run_grpo_baseline.sh"

        echo ">>> Step 2/4: GSPO baseline"
        bash "$SCRIPT_DIR/scripts/run_gspo_baseline.sh"

        echo ">>> Step 3/4: Strategy comparison (all 6 strategies)"
        bash "$SCRIPT_DIR/scripts/run_ablation_k.sh" strategies

        echo ">>> Step 4/4: Truncated window k ablation"
        bash "$SCRIPT_DIR/scripts/run_ablation_k.sh" window

        echo ">>> All experiments complete. Check project: state_ratio_experiment"
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash run_all.sh [grpo|gspo|sc|strategies|ablation|all]"
        exit 1
        ;;
esac
