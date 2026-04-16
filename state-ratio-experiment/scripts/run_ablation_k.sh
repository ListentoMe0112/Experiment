#!/usr/bin/env bash
# =============================================================================
# Ablation: Sweep over prefix correction strategies and their hyperparameters
#
# Strategy sweep:
#   none              → GRPO + clip (no state correction baseline)
#   truncated_window  → k ∈ {3, 5, 10, -1}
#   min_prefix        → MinPRO-style min prefix ratio
#   vtrace            → c̄ ∈ {0.5, 1.0, 2.0}
#   log_ema           → α ∈ {0.8, 0.9, 0.95}
#   self_normalized   → group-normalized IS
#
# Usage:
#   bash run_ablation_strategies.sh              # run all strategies
#   bash run_ablation_strategies.sh window       # only truncated window sweep
#   bash run_ablation_strategies.sh strategies   # only strategy comparison (best hyperparams)
# =============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET=${1:-all}

run_sc() {
    local strategy=$1
    shift
    echo "============================================="
    echo "Running SC-GRPO: strategy=$strategy $*"
    echo "============================================="
    # Export all KEY=VALUE pairs as environment variables
    for arg in "$@"; do
        export "$arg"
    done
    SC_STRATEGY=$strategy bash "$SCRIPT_DIR/run_state_corrected.sh"
}

case $TARGET in
    window)
        # Ablation over lookback window k (truncated_window strategy)
        for k in 3 5 10 -1; do
            run_sc truncated_window LOOKBACK_K=$k
        done
        ;;
    strategies)
        # Compare all strategies with their best/default hyperparameters
        run_sc none
        run_sc truncated_window LOOKBACK_K=5
        run_sc min_prefix
        run_sc vtrace VTRACE_C=1.0
        run_sc log_ema EMA_ALPHA=0.9
        run_sc self_normalized
        ;;
    vtrace)
        # Ablation over V-trace truncation threshold c̄
        for c in 0.5 1.0 2.0; do
            run_sc vtrace VTRACE_C=$c
        done
        ;;
    ema)
        # Ablation over EMA smoothing factor α
        for a in 0.8 0.9 0.95; do
            run_sc log_ema EMA_ALPHA=$a
        done
        ;;
    all)
        # Full sweep: strategy comparison + window ablation
        echo ">>> Phase 1: Strategy comparison"
        run_sc none
        run_sc truncated_window LOOKBACK_K=5
        run_sc min_prefix
        run_sc vtrace VTRACE_C=1.0
        run_sc log_ema EMA_ALPHA=0.9
        run_sc self_normalized

        echo ">>> Phase 2: Truncated window k ablation"
        for k in 3 10 -1; do
            run_sc truncated_window LOOKBACK_K=$k
        done
        ;;
    *)
        echo "Unknown target: $TARGET"
        echo "Usage: bash run_ablation_strategies.sh [window|strategies|vtrace|ema|all]"
        exit 1
        ;;
esac
