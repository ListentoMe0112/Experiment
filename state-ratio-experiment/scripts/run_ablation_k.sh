#!/usr/bin/env bash
# =============================================================================
# Ablation: Sweep over lookback window k for State-Corrected GRPO
#
# k=0  → GRPO (no state correction)
# k=3  → mild correction
# k=5  → moderate correction (default)
# k=10 → aggressive correction
# k=-1 → full trajectory IS (exact, high variance)
#
# Usage:
#   bash run_ablation_k.sh          # run all k values sequentially
#   bash run_ablation_k.sh 5        # run only k=5
# =============================================================================
set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -ge 1 ]; then
    K_VALUES=("$1")
else
    K_VALUES=(0 3 5 10 -1)
fi

for k in "${K_VALUES[@]}"; do
    echo "============================================="
    echo "Running State-Corrected GRPO with k=$k"
    echo "============================================="
    LOOKBACK_K=$k bash "$SCRIPT_DIR/run_state_corrected.sh"
done
