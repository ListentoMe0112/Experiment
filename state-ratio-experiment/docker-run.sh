#!/usr/bin/env bash
# =============================================================================
# Launch the training Docker container with GPU access
#
# Usage:
#   bash docker-run.sh                  # build + run interactive shell
#   bash docker-run.sh bash run_all.sh  # build + run all experiments
#   bash docker-run.sh bash run_all.sh grpo  # run only GRPO
#
# Environment variables:
#   MODEL_DIR  — host path to model cache (default: $HOME/models)
#   DATA_DIR   — host path to data cache  (default: $HOME/data)
#   OUTPUT_DIR — host path for checkpoints (default: $HOME/output)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="state-ratio-experiment"

MODEL_DIR="${MODEL_DIR:-$HOME/models}"
DATA_DIR="${DATA_DIR:-$HOME/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/output}"

# Create host directories if they don't exist
mkdir -p "$MODEL_DIR" "$DATA_DIR" "$OUTPUT_DIR"

# Build the Docker image
echo ">>> Building Docker image: $IMAGE_NAME ..."
docker build -t "$IMAGE_NAME" "$SCRIPT_DIR"

# Run the container
echo ">>> Starting container with ${GPUS:-all} GPUs ..."
docker run --rm -it \
    --gpus "${GPUS:-all}" \
    --shm-size=16g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "$MODEL_DIR:/root/models" \
    -v "$DATA_DIR:/root/data" \
    -v "$OUTPUT_DIR:/root/output" \
    -e MODEL_PATH=/root/models/Qwen2.5-1.5B-Instruct \
    -e HF_HOME=/root/models/hf_cache \
    "$IMAGE_NAME" \
    "${@:-bash}"
