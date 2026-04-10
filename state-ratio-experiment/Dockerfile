FROM verlai/verl:vllm018.dev1

# The base image includes: vllm, torch, flash-attn, transformers, etc.
# We still need: verl (RL training framework) and datasets (data preprocessing)
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple \
    "verl>=0.7.1" datasets

# Set working directory
WORKDIR /workspace/state-ratio-experiment

# Copy project files into the image
COPY . .

# Make scripts executable
RUN chmod +x scripts/*.sh run_all.sh setup.sh

# Add project root to PYTHONPATH so state_corrected_loss.py is importable
ENV PYTHONPATH="/workspace/state-ratio-experiment:${PYTHONPATH:-}"

# Use HuggingFace mirror for faster downloads in China
ENV HF_ENDPOINT=https://hf-mirror.com
