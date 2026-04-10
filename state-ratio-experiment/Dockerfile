FROM verlai/verl:vllm018.dev1

# The base image already includes: verl, vllm, torch, flash-attn, transformers, etc.
# We only need to ensure datasets is available for data preprocessing.
RUN pip install --no-cache-dir -i https://mirrors.cloud.tencent.com/pypi/simple datasets

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
