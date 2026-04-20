#!/usr/bin/env bash
# =============================================================================
# Experiment: State-Corrected GRPO (prefix IS correction)
# Model: Qwen2.5-1.5B-Instruct | 8× H100 80GB | Docker: verlai/verl:vllm018.dev1
#
# This implements the prefix state-ratio correction from the Outlook section.
# In autoregressive LLMs, transitions are deterministic, so:
#   d^{π_θ}(s_t) / d^{π_β}(s_t) = ∏_{j=1}^{t-1} r_j(θ)
#
# Supported strategies (set via SC_STRATEGY):
#   truncated_window   — w_t = ∏_{j=max(1,t-k)}^{t-1} r_j  (default)
#   min_prefix         — w_t = min_{j<t} r_j  (MinPRO-style)
#   vtrace             — w_t = ∏_{j<t} min(c̄, r_j)  (V-trace truncated IS)
#   log_ema            — w_t = exp(EMA of log r_j)  (smooth tracking)
#   self_normalized    — w_t = ρ_{1:t} / Σ_i ρ_{1:t}^{(i)}  (batch-normalized)
#   baseline_corrected — w_t = r_t · ρ_{1:t-1} / GeoMean_group(ρ_{1:t-1})
#                        (cross-group control variate; unbiased, corrects timestep drift)
#   none               — raw full product (no variance reduction, high variance)
#   identity           — w_t = 1 (pure REINFORCE, completely eliminates state correction)
#
# Key: old_log_prob = π_β (rollout policy). We use bypass_mode to skip
# _compute_old_log_prob() and directly use rollout engine's log_probs.
# This ensures old_log_probs IS EXACTLY π_β, not a recomputed approximation.
# =============================================================================
set -xuo pipefail

# ========================= Output & Data Paths ===============================
# OUTPUT_DIR must match the Docker volume mount (docker-run.sh: -v $HOME/output:/root/output)
OUTPUT_DIR=${OUTPUT_DIR:-$HOME/output}
mkdir -p "$OUTPUT_DIR"

# File logger: write jsonl logs under OUTPUT_DIR for persistent error capture
export VERL_FILE_LOGGER_ROOT="$OUTPUT_DIR/logs"

gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# ========================= Shared Hyperparameters ============================
MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen2.5-1.5B-Instruct}
GPUS_PER_NODE=8
NNODES=1

# Training (8× H100 80GB) - Optimized for 1.5B model
train_batch_size=1024
ppo_mini_batch_size=1024
ppo_micro_batch_size_per_gpu=4
max_prompt_length=1024
max_response_length=2048
n_resp_per_prompt=8
ppo_epochs=4              # offline updates per rollout batch
total_epochs=15
lr=2e-6

# Rollout
rollout_tp=1
gpu_memory_utilization=0.4
temperature=1.0
top_p=1.0

# ========================= State Correction Hyperparameters ==================
# Strategy: truncated_window | min_prefix | vtrace | log_ema | self_normalized
#         | baseline_corrected | none | identity
SC_STRATEGY=${SC_STRATEGY:-truncated_window}

# For truncated_window: lookback window k
# k=0 → no state correction, k=5 → moderate, k=-1 → full trajectory
LOOKBACK_K=${LOOKBACK_K:-5}

# For vtrace: truncation threshold c̄
VTRACE_C=${VTRACE_C:-1.0}

# For log_ema: smoothing factor α
EMA_ALPHA=${EMA_ALPHA:-0.9}

# For baseline_corrected: group size (number of rollouts per prompt).
# MUST equal actor_rollout_ref.rollout.n so the (B, T) batch reshapes cleanly
# into (num_prompts, G, T) for per-prompt geometric centering of ρ_{1:t-1}.
SC_GROUP_SIZE=${SC_GROUP_SIZE:-$n_resp_per_prompt}

# Weight clipping bounds
MAX_STATE_WEIGHT=${MAX_STATE_WEIGHT:-5.0}
MIN_STATE_WEIGHT=${MIN_STATE_WEIGHT:-0.2}

# PPO clip ratios (asymmetric: wider for exploration)
CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-0.2}
CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-0.28}

# ========================= Run ===============================================
# Export state correction hyperparameters as environment variables
export SC_STRATEGY=$SC_STRATEGY
export SC_LOOKBACK_K=$LOOKBACK_K
export SC_VTRACE_C=$VTRACE_C
export SC_EMA_ALPHA=$EMA_ALPHA
export SC_GROUP_SIZE=$SC_GROUP_SIZE
export SC_MAX_STATE_WEIGHT=$MAX_STATE_WEIGHT
export SC_MIN_STATE_WEIGHT=$MIN_STATE_WEIGHT

# Sanity check: baseline_corrected requires group_size == rollout.n
if [ "$SC_STRATEGY" = "baseline_corrected" ] && [ "$SC_GROUP_SIZE" != "$n_resp_per_prompt" ]; then
    echo "ERROR: SC_GROUP_SIZE ($SC_GROUP_SIZE) must equal n_resp_per_prompt ($n_resp_per_prompt) for baseline_corrected."
    exit 1
fi

# Build experiment name
if [ "$SC_STRATEGY" = "truncated_window" ]; then
    EXP_NAME="sc_${SC_STRATEGY}_k${LOOKBACK_K}_qwen2.5_1.5b"
elif [ "$SC_STRATEGY" = "vtrace" ]; then
    EXP_NAME="sc_${SC_STRATEGY}_c${VTRACE_C}_qwen2.5_1.5b"
elif [ "$SC_STRATEGY" = "log_ema" ]; then
    EXP_NAME="sc_${SC_STRATEGY}_a${EMA_ALPHA}_qwen2.5_1.5b"
elif [ "$SC_STRATEGY" = "baseline_corrected" ]; then
    EXP_NAME="sc_${SC_STRATEGY}_g${SC_GROUP_SIZE}_qwen2.5_1.5b"
else
    EXP_NAME="sc_${SC_STRATEGY}_qwen2.5_1.5b"
fi

# Verify the custom loss module is importable
python3 -c "import state_corrected_loss; print('Registered state_corrected_grpo loss')" || {
    echo 'ERROR: Cannot import state_corrected_loss.py. Make sure it is in PYTHONPATH or cwd.'
    exit 1
}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=state_corrected_grpo \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.rollout_correction.bypass_mode=True \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.external_lib=state_corrected_loss \
    +actor_rollout_ref.model.override_config.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.ppo_epochs=$ppo_epochs \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_RATIO_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_RATIO_HIGH \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","file"]' \
    trainer.project_name='state_ratio_experiment' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=True \
    trainer.default_local_dir="$OUTPUT_DIR/checkpoints/$EXP_NAME" \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs \
    "$@"
