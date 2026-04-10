#!/usr/bin/env bash
# =============================================================================
# Experiment 2: GSPO Baseline (sequence-level geometric mean ratio clip)
# Model: Qwen2.5-7B-Instruct | 8 GPUs | Math reasoning
# =============================================================================
set -xeuo pipefail

# Use HuggingFace mirror for mainland China
export HF_ENDPOINT=https://hf-mirror.com
export VLLM_USE_V1=1

# Project root (where pyproject.toml lives)
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ========================= Data Paths ========================================
gsm8k_train_path=$HOME/data/gsm8k/train.parquet
gsm8k_test_path=$HOME/data/gsm8k/test.parquet
math_train_path=$HOME/data/math/train.parquet
math_test_path=$HOME/data/math/test.parquet

train_files="['$gsm8k_train_path', '$math_train_path']"
test_files="['$gsm8k_test_path', '$math_test_path']"

# ========================= Shared Hyperparameters ============================
MODEL_PATH=${MODEL_PATH:-$HOME/models/Qwen2.5-7B-Instruct}
GPUS_PER_NODE=8
NNODES=1

# Training
train_batch_size=512
ppo_mini_batch_size=128
ppo_micro_batch_size_per_gpu=4
max_prompt_length=1024
max_response_length=2048
n_resp_per_prompt=8
total_epochs=15
total_training_steps=-1
lr=1e-6

# GSPO-specific: tighter clip bounds as recommended by the paper (Sec. 5.1)
clip_ratio_low=0.0003
clip_ratio_high=0.0004

# Rollout
rollout_tp=2
gpu_memory_utilization=0.4
temperature=1.0
top_p=1.0

# ========================= Run ===============================================
uv run --project "$PROJECT_ROOT" python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=$lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 2)) \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((max_prompt_length + max_response_length) * 3)) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.temperature=$temperature \
    actor_rollout_ref.rollout.top_p=$top_p \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='state_ratio_experiment' \
    trainer.experiment_name='gspo_baseline_qwen2.5_7b' \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.val_before_train=True \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=$total_epochs \
    trainer.total_training_steps=$total_training_steps \
    "$@"
