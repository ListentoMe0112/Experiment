"""
State-Corrected GRPO: Recovering the State Distribution Ratio via Deterministic Transitions.

This module implements the truncated prefix IS correction for LLM RL training,
as described in the "Outlook: Recovering the State Ratio via Deterministic Transitions"
section. In autoregressive LLMs, transitions are deterministic, so the state
distribution ratio decomposes exactly into a product of per-token action ratios:

    d^{π_θ}(s_t) / d^{π_β}(s_t) = ∏_{j=1}^{t-1} r_j(θ)

The full importance weight becomes:

    w(s_t, a_t) = ∏_{j=1}^{t} r_j(θ)

To control variance, we use truncated prefix correction with lookback window k:

    w_t^{(k)}(θ) = ∏_{j=max(1, t-k)}^{t} r_j(θ)

This creates a bias-variance tradeoff:
    k=0  → GRPO/PPO (state ratio dropped entirely)
    k=3~10 → correct recent state mismatch
    k=T  → exact full trajectory IS (explosive variance)

Usage:
    Import this module before training to register the loss function, then set:
        actor_rollout_ref.actor.policy_loss.loss_mode=state_corrected_grpo
"""

from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


@register_policy_loss("state_corrected_grpo")
def compute_policy_loss_state_corrected_grpo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute policy loss with truncated prefix state-ratio correction.

    This implements the key insight from the Outlook section: in LLMs, the state
    distribution ratio is exactly ∏_{j=1}^{t-1} r_j(θ), and we can use a truncated
    window k to trade off bias vs variance.

    The surrogate objective is:
        L(θ) = E[ Σ_t w_t^{(k)}(θ) · clip(r_t(θ)) · A_t ]

    where w_t^{(k)}(θ) = ∏_{j=max(1,t-k)}^{t-1} r_j(θ)  (state correction)
    and clip(r_t(θ)) is the standard PPO clip on the action ratio.

    The state correction weight is stop-gradiented to avoid compounding gradient issues.

    Args:
        old_log_prob: Log-probs under old policy, shape (batch_size, response_length).
        log_prob: Log-probs under current policy, shape (batch_size, response_length).
        advantages: Advantage estimates, shape (batch_size, response_length).
        response_mask: Token mask, shape (batch_size, response_length).
        loss_agg_mode: Aggregation mode for loss.
        config: ActorConfig with clip_ratio, clip_ratio_low, clip_ratio_high,
                and state_correction_* fields.
        rollout_is_weights: Optional external IS weights.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    assert config is not None
    assert isinstance(config, ActorConfig)

    clip_ratio = config.clip_ratio
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    # --- Hyperparameters for state correction ---
    # Lookback window k: how many previous tokens to include in state correction
    # k=0 → standard GRPO (no state correction)
    # k=5 → correct state mismatch from last 5 tokens
    # k=-1 → full trajectory (exact but high variance)
    lookback_k = config.get("state_correction_lookback_k", 5)

    # Max allowed state correction weight (truncation for stability)
    max_state_weight = config.get("state_correction_max_weight", 5.0)

    # Min allowed state correction weight
    min_state_weight = config.get("state_correction_min_weight", 0.2)

    # --- Per-token log importance ratio ---
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    log_ratio = negative_approx_kl  # log(π_θ / π_β) per token

    # --- Per-token action ratio (for PPO clip) ---
    ratio = torch.exp(log_ratio)

    # --- Compute truncated prefix state correction weight ---
    # w_t^{(k)} = ∏_{j=max(1,t-k)}^{t-1} r_j = exp(Σ_{j=max(1,t-k)}^{t-1} log r_j)
    # This is the state distribution ratio correction.
    # We detach it so gradients only flow through the action ratio.

    # Cumulative sum of log ratios (masked)
    masked_log_ratio = log_ratio * response_mask  # (B, T)
    cumsum_log_ratio = torch.cumsum(masked_log_ratio, dim=-1)  # (B, T)

    # cumsum_log_ratio[t] = Σ_{j=0}^{t} log r_j
    # We need Σ_{j=max(0,t-k)}^{t-1} log r_j for state correction (excluding current token t)

    # Shift right by 1: prefix sum up to t-1
    # prefix_sum[t] = Σ_{j=0}^{t-1} log r_j
    prefix_sum = torch.zeros_like(cumsum_log_ratio)
    prefix_sum[:, 1:] = cumsum_log_ratio[:, :-1]

    T = log_ratio.shape[1]

    if lookback_k < 0 or lookback_k >= T:
        # Full trajectory correction: w_t = ∏_{j=0}^{t-1} r_j
        log_state_weight = prefix_sum
    elif lookback_k == 0:
        # No state correction (standard GRPO)
        log_state_weight = torch.zeros_like(prefix_sum)
    else:
        # Truncated: w_t^{(k)} = ∏_{j=max(0,t-k)}^{t-1} r_j
        # = exp(prefix_sum[t] - prefix_sum[max(0, t-k)])
        # prefix_sum[t] - prefix_sum[t-k] = Σ_{j=t-k}^{t-1} log r_j
        shifted_prefix = torch.zeros_like(prefix_sum)
        if lookback_k < T:
            shifted_prefix[:, lookback_k:] = prefix_sum[:, :-lookback_k]
        # For t < k, shifted_prefix[t] = 0, so log_state_weight[t] = prefix_sum[t]
        # which is the full prefix (correct, since we can't look back further)
        log_state_weight = prefix_sum - shifted_prefix

    # Clamp log weights for numerical stability before exp
    log_state_weight = torch.clamp(log_state_weight, min=-5.0, max=5.0)
    state_weight = torch.exp(log_state_weight).detach()  # stop gradient!

    # Truncate extreme weights
    state_weight = torch.clamp(state_weight, min=min_state_weight, max=max_state_weight)

    # --- Standard PPO clip on action ratio ---
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses_clipped = torch.maximum(pg_losses1, pg_losses2)

    # --- Apply state correction weight ---
    pg_losses = pg_losses_clipped * state_weight

    # Apply external rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # --- Aggregate loss ---
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # --- Metrics ---
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    # State correction diagnostics
    mean_state_weight = verl_F.masked_mean(state_weight, response_mask)
    max_state_weight_val = (state_weight * response_mask).max()
    std_state_weight = verl_F.masked_mean((state_weight - mean_state_weight) ** 2, response_mask).sqrt()

    # Fraction of tokens where state weight was clamped
    clamped_high = (state_weight >= max_state_weight - 1e-6).float()
    clamped_low = (state_weight <= min_state_weight + 1e-6).float()
    clamp_frac = verl_F.masked_mean((clamped_high + clamped_low), response_mask)

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": 0.0,
        # State correction specific metrics
        "actor/state_weight_mean": mean_state_weight.detach().item(),
        "actor/state_weight_max": max_state_weight_val.detach().item(),
        "actor/state_weight_std": std_state_weight.detach().item(),
        "actor/state_weight_clamp_frac": clamp_frac.detach().item(),
    }

    return pg_loss, pg_metrics
