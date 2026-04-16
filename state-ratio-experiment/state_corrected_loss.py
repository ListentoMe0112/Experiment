"""
State-Corrected GRPO: Recovering the State Distribution Ratio via Deterministic Transitions.

This module implements prefix IS correction strategies for LLM RL training,
as described in the "Outlook" section of the notes. In autoregressive LLMs,
transitions are deterministic, so the state distribution ratio decomposes
exactly into a product of per-token action ratios:

    d^{π_θ}(s_t) / d^{π_β}(s_t) = ∏_{j=1}^{t-1} r_j(θ)

where r_j(θ) = π_θ(o_j|s_j) / π_β(o_j|s_j) and π_β is the rollout policy.

IMPORTANT: For state correction to be meaningful, old_log_prob MUST be the
log-probabilities from the rollout policy π_β (the policy that generated the
data), NOT a recomputed π_old from a later checkpoint.

To ensure this, we use verl's bypass_mode with a monkey-patch:
  1. Set actor_rollout_ref.rollout.calculate_log_probs=True
     → rollout engine (vllm) returns per-token log_probs during generation
  2. Set +algorithm.rollout_correction.bypass_mode=True
     → trainer sets old_log_probs = rollout_log_probs (skips _compute_old_log_prob)
  3. This module patches apply_bypass_mode() to NOT override loss_mode,
     so our state_corrected_grpo loss is preserved.

This saves one full actor forward pass AND guarantees old_log_probs = π_β exactly.

Supported correction strategies (set via SC_STRATEGY env var):
    All strategies compute w_t = state_ratio × action_ratio, i.e. the product
    ∏_{j=0}^{t} r_j covers tokens 0..t (inclusive of current token t).
    The loss is REINFORCE-style: L = -Σ_t w_t · log π_θ(a_t|s_t) · A_t

    - "truncated_window" : w_t = ∏_{j=max(0,t-k)}^{t} r_j  (bias-variance via k)
    - "min_prefix"       : w_t = min_{j≤t} r_j              (MinPRO-style)
    - "vtrace"           : w_t = ∏_{j=0}^{t} min(c̄, r_j)   (V-trace truncated IS)
    - "log_ema"          : w_t = exp(EMA of log r_j for j≤t) (smooth tracking)
    - "self_normalized"  : w_t = ρ_{0:t} / Σ_i ρ_{0:t}^{(i)} (group-normalized)
    - "clamp_only"       : w_t = r_t                         (single token ratio, no prefix)
    - "none"             : w_t = ∏_{j=0}^{t} r_j             (raw full product, no variance reduction)
    - "identity"         : w_t = 1                            (pure REINFORCE, no IS)

Usage:
    Import this module before training to register the loss function, then set:
        actor_rollout_ref.actor.policy_loss.loss_mode=state_corrected_grpo
    Configure via environment variables:
        SC_STRATEGY=truncated_window  (default)
        SC_LOOKBACK_K=5              (for truncated_window)
        SC_VTRACE_C=1.0              (for vtrace)
        SC_EMA_ALPHA=0.9             (for log_ema)
        SC_MAX_STATE_WEIGHT=5.0
        SC_MIN_STATE_WEIGHT=0.2

"""

import math
import os
from typing import Any, Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import agg_loss, register_policy_loss
from verl.workers.config import ActorConfig


# =============================================================================
# Monkey-patch: make bypass_mode use rollout_log_probs as old_log_probs
# WITHOUT overriding loss_mode. This lets us combine:
#   - algorithm.rollout_correction.bypass_mode=True  (skip _compute_old_log_prob)
#   - actor_rollout_ref.actor.policy_loss.loss_mode=state_corrected_grpo  (our loss)
#
# The original apply_bypass_mode() forces loss_mode="bypass_mode", which would
# override our custom loss. We patch it to preserve the user-specified loss_mode.
# =============================================================================
def _patched_apply_bypass_mode(batch, rollout_corr_config=None, policy_loss_config=None):
    """Bypass mode: use rollout_log_probs as old_log_probs, preserve loss_mode."""
    from omegaconf import open_dict

    if "rollout_log_probs" not in batch.batch:
        raise ValueError(
            "bypass_mode=True requires rollout_log_probs in batch. "
            "Ensure rollout worker is configured with "
            "actor_rollout_ref.rollout.calculate_log_probs=true."
        )

    # Use rollout log probs as old log probs (zero-cost substitution)
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]

    with open_dict(policy_loss_config):
        # Pass rollout_correction config to actor for loss computation and metrics
        policy_loss_config["rollout_correction"] = rollout_corr_config
        # NOTE: We intentionally do NOT override loss_mode here.
        # The original apply_bypass_mode sets loss_mode="bypass_mode",
        # but we want to keep the user-specified loss_mode (e.g. "state_corrected_grpo").


try:
    import verl.trainer.ppo.rollout_corr_helper as _rollout_corr_mod
    _rollout_corr_mod.apply_bypass_mode = _patched_apply_bypass_mode
except ImportError:
    pass  # rollout_corr_helper not available, skip patching


# =============================================================================
# Prefix correction strategies
# =============================================================================

def _compute_state_weight_truncated_window(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    lookback_k: int,
) -> torch.Tensor:
    """
    Truncated window: w_t^{(k)} = ∏_{j=max(0,t-k)}^{t} r_j

    Includes current token t in the product (full IS weight from t-k to t).

    k=0  → w_t = r_t only (no state correction, just current action ratio)
    k>0  → correct recent k+1 tokens of state mismatch (including current)
    k=-1 → full trajectory ∏_{j=0}^{t} r_j (exact but high variance)
    """
    masked_log_ratio = log_ratio * response_mask
    # cumsum[t] = Σ_{j=0}^{t} log r_j (inclusive of current token)
    cumsum = torch.cumsum(masked_log_ratio, dim=-1)

    T = log_ratio.shape[1]

    if lookback_k == 0:
        # w_t = r_t only → log_weight = log_ratio
        return masked_log_ratio
    elif lookback_k < 0 or lookback_k >= T:
        # Full trajectory: w_t = ∏_{j=0}^{t} r_j
        return cumsum
    else:
        # Window [t-k, t]: cumsum[t] - cumsum[t-k-1]
        shifted = torch.zeros_like(cumsum)
        shifted[:, lookback_k + 1:] = cumsum[:, :-lookback_k - 1]
        return cumsum - shifted


def _compute_state_weight_min_prefix(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    MinPRO-style: ρ̲_t = min_{j≤t} r_j

    Uses the minimum token ratio up to and including current token as a
    stable, non-cumulative surrogate for the full prefix product. When any
    token (including current) has drifted far from the rollout policy, the
    gradient is attenuated.

    Returns log(ρ̲_t) for numerical stability.
    """
    B, T = log_ratio.shape

    # Replace masked positions with +inf so they don't affect the min
    large_val = torch.tensor(1e9, device=log_ratio.device, dtype=log_ratio.dtype)
    masked_log_ratio = torch.where(response_mask.bool(), log_ratio, large_val)

    # Compute running min of log_ratio for positions j ≤ t (including current)
    # cummin_vals[t] = min(log_ratio[0], ..., log_ratio[t])
    cummin_vals, _ = torch.cummin(masked_log_ratio, dim=-1)

    return cummin_vals


def _compute_state_weight_vtrace(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    c_bar: float = 1.0,
) -> torch.Tensor:
    """
    V-trace style: w_t = ∏_{j=0}^{t} min(c̄, r_j)

    Clip each factor BEFORE multiplying (including current token).
    With c̄ ≤ 1, the product is monotonically non-increasing, bounding
    variance growth.

    Returns log(w_t).
    """
    # Clip log_ratio so that r_j ≤ c_bar → log_ratio ≤ log(c_bar)
    log_c_bar = torch.tensor(c_bar, device=log_ratio.device, dtype=log_ratio.dtype).log()
    clipped_log_ratio = torch.clamp(log_ratio, max=log_c_bar.item())
    masked = clipped_log_ratio * response_mask

    # Inclusive cumsum: cumsum[t] = Σ_{j=0}^{t} clipped_log_ratio[j]
    log_state_weight = torch.cumsum(masked, dim=-1)

    return log_state_weight


def _compute_state_weight_log_ema(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.9,
) -> torch.Tensor:
    """
    Exponential moving average in log-space:
        l̄_t = α · l̄_{t-1} + (1-α) · log r_t
    Uses exp(l̄_t) as the state weight for token t (including current token).

    Smoother than min, captures gradual drift rather than only worst-case.

    Returns log(state_weight) = l̄_t for each position t.
    """
    B, T = log_ratio.shape
    log_state_weight = torch.zeros_like(log_ratio)
    ema = torch.zeros(B, device=log_ratio.device, dtype=log_ratio.dtype)

    for t in range(T):
        # Update EMA including current token, then record
        mask_t = response_mask[:, t]
        ema = mask_t * (alpha * ema + (1 - alpha) * log_ratio[:, t]) + (1 - mask_t) * ema
        log_state_weight[:, t] = ema

    return log_state_weight


def _compute_state_weight_self_normalized(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Self-normalized IS: w̃_t^{(i)} = ρ_{0:t}^{(i)} / Σ_i ρ_{0:t}^{(i)}

    Includes current token in the product. Weights sum to 1 by construction,
    preventing explosion. This naturally fits GRPO's group structure.

    Returns log(w̃_t) (approximately, via log-sum-exp normalization).

    NOTE: This operates across the batch dimension, assuming samples in the
    same batch are from the same prompt group.
    """
    masked = log_ratio * response_mask
    # Inclusive cumsum: log ρ_{0:t} = Σ_{j=0}^{t} log r_j
    log_cumulative = torch.cumsum(masked, dim=-1)

    # Self-normalize across batch dimension at each position
    # log(Σ_i exp(log_cumulative_i)) via logsumexp
    log_sum = torch.logsumexp(log_cumulative, dim=0, keepdim=True)  # (1, T)
    log_state_weight = log_cumulative - log_sum + torch.tensor(log_cumulative.shape[0], device=log_ratio.device).float().log()

    return log_state_weight


# =============================================================================
# Main loss function
# =============================================================================

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
    Compute policy loss with prefix state-ratio correction.

    The true off-policy surrogate for LLMs is:
        L(θ) = E_{o~π_β}[ Σ_t (d^{π_θ}(s_t)/d^{π_β}(s_t)) · (π_θ(a_t|s_t)/π_β(a_t|s_t)) · A_t ]

    Because LLM transitions are deterministic:
        d^{π_θ}(s_t)/d^{π_β}(s_t) = ∏_{j=1}^{t-1} π_θ(o_j|s_j)/π_β(o_j|s_j)

    All strategies unify state_ratio and action_ratio into a single weight:
        w_t = (d^{π_θ}(s_t)/d^{π_β}(s_t)) · (π_θ(a_t|s_t)/π_β(a_t|s_t))
            = ∏_{j=0}^{t} r_j  (inclusive of current token)

    This function computes:
        - log_w_t = f(Σ_{j=0}^{t} log r_j)  where f is the chosen strategy
        - w_t = clamp(exp(log_w_t), min_w, max_w)  (detached)
        - loss = -Σ_t w_t · log π_θ(a_t|s_t) · A_t
        - Gradient flows only through log π_θ (w_t is detached, REINFORCE-style)

    CRITICAL: old_log_prob must be from the ROLLOUT policy π_β.
    In verl's default mode, old_log_probs is computed before actor updates,
    so it equals π_β. This is correct for state correction.

    Args:
        old_log_prob: Log-probs under rollout policy π_β, shape (B, T).
        log_prob: Log-probs under current policy π_θ, shape (B, T).
        advantages: Advantage estimates, shape (B, T).
        response_mask: Token mask, shape (B, T).
        loss_agg_mode: Aggregation mode for loss.
        config: ActorConfig.
        rollout_is_weights: Optional external IS weights (unused, kept for API compat).

    Returns:
        Tuple of (loss, metrics_dict).
    """
    assert config is not None, "config must be provided"
    # Accept both ActorConfig and mock configs (for testing)

    # --- Read hyperparameters from environment variables ---
    strategy = os.environ.get("SC_STRATEGY", "truncated_window")
    lookback_k = int(os.environ.get("SC_LOOKBACK_K", "5"))
    vtrace_c = float(os.environ.get("SC_VTRACE_C", "1.0"))
    ema_alpha = float(os.environ.get("SC_EMA_ALPHA", "0.9"))
    max_state_weight = float(os.environ.get("SC_MAX_STATE_WEIGHT", "5.0"))
    min_state_weight = float(os.environ.get("SC_MIN_STATE_WEIGHT", "0.2"))

    # --- Per-token log importance ratio: log(π_θ / π_β) ---
    # This is the ratio between current policy and ROLLOUT policy.
    # old_log_prob = log π_β (rollout policy that generated the data)
    # log_prob = log π_θ (current policy being optimized)
    log_ratio = log_prob - old_log_prob

    # --- Compute IS weight = state_ratio × action_ratio using chosen strategy ---
    # All strategies return log(w_t) = log(∏_{j=0}^{t} r_j variant) inclusive of current token
    if strategy == "truncated_window":
        log_state_weight = _compute_state_weight_truncated_window(
            log_ratio, response_mask, lookback_k
        )
    elif strategy == "min_prefix":
        log_state_weight = _compute_state_weight_min_prefix(
            log_ratio, response_mask
        )
    elif strategy == "vtrace":
        log_state_weight = _compute_state_weight_vtrace(
            log_ratio, response_mask, c_bar=vtrace_c
        )
    elif strategy == "log_ema":
        log_state_weight = _compute_state_weight_log_ema(
            log_ratio, response_mask, alpha=ema_alpha
        )
    elif strategy == "self_normalized":
        log_state_weight = _compute_state_weight_self_normalized(
            log_ratio, response_mask
        )
    elif strategy == "none":
        # Raw full product: w_t = ∏_{j=0}^{t} r_j = state_ratio × action_ratio
        # No variance reduction (no truncation, no min, no EMA).
        # Only the outer clamp applies.
        masked_log_ratio = log_ratio * response_mask
        log_state_weight = torch.cumsum(masked_log_ratio, dim=-1)
    else:
        raise ValueError(f"Unknown SC_STRATEGY: {strategy}. "
                         f"Choose from: truncated_window, min_prefix, vtrace, log_ema, "
                         f"self_normalized, clamp_only, none, identity")

    # --- Clamp IS weight (state_ratio × action_ratio) and detach ---
    # w_t covers t=0..T (inclusive: state_ratio × action_ratio).
    # Clamp in log-space then exponentiate. Detach so gradient flows
    # only through log_prob (REINFORCE-style).
    log_min = math.log(min_state_weight)
    log_max = math.log(max_state_weight)
    log_state_weight = torch.clamp(log_state_weight, min=log_min, max=log_max)
    state_weight = torch.exp(log_state_weight).detach()  # stop gradient!

    # --- Final loss: REINFORCE-style ---
    # loss = -w_t · log π_θ(a_t|s_t) · A_t  where w_t = state_ratio × action_ratio
    # Gradient: ∇_θ L = -w_t · A_t · ∇_θ log π_θ  (w_t detached)
    pg_losses = -state_weight * log_prob * advantages

    # NaN/Inf safety
    if torch.isnan(pg_losses).any() or torch.isinf(pg_losses).any():
        pg_losses = torch.nan_to_num(pg_losses, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Aggregate loss ---
    pg_loss = agg_loss(
        loss_mat=pg_losses,
        loss_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        **config.global_batch_info,
    )

    # --- Metrics ---
    with torch.no_grad():
        # KL divergence (approx)
        ppo_kl = verl_F.masked_mean(-log_ratio, response_mask)

        # IS weight diagnostics (w_t = state_ratio × action_ratio)
        mean_sw = verl_F.masked_mean(state_weight, response_mask)
        max_sw = (state_weight * response_mask).max()
        std_sw = verl_F.masked_mean((state_weight - mean_sw) ** 2, response_mask).sqrt()

        # Fraction of tokens where state weight was clamped
        clamped_high = (state_weight >= max_state_weight - 1e-6).float()
        clamped_low = (state_weight <= min_state_weight + 1e-6).float()
        clamp_frac = verl_F.masked_mean(clamped_high + clamped_low, response_mask)

        # Per-token action ratio diagnostics (for monitoring)
        ratio = torch.exp(log_ratio)
        ratio_mean = verl_F.masked_mean(ratio, response_mask)
        ratio_max = (ratio * response_mask).max()

    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        # IS weight (w_t = state_ratio × action_ratio, inclusive of current token)
        "actor/state_weight_mean": mean_sw.detach().item(),
        "actor/state_weight_max": max_sw.detach().item(),
        "actor/state_weight_std": std_sw.detach().item(),
        "actor/state_weight_clamp_frac": clamp_frac.detach().item(),
        # Per-token action ratio (for monitoring)
        "actor/ratio_mean": ratio_mean.detach().item(),
        "actor/ratio_max": ratio_max.detach().item(),
    }

    return pg_loss, pg_metrics
