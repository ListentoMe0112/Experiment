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
    UPDATED CONTRACT: Every strategy returns log_state_weight[t] = log w_t,
    where w_t is the EXCLUSIVE state importance weight covering tokens 0..t-1,
    NOT including the current action ratio r_t. At t=0 the state part is an
    empty product (= 1), so every strategy yields w_0 = 1 (log w_0 = 0).

    The loss is REINFORCE-style with separate state and action components,
    AND both SC clamp and PPO clip are applied as multiplicative GRADIENT MASKS
    (Plan A — unified gating). See `compute_policy_loss_state_corrected_grpo`
    for details.

        L = -Σ_t w_t · r_t · log π_θ(a_t|s_t) · A_t · m_t

    where
        w_t           = state_ratio  = ∏_{j=0}^{t-1} r_j  (detached)
        r_t           = action_ratio = π_θ/π_β            (detached, raw; NO in-place clip)
        m_t           = 1 iff SC clamp AND PPO clip both passed, else 0

    - "truncated_window"   : w_t = ∏_{j=max(0,t-k)}^{t-1} r_j   (bias-variance via k)
    - "min_prefix"         : w_t = min_{j<t} r_j               (MinPRO-style surrogate)
    - "vtrace"             : w_t = ∏_{j=0}^{t-1} min(c̄, r_j)    (V-trace truncated IS)
    - "log_ema"            : w_t = exp(EMA of log r_j, j<t)    (smooth tracking, l̄_0=0)
    - "self_normalized"    : w_t = ρ_{0:t-1} / mean_i ρ_{0:t-1}^{(i)}         (batch-normalized)
    - "baseline_corrected" : w_t = ρ_{0:t-1} / GeoMean_group(ρ_{0:t-1})
                             (cross-group CV, unbiased, corrects timestep drift)
    - "none"               : w_t = ∏_{j=0}^{t-1} r_j              (raw full product, no variance reduction)
    - "identity"           : w_t = 1                             (pure REINFORCE, no state correction)

Usage:
    Import this module before training to register the loss function, then set:
        actor_rollout_ref.actor.policy_loss.loss_mode=state_corrected_grpo
    Configure via environment variables:
        SC_STRATEGY=truncated_window  (default)
        SC_LOOKBACK_K=5              (for truncated_window)
        SC_VTRACE_C=1.0              (for vtrace)
        SC_EMA_ALPHA=0.9             (for log_ema)
        SC_GROUP_SIZE=8              (for baseline_corrected; must match rollout.n)
        SC_MAX_STATE_WEIGHT=5.0
        SC_MIN_STATE_WEIGHT=0.2
        SC_PPO_CLIP_EPSILON=0.2      (PPO clip epsilon for action ratio r_t)

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
    Truncated window: w_t^{(k)} = ∏_{j=max(0,t-k)}^{t-1} r_j  (EXCLUDING current token t)

    UPDATED CONTRACT (shared by ALL strategies in this file):
        log_state_weight[t] = log w_t, where w_t is the EXCLUSIVE state ratio
        that covers tokens 0..t-1 (NOT including the current token t).
        The current action ratio r_t is handled separately in the loss function.

    Boundary: at t=0 the state part is an empty product (= 1), so every
    strategy must yield w_0 = 1 (equivalently log_state_weight[:, 0] = 0).

    k=0  → w_t = 1 only (no state correction, empty product at all t)
    k>0  → correct recent k tokens of state mismatch (excluding current)
    k=-1 → full trajectory w_t = ∏_{j=0}^{t-1} r_j (exact but high variance)
    """
    masked_log_ratio = log_ratio * response_mask
    # cumsum[t] = Σ_{j=0}^{t} log r_j (inclusive of current token)
    cumsum = torch.cumsum(masked_log_ratio, dim=-1)

    T = log_ratio.shape[1]

    if lookback_k == 0:
        # w_t = 1 only → log_weight = 0 (empty product at all t)
        return torch.zeros_like(log_ratio)
    elif lookback_k < 0 or lookback_k >= T:
        # Full trajectory: w_t = ∏_{j=0}^{t-1} r_j (exclusive of current token)
        # = cumsum[t-1] if t>0, or 0 if t=0
        exclusive_cumsum = cumsum - masked_log_ratio  # cumsum[t] - log_ratio[t]
        return exclusive_cumsum
    else:
        # Window [t-k, t-1]: cumsum[t-1] - cumsum[t-k-1]
        # For t < k+1, window starts at 0
        exclusive_cumsum = cumsum - masked_log_ratio
        shifted = torch.zeros_like(cumsum)
        shifted[:, lookback_k:] = cumsum[:, :-lookback_k]
        return exclusive_cumsum - shifted


def _compute_state_weight_min_prefix(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    MinPRO-style surrogate: w_t = min_{j < t} r_j  (EXCLUDING current token t)

    UPDATED CONTRACT (shared by ALL strategies in this file):
        log_state_weight[t] = log w_t, exclusive of current token t.
        Here w_t is a heuristic surrogate for the exclusive prefix product.

    Uses the minimum token ratio up to but NOT including current token as a
    stable, non-cumulative surrogate for the exclusive prefix product.

    Returns log(min_{j<t} r_j) for numerical stability.
    """
    B, T = log_ratio.shape

    # Replace masked positions with +inf so they don't affect the min
    large_val = torch.tensor(1e9, device=log_ratio.device, dtype=log_ratio.dtype)
    masked_log_ratio = torch.where(response_mask.bool(), log_ratio, large_val)

    # Compute running min of log_ratio for positions j < t (excluding current)
    # cummin_vals[t] = min(log_ratio[0], ..., log_ratio[t-1])
    # For t=0, use a large value (no valid j < 0)
    cummin_vals = torch.full_like(log_ratio, large_val)
    for t in range(1, T):
        cummin_vals[:, t] = torch.min(cummin_vals[:, t-1], masked_log_ratio[:, t-1])
    
    # For t=0, set to 0 (log(1) = 0 for empty product)
    cummin_vals[:, 0] = 0.0
    
    return cummin_vals


def _compute_state_weight_vtrace(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    c_bar: float = 1.0,
) -> torch.Tensor:
    """
    V-trace style: w_t = ∏_{j=0}^{t-1} min(c̄, r_j)  (EXCLUDING current token t)

    UPDATED CONTRACT (shared by ALL strategies in this file):
        log_state_weight[t] = log w_t, exclusive of current token t.
        Each factor (excluding r_t at position t) is clipped to c̄ before
        being multiplied into the cumulative product.

    Clip each factor BEFORE multiplying. With c̄ ≤ 1, the product is
    monotonically non-increasing, bounding variance growth.

    Returns log(w_t).
    """
    # Clip log_ratio so that r_j ≤ c_bar → log_ratio ≤ log(c_bar)
    log_c_bar = torch.tensor(c_bar, device=log_ratio.device, dtype=log_ratio.dtype).log()
    clipped_log_ratio = torch.clamp(log_ratio, max=log_c_bar.item())
    masked = clipped_log_ratio * response_mask

    # Exclusive cumsum: cumsum[t-1] = Σ_{j=0}^{t-1} clipped_log_ratio[j]
    # = inclusive_cumsum[t] - masked[t]
    inclusive_cumsum = torch.cumsum(masked, dim=-1)
    log_state_weight = inclusive_cumsum - masked

    return log_state_weight


def _compute_state_weight_log_ema(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    alpha: float = 0.9,
) -> torch.Tensor:
    """
    Exponential moving average in log-space (EXCLUDING current token t):

        l̄_t = α · l̄_{t-1} + (1-α) · log r_{t-1},   with l̄_0 := 0

    Uses exp(l̄_t) as the state weight for token t.

    UPDATED CONTRACT (shared by ALL strategies in this file):
        log_state_weight[t] = log w_t, exclusive of current token t.
        Here w_t is a smoothed surrogate for the exclusive prefix product.

    Boundary: at t=0 the state part is an empty product (= 1), so we
    initialize l̄_0 = 0 (log(1) = 0), consistent with all strategies.

    Smoother than min, captures gradual drift rather than only worst-case.

    Returns log(state_weight) = l̄_t for each position t.
    """
    B, T = log_ratio.shape
    log_state_weight = torch.zeros_like(log_ratio)
    # Initialize EMA with 0 so that at t=0: l̄_0 = 0 (log(1) = 0 for empty product)
    ema = torch.zeros(B, device=log_ratio.device, dtype=log_ratio.dtype)

    for t in range(T):
        # For t=0: keep ema = 0 (empty product)
        # For t>0: update EMA with previous token's ratio
        if t > 0:
            mask_t_minus_1 = response_mask[:, t-1]
            ema = mask_t_minus_1 * (alpha * ema + (1 - alpha) * log_ratio[:, t-1]) + (1 - mask_t_minus_1) * ema
        
        log_state_weight[:, t] = ema

    return log_state_weight


def _compute_state_weight_self_normalized(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Self-normalized IS: w̃_t^{(i)} = ρ_{0:t-1}^{(i)} / mean_i ρ_{0:t-1}^{(i)}  (EXCLUDING current token t)

    UPDATED CONTRACT (shared by ALL strategies in this file):
        log_state_weight[t] = log w_t, exclusive of current token t.
        The cumulative product ρ_{0:t-1} = ∏_{j=0}^{t-1} r_j excludes r_t.

    Normalization is done across the batch dimension at each position (we
    subtract logsumexp and add log(batch_size), so the mean is 1). This
    naturally fits GRPO's group structure and prevents weight explosion.

    Returns log(w̃_t).

    NOTE: This operates across the WHOLE batch dimension, not per-prompt
    group. If you want per-group normalization, use baseline_corrected instead.
    """
    masked = log_ratio * response_mask
    # Exclusive cumsum: log ρ_{0:t-1} = Σ_{j=0}^{t-1} log r_j
    # = inclusive_cumsum[t] - masked[t]
    inclusive_cumsum = torch.cumsum(masked, dim=-1)
    log_cumulative = inclusive_cumsum - masked

    # Self-normalize across batch dimension at each position
    # log(Σ_i exp(log_cumulative_i)) via logsumexp
    log_sum = torch.logsumexp(log_cumulative, dim=0, keepdim=True)  # (1, T)
    log_state_weight = log_cumulative - log_sum + torch.tensor(log_cumulative.shape[0], device=log_ratio.device).float().log()

    return log_state_weight


def _compute_state_weight_baseline_corrected(
    log_ratio: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Cross-group baseline subtraction (Control Variate, CV) in log-space:

        w_t^{(i)} = ρ_{0:t-1}^{(i)} / GeoMean_{j in group(i)}( ρ_{0:t-1}^{(j)} )

    UPDATED CONTRACT (shared by ALL strategies in this file):
        log_state_weight[t] = log w_t, exclusive of current token t.
        Here w_t is the state ratio part only, excluding the current action ratio r_t.

    Rationale (see notes.md §8.1.6):
      • ρ_{0:t-1} = ∏_{j<t} r_j is the state-ratio part (only depends on s_t).
      • Subtracting a state-dependent baseline b(s_t) preserves unbiasedness
        under the off-policy score function identity:
              E_{π_β}[ r_t · b(s_t) · ∇ log π_θ ] = 0  for any b(s_t).
      • "Cross-group" means: the baseline is computed per-position t, averaged
        over the G rollouts that share the same prompt (GRPO group). This:
          (1) respects s_t-measurability (no future actions leaked),
          (2) corrects timestep drift E[log ρ_{0:t-1}] ≈ -t · KL,
          (3) is naturally compatible with GRPO's group structure.
      • Geometric centering (subtract-in-log, divide-in-original) preserves
        the multiplicative structure of IS ratios and is robust to outliers.

    LIMITATION: This corrects "timestep drift" (mean shift across t), but does
    NOT reduce "sample spread" (within-group variance). Must be combined with
    the outer clamp [min_state_weight, max_state_weight] to control outliers.

    Args:
        log_ratio: log(π_θ / π_β) per token, shape (B, T) where B = num_prompts * G.
        response_mask: Valid-token mask, shape (B, T).
        group_size: G, number of rollouts per prompt. Must divide B.

    Returns:
        log_state_weight: log(w_t), shape (B, T).
    """
    B, T = log_ratio.shape
    assert B % group_size == 0, (
        f"baseline_corrected requires B ({B}) divisible by group_size ({group_size}). "
        f"Check actor_rollout_ref.rollout.n == SC_GROUP_SIZE."
    )
    num_groups = B // group_size

    masked_log_ratio = log_ratio * response_mask

    # Exclusive cumulative log-ratio: log ρ_{0:t-1} = Σ_{j=0}^{t-1} log r_j
    # (i.e., prefix does NOT include the current token t; at t=0 this is 0,
    # the empty product, which correctly gives w_0 = 1).
    # Standard trick: exclusive cumsum = inclusive_cumsum - current_value
    inclusive_cumsum = torch.cumsum(masked_log_ratio, dim=-1)
    log_prefix = inclusive_cumsum - masked_log_ratio  # shape (B, T)

    # Reshape to (num_groups, G, T) for cross-group (within-prompt) averaging
    log_prefix_grouped = log_prefix.view(num_groups, group_size, T)
    mask_grouped = response_mask.view(num_groups, group_size, T)

    # Mask-aware mean across the G samples within each group, at each position t.
    # Only count samples where token t is valid (response_mask == 1).
    # If no valid samples at position t in a group, fall back to 0 (no centering).
    mask_sum = mask_grouped.sum(dim=1, keepdim=True)  # (num_groups, 1, T)
    safe_mask_sum = torch.where(mask_sum > 0, mask_sum, torch.ones_like(mask_sum))
    log_prefix_mean = (log_prefix_grouped * mask_grouped).sum(dim=1, keepdim=True) / safe_mask_sum
    # Where the group has zero valid samples at t, suppress centering (avoid NaN)
    log_prefix_mean = torch.where(mask_sum > 0, log_prefix_mean, torch.zeros_like(log_prefix_mean))

    # Geometric centering: subtract group-mean in log-space
    # Equivalent to dividing by the group's geometric mean of ρ_{0:t-1}.
    log_prefix_centered = (log_prefix_grouped - log_prefix_mean).view(B, T)

    # Final log-weight: centered log prefix (state ratio only, excluding r_t)
    # Contract check: at t=0, log_prefix = 0 and its group-mean is also 0,
    # so log_state_weight[0] = 0 = log(1) (consistent with all strategies).
    log_state_weight = log_prefix_centered * response_mask

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

    UPDATED IMPLEMENTATION (Plan A):
    Strategies separate state_ratio and action_ratio into distinct components,
    AND both SC clamp and PPO clip are implemented as multiplicative masks
    on the loss (unified gating style).

        w_t = state_ratio part = ∏_{j=0}^{t-1} r_j   (EXCLUDING current token t)
        r_t = action_ratio     = π_θ(a_t|s_t)/π_β(a_t|s_t)   (raw, detached; NO
                                 in-place clip applied inside the loss)
        combined_mask[t] = 1 iff token t survives BOTH
             (a) SC clamp:  log w_t ∈ [log min_w, log max_w]
             (b) PPO clip:  r_t ∈ [1 - ε, 1 + ε]

    The loss function is:
        loss = -Σ_t w_t · r_t · log π_θ(a_t|s_t) · A_t · combined_mask[t]

    Rationale for Plan A:
      • SC clamp and PPO clip both express "this token is too off-policy to
        trust" — treating them the same way (as gradient gates) is cleaner
        than mixing a clamp-and-continue (PPO) with a drop (SC).
      • On surviving tokens, r_t is already within [1-ε, 1+ε] by definition,
        so using the raw ratio here is numerically identical to clipping it
        in-place, but with clearer semantics.
      • Unified masking enables a single `joint_active_frac` metric that
        faithfully reflects which tokens drive the gradient.

    Gradient flow:
        ∇_θ L = -w_t · r_t · A_t · combined_mask[t] · ∇_θ log π_θ
                (w_t and r_t are both detached; REINFORCE-style.)

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
    # PPO clip epsilon for action ratio r_t
    ppo_clip_epsilon = float(os.environ.get("SC_PPO_CLIP_EPSILON", "0.2"))
    # Group size for baseline_corrected strategy: number of rollouts per prompt
    # Must match actor_rollout_ref.rollout.n in the verl config.
    group_size = int(os.environ.get("SC_GROUP_SIZE", "8"))

    # --- Per-token log importance ratio: log(π_θ / π_β) ---
    # This is the ratio between current policy and ROLLOUT policy.
    # old_log_prob = log π_β (rollout policy that generated the data)
    # log_prob = log π_θ (current policy being optimized)
    log_ratio = log_prob - old_log_prob
    
    # Apply PPO-style clip to action ratio r_t
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, min=1.0 - ppo_clip_epsilon, max=1.0 + ppo_clip_epsilon)
    clipped_log_ratio = torch.log(clipped_ratio)

    # --- Compute state weight (state_ratio only, excluding current action) using chosen strategy ---
    #
    # =========================== UPDATED CONTRACT ===========================
    # Every `_compute_state_weight_*` helper now returns a tensor `log_state_weight`
    # of shape (B, T) that satisfies:
    #
    #   log_state_weight[b, t]  ==  log w_t^{(b)}
    #
    # where w_t is the state importance weight at token t, covering ONLY the
    # state ratio part (prefix tokens 0..t-1), EXCLUDING the current action ratio r_t:
    #
    #   w_t  =  state_ratio part depending on o_{<t} = ∏_{j=0}^{t-1} r_j
    #
    # Boundary: at t=0 the state part is an empty product (= 1), so every
    # strategy must yield w_0 = 1 (equivalently log_state_weight[:, 0] = 0).
    # ========================================================================
    if strategy == "truncated_window":
        log_state_weight = _compute_state_weight_truncated_window(
            clipped_log_ratio, response_mask, lookback_k
        )
    elif strategy == "min_prefix":
        log_state_weight = _compute_state_weight_min_prefix(
            clipped_log_ratio, response_mask
        )
    elif strategy == "vtrace":
        log_state_weight = _compute_state_weight_vtrace(
            clipped_log_ratio, response_mask, c_bar=vtrace_c
        )
    elif strategy == "log_ema":
        log_state_weight = _compute_state_weight_log_ema(
            clipped_log_ratio, response_mask, alpha=ema_alpha
        )
    elif strategy == "self_normalized":
        log_state_weight = _compute_state_weight_self_normalized(
            clipped_log_ratio, response_mask
        )
    elif strategy == "baseline_corrected":
        # Cross-group baseline subtraction (control variate) in log-space.
        # w_t = ρ_{1:t-1} / GeoMean_group(ρ_{1:t-1})
        log_state_weight = _compute_state_weight_baseline_corrected(
            clipped_log_ratio, response_mask, group_size=group_size
        )
    elif strategy == "none":
        # Raw full product: w_t = ∏_{j=0}^{t-1} r_j = state_ratio only
        masked_log_ratio = clipped_log_ratio * response_mask
        log_state_weight = torch.cumsum(masked_log_ratio, dim=-1) - masked_log_ratio  # exclusive cumsum
    elif strategy == "identity":
        # No state correction: w_t = 1 for all tokens
        log_state_weight = torch.zeros_like(clipped_log_ratio)
    else:
        raise ValueError(f"Unknown SC_STRATEGY: {strategy}. "
                         f"Choose from: truncated_window, min_prefix, vtrace, log_ema, "
                         f"self_normalized, baseline_corrected, none, identity")

    # --- Clamp state weight (state_ratio only) and detach ---
    # w_t covers t=0..T (exclusive of current action ratio).
    # Clamp in log-space then exponentiate. Detach so gradient flows
    # only through log_prob (REINFORCE-style).
    log_min = math.log(min_state_weight)
    log_max = math.log(max_state_weight)

    # Identify tokens where state_weight would be clamped (off-policy too far).
    # For these tokens, we detach log_prob so no gradient flows through them,
    # but the loss VALUE is preserved (keeps loss magnitude stable for agg_loss).
    with torch.no_grad():
        sc_below_mask = (log_state_weight < log_min).float()  # clamped at lower bound
        sc_above_mask = (log_state_weight > log_max).float()  # clamped at upper bound
        not_clamped_mask = (
            (log_state_weight >= log_min) & (log_state_weight <= log_max)
        ).float()  # 1.0 = within bounds (gradient flows), 0.0 = clamped (no gradient)

        # === Plan A: PPO clip as a MASK (same style as SC clamp) ===
        # A token is PPO-clipped when its raw ratio falls outside [1-ε, 1+ε].
        # We treat such tokens exactly like SC-clamped tokens: zero out their
        # gradient contribution via multiplicative masking. This unifies the
        # two gating mechanisms under a single "combined_mask" and avoids the
        # subtle issue of letting saturated ratios leak through the loss.
        clip_lower_mask = (ratio < 1.0 - ppo_clip_epsilon)
        clip_upper_mask = (ratio > 1.0 + ppo_clip_epsilon)
        clip_mask = clip_lower_mask | clip_upper_mask        # True = clipped
        not_clipped_mask = (~clip_mask).float()              # 1.0 = within PPO band

        # Combined gating: a token contributes to the gradient ONLY if it
        # survives BOTH the SC clamp and the PPO clip.
        combined_mask = not_clamped_mask * not_clipped_mask

    log_state_weight = torch.clamp(log_state_weight, min=log_min, max=log_max)
    state_weight = torch.exp(log_state_weight).detach()  # stop gradient!

    # --- Final loss: REINFORCE-style with separate state_weight and action_ratio ---
    # loss = -w_t · r_t · log π_θ(a_t|s_t) · A_t · combined_mask
    # where w_t           = state_ratio (excluding current action), detached
    #       r_t           = action_ratio π_θ/π_β, detached  (NO in-place clip;
    #                       out-of-band tokens are masked out instead)
    #       combined_mask = 1 iff token survives both SC clamp and PPO clip
    #
    # For survived tokens (combined_mask == 1), r_t is guaranteed to be within
    # [1-ε, 1+ε] by construction, so using `ratio` or `clipped_ratio` here is
    # numerically identical. Using the raw `ratio` keeps the semantics cleaner.
    pg_losses = -state_weight * ratio.detach() * log_prob * advantages * combined_mask

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

        # State weight diagnostics (w_t = state_ratio only, excluding current action)
        mean_sw = verl_F.masked_mean(state_weight, response_mask)
        max_sw = (state_weight * response_mask).max()
        std_sw = verl_F.masked_mean((state_weight - mean_sw) ** 2, response_mask).sqrt()

        # Action ratio diagnostics (r_t raw; `clipped_ratio` kept for comparison only,
        # note: under Plan A it is NO LONGER used inside pg_losses).
        ratio_mean = verl_F.masked_mean(ratio, response_mask)
        ratio_max = (ratio * response_mask).max()
        clipped_ratio_mean = verl_F.masked_mean(clipped_ratio, response_mask)
        clipped_ratio_max = (clipped_ratio * response_mask).max()

        # PPO clip statistics (mask-based gating, computed above before the loss).
        clip_frac = verl_F.masked_mean(clip_mask.float(), response_mask)
        clip_lower_frac = verl_F.masked_mean(clip_lower_mask.float(), response_mask)
        clip_upper_frac = verl_F.masked_mean(clip_upper_mask.float(), response_mask)

        # SC clamp statistics (also mask-based).
        clamp_frac = verl_F.masked_mean(1.0 - not_clamped_mask, response_mask)
        sc_clamp_lower_frac = verl_F.masked_mean(sc_below_mask, response_mask)
        sc_clamp_upper_frac = verl_F.masked_mean(sc_above_mask, response_mask)

        # === Joint active-ratio statistics: SC-clamp ∩ PPO-clip both passed ===
        # Under Plan A, `combined_mask` is the SOLE gate controlling whether a
        # token contributes to the policy gradient. `joint_active_frac` is the
        # fraction of tokens that survive BOTH gates, matching the actual loss
        # behavior. It is paired with `joint_excluded_frac` (they sum to 1).
        joint_active_frac = verl_F.masked_mean(combined_mask, response_mask)
        joint_excluded_frac = verl_F.masked_mean(1.0 - combined_mask, response_mask)

        # Exclusive breakdown (sum = joint_excluded_frac, no double counting)
        only_sc_excluded_mask = (1.0 - not_clamped_mask) * not_clipped_mask       # SC-only
        only_ppo_clipped_mask = not_clamped_mask * (1.0 - not_clipped_mask)       # PPO-only
        both_excluded_mask = (1.0 - not_clamped_mask) * (1.0 - not_clipped_mask)  # both
        only_sc_frac = verl_F.masked_mean(only_sc_excluded_mask, response_mask)
        only_ppo_frac = verl_F.masked_mean(only_ppo_clipped_mask, response_mask)
        both_frac = verl_F.masked_mean(both_excluded_mask, response_mask)

        # Log-space diagnostics: useful for baseline_corrected / log_ema / vtrace.
        # For baseline_corrected, log_state_weight should be centered near 0 and
        # tight around log r_t, confirming timestep drift has been removed.
        log_sw_mean = verl_F.masked_mean(log_state_weight, response_mask)
        log_sw_std = verl_F.masked_mean(
            (log_state_weight - log_sw_mean) ** 2, response_mask
        ).sqrt()

    pg_metrics = {
        "actor/ppo_kl": ppo_kl.detach().item(),
        # State weight (w_t = state_ratio only, excluding current action)
        "actor/state_weight_mean": mean_sw.detach().item(),
        "actor/state_weight_max": max_sw.detach().item(),
        "actor/state_weight_std": std_sw.detach().item(),
        "actor/state_weight_clamp_frac": clamp_frac.detach().item(),
        "actor/state_weight_clamp_lower_frac": sc_clamp_lower_frac.detach().item(),
        "actor/state_weight_clamp_upper_frac": sc_clamp_upper_frac.detach().item(),
        # === Joint-gate diagnostics (SC clamp ∩ PPO clip both passed) ===
        # Fraction of tokens that survive BOTH SC clamp AND PPO clip
        # (paired with joint_excluded_frac below; they sum to 1).
        "actor/joint_active_frac": joint_active_frac.detach().item(),
        # Action ratio diagnostics (raw r_t; under Plan A no in-place clip is applied
        # to r_t inside the loss — out-of-band tokens are masked out instead.)
        "actor/ratio_mean": ratio_mean.detach().item(),
        "actor/ratio_max": ratio_max.detach().item(),
        "actor/clipped_ratio_mean": clipped_ratio_mean.detach().item(),
        "actor/clipped_ratio_max": clipped_ratio_max.detach().item(),
        "actor/ppo_clip_frac": clip_frac.detach().item(),
        "actor/ppo_clip_lower_frac": clip_lower_frac.detach().item(),
        "actor/ppo_clip_upper_frac": clip_upper_frac.detach().item(),
        # Tokens excluded by either gate (union): 1 - joint_active_frac.
        "actor/joint_excluded_frac": joint_excluded_frac.detach().item(),
        # Exclusive breakdown (sum == joint_excluded_frac, no double counting)
        "actor/excluded_sc_only_frac": only_sc_frac.detach().item(),
        "actor/excluded_ppo_only_frac": only_ppo_frac.detach().item(),
        "actor/excluded_both_frac": both_frac.detach().item(),
        # Log-space diagnostics (drift/spread decomposition)
        "actor/log_state_weight_mean": log_sw_mean.detach().item(),
        "actor/log_state_weight_std": log_sw_std.detach().item(),
    }

    return pg_loss, pg_metrics
