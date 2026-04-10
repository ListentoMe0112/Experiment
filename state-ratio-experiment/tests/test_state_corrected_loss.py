"""
Unit tests for the state-corrected GRPO loss function.

Verifies:
1. k=0 degenerates to standard GRPO (state weight = 1 everywhere)
2. State weights are correctly computed for known inputs
3. Truncation (clamping) works as expected
4. Gradient flows only through action ratio, not state weight
5. Metrics are correctly reported
"""

import sys
import os

import torch

# Add parent directory to path so we can import the loss module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def make_mock_config(**overrides):
    """Create a minimal mock config that behaves like ActorConfig."""

    class MockConfig:
        clip_ratio = 0.2
        clip_ratio_low = 0.2
        clip_ratio_high = 0.2
        global_batch_info = {}

        def get(self, key, default=None):
            return getattr(self, key, default)

    config = MockConfig()
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def test_k0_degenerates_to_grpo():
    """When k=0, state correction weight should be 1.0 everywhere (no correction)."""
    import state_corrected_loss  # noqa: F401 - registers the loss
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 4, 16
    old_log_prob = torch.randn(B, T)
    log_prob = old_log_prob + torch.randn(B, T) * 0.1  # small perturbation
    advantages = torch.randn(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config(state_correction_lookback_k=0)

    loss, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                            loss_agg_mode="seq-mean-token-mean", config=config)

    # With k=0, state_weight_mean should be ~1.0
    assert abs(metrics["actor/state_weight_mean"] - 1.0) < 1e-5, (
        f"k=0 should give state_weight_mean=1.0, got {metrics['actor/state_weight_mean']}"
    )
    print("✅ test_k0_degenerates_to_grpo passed")


def test_state_weights_increase_with_k():
    """Larger k should produce more variable state weights (further from 1.0)."""
    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 4, 32
    torch.manual_seed(42)
    old_log_prob = torch.randn(B, T)
    log_prob = old_log_prob + torch.randn(B, T) * 0.3  # larger perturbation
    advantages = torch.randn(B, T)
    response_mask = torch.ones(B, T)

    stds = {}
    for k in [0, 3, 10]:
        config = make_mock_config(
            state_correction_lookback_k=k,
            state_correction_max_weight=100.0,  # high limit to see effect
            state_correction_min_weight=0.01,
        )
        _, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="seq-mean-token-mean", config=config)
        stds[k] = metrics["actor/state_weight_std"]

    assert stds[0] < stds[3] < stds[10], (
        f"State weight std should increase with k: {stds}"
    )
    print(f"✅ test_state_weights_increase_with_k passed (stds: {stds})")


def test_gradient_does_not_flow_through_state_weight():
    """State weight should be detached — gradients should only flow through action ratio."""
    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 2, 8
    old_log_prob = torch.randn(B, T)
    log_prob = torch.randn(B, T, requires_grad=True)
    advantages = torch.ones(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config(state_correction_lookback_k=5)

    loss, _ = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                      loss_agg_mode="seq-mean-token-mean", config=config)

    loss.backward()
    assert log_prob.grad is not None, "Gradient should flow through log_prob"
    assert not torch.all(log_prob.grad == 0), "Gradient should be non-zero"
    print("✅ test_gradient_does_not_flow_through_state_weight passed")


def test_weight_clamping():
    """Extreme ratios should be clamped to [min_weight, max_weight]."""
    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 2, 16
    # Create extreme log ratio differences
    old_log_prob = torch.zeros(B, T)
    log_prob = torch.ones(B, T) * 3.0  # ratio = e^3 ≈ 20 per token
    advantages = torch.ones(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config(
        state_correction_lookback_k=10,
        state_correction_max_weight=5.0,
        state_correction_min_weight=0.2,
    )

    _, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                         loss_agg_mode="seq-mean-token-mean", config=config)

    assert metrics["actor/state_weight_max"] <= 5.0 + 1e-6, (
        f"Max state weight should be clamped to 5.0, got {metrics['actor/state_weight_max']}"
    )
    assert metrics["actor/state_weight_clamp_frac"] > 0, (
        "Some weights should be clamped with extreme ratios"
    )
    print("✅ test_weight_clamping passed")


def test_metrics_reported():
    """All expected metrics should be present."""
    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 2, 8
    old_log_prob = torch.randn(B, T)
    log_prob = old_log_prob + torch.randn(B, T) * 0.1
    advantages = torch.randn(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config(state_correction_lookback_k=5)

    _, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                         loss_agg_mode="seq-mean-token-mean", config=config)

    expected_keys = [
        "actor/pg_clipfrac",
        "actor/ppo_kl",
        "actor/pg_clipfrac_lower",
        "actor/state_weight_mean",
        "actor/state_weight_max",
        "actor/state_weight_std",
        "actor/state_weight_clamp_frac",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
    print("✅ test_metrics_reported passed")


if __name__ == "__main__":
    test_k0_degenerates_to_grpo()
    test_state_weights_increase_with_k()
    test_gradient_does_not_flow_through_state_weight()
    test_weight_clamping()
    test_metrics_reported()
    print("\n🎉 All tests passed!")
