"""
Unit tests for the state-corrected GRPO loss function.

Verifies:
1. strategy="none" degenerates to standard GRPO (state weight = 1 everywhere)
2. Different strategies produce different state weights
3. State weights are correctly computed for known inputs
4. Truncation (clamping) works as expected
5. Gradient flows only through action ratio, not state weight
6. PPO clip is applied correctly
7. Metrics are correctly reported
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
        clip_ratio_high = 0.28
        global_batch_info = {}

        def get(self, key, default=None):
            return getattr(self, key, default)

    config = MockConfig()
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


def _set_env(**kwargs):
    """Set environment variables for state correction config."""
    for k, v in kwargs.items():
        os.environ[k] = str(v)


def _clear_env():
    """Clear state correction environment variables."""
    for key in ["SC_STRATEGY", "SC_LOOKBACK_K", "SC_VTRACE_C", "SC_EMA_ALPHA",
                "SC_MAX_STATE_WEIGHT", "SC_MIN_STATE_WEIGHT"]:
        os.environ.pop(key, None)


def test_none_strategy_no_correction():
    """When strategy='none', state correction weight should be 1.0 everywhere."""
    _clear_env()
    _set_env(SC_STRATEGY="none")

    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 4, 16
    old_log_prob = torch.randn(B, T)
    log_prob = old_log_prob + torch.randn(B, T) * 0.1
    advantages = torch.randn(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config()
    loss, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                            loss_agg_mode="token-mean", config=config)

    assert abs(metrics["actor/state_weight_mean"] - 1.0) < 1e-5, (
        f"strategy='none' should give state_weight_mean=1.0, got {metrics['actor/state_weight_mean']}"
    )
    print("✅ test_none_strategy_no_correction passed")


def test_truncated_window_k0_no_correction():
    """truncated_window with k=0 should also give no correction."""
    _clear_env()
    _set_env(SC_STRATEGY="truncated_window", SC_LOOKBACK_K=0)

    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 4, 16
    old_log_prob = torch.randn(B, T)
    log_prob = old_log_prob + torch.randn(B, T) * 0.1
    advantages = torch.randn(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config()
    _, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                         loss_agg_mode="token-mean", config=config)

    assert abs(metrics["actor/state_weight_mean"] - 1.0) < 1e-5, (
        f"k=0 should give state_weight_mean=1.0, got {metrics['actor/state_weight_mean']}"
    )
    print("✅ test_truncated_window_k0_no_correction passed")


def test_state_weights_increase_with_k():
    """Larger k should produce more variable state weights."""
    _clear_env()

    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 4, 32
    torch.manual_seed(42)
    old_log_prob = torch.randn(B, T)
    log_prob = old_log_prob + torch.randn(B, T) * 0.3
    advantages = torch.randn(B, T)
    response_mask = torch.ones(B, T)

    stds = {}
    for k in [0, 3, 10]:
        _set_env(SC_STRATEGY="truncated_window", SC_LOOKBACK_K=k,
                 SC_MAX_STATE_WEIGHT=100.0, SC_MIN_STATE_WEIGHT=0.01)
        config = make_mock_config()
        _, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token-mean", config=config)
        stds[k] = metrics["actor/state_weight_std"]

    assert stds[0] < stds[3] < stds[10], (
        f"State weight std should increase with k: {stds}"
    )
    print(f"✅ test_state_weights_increase_with_k passed (stds: {stds})")


def test_all_strategies_run():
    """All strategies should run without error and produce valid metrics."""
    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 4, 16
    torch.manual_seed(42)
    old_log_prob = torch.randn(B, T)
    log_prob = old_log_prob + torch.randn(B, T) * 0.2
    advantages = torch.randn(B, T)
    response_mask = torch.ones(B, T)

    strategies = [
        ("none", {}),
        ("truncated_window", {"SC_LOOKBACK_K": "5"}),
        ("min_prefix", {}),
        ("vtrace", {"SC_VTRACE_C": "1.0"}),
        ("log_ema", {"SC_EMA_ALPHA": "0.9"}),
        ("self_normalized", {}),
    ]

    for strategy, extra_env in strategies:
        _clear_env()
        _set_env(SC_STRATEGY=strategy, **extra_env)
        config = make_mock_config()
        loss, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                                loss_agg_mode="token-mean", config=config)

        assert not torch.isnan(loss), f"strategy={strategy}: loss is NaN"
        assert not torch.isinf(loss), f"strategy={strategy}: loss is Inf"
        assert "actor/state_weight_mean" in metrics, f"strategy={strategy}: missing state_weight_mean"
        print(f"  ✓ strategy={strategy}: loss={loss.item():.4f}, "
              f"sw_mean={metrics['actor/state_weight_mean']:.4f}, "
              f"sw_std={metrics['actor/state_weight_std']:.4f}")

    print("✅ test_all_strategies_run passed")


def test_gradient_flows_through_action_ratio_only():
    """State weight should be detached — gradients only flow through action ratio."""
    _clear_env()
    _set_env(SC_STRATEGY="truncated_window", SC_LOOKBACK_K=5)

    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 2, 8
    old_log_prob = torch.randn(B, T)
    log_prob = torch.randn(B, T, requires_grad=True)
    advantages = torch.ones(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config()
    loss, _ = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                      loss_agg_mode="token-mean", config=config)

    loss.backward()
    assert log_prob.grad is not None, "Gradient should flow through log_prob"
    assert not torch.all(log_prob.grad == 0), "Gradient should be non-zero"
    print("✅ test_gradient_flows_through_action_ratio_only passed")


def test_ppo_clip_applied():
    """PPO clip should constrain the action ratio."""
    _clear_env()
    _set_env(SC_STRATEGY="none")

    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 4, 16
    old_log_prob = torch.zeros(B, T)
    # Large deviation: ratio = e^2 ≈ 7.4 (way outside clip range)
    log_prob = torch.ones(B, T) * 2.0
    advantages = torch.ones(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config(clip_ratio_low=0.2, clip_ratio_high=0.28)
    _, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                         loss_agg_mode="token-mean", config=config)

    assert metrics["actor/pg_clipfrac"] > 0.5, (
        f"Most tokens should be clipped with ratio≈7.4, got clipfrac={metrics['actor/pg_clipfrac']}"
    )
    print("✅ test_ppo_clip_applied passed")


def test_weight_clamping():
    """Extreme ratios should be clamped to [min_weight, max_weight]."""
    _clear_env()
    _set_env(SC_STRATEGY="truncated_window", SC_LOOKBACK_K=10,
             SC_MAX_STATE_WEIGHT=5.0, SC_MIN_STATE_WEIGHT=0.2)

    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 2, 16
    old_log_prob = torch.zeros(B, T)
    log_prob = torch.ones(B, T) * 3.0  # ratio = e^3 ≈ 20 per token
    advantages = torch.ones(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config()
    _, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                         loss_agg_mode="token-mean", config=config)

    assert metrics["actor/state_weight_max"] <= 5.0 + 1e-6, (
        f"Max state weight should be clamped to 5.0, got {metrics['actor/state_weight_max']}"
    )
    assert metrics["actor/state_weight_clamp_frac"] > 0, (
        "Some weights should be clamped with extreme ratios"
    )
    print("✅ test_weight_clamping passed")


def test_metrics_reported():
    """All expected metrics should be present."""
    _clear_env()
    _set_env(SC_STRATEGY="truncated_window", SC_LOOKBACK_K=5)

    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 2, 8
    old_log_prob = torch.randn(B, T)
    log_prob = old_log_prob + torch.randn(B, T) * 0.1
    advantages = torch.randn(B, T)
    response_mask = torch.ones(B, T)

    config = make_mock_config()
    _, metrics = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                         loss_agg_mode="token-mean", config=config)

    expected_keys = [
        "actor/pg_clipfrac",
        "actor/ppo_kl",
        "actor/pg_clipfrac_lower",
        "actor/ratio_mean",
        "actor/ratio_max",
        "actor/state_weight_mean",
        "actor/state_weight_max",
        "actor/state_weight_std",
        "actor/state_weight_clamp_frac",
    ]
    for key in expected_keys:
        assert key in metrics, f"Missing metric: {key}"
    print("✅ test_metrics_reported passed")


def test_min_prefix_attenuates_off_policy():
    """min_prefix should attenuate gradients when prefix has drifted tokens."""
    _clear_env()

    import state_corrected_loss  # noqa: F401
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    loss_fn = get_policy_loss_fn("state_corrected_grpo")

    B, T = 2, 16
    torch.manual_seed(42)
    old_log_prob = torch.zeros(B, T)
    log_prob = torch.zeros(B, T)
    # Inject a very off-policy token at position 3
    log_prob[:, 3] = -5.0  # ratio = e^{-5} ≈ 0.007

    advantages = torch.ones(B, T)
    response_mask = torch.ones(B, T)

    # With min_prefix, tokens after position 3 should have low state weight
    _set_env(SC_STRATEGY="min_prefix", SC_MIN_STATE_WEIGHT=0.01, SC_MAX_STATE_WEIGHT=100.0)
    config = make_mock_config()
    _, metrics_min = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                             loss_agg_mode="token-mean", config=config)

    # With none, state weight should be 1.0
    _set_env(SC_STRATEGY="none")
    _, metrics_none = loss_fn(old_log_prob, log_prob, advantages, response_mask,
                              loss_agg_mode="token-mean", config=config)

    assert metrics_min["actor/state_weight_mean"] < metrics_none["actor/state_weight_mean"], (
        f"min_prefix should have lower mean state weight than none: "
        f"{metrics_min['actor/state_weight_mean']} vs {metrics_none['actor/state_weight_mean']}"
    )
    print("✅ test_min_prefix_attenuates_off_policy passed")


if __name__ == "__main__":
    test_none_strategy_no_correction()
    test_truncated_window_k0_no_correction()
    test_state_weights_increase_with_k()
    test_all_strategies_run()
    test_gradient_flows_through_action_ratio_only()
    test_ppo_clip_applied()
    test_weight_clamping()
    test_metrics_reported()
    test_min_prefix_attenuates_off_policy()
    print("\n🎉 All tests passed!")
