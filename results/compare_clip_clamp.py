#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare PPO clip / SC clamp / joint-excluded fractions across the runs:
  - GRPO baseline         (verl logs `actor/pg_clipfrac`, `actor/pg_clipfrac_lower`)
  - GSPO baseline         (same clip metrics as GRPO; sequence-level ratio)
  - SC-mask             (logs both `ppo_clip_*` and `state_weight_clamp_*`)
  - SC (min_prefix / identity)

Layout (2 x 3):
  Row 1
    [0] Total excluded fraction (union)
        - GRPO baseline     : actor/pg_clipfrac      (only PPO clip exists)
        - SC (*)            : actor/joint_excluded_frac  (PPO clip OR SC clamp)
    [1] PPO clip fraction  (apples-to-apples PPO side)
        - GRPO baseline     : actor/pg_clipfrac
        - SC (*)            : actor/ppo_clip_frac
    [2] SC state-weight clamp fraction  (SC-only)
        - SC (*)            : actor/state_weight_clamp_frac
  Row 2  (upper/lower decomposition)
    [3] PPO clip — upper   (positive-advantage tokens clipped at 1+eps)
    [4] PPO clip — lower   (negative-advantage tokens clipped at 1-eps)
    [5] SC clamp — upper/lower  (both SC runs, two lines each)

Output: results/comparison_clip_clamp.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent

RUNS = {
    "GRPO baseline": RESULTS_DIR / "grpo_baseline_qwen2.5_1.5b.jsonl",
    "GSPO baseline": RESULTS_DIR / "gspo_baseline_qwen2.5_1.5b.jsonl",
    "SC-mask (c=2)": RESULTS_DIR / "sc_none_qwen2.5_1.5b_clamp2.jsonl",
    "SC-mask (c=4)": RESULTS_DIR / "sc_none_qwen2.5_1.5b_clamp4.jsonl",
    "SC-mask (c=1e5)": RESULTS_DIR / "sc_none_qwen2.5_1.5b_clamp100000.jsonl",
    "SC-min-prefix":
        RESULTS_DIR / "sc_min_prefix_qwen2.5_1.5b.jsonl",
    "SC-identity":
        RESULTS_DIR / "sc_identity_qwen2.5_1.5b.jsonl",
}

COLORS = {
    "GRPO baseline":                "#1f77b4",
    "GSPO baseline":                "#17becf",
    "SC-mask (c=2)":             "#d62728",
    "SC-mask (c=4)":             "#e377c2",
    "SC-mask (c=1e5)":           "#8b1a1a",
    "SC-min-prefix":              "#ff7f0e",
    "SC-identity":                "#9467bd",
}

# All metric keys we may look up (missing keys are just skipped per-run).
METRIC_KEYS = [
    "actor/pg_clipfrac",
    "actor/pg_clipfrac_lower",
    "actor/ppo_clip_frac",
    "actor/ppo_clip_lower_frac",
    "actor/ppo_clip_upper_frac",
    "actor/state_weight_clamp_frac",
    "actor/state_weight_clamp_lower_frac",
    "actor/state_weight_clamp_upper_frac",
    "actor/joint_excluded_frac",
]


def load_run(path: Path):
    """Return dict[metric] -> (steps, values)."""
    buckets = {k: ([], []) for k in METRIC_KEYS}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            data = rec.get("data", {})
            gs = data.get("training/global_step")
            if gs is None:
                continue
            for k in METRIC_KEYS:
                if k in data and data[k] is not None:
                    buckets[k][0].append(gs)
                    buckets[k][1].append(data[k])
    return buckets


def smooth(y, k=5):
    y = np.asarray(y, dtype=float)
    if len(y) < k or k <= 1:
        return y
    kernel = np.ones(k) / k
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")[: len(y)]


def plot_metric(ax, runs_data, mapping, title, ylabel="fraction"):
    """
    mapping: dict[run_name] -> metric_key (or list of (metric_key, line_label_suffix, linestyle))
    """
    for name, spec in mapping.items():
        bkt = runs_data[name]
        if isinstance(spec, str):
            specs = [(spec, "", "-")]
        else:
            specs = spec
        for key, suffix, ls in specs:
            if key not in bkt:
                continue
            xs, ys = bkt[key]
            if not ys:
                continue
            label = name + (f" {suffix}" if suffix else "")
            ax.plot(xs, ys, color=COLORS[name], alpha=0.20, linewidth=1,
                    linestyle=ls)
            ax.plot(xs, smooth(ys, k=5), color=COLORS[name], linewidth=2,
                    linestyle=ls, label=label)
    ax.set_title(title)
    ax.set_xlabel("global step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


def main():
    runs_data = {name: load_run(p) for name, p in RUNS.items()}

    # Sanity print
    print("=" * 70)
    print("Clip / clamp fractions — averaged over the run")
    print("=" * 70)
    for name, bkt in runs_data.items():
        print(f"\n[{name}]")
        for k, (_, ys) in bkt.items():
            if ys:
                print(f"  {k:<42s} mean={np.mean(ys):.4f}  "
                      f"max={np.max(ys):.4f}  n={len(ys)}")

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.flatten()

    # --- [0] Total excluded fraction (union) -------------------------------
    # GRPO / GSPO baselines have no SC side, so their "total" == PPO clip itself.
    plot_metric(
        axes[0], runs_data,
        {
            "GRPO baseline":                "actor/pg_clipfrac",
            "GSPO baseline":                "actor/pg_clipfrac",
            "SC-mask (c=2)":             "actor/joint_excluded_frac",
            "SC-mask (c=4)":             "actor/joint_excluded_frac",
            "SC-mask (c=1e5)":             "actor/joint_excluded_frac",
            "SC-min-prefix":              "actor/joint_excluded_frac",
            "SC-identity":                "actor/joint_excluded_frac",
        },
        title="Total excluded fraction  (PPO clip ∪ SC clamp)",
    )

    # --- [1] PPO clip fraction --------------------------------------------
    plot_metric(
        axes[1], runs_data,
        {
            "GRPO baseline":                "actor/pg_clipfrac",
            "GSPO baseline":                "actor/pg_clipfrac",
            "SC-mask (c=2)":             "actor/ppo_clip_frac",
            "SC-mask (c=4)":             "actor/ppo_clip_frac",
            "SC-mask (c=1e5)":             "actor/ppo_clip_frac",
            "SC-min-prefix":              "actor/ppo_clip_frac",
            "SC-identity":                "actor/ppo_clip_frac",
        },
        title="PPO clip fraction  (total)",
    )

    # --- [2] SC clamp fraction (SC runs only) -----------------------------
    plot_metric(
        axes[2], runs_data,
        {
            "SC-mask (c=2)":             "actor/state_weight_clamp_frac",
            "SC-mask (c=4)":             "actor/state_weight_clamp_frac",
            "SC-mask (c=1e5)":             "actor/state_weight_clamp_frac",
            "SC-min-prefix":              "actor/state_weight_clamp_frac",
            "SC-identity":                "actor/state_weight_clamp_frac",
        },
        title="SC state-weight clamp fraction  (total)",
    )

    # --- [3] PPO clip — upper (positive adv clipped) ----------------------
    # GRPO / GSPO baselines do NOT log the upper half separately (verl only logs
    # pg_clipfrac and pg_clipfrac_lower); we approximate upper = total - lower.
    ax = axes[3]
    for bname in ("GRPO baseline", "GSPO baseline"):
        b_bkt = runs_data[bname]
        xs_tot, ys_tot = b_bkt.get("actor/pg_clipfrac", ([], []))
        xs_low, ys_low = b_bkt.get("actor/pg_clipfrac_lower", ([], []))
        if xs_tot and xs_low and xs_tot == xs_low:
            ys_upper = [max(t - l, 0.0) for t, l in zip(ys_tot, ys_low)]
        else:
            ys_upper = []
        if ys_upper:
            ax.plot(xs_tot, ys_upper, color=COLORS[bname],
                    alpha=0.20, linewidth=1)
            ax.plot(xs_tot, smooth(ys_upper, k=5),
                    color=COLORS[bname], linewidth=2,
                    label=f"{bname}  (= total − lower)")
    for name in ["SC-mask (c=2)", "SC-mask (c=4)", "SC-mask (c=1e5)", "SC-min-prefix", "SC-identity"]:
        xs, ys = runs_data[name].get("actor/ppo_clip_upper_frac", ([], []))
        if ys:
            ax.plot(xs, ys, color=COLORS[name], alpha=0.20, linewidth=1)
            ax.plot(xs, smooth(ys, k=5), color=COLORS[name], linewidth=2,
                    label=name)
    ax.set_title("PPO clip — upper  (r > 1+ε, positive adv)")
    ax.set_xlabel("global step"); ax.set_ylabel("fraction")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # --- [4] PPO clip — lower ---------------------------------------------
    plot_metric(
        axes[4], runs_data,
        {
            "GRPO baseline":                "actor/pg_clipfrac_lower",
            "GSPO baseline":                "actor/pg_clipfrac_lower",
            "SC-mask (c=2)":             "actor/ppo_clip_lower_frac",
            "SC-mask (c=4)":             "actor/ppo_clip_lower_frac",
            "SC-mask (c=1e5)":             "actor/ppo_clip_lower_frac",
            "SC-min-prefix":              "actor/ppo_clip_lower_frac",
            "SC-identity":                "actor/ppo_clip_lower_frac",
        },
        title="PPO clip — lower  (r < 1−ε, negative adv)",
    )

    # --- [5] SC clamp — upper / lower (SC runs only) ----------------------
    ax = axes[5]
    for name in ["SC-mask (c=2)", "SC-mask (c=4)", "SC-mask (c=1e5)", "SC-min-prefix", "SC-identity"]:
        bkt = runs_data[name]
        xs_up, ys_up = bkt.get("actor/state_weight_clamp_upper_frac", ([], []))
        xs_lo, ys_lo = bkt.get("actor/state_weight_clamp_lower_frac", ([], []))
        if ys_up:
            ax.plot(xs_up, smooth(ys_up, k=5), color=COLORS[name], linewidth=2,
                    linestyle="-", label=f"{name}  upper")
        if ys_lo:
            ax.plot(xs_lo, smooth(ys_lo, k=5), color=COLORS[name], linewidth=2,
                    linestyle="--", label=f"{name}  lower")
    ax.set_title("SC clamp — upper (solid) vs lower (dashed)")
    ax.set_xlabel("global step"); ax.set_ylabel("fraction")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(
        "Clip & clamp fractions  —  GRPO / GSPO baselines vs SC-mask / SC-min-prefix / SC-identity",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = RESULTS_DIR / "comparison_clip_clamp.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved figure -> {out_path}")


if __name__ == "__main__":
    main()
