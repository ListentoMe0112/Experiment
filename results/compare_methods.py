#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare multiple training runs:
  - grpo_baseline_qwen2.5_1.5b.jsonl        (vanilla GRPO baseline)
  - gspo_baseline_qwen2.5_1.5b.jsonl        (GSPO sequence-ratio baseline)
  - sc_none_qwen2.5_1.5b.jsonl              (State-Corrected loss, no KL)
  - sc_min_prefix_qwen2.5_1.5b.jsonl
  - sc_min_prefix_qwen2.5_1.5b.jsonl
  - sc_identity_qwen2.5_1.5b.jsonl

Parse the jsonl logs and plot several side-by-side comparison figures:
  1. Validation accuracy on MATH (val-core/.../acc/mean@1)
  2. Training reward / critic score (critic/score/mean)
  3. Policy gradient loss (actor/pg_loss)
  4. Gradient norm (actor/grad_norm)
  5. Mean response length (response_length/mean)
  6. Per-step wall-clock time (timing_s/step)

Output: results/comparison.png  (a 2x3 grid)
"""

import json
import os
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

VAL_ACC_KEY = "val-core/DigitalLearningGmbH/MATH-lighteval/acc/mean@1"

# Different runs may log the validation metric under slightly different dataset
# names (e.g. "DigitalLearningGmbH/MATH-lighteval" vs "lighteval/MATH"), depending
# on which mirror was used at preprocessing time. Anything matching the prefix +
# suffix below is treated as the MATH val-acc.
VAL_ACC_KEY_PREFIX = "val-core/"
VAL_ACC_KEY_SUFFIX = "/acc/mean@1"


def _find_val_acc(data: dict):
    """Return the val-acc value if any matching key is present, else None."""
    if VAL_ACC_KEY in data and data[VAL_ACC_KEY] is not None:
        return data[VAL_ACC_KEY]
    for k, v in data.items():
        if (
            isinstance(k, str)
            and k.startswith(VAL_ACC_KEY_PREFIX)
            and k.endswith(VAL_ACC_KEY_SUFFIX)
            and v is not None
        ):
            return v
    return None

# Metrics we want to track from the "training" stream (keyed by global_step).
TRAIN_METRICS = [
    ("critic/score/mean",      "Training reward (critic/score/mean)"),
    ("actor/pg_loss",          "Actor PG loss"),
    ("actor/grad_norm",        "Actor grad norm"),
    ("response_length/mean",   "Mean response length (tokens)"),
    ("timing_s/step",          "Wall-clock per step (s)"),
]


def load_run(path: Path):
    """Return (val_points, train_series).

    val_points  : list of (step, acc)      - validation accuracy over time
    train_series: dict[metric] -> (steps, values)
    """
    val_points = []
    train_buckets = {k: ([], []) for k, _ in TRAIN_METRICS}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            step = rec.get("step", 0)
            data = rec.get("data", {})

            # Validation accuracy can appear either at step 0 or embedded in
            # training steps (every N steps) -- capture it in both cases.
            val_v = _find_val_acc(data)
            if val_v is not None:
                gs = data.get("training/global_step", step if step == 0 else None)
                if gs is None:
                    gs = step
                val_points.append((gs, val_v))

            # Training-side metrics are only meaningful when a global_step exists
            gs = data.get("training/global_step")
            if gs is None:
                continue
            for key, _ in TRAIN_METRICS:
                if key in data and data[key] is not None:
                    xs, ys = train_buckets[key]
                    xs.append(gs)
                    ys.append(data[key])

    # Deduplicate / sort validation points by step (keep first occurrence).
    seen = {}
    for s, v in val_points:
        seen.setdefault(s, v)
    val_points = sorted(seen.items())

    return val_points, train_buckets


def smooth(y, k=5):
    """Simple moving-average smoothing for noisy curves."""
    y = np.asarray(y, dtype=float)
    if len(y) < k or k <= 1:
        return y
    kernel = np.ones(k) / k
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(ypad, kernel, mode="valid")[: len(y)]


def main():
    runs_data = {name: load_run(p) for name, p in RUNS.items()}

    # ---- Print a quick textual summary ---------------------------------
    print("=" * 70)
    print("Summary of the two runs")
    print("=" * 70)
    for name, (val_points, train_buckets) in runs_data.items():
        print(f"\n[{name}]")
        if val_points:
            v0 = val_points[0][1]
            vmax = max(v for _, v in val_points)
            vlast = val_points[-1][1]
            step_max = max(val_points, key=lambda t: t[1])[0]
            print(f"  val acc @ step {val_points[0][0]:>3}: {v0:.4f}")
            print(f"  val acc final (step {val_points[-1][0]:>3}): {vlast:.4f}")
            print(f"  val acc best   (step {step_max:>3}): {vmax:.4f}")
        xs, ys = train_buckets["critic/score/mean"]
        if ys:
            print(f"  train reward first / last / max : "
                  f"{ys[0]:.4f} / {ys[-1]:.4f} / {max(ys):.4f}")
        xs, ys = train_buckets["timing_s/step"]
        if ys:
            print(f"  mean step time                  : {np.mean(ys):.1f} s")
        xs, ys = train_buckets["response_length/mean"]
        if ys:
            print(f"  mean response length (avg)      : {np.mean(ys):.1f} tok")

    # ---- Plot ---------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    colors = {
        "GRPO baseline":                  "#1f77b4",  # blue
        "GSPO baseline":                  "#17becf",  # cyan
        "SC-mask (c=2)":               "#d62728",  # red
        "SC-mask (c=4)":               "#e377c2",  # pink
        "SC-mask (c=1e5)":             "#8b1a1a",  # dark red (effectively no clamp)
        "SC-min-prefix":                "#ff7f0e",  # orange
        "SC-identity":                  "#9467bd",  # purple
    }

    # Panel 0 : validation accuracy
    ax = axes[0]
    for name, (val_points, _) in runs_data.items():
        if not val_points:
            continue
        xs = [s for s, _ in val_points]
        ys = [v for _, v in val_points]
        ax.plot(xs, ys, "-o", label=name, color=colors[name], linewidth=2)
    ax.set_title("Validation accuracy  (MATH-lighteval, mean@1)")
    ax.set_xlabel("global step")
    ax.set_ylabel("accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Panels 1..5 : training metrics
    for idx, (key, title) in enumerate(TRAIN_METRICS, start=1):
        ax = axes[idx]
        for name, (_, train_buckets) in runs_data.items():
            xs, ys = train_buckets[key]
            if not ys:
                continue
            ax.plot(xs, ys, color=colors[name], alpha=0.25, linewidth=1)
            ax.plot(xs, smooth(ys, k=5), color=colors[name], linewidth=2,
                    label=name)
        ax.set_title(title)
        ax.set_xlabel("global step")
        ax.grid(True, alpha=0.3)
        if idx == 1:
            ax.legend()

    fig.suptitle(
        "GRPO / GSPO baselines  vs.  SC-mask / SC-min-prefix / SC-identity   —   Qwen2.5-1.5B on MATH",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = RESULTS_DIR / "comparison.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved figure -> {out_path}")


if __name__ == "__main__":
    main()
