#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot rollout-vs-training off-policy drift metrics across the SC runs.

`rollout_corr/chi2_seq` quantifies the chi-squared divergence between the
rollout policy π_β (vllm-side, lower precision) and the training-side
recompute under π_θ at the *sequence* level. Higher = more off-policy drift
between rollout collection and training. It is the most direct numeric
proxy for "how off-policy is this batch?", complementary to per-token
metrics like `kl`, `chi2_token`, and `ppl_ratio`.

Layout (2 x 2):
    [0] chi2_seq  (sequence-level χ² IS-variance proxy)         — *headline*
    [1] chi2_token (per-token χ²)
    [2] kl  (rollout vs training KL)
    [3] ppl_ratio (training PPL / rollout PPL)

Output: results/comparison_off_policy_drift.png

Note: GRPO / GSPO baselines do not log `rollout_corr/*`, so they are not
shown here — this comparison is among SC variants only.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent

RUNS = {
    "SC-mask (c=2)":             RESULTS_DIR / "sc_none_qwen2.5_1.5b_clamp2.jsonl",
    "SC-mask (c=4)":             RESULTS_DIR / "sc_none_qwen2.5_1.5b_clamp4.jsonl",
    "SC-mask (c=1e5)":           RESULTS_DIR / "sc_none_qwen2.5_1.5b_clamp100000.jsonl",
    "SC-min-prefix":               RESULTS_DIR / "sc_min_prefix_qwen2.5_1.5b.jsonl",
    "SC-identity":                 RESULTS_DIR / "sc_identity_qwen2.5_1.5b.jsonl",
}

COLORS = {
    "SC-mask (c=2)":             "#d62728",  # red
    "SC-mask (c=4)":             "#e377c2",  # pink
    "SC-mask (c=1e5)":           "#8b1a1a",  # dark red (effectively no clamp)
    "SC-min-prefix":               "#ff7f0e",  # orange
    "SC-identity":                 "#9467bd",  # purple
}

PANELS = [
    ("rollout_corr/chi2_seq",
     r"$\chi^2_{\mathrm{seq}}$  (sequence-level χ²)",
     "off-policy drift (higher = more drifted)"),
    ("rollout_corr/chi2_token",
     r"$\chi^2_{\mathrm{token}}$  (per-token χ²)",
     "off-policy drift"),
    ("rollout_corr/kl",
     r"KL$(\pi_\beta\|\pi_\theta)$  (rollout vs training)",
     "KL"),
    ("rollout_corr/ppl_ratio",
     r"PPL ratio  ($\mathrm{PPL}_\theta/\mathrm{PPL}_\beta$)",
     "ratio"),
]


def load_run(path: Path, keys):
    """Return dict[key] -> (steps, values), only for steps with a global_step."""
    buckets = {k: ([], []) for k in keys}
    if not path.exists():
        return buckets
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
            for k in keys:
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


def main():
    keys = [k for k, _, _ in PANELS]
    runs_data = {name: load_run(p, keys) for name, p in RUNS.items()}

    # --- Textual summary ---------------------------------------------------
    print("=" * 78)
    print("Off-policy drift metrics (rollout π_β  vs  training-recompute π_θ)")
    print("=" * 78)
    for name, bkt in runs_data.items():
        # Use any populated key to get the run length (not all runs are same length)
        run_steps = max((len(v[0]) for v in bkt.values()), default=0)
        print(f"\n[{name}]  (steps logged: {run_steps})")
        for key, _, _ in PANELS:
            xs, ys = bkt[key]
            if not ys:
                print(f"  {key:<32s} (not logged)")
                continue
            ys_arr = np.asarray(ys, dtype=float)
            print(f"  {key:<32s} "
                  f"first={ys_arr[0]:.4g}  "
                  f"last={ys_arr[-1]:.4g}  "
                  f"mean={ys_arr.mean():.4g}  "
                  f"max={ys_arr.max():.4g}")

    # --- Plot --------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for idx, (key, title, ylabel) in enumerate(PANELS):
        ax = axes[idx]
        for name, bkt in runs_data.items():
            xs, ys = bkt[key]
            if not ys:
                continue
            ax.plot(xs, ys, color=COLORS[name], alpha=0.20, linewidth=1)
            ax.plot(xs, smooth(ys, k=5), color=COLORS[name], linewidth=2,
                    label=name)
        ax.set_title(title)
        ax.set_xlabel("global step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        # χ² and PPL-ratio span many orders of magnitude across runs;
        # KL also blows up for collapsed runs (identity/min_prefix). Use a
        # symlog-ish log scale wherever values can grow large.
        if key in ("rollout_corr/chi2_seq",
                   "rollout_corr/chi2_token",
                   "rollout_corr/kl",
                   "rollout_corr/ppl_ratio"):
            ax.set_yscale("log")
        if idx == 0:
            ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        "Rollout-vs-training off-policy drift  —  SC variants on Qwen2.5-1.5B / MATH",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = RESULTS_DIR / "comparison_off_policy_drift.png"
    fig.savefig(out_path, dpi=130)
    print(f"\nSaved figure -> {out_path}")


if __name__ == "__main__":
    main()
