# Policy Gradient: On-Policy vs Off-Policy Objective

> Color convention: $\color{blue}{\text{new / current policy } \pi_\theta}$ vs $\color{red}{\text{old / behavior policy } \pi_\beta}$

---

## On-Policy

$$
\nabla_\theta \eta(\pi_\theta) = \mathbb{E}_{s \sim \color{blue}{d^{\pi_\theta}}, \, a \sim \color{blue}{\pi_\theta}} \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s,a) \right]
$$

- Everything under $\color{blue}{\pi_\theta}$: state distribution, action sampling — all from **current** policy
- $\hat{A}(s,a)$: advantage estimated from sampled rewards (policy-independent)
- ✅ Unbiased · ❌ Must re-sample every update

---

## Off-Policy / Offline

$$
\nabla_\theta L^{\text{IS}}(\theta) = \mathbb{E}_{s \sim \color{red}{d^{\pi_\beta}}, \, a \sim \color{red}{\pi_\beta}} \left[ \frac{\color{blue}{\pi_\theta(a|s)}}{\color{red}{\pi_\beta(a|s)}} \nabla_\theta \log \pi_\theta(a|s) \cdot \hat{A}(s,a) \right]
$$

- Sampling from $\color{red}{\pi_\beta}$ (old), corrected toward $\color{blue}{\pi_\theta}$ (new) via IS ratio $\frac{\color{blue}{\pi_\theta}}{\color{red}{\pi_\beta}}$
- ⚠️ State distribution $\color{red}{d^{\pi_\beta}}$ remains uncorrected → biased
- ❌ Biased (state mismatch) · ✅ Can reuse old data

---

## The True Objective and How Methods Approximate It

### Notation

We use a unified notation throughout. In the LLM setting, for a prompt $q$ and a sampled response $o = (o_1, o_2, \ldots, o_T)$:

| Symbol | Meaning |
|---|---|
| $s_t = (q, o_{<t})$ | State at step $t$ (prompt + generated tokens so far) |
| $a_t = o_t$ | Action at step $t$ (the token generated) |
| $\color{blue}{\pi_\theta}$ | Current / new policy being optimized |
| $\color{red}{\pi_\beta}$ | Behavior / old policy that generated the rollout |
| $r_t(\theta) = \frac{\color{blue}{\pi_\theta(a_t \vert s_t)}}{\color{red}{\pi_\beta(a_t \vert s_t)}}$ | Per-token importance sampling ratio |
| $\hat{A}_t$ | Advantage estimate at step $t$ |

### The True Off-Policy Surrogate

The exact off-policy surrogate requires correcting **both** the state distribution and the action distribution:

$$
L^{\text{true}}(\theta) = \mathbb{E}_{s \sim \color{red}{d^{\pi_\beta}}, \, a \sim \color{red}{\pi_\beta}} \left[ \frac{\color{blue}{d^{\pi_\theta}(s)}}{\color{red}{d^{\pi_\beta}(s)}} \cdot \frac{\color{blue}{\pi_\theta(a|s)}}{\color{red}{\pi_\beta(a|s)}} \cdot \hat{A}(s,a) \right]
$$

The full importance weight has two factors:

$$
w(s, a) = \underbrace{\frac{\color{blue}{d^{\pi_\theta}(s)}}{\color{red}{d^{\pi_\beta}(s)}}}_{\text{state correction}} \cdot \underbrace{\frac{\color{blue}{\pi_\theta(a|s)}}{\color{red}{\pi_\beta(a|s)}}}_{\text{action correction}} = \underbrace{\frac{\color{blue}{d^{\pi_\theta}(s)}}{\color{red}{d^{\pi_\beta}(s)}}}_{\text{state correction}} \cdot \; r_t(\theta)
$$

**Nobody computes $\frac{d^{\pi_\theta}(s)}{d^{\pi_\beta}(s)}$ directly** — it requires knowing the full state visitation distributions, which is intractable. **Every method below drops the state ratio entirely**, and instead tries to keep $\pi_\theta$ close enough to $\pi_\beta$ so that $d^{\pi_\theta} \approx d^{\pi_\beta}$ and the dropped factor is approximately 1. The common surrogate becomes:

$$
L^{\text{IS}}(\theta) = \mathbb{E}_{s \sim \color{red}{d^{\pi_\beta}}, \, a \sim \color{red}{\pi_\beta}} \left[ r_t(\theta) \cdot \hat{A}_t \right]
$$

The question then becomes: *how does each method constrain $r_t(\theta)$ — i.e., how does it implement the trust region on the action distribution?*

---

### TRPO — Constrain so the ratio ≈ 1 ✅

**Surrogate**:

$$
\max_\theta \; \mathbb{E}_{s_t, a_t \sim \color{red}{\pi_\beta}} \left[ r_t(\theta) \cdot \hat{A}_t \right] \quad \text{s.t.} \quad \mathbb{E}_{s_t}\left[D_{KL}\big(\color{red}{\pi_\beta}(\cdot|s_t) \| \color{blue}{\pi_\theta}(\cdot|s_t)\big)\right] \le \delta
$$

**Strategy**: maximize the IS surrogate, but enforce a hard KL constraint so that $\pi_\theta \approx \pi_\beta$ everywhere.

- KL is a **distribution-level** quantity — sums over the **entire** $\mathcal{A}$: $D_{KL} = \sum_{a \in \mathcal{A}} \pi_\beta(a|s) \log \frac{\pi_\beta(a|s)}{\pi_\theta(a|s)}$
- Via Pinsker: $D_{KL}$ small → $D_{TV}$ small → state distributions stay close → ratio ≈ 1
- ✅ Theoretically sound · ❌ Requires second-order optimization (Fisher matrix, conjugate gradient)

---

### PPO — Clip the action ratio, hope the state ratio ≈ 1 ⚠️

**Surrogate**:

$$
\max_\theta \; \mathbb{E}_{s_t, a_t \sim \color{red}{\pi_\beta}} \left[ \min\Big( r_t(\theta) \cdot \hat{A}_t, \;\; \text{clip}\big(r_t(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big) \cdot \hat{A}_t \Big) \right]
$$

**Strategy**: replace TRPO's KL constraint with a clip on $r_t(\theta)$ at a **single sampled action**, hoping this is representative of the overall distribution shift.

#### Why this fails in large action spaces

TRPO's KL constraint controls $D_{TV}(\pi_\beta(\cdot|s), \pi_\theta(\cdot|s)) = \frac{1}{2}\sum_{\text{all } a \in \mathcal{A}} |\pi_\theta(a|s) - \pi_\beta(a|s)|$ — it sees the **entire** distribution. PPO's clip only sees **one sample** from it:

| $\vert\mathcal{A}\vert$ | Mass per action | Clip as proxy for $D_{TV}$ |
|---|---|---|
| Small (e.g. 4) | ~25% each | ✅ One sample covers meaningful mass |
| LLM (~$10^5$) | ~$10^{-5}$ each | ❌ One token = $0.001\%$ of the simplex |

With $|\mathcal{A}| \approx 10^5$, the clip constrains one token's ratio while the remaining $10^5 - 1$ tokens shift freely and invisibly. The assumption "$r_t$ clipped → $\pi_\theta \approx \pi_\beta$" breaks — so the assumption "$\frac{d^{\pi_\theta}}{d^{\pi_\beta}} \approx 1$" also breaks. **The surrogate reliability is gone.**

---

### GRPO — Same clip, different advantage ⚠️

> DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025.

GRPO's innovation is in **advantage estimation** — it drops the value function entirely and uses group-relative normalization $\hat{A}_i = \frac{r_i - \mu}{\sigma}$ across $G$ sampled responses. This is a significant simplification for LLMs where only a single end-of-sequence reward is available.

**Surrogate**:

$$
\max_\theta \; \mathbb{E}_{q \sim \mathcal{D},\, \{o_i\}_{i=1}^G \sim \color{red}{\pi_\beta}} \left[ \frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} \min\Big( r_{i,t}(\theta) \cdot \hat{A}_{i}, \;\; \text{clip}\big(r_{i,t}(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big) \cdot \hat{A}_{i} \Big) \right]
$$

where $r_{i,t}(\theta) = \frac{\color{blue}{\pi_\theta(a_{i,t}|s_{i,t})}}{\color{red}{\pi_\beta(a_{i,t}|s_{i,t})}}$ and $\hat{A}_i = \frac{R_i - \text{mean}(\{R_1,\ldots,R_G\})}{\text{std}(\{R_1,\ldots,R_G\})}$.

From the trust region perspective, **GRPO inherits PPO's clip mechanism unchanged** — same per-token ratio, same symmetric band $[1-\varepsilon, 1+\varepsilon]$. All of PPO's trust region problems carry over:

- Still clips a **single sampled token** — blind to the rest of the $10^5$-dimensional simplex
- Still over-constrains low-probability tokens, under-constrains high-probability ones
- No improvement in how the action distribution shift is measured or controlled

**GRPO's contribution is orthogonal to the trust region problem.** It solves "how to estimate advantage without a value function" — not "how to keep $\pi_\theta$ close to $\pi_\beta$." The methods that follow address the latter.

- ✅ Eliminates value function overhead · ✅ Better advantage estimation for sparse rewards
- ❌ Same broken trust region as PPO in large action spaces

---

### DAPO (Clip-Higher) — Loosen the action ratio constraint asymmetrically

> ByteDance Seed et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." 2025.

**Surrogate**:

$$
\max_\theta \; \mathbb{E}_{s_t, a_t \sim \color{red}{\pi_\beta}} \left[ \min\Big( r_t(\theta) \cdot \hat{A}_t, \;\; \text{clip}\big(r_t(\theta),\, 1-\varepsilon_{\text{low}},\, 1+\varepsilon_{\text{high}}\big) \cdot \hat{A}_t \Big) \right], \quad \varepsilon_{\text{high}} > \varepsilon_{\text{low}}
$$

**Strategy**: same structure as PPO, but with **asymmetric** clip bounds. Recognizes that PPO's symmetric clip has an asymmetric effect in large action spaces: it over-constrains rare (exploration) tokens while under-constraining dominant ones.

- Raises $\varepsilon_{\text{high}}$ so low-probability tokens can increase more freely
- Doesn't fix the fundamental blindness of per-token clipping, but mitigates one of its worst consequences
- DAPO also includes dynamic sampling (filter zero-advantage prompts), token-level normalization, and overlong reward shaping — but from the trust region perspective, Clip-Higher is the relevant piece
- ⚠️ Still a per-token heuristic · ✅ Better exploration dynamics

---

### GSPO — Approximate the full ratio at sequence level

> Zhang et al. "GSPO: Geometric-Mean Sequence-Level Policy Optimization." 2025.

Define the **sequence-level ratio** as the geometric mean of per-token ratios:

$$
s_i(\theta) = \left(\prod_{t=1}^{|o_i|} r_{i,t}(\theta)\right)^{1/|o_i|} = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log r_{i,t}(\theta)\right)
$$

**Surrogate**:

$$
\max_\theta \; \mathbb{E}_{q \sim \mathcal{D},\, \{o_i\}_{i=1}^G \sim \color{red}{\pi_\beta}} \left[ \frac{1}{G}\sum_{i=1}^{G} \min\Big( s_i(\theta) \cdot \hat{A}_i, \;\; \text{clip}\big(s_i(\theta),\, 1-\varepsilon,\, 1+\varepsilon\big) \cdot \hat{A}_i \Big) \right]
$$

**Strategy**: instead of clipping per-token $r_{i,t}$, clip the **sequence-level** $s_i$ — a single scalar that aggregates information across all tokens. By treating the entire response as one action, the clip on $s_i$ becomes a more meaningful proxy for keeping $\pi_\theta$ close to $\pi_\beta$ — which in turn keeps $\frac{d^{\pi_\theta}}{d^{\pi_\beta}} \approx 1$.

- ✅ Aggregates information across all tokens → better proxy for the true ratio
- ✅ More stable than per-token IS

---

### SAPO — Smooth soft gate as continuous trust region

> Gao et al. "Soft Adaptive Policy Optimization." Qwen Team, Alibaba, 2025.

**Surrogate**: replace the hard clip with a smooth sigmoid gate $f$ applied to the same per-token ratio $r_{i,t}(\theta)$:

$$
\max_\theta \; \mathbb{E}_{s_t, a_t \sim \color{red}{\pi_\beta}} \left[ f_{i,t}\big(r_{i,t}(\theta)\big) \cdot \hat{A}_t \right]
$$

where

$$
f_{i,t}(r_{i,t}(\theta)) = \sigma\big(\tau_{i,t}(r_{i,t}(\theta) - 1)\big) \cdot \frac{4}{\tau_{i,t}}
$$

The effective gradient weight becomes:

$$
w_{i,t}(\theta) = 4\, p_{i,t}(\theta)\big(1 - p_{i,t}(\theta)\big), \quad p_{i,t}(\theta) = \sigma\big(\tau_{i,t}(r_{i,t}(\theta) - 1)\big)
$$

**Strategy**: $w_{i,t}$ peaks at $r_{i,t} = 1$ (on-policy) and decays smoothly as the ratio deviates — implementing a **continuous trust region**:
- Near on-policy → gradients preserved (encourages useful updates)
- Far off-policy → gradients attenuated (prevents instability)
- At $r_{i,t} = 1$: gradient equals the unclipped objective, regardless of $\tau$

SAPO is both **sequence-coherent** and **token-adaptive**:
- Under mild assumptions (small steps + low intra-sequence dispersion), the average token gate concentrates to a sequence-level gate $g(\log s_i) = \text{sech}^2(\frac{\tau_i}{2} \log s_i)$ — reducing to a GSPO-like formulation but with smooth boundaries
- When a few tokens are highly off-policy, GSPO suppresses **all** gradients for that sequence; SAPO selectively down-weights only the offending tokens

Asymmetric temperatures: $\tau_{\text{neg}} > \tau_{\text{pos}}$ — negative-token updates (which increase logits of many inappropriate tokens) decay faster, reflecting their greater instability risk.

- ✅ Smooth trust region (no gradient discontinuity) · ✅ Token-adaptive + sequence-coherent
- ⚠️ Per-token ratio, but with better gating

---

### DPPO — Directly estimate the divergence, not the ratio ✅

> Qi et al. "Rethinking the Trust Region in LLM Reinforcement Learning." 2026.

DPPO identifies the root cause: PPO's ratio $r_t(\theta)$ is a **noisy single-sample Monte Carlo estimate** of the true policy divergence. The ratio is hypersensitive to probability magnitude — a rare token going from $10^{-5}$ to $10^{-3}$ produces $r_t = 100$ (triggers clip), yet the actual $D_{TV}$ change is negligible. Conversely, a dominant token dropping from $0.99$ to $0.8$ has $r_t \approx 0.81$ (no clip), yet the divergence is catastrophic.

**Surrogate**: replace the ratio clip with a **direct divergence constraint** — bringing back TRPO's distribution-level guarantee using only first-order optimization:

$$
\max_\theta \; \mathbb{E}_{s_t, a_t \sim \color{red}{\pi_\beta}} \left[ \big(r_t(\theta) - 1\big) \cdot \hat{A}_t \right] \quad \text{s.t.} \quad \max_t \; D_{TV}\big(\color{red}{\pi_\beta}(\cdot|s_t) \| \color{blue}{\pi_\theta}(\cdot|s_t)\big) \le \delta
$$

Note: the surrogate uses $(r_t - 1)$ instead of $r_t$ — they differ by a constant and have the same gradient, but this form makes the connection to the performance difference identity clearer.

To avoid storing the full vocabulary distribution (memory-prohibitive for LLMs), DPPO introduces efficient approximations:

- **Binary-TV**: approximate $D_{TV}(\pi_\beta(\cdot|s_t), \pi_\theta(\cdot|s_t)) \approx |\pi_\beta(y_t|s_t) - \pi_\theta(y_t|s_t)|$ — collapse the vocabulary into "sampled token" vs "everything else"
- **Top-K**: compute divergence over the top-K most probable tokens, capturing the essential distributional shift with negligible overhead

Key findings from DPPO:
1. **Trust region is essential** even at tiny learning rates ($10^{-6}$) — without it, training-inference mismatch accumulates and collapses
2. **Anchor to rollout policy $\mu_{\theta'}$**, not recomputed $\pi_{\theta'}$ — using recomputed policy as anchor leads to instability
3. **Primary instability source**: a tiny fraction ($\le 0.5\%$) of updates on negative samples that push the policy far outside the trust region

- ✅ Distribution-level trust region (like TRPO) · ✅ First-order optimization (like PPO)
- ✅ Correctly distinguishes safe vs unsafe updates regardless of token probability

---

### Summary: How each method handles $w(s,a) = \frac{d^{\pi_\theta}(s)}{d^{\pi_\beta}(s)} \cdot \frac{\pi_\theta(a|s)}{\pi_\beta(a|s)}$

| Method | Action ratio $\frac{\pi_\theta}{\pi_\beta}$ | Trust region mechanism | Guarantee |
|--------|---|---|---|
| **TRPO** | Full distribution | Hard KL constraint (2nd-order) | ✅ Theoretical bound |
| **PPO** | Single sampled token | Ratio clip at one sample | ❌ Fails for $|\mathcal{A}| \gg 1$ |
| **GRPO** | Single sampled token (same as PPO) | Same ratio clip | ❌ Same failure (innovation is in advantage, not trust region) |
| **DAPO** (Clip-Higher) | Single token, asymmetric bounds | Asymmetric ratio clip | ⚠️ Heuristic improvement |
| **GSPO** | Geometric mean over all tokens | Clip on sequence-level ratio | ✅ Better proxy |
| **SAPO** | Per-token with smooth gating | Sigmoid soft gate ($\tau$-controlled) | ✅ Smooth + token-adaptive |
| **DPPO** | Full distribution (approx.) | Direct $D_{TV}$/$D_{KL}$ estimate (1st-order) | ✅ Distribution-level, scalable |

---

## Outlook: Recovering the State Ratio via Deterministic Transitions

All methods above share a common blind spot: **the state distribution ratio $\frac{d^{\pi_\theta}(s)}{d^{\pi_\beta}(s)}$ is dropped entirely**. Each method then focuses exclusively on constraining the action ratio $r_t(\theta)$, hoping that a tight action-level trust region implies $d^{\pi_\theta} \approx d^{\pi_\beta}$. But none of them attempt to estimate or correct the state ratio itself.

This is well-justified in general MDPs, where the transition $P(s'|s,a)$ is stochastic and the state visitation distribution $d^\pi(s)$ requires intractable integration over all possible trajectories. However, **LLMs have a special structural property that all existing methods overlook**:

### LLM transitions are deterministic

In autoregressive generation, the transition is **deterministic**: given state $s_t = (q, o_{<t})$ and action $a_t = o_t$, the next state is uniquely determined:

$$
P(s_{t+1} | s_t, a_t) = \mathbf{1}[s_{t+1} = (q, o_{\le t})]
$$

This means the state distribution ratio **decomposes exactly** into a product of per-token action ratios:

$$
\frac{d^{\pi_\theta}(s_t)}{d^{\pi_\beta}(s_t)} = \frac{\pi_\theta(o_1|q) \cdot \pi_\theta(o_2|q,o_1) \cdots \pi_\theta(o_{t-1}|q,o_{<t-1})}{\pi_\beta(o_1|q) \cdot \pi_\beta(o_2|q,o_1) \cdots \pi_\beta(o_{t-1}|q,o_{<t-1})} = \prod_{j=1}^{t-1} r_j(\theta)
$$

This identity is **exact** — not an approximation. It does not hold in general MDPs (where stochastic transitions break the product structure), but it holds in LLMs because the environment is fully determined by the agent's own actions.

### The true objective becomes computable

Substituting back into the full importance weight:

$$
w(s_t, a_t) = \underbrace{\prod_{j=1}^{t-1} r_j(\theta)}_{\text{state correction (exact)}} \cdot \underbrace{r_t(\theta)}_{\text{action correction}} = \prod_{j=1}^{t} r_j(\theta)
$$

The true off-policy surrogate is therefore:

$$
L^{\text{true}}(\theta) = \mathbb{E}_{o \sim \color{red}{\pi_\beta}} \left[ \sum_{t=1}^{T} \left(\prod_{j=1}^{t} r_j(\theta)\right) \cdot \hat{A}_t \right]
$$

This is no longer intractable — every $r_j(\theta)$ is already computed during training. The state ratio, which all existing methods drop, is **free** in the LLM setting.

### The variance problem and potential solutions

The obvious challenge: $\prod_{j=1}^{t} r_j(\theta)$ suffers from **exponential variance growth** — the classic curse of trajectory-level importance sampling (Precup et al., 2000). Even small per-token deviations compound multiplicatively over hundreds of tokens.

A natural mitigation is **truncated prefix correction** with a lookback window $k$:

$$
w_t^{(k)}(\theta) = \prod_{j=\max(1,\, t-k)}^{t} r_j(\theta)
$$

This creates a bias-variance tradeoff controlled by a single hyperparameter:

| $k$ | Behavior | Bias | Variance |
|---|---|---|---|
| $0$ | PPO / GRPO (state ratio dropped) | High | Low |
| $3 \sim 10$ | Correct recent state mismatch | Moderate | Moderate |
| $T$ | Exact (full trajectory IS) | Zero | Explosive |

The truncated weight can be further stabilized by combining with existing trust region mechanisms — e.g., applying SAPO's soft gate to $w_t^{(k)}$, or using DPPO's divergence constraint alongside the prefix correction.

### Connection to existing methods

- **GSPO** already computes $\prod_{t=1}^{|o_i|} r_{i,t}(\theta)$ — but uses its geometric mean $(\prod_t r_t)^{1/|o_i|}$ as a **clip target**, not as a state correction weight. The mathematical object is the same; the usage is fundamentally different.
- **V-trace** (Espeholt et al., 2018) uses truncated cumulative IS products $\prod_{j=s}^{t-1} \min(\bar{c}, r_j)$ — but for **value estimation**, not policy gradient correction, and under stochastic transitions where the product is only an approximation.
- **Per-decision IS** (Precup et al., 2000) established the theoretical foundation for step-wise IS correction in RL, but was never applied to the LLM setting where deterministic transitions make it exact.

### Why this direction is promising

1. **Theoretically clean** — it recovers the exact state ratio rather than heuristically hoping it is ≈ 1
2. **Zero overhead** — all $r_j(\theta)$ values are already computed; the prefix product is a cumulative multiplication
3. **Orthogonal to existing methods** — can be combined with any trust region mechanism (clip, soft gate, divergence constraint) as an additional correction factor
4. **Clear ablation axis** — the lookback window $k$ provides a continuous spectrum from PPO ($k=0$) to exact IS ($k=T$), enabling systematic study of the bias-variance tradeoff in state correction
