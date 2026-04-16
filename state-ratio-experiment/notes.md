# From PPO to GRPO and Beyond: A Unified View of RL for Large Language Models

> This document consolidates and reorganizes material from multiple notes on GRPO variants, policy gradient objectives, and the theoretical foundations of trust regions in the LLM setting. All errors from the original documents have been corrected.

---

## Table of Contents

1. [Background: LLMs as RL Problems](#1-background-llms-as-rl-problems)
2. [Theoretical Foundations](#2-theoretical-foundations)
3. [PPO: The Baseline and Its Structural Failure](#3-ppo-the-baseline-and-its-structural-failure)
4. [GRPO: Dropping the Value Function](#4-grpo-dropping-the-value-function)
5. [Axis 1 — Advantage & Training Improvements](#5-axis-1--advantage--training-improvements)
6. [Axis 2 — Trust Region Improvements](#6-axis-2--trust-region-improvements)
7. [Summary](#7-summary)
8. [Outlook](#8-outlook)

> **Methods covered:** TRPO, PPO, GRPO, DAPO, Entropy Mechanism, NLF, GSPO, SAPO, DPPO, MinPRO

---

# 1. Background: LLMs as RL Problems

LLMs aren't your typical RL problems — they've got their own quirky personality that makes them fascinatingly different. To truly understand these algorithms and how they dance together, we need to embrace these unique characteristics:

- **LLMs as actors**: Their policies are stubbornly deterministic thanks to pre-training. We've got to work extra hard to keep them exploring the vast space of possibilities.

- **Action space drama**: You can think of an LLM's action space as either the entire response (sparse and massive) or individual tokens (sparse but manageable). With $|\mathcal{A}| \approx 10^5$ (vocabulary size), this creates fundamental challenges for trust region methods.

- **High-quality samples**: Unlike training AI agents from scratch, LLMs already speak fluent language. They're not starting from zero — they're starting from "already pretty good."

- **Sample efficiency matters**: Because generating samples is expensive and time-consuming, we need to squeeze every drop of wisdom out of each one without compromising the results.

- **Full observability**: LLMs don't live in a POMDP world — they're in a fully observable MDP. A good LLM should be able to solve most problems if given enough context.

- **Deterministic transitions**: Given state $s_t = (q, o_{<t})$ and action $a_t = o_t$, the next state $s_{t+1} = (q, o_{\le t})$ is uniquely determined. This structural property — overlooked by all existing methods — has profound implications for importance sampling (see [§8](#8-outlook)).

- **Sparse, end-of-sequence rewards**: LLM responses get rewards only at the end — "did you solve the problem or not?" There's no temporal structure to exploit.

**Bottom line**: Think of an LLM as a high school student who's already got excellent writing skills but might need some extra tutoring for specific subjects like mathematics. Understanding each algorithm through this lens makes everything click.

---

# 2. Theoretical Foundations

## 2.1 The RL Objective

The goal is to maximize the expected discounted return under policy $\pi$:

$$
\eta(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right]
$$

This depends on the **state visitation distribution** $d_\pi(s) = (1-\gamma)\sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s \mid \pi)$, which changes every time $\theta$ changes and has no closed-form gradient.

## 2.2 Policy Gradient and REINFORCE

The policy gradient theorem gives:

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot A(s_t, a_t)\right]
$$

where $A(s,a) = Q^\pi(s,a) - V^\pi(s)$ is the advantage function. REINFORCE always samples from the **current** policy $\pi_\theta$ — no distribution mismatch, but old samples cannot be reused. This on-policy bottleneck motivates everything that follows.

## 2.3 The Surrogate Objective

> Color convention: $\color{blue}{\text{new / current policy } \pi_\theta}$ vs $\color{red}{\text{old / behavior policy } \pi_\beta}$

### Performance-Difference Identity (Exact)

$$
\eta(\pi) = \eta(\pi_{old}) + \frac{1}{1-\gamma} \mathbb{E}_{\substack{s \sim d_{\pi} \\ a \sim \pi(\cdot \mid s)}}[A(s,a)]
$$

This is exact but unusable: the expectation requires $s \sim d_\pi$, the state distribution of the new policy we are optimizing.

### The Surrogate (Approximate)

Replace $d_\pi$ with the available $d_{\pi_{old}}$, and use importance sampling for actions:

$$
L_{\pi_{old}}(\pi) = \eta(\pi_{old}) + \frac{1}{1-\gamma} \mathbb{E}_{\substack{s \sim \color{red}{d_{\pi_{old}}} \\ a \sim \color{red}{\pi_{old}}}}\left[\frac{\color{blue}{\pi(a \mid s)}}{\color{red}{\pi_{old}(a \mid s)}} A(s,a)\right]
$$

At $\pi = \pi_{old}$, the surrogate exactly matches the true objective. But away from $\pi_{old}$, the state distribution substitution introduces error.

## 2.4 The Approximation Error

### Bounding the Error

$$
|\eta(\pi) - L_{\pi_{old}}(\pi)| \le \frac{\epsilon}{1-\gamma} \sum_s |d_{\pi}(s) - d_{\pi_{old}}(s)|
$$

where $\epsilon = \max_{s,a}|A(s,a)|$. **The surrogate is reliable only when $d_\pi \approx d_{\pi_{old}}$.**

### From State Mismatch to Action Distribution Shift

Let $\alpha = \max_s D_{TV}(\pi_{old}(\cdot|s), \pi(\cdot|s))$. By induction over time steps (using the triangle inequality and the normalization of the transition kernel $\sum_{s'} P(s'|s,a) = 1$):

$$
\sum_s |d_\pi^{(t)}(s) - d_{\pi_{old}}^{(t)}(s)| \le 2t\alpha \quad \text{for all } t \ge 0
$$

Taking the discounted weighted sum:

$$
\sum_s |d_\pi(s) - d_{\pi_{old}}(s)| \le \frac{2\gamma\alpha}{1-\gamma}
$$

### The Full Chain

Via Pinsker's inequality ($D_{TV} \le \sqrt{\frac{1}{2} D_{KL}}$):

$$
\underbrace{\mathbb{E}_s[D_{KL}]}_{\text{KL constraint}} \xrightarrow{\text{Pinsker}} \underbrace{D_{TV}(\pi_{old}(\cdot|s), \pi(\cdot|s))}_{\text{action distribution shift}} \xrightarrow{\text{propagation}} \underbrace{\|d_\pi - d_{\pi_{old}}\|_1}_{\text{state mismatch}} \xrightarrow{} \underbrace{|\eta - L|}_{\text{surrogate error}}
$$

**Conclusion:** keeping $\mathbb{E}_s[D_{KL}(\pi_{old}(\cdot|s) \| \pi(\cdot|s))]$ small guarantees that the surrogate is reliable. Notice that $D_{KL}$ is a **distribution-level** quantity, summing over the **entire action space** $\mathcal{A}$. This distinction is what PPO's clipping will fail to respect.

## 2.5 The True Off-Policy Surrogate

The exact off-policy surrogate requires correcting **both** the state distribution and the action distribution:

$$
L^{\text{true}}(\theta) = \mathbb{E}_{s \sim \color{red}{d^{\pi_\beta}}, \, a \sim \color{red}{\pi_\beta}} \left[ \underbrace{\frac{\color{blue}{d^{\pi_\theta}(s)}}{\color{red}{d^{\pi_\beta}(s)}}}_{\text{state correction}} \cdot \underbrace{\frac{\color{blue}{\pi_\theta(a|s)}}{\color{red}{\pi_\beta(a|s)}}}_{\text{action correction}} \cdot \hat{A}(s,a) \right]
$$

**Nobody computes $\frac{d^{\pi_\theta}(s)}{d^{\pi_\beta}(s)}$ directly** — it requires knowing the full state visitation distributions, which is intractable in general MDPs. **Every method below drops the state ratio entirely**, and instead tries to keep $\pi_\theta$ close enough to $\pi_\beta$ so that $d^{\pi_\theta} \approx d^{\pi_\beta}$. The common surrogate becomes:

$$
L^{\text{IS}}(\theta) = \mathbb{E}_{s \sim \color{red}{d^{\pi_\beta}}, \, a \sim \color{red}{\pi_\beta}} \left[ r_t(\theta) \cdot \hat{A}_t \right]
$$

where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_\beta(a_t|s_t)}$ is the per-token importance sampling ratio.

The question then becomes: *how does each method constrain $r_t(\theta)$ — i.e., how does it implement the trust region on the action distribution?*

## 2.6 Notation

| Symbol | Meaning |
|---|---|
| $s_t = (q, o_{<t})$ | State at step $t$ (prompt + generated tokens so far) |
| $a_t = o_t$ | Action at step $t$ (the token generated) |
| $\color{blue}{\pi_\theta}$ | Current / new policy being optimized |
| $\color{red}{\pi_\beta}$ | Behavior / old policy that generated the rollout |
| $r_t(\theta) = \frac{\color{blue}{\pi_\theta(a_t \vert s_t)}}{\color{red}{\pi_\beta(a_t \vert s_t)}}$ | Per-token importance sampling ratio |
| $\hat{A}_t$ | Advantage estimate at step $t$ |

---

# 3. PPO: The Baseline and Its Structural Failure

> **Citation:** Schulman et al. "Proximal Policy Optimization Algorithms." *OpenAI*, 2017.

## 3.1 TRPO: The Theoretically Sound Baseline

$$
\max_\theta \; \mathbb{E}_{s_t, a_t \sim \color{red}{\pi_\beta}} \left[ r_t(\theta) \cdot \hat{A}_t \right] \quad \text{s.t.} \quad \mathbb{E}_{s_t}\left[D_{KL}\big(\color{red}{\pi_\beta}(\cdot|s_t) \| \color{blue}{\pi_\theta}(\cdot|s_t)\big)\right] \le \delta
$$

TRPO enforces a hard KL constraint — a **distribution-level** quantity that sums over the **entire** $\mathcal{A}$. Via Pinsker: $D_{KL}$ small → $D_{TV}$ small → state distributions stay close → surrogate is reliable.

- ✅ Theoretically sound · ❌ Requires second-order optimization (Fisher matrix, conjugate gradient)

## 3.2 The PPO Objective

PPO replaces TRPO's hard KL constraint with a simpler mechanism — **clipping the importance ratio**:

$$
\mathcal{J}_{PPO}(\theta) = \mathbb{E}_{(s,a) \sim \pi_{old}} \Big[ \min\big( r_t(\theta) \cdot A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot A_t \big) \Big]
$$

Where:
- $r_t(\theta) = \frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}$ is the importance sampling ratio
- $A(s,a)$ is the advantage
- The clip keeps the ratio in $[1-\epsilon, 1+\epsilon]$ (typically $\epsilon \in [0.1, 0.2]$)

**Case $A_t > 0$:** we want to increase $r_t(\theta)$, but the clip caps it at $1+\varepsilon$. **Case $A_t < 0$:** we want to decrease $r_t(\theta)$, but the clip floors it at $1-\varepsilon$. The `min` ensures we always take the more conservative value.

### Full PPO Loss

$$
\mathcal{L}_{PPO}(\theta) = \mathcal{L}^{CLIP}(\theta) - c_v \mathcal{L}^{VF}(\theta) + c_e \mathcal{H}(\pi_\theta)
$$

Where $\mathcal{L}^{VF} = \mathbb{E}_t[(V_\theta(s_t) - V_t^{target})^2]$ is the critic loss, $\mathcal{H}(\pi_\theta)$ is the entropy bonus, and $c_v, c_e$ are balancing coefficients.

### Generalized Advantage Estimation (GAE)

$$
A_t^{GAE} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

In practice, for LLM tasks where we only get a single reward at the end: $A^{GAE} \approx r_{final} - V(s)$.

## 3.3 Why PPO's Clip Breaks for LLMs

### What the Theory Needs vs. What PPO Does

The surrogate theory requires controlling:

$$
D_{TV}(\pi_{old}(\cdot|s),\, \pi_\theta(\cdot|s)) = \frac{1}{2}\sum_{a \in \mathcal{A}} |\pi_\theta(a|s) - \pi_{old}(a|s)|
$$

This is a sum over **all actions** in $\mathcal{A}$. PPO's clip constrains $r_t(\theta)$ at a **single sampled action** $a_t \sim \pi_{old}(\cdot|s_t)$.

### The Representativeness Problem

| $\vert\mathcal{A}\vert$ | Mass per action | Clip as proxy for $D_{TV}$ |
|---|---|---|
| Small (e.g. 4) | ~25% each | ✅ One sample covers meaningful mass |
| LLM (~$10^5$) | ~$10^{-5}$ each | ❌ One token = $0.001\%$ of the simplex |

With $|\mathcal{A}| \approx 10^5$, the clip constrains one token's ratio while the remaining $10^5 - 1$ tokens shift freely and invisibly. The logical chain:

$$
\underbrace{\text{clip ratio at one sampled token}}_{\text{what PPO does}}
\;\longrightarrow\;
\underbrace{D_{TV}(\pi_{old}(\cdot|s), \pi_\theta(\cdot|s)) \text{ is small}}_{\text{what the theory needs}}
\;\longrightarrow\;
\underbrace{|\eta - L| \text{ is small}}_{\text{surrogate is reliable}}
$$

**breaks at the first arrow** when $|\mathcal{A}| \approx 10^5$.

> **Note: softmax coupling does not save us.** The gradient update propagates through the softmax to affect all token logits — but the clip constraint is only evaluated at $a_t$. If the top token shifts from $\pi_{old}(a_1|s) = 0.6$ to $\pi_\theta(a_1|s) = 0.5$, the ratio $r = 0.83$ stays within $[1-\varepsilon, 1+\varepsilon]$ and no clip fires — yet the $0.1$ probability mass has been redistributed across $10^5 - 1$ tokens in a way entirely invisible to the clip.

### PPO's Problems for LLMs (Summary)

| Issue | PPO's Behavior | Why It Hurts for LLMs |
|-------|---------------|----------------------|
| **Value function overhead** | Requires training a separate critic network | Extra parameters, extra complexity |
| **Memory inefficiency** | Stores $V(s)$ for every state in the rollout | LLMs generate thousands of tokens |
| **Reward sparsity** | Expects frequent reward signals | LLM responses get rewards only at the end |
| **Trust region failure** | Per-token clip at one sampled action | Blind to $10^5$-dim distribution shift |

---

# 4. GRPO: Dropping the Value Function

> **Citation:** DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." *DeepSeek*, 2025.

## 4.1 The Key Innovation: Group Relative Advantages

**GRPO throws out the value function entirely.** No critic network. No separate $V(s)$ estimation. For each question $q$, GRPO samples $G$ responses $\{o_1, o_2, \ldots, o_G\}$ and compares them against each other:

$$
A_i = \frac{r_i - \mu(\{r_1, r_2, \ldots, r_G\})}{\sigma(\{r_1, r_2, \ldots, r_G\})}
$$

**Why this is brilliant**: If you're comparing $G$ responses to the *same* question, the absolute value doesn't matter — what matters is how each compares to the group average. This relative ranking is all you need when the question is fixed!

### Why Drop the Value Function?

PPO's value function exists because: (1) it provides a **baseline** to reduce variance in advantage estimation, and (2) it enables **temporal difference learning** for multi-step rewards.

But for LLMs? **We get only ONE reward at the end.** There's no temporal structure to exploit — just "did you solve it or not?" The value function just learns a baseline — something GRPO gets for free from the group mean.

## 4.2 GRPO vs PPO

| Aspect | PPO | GRPO | Why GRPO Wins for LLMs |
|--------|-----|------|------------------------|
| **Advantage estimation** | $A = r + \gamma V(s') - V(s)$ | $A_i = \frac{r_i - \mu}{\sigma}$ (group-normalized) | No value function needed! |
| **Baseline** | Learns $V(s)$ from data | Uses group mean as baseline | Zero extra parameters |
| **Policy constraint** | Clip on importance ratio | Clip + KL divergence to reference | More principled regularization |
| **Memory footprint** | Stores $V(s)$ for all states | Just stores rewards | Huge memory savings |
| **Trust region** | Per-token clip | Same per-token clip (inherited) | ❌ Same structural failure |

## 4.3 Sequence-Level Formulation

$$ \begin{align}
\mathcal{J}_{GRPO}(\theta) &= \mathbb{E}_{(q,a) \sim D,\; \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)} \frac{1}{G} \sum^{G}_{i=1}\Big(\min\big(\frac{\pi_\theta(o_i | q)}{\pi_{\theta_{old}} (o_i|q)} A_i,\; \text{clip}(\frac{\pi_\theta(o_i | q)}{\pi_{\theta_{old}} (o_i |q)}, 1 - \epsilon, 1 + \epsilon)A_i\big) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\Big) \\
\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) &= \frac{\pi_{ref}(o_i | q)}{\pi_{\theta}(o_i | q)}  - \log\frac{\pi_{ref}(o_i | q)}{\pi_{\theta}(o_i | q)} - 1 \\
A_i &= \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G \})}{\text{std}(\{r_1, r_2, \dots, r_G \})}
\end{align} $$

## 4.4 Token-Level Formulation

$$ \begin{align}
\mathcal{J}_{GRPO}(\theta) &= \mathbb{E}_{(q,a) \sim D,\; \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)} \frac{1}{G} \sum^{G}_{i=1} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|}
\Big(\min\big(\frac{\pi_\theta(o_{i,t} | q, o_{i<t})}{\pi_{\theta_{old}} (o_{i,t}|q, o_{i<t})} A_{i,t},\; \text{clip}(\frac{\pi_\theta(o_{i,t} | q, o_{i<t})}{\pi_{\theta_{old}} (o_{i,t} |q, o_{i<t})}, 1 - \epsilon, 1 + \epsilon)A_{i,t}\big) - \beta \mathbb{D}_{KL}(\pi_\theta \| \pi_{ref})\Big) \\
\mathbb{D}_{KL}(\pi_\theta \| \pi_{ref}) &= \frac{\pi_{ref}(o_{i,t} | q, o_{i<t})}{\pi_{\theta}(o_{i,t} | q, o_{i<t})}  - \log\frac{\pi_{ref}(o_{i,t} | q, o_{i<t})}{\pi_{\theta}(o_{i,t} | q, o_{i<t})} - 1 \\
A_{i,t} &= \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G \})}{\text{std}(\{r_1, r_2, \dots, r_G \})}
\end{align} $$

## 4.5 GRPO's Position in the Landscape

From the trust region perspective, **GRPO inherits PPO's clip mechanism unchanged** — same per-token ratio, same symmetric band $[1-\varepsilon, 1+\varepsilon]$. All of PPO's trust region problems carry over.

**GRPO's contribution is orthogonal to the trust region problem.** It solves "how to estimate advantage without a value function" — not "how to keep $\pi_\theta$ close to $\pi_\beta$." The methods that follow address either the trust region, or the advantage/training pipeline, or both.

- ✅ Eliminates value function overhead · ✅ Better advantage estimation for sparse rewards
- ❌ Same broken trust region as PPO in large action spaces

---

# 5. Axis 1 — Advantage & Training Improvements

These methods improve **what the model learns from** — better advantage estimation, smarter sampling, richer feedback signals — without fundamentally changing the trust region mechanism.

---

## 5.1 DAPO: The Optimizer

> **Citation:** ByteDance Seed et al. "DAPO: An Open-Source LLM Reinforcement Learning System at Scale." *arXiv*, 2025.

### What GRPO Problems Does DAPO Solve?

| GRPO Problem | How DAPO Fixes It |
|-------------|-------------------|
| **KL constraint holds the model back** | Removes KL constraint entirely |
| **Wastes compute on already-perfect samples** | Dynamic sampling filters out zero-advantage prompts |
| **Symmetric clipping smothers exploration** | Asymmetric clipping: wider leash for low-prob tokens |
| **Long responses get diluted in token-level loss** | Normalize by total tokens across batch, not per-sample |
| **No penalty for overlong responses** | Length-aware reward shaping punishes truncation |

### Core Ideas

- Don't waste compute on samples that teach us nothing.
- Encourage exploration — let the model venture into the unknown!

### Removing KL Constraint (✦)

The goal of modern RL isn't just to mimic human behavior or stay close to some reference model. These days, we're pushing models to go beyond — extending their capabilities far beyond what they learned in pre-training. In most cases, clinging to a reference model just holds us back.

### Dynamic Sampling (✦✦✦✦✦)

Keep only the prompts in our batch that actually carry learning signals, while maintaining a consistent batch size.

If all $G$ outputs $\{o_i\}_{i=1}^G$ for a particular prompt are correct and receive identical rewards, that prompt is essentially useless — a zero advantage means zero gradient. So we shuffle it out of the dataset and move on.

### Asymmetric Clipping (✦✦✦)

The intuition here is elegant: "exploitation" tokens that already have high probability (say, 0.9) shouldn't be artificially capped from getting even more likely (like 0.999). Meanwhile, low-probability "exploration" tokens struggle to gain probability — it's inherently harder to bootstrap the unlikely.

The fix? Raise $\epsilon_{high}$ to give exploration tokens more room to breathe.

> **Trust region perspective:** From the trust region angle, asymmetric clipping recognizes that PPO's symmetric clip has an asymmetric effect in large action spaces: it over-constrains rare (exploration) tokens while under-constraining dominant ones. It doesn't fix the fundamental blindness of per-token clipping, but mitigates one of its worst consequences.

### Token-Level Policy Gradient Loss (✦✦)

The original GRPO uses sample-level loss: average losses by token within each sample, then aggregate across samples. This gives every sample equal weight — problematic for long responses.

Here's why: when all samples have equal weight, tokens in longer responses contribute disproportionately less to the total loss. Each token gets diluted by its peers. Not ideal when you want every token to matter.

### Overlong Reward Shaping (✦✦)

A clever length-aware penalty for truncated responses:

$$
R_{\text{length}}(y) =
\begin{cases}
0, & |y| \le L_{\max} - L_{\text{cache}} \\
\frac{(L_{\max} - L_{\text{cache}}) - |y|}{L_{\text{cache}}}, & L_{\max} - L_{\text{cache}} < |y| \le L_{\max} \\
-1, & L_{\max} < |y|
\end{cases}
$$

### DAPO Objective

$$ \begin{align}
\mathcal{J}_{DAPO}(\theta) =& \mathbb{E}_{(q,a) \sim D,\; \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)} \frac{1}{\sum^{G}_{i=1} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|}
\min\big(\frac{\pi_\theta(o_{i,t} | q, o_{i<t})}{\pi_{\theta_{old}} (o_{i,t}|q, o_{i<t})} A_{i,t},\; \text{clip}(\frac{\pi_\theta(o_{i,t} | q, o_{i<t})}{\pi_{\theta_{old}} (o_{i,t} |q, o_{i<t})}, 1 - \epsilon_{low}, 1 + \epsilon_{high})A_{i,t}\big) \\
\text{s.t.} \quad &0 < | \{o_i \mid \text{is\_equivalent}(a, o_i)\} | < G \\
A_{i,t} =& \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G \})}{\text{std}(\{r_1, r_2, \dots, r_G \})}
\end{align} $$

### Algorithm

**Input** Initial policy $\pi_{\theta}$; reward model $R$; task prompts $\mathcal{D}$; hyperparameters $\epsilon_{low}$, $\epsilon_{high}$

1: **for** step = 1, ... , M **do**
2: &ensp; Sample a batch $\mathcal{D}_b$ from $\mathcal{D}$
3: &ensp; Update old policy: $\pi_{\theta_{old}} \leftarrow \pi_{\theta}$
4: &ensp; For each question $q \in \mathcal{D}_b$, sample $G$ outputs $\{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot | q)$
5: &ensp; Compute rewards $\{r_i\}_{i=1}^G$ for each output $o_i$ by running $R$
6: &ensp; Filter out useless $o_i$ and add the rest to the dynamic sampling buffer
7: &ensp; **if** buffer size $n_b < N$: **continue**
8: &ensp; For each $o_i$ in buffer, compute advantage $A_{i,t}$ for the $t$-th token
9: &ensp; **for** iteration = 1, ... , $\mu$ **do**
10: &ensp;&ensp; Update policy $\pi_\theta$ by maximizing the DAPO objective

**Output** $\pi_{\theta}$

### Takeaway

Why waste cycles on trivial math like 1+1? Just toss those samples out and focus on problems that actually challenge the model.

---

## 5.2 Entropy Mechanism: The Explorer

### What GRPO Problem Does It Solve?

| GRPO Problem | How It Fixes It |
|-------------|----------------|
| **Entropy collapses → exploration dies** | Clip high-covariance samples so they don't dominate |
| **High covariance between Adv and log-prob kills diversity** | Selectively detach gradients from samples that cause entropy crash |
| **Model converges to narrow strategies** | Keep exploring diverse paths by preserving low-covariance samples |

### Core Idea

High covariance between advantages and log-probabilities causes entropy to plummet — which kills exploration. The fix? Clip these problematic samples to keep the model curious.

### Covariance of Advantage and Token Log-Prob

Consider a batch of $N$ rollout tokens. Let $\pi_{\theta}(o_{i,t})$ denote the output probability of the policy model for token $o_{i,t}$ given its prefix.

$$\begin{align}
\text{Cov}(o_{i,t}) =
    \Big(
        \log \pi_{\theta}(o_{i,t}) -
        \frac{1}{N}\sum_{g=1}^{N}\log \pi_{\theta}(o_{g,t})
    \Big)
    \cdot
    \Big(
        A(o_{i,t}) -
        \frac{1}{N}\sum_{g=1}^{N}A(o_{g,t})
    \Big)
\end{align}$$

### Algorithm

```python
def compute_policy_loss(old_log_prob, log_prob, advantages, select_ratio,
                        e_lb, e_ub, cov_lb, cov_ub):
    """
    old_log_prob: old_policy log, shape [bs, response_len]
    log_prob:     cur_policy log, shape [bs, response_len]
    advantages:   adv, shape [bs, response_len]
    select_ratio: scalar
    e_lb:   origin ppo clip ratio (lower)
    e_ub:   origin ppo clip ratio (upper)
    cov_lb: clip_cov lower bound
    cov_ub: clip_cov upper bound
    """
    ratio = exp(log_prob - old_log_prob)
    pg_losses1 = -ratio * advantages
    covs = (log_prob - log_prob.mean()) * (advantages - advantages.mean())
    select_num = int(select_ratio * len(pg_losses1))
    pg_losses2 = -clip(ratio, 1 - e_lb, 1 + e_ub) * advantages
    clip_idx = random_select(
        covs[(covs > 1 - cov_lb) & (covs < 1 + cov_ub)], num=select_num
    )
    pg_losses1[clip_idx].detach_()
    pg_losses2[clip_idx].detach_()
    pg_loss = maximum(pg_losses1, pg_losses2).mean()
```

### Takeaway

To preserve entropy (that precious ability to explore), we need to ditch samples that cause entropy to crash too fast. If you're trying to crack a math problem but always approach it the same wrong way, no amount of practice will save you. Keep exploring diverse paths!

---

## 5.3 Natural Language Feedback: The Mentor

> **Citation:** Huang et al. "Bootstrapping Exploration with Group-Level Natural Language Feedback in Reinforcement Learning." *Tsinghua University & ByteDance*, 2025.

### What GRPO Problem Does It Solve?

| GRPO Problem | How NLF Fixes It |
|-------------|------------------|
| **Scalar reward (right/wrong) is too sparse** | Rich language critiques instead of just a number |
| **No signal for *why* a response failed** | Critique explains mistakes, not just their presence |
| **Policy gets stuck in low-reward regions** | Adaptive injection of off-policy refinement scaffolds |
| **Only learns from final outcome** | Aggregates feedback across multiple failed attempts |

### Core Ideas

- **Aggregated Feedback Refinement**: Combine external critiques with intra-group comparisons to produce refined responses for failed attempts. Not only are errors corrected, but diverse reasoning paths are explored.

- **Adaptive Refinement Injection**: When the policy gets stuck in sparse-reward regions, adaptively inject high-quality refinements as off-policy scaffolds — targeted guidance without killing exploration.

- **Joint Optimization of Generation and Refinement**: Optimize both generation and refinement together in one unified RL loop. Self-refinement improves → scaffolds get better → exploration gets better. A beautiful virtuous cycle!

### Group-Level Feedback Aggregated Refinement

For each prompt $x \sim \mathcal{D}$, sample $N$ responses: $\mathcal{G}_{gen}(x) = \{y^{(i)}\}_{i=1}^N$, then query the reward model for both a scalar reward and a critique: $(r^{(i)}, c^{(i)}) = R(x, y^{(i)})$.

Two flavors of feedback:
- **External feedback**: The critique $c^{(i)}$ attached to a specific response $y^{(i)}$
- **Intra-group feedback**: Alternative responses within $\mathcal{G}_{gen}(x)$ that often contain complementary partial ideas

Instead of fixing each failure in isolation, we throw them all together with their critiques into one big context. Concretely, we gather the failure set $\mathcal{F} = \{(y^{(i)}, c^{(i)}) \mid r^{(i)} = 0\}$ and construct an aggregated refinement prompt:

$$p_{agg}(x) = \text{CONCAT}(x, \mathcal{F}(x))$$

Conditioned on $p_{agg}(x)$, we sample a refinement group $\mathcal{G}_{refine}(x) = \{\tilde{y}^{(j)}\}_{j=1}^{N}$ with $\tilde{y} \sim \pi_{old}(\cdot|p_{agg}(x))$.

### Mixed Policy Optimization

During post-training, LLMs learn to solve problems via RL. But test-time self-refinement? That's not explicitly trained for. Empirically, standard RL fine-tuning can actually *hurt* when combined with test-time self-refinement. To fix this, we explicitly train the LLM to improve both direct problem-solving AND feedback-conditioned refinement in one integrated RL process.

$$\begin{align}
\mathcal{J}_{Mixed}(\theta) &= 
\frac{1}{Z}\Big[
    \sum_{i=1}^{N_{on}}
    \sum_{t=1}^{|o_i|}
    \text{CLIP}(r_{i,t}^{on}(\theta),\; \tilde{A}_i,\; \epsilon)
     + 
    \sum_{i=1}^{N_{off}}
    \sum_{t=1}^{|o_i|}
    \text{CLIP}(f(r_{i,t}^{off}(\theta)),\; \tilde{A}_i,\; \epsilon)
    \Big]  \\
Z &= \sum_{i=1}^{N_{on}} |o_i| + \sum_{i=1}^{N_{off}} |o_i| \\
r_{i,t}^{on}(\theta) &= \frac{\pi_\theta(o_{i,t} | q, o_{i<t})}{\pi_{\theta_{old}} (o_{i,t} | q, o_{i<t})} \\
r_{i,t}^{off}(\theta) &= \frac{\pi_\theta(o_{i,t} | q, o_{i<t})}{\pi_{\theta_{old}} (o_{i,t} | p_{agg}(x), o_{i<t})} \\
\tilde{A}_i &= R(o_i) - \text{mean}(\mathcal{G}_{on}(x) \cup \mathcal{G}_{off}(x)) \\
f(\mu) &= \mu / (\mu + \lambda)
\end{align}$$

### Takeaway

When you make mistakes, you take notes, reflect on how to avoid repeating them, and then try again with a better approach. That's essentially what's happening here — this process improves the actor's exploration capabilities and boosts training efficiency by learning from the feedback.

---

# 6. Axis 2 — Trust Region Improvements

These methods address the fundamental question: **how do you measure "close" in a $10^5$-dimensional action space?** They improve the trust region mechanism itself — how the policy update is constrained to keep the surrogate reliable.

---

## 6.1 GSPO: The Stabilizer

> **Citation:** Zhang et al. "GSPO: Geometric-Mean Sequence-Level Policy Optimization." 2025.

### What GRPO Problem Does GSPO Solve?

| GRPO Problem | How GSPO Fixes It |
|-------------|-------------------|
| **Token-level importance weights = high variance noise** | Switch to sequence-level importance weight |
| **Each token's weight based on single sample from $\pi_{old}$** | Use geometric mean across all tokens in the sequence |
| **Token-level weights fail at correcting distribution shift** | Sequence-level treats the whole response as one action |

### Core Idea

Importance ratios should really be computed over multiple samples ($N \gg 1$). But GRPO attaches the importance weight $\frac{\pi_\theta(o_{i,t} | q, o_{i<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i<t})}$ at each individual token position $t$. Since this weight is based on just a single sample from each next-token distribution, it fails at its intended job of correcting for distribution shift. Instead? It injects high-variance noise straight into the training gradients.

### Sequence-Level Formulation

Define the **sequence-level ratio** as the geometric mean of per-token ratios:

$$
s_i(\theta) = \left(\prod_{t=1}^{|o_i|} r_{i,t}(\theta)\right)^{1/|o_i|} = \exp\left(\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}\log r_{i,t}(\theta)\right)
$$

**Surrogate:**

$$ \begin{align}
\mathcal{J}_{GSPO}(\theta) &= \mathbb{E}_{(q,a) \sim D,\; \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)} \frac{1}{G} \sum^{G}_{i=1}\Big(\min\big(s_i(\theta) \cdot A_i,\; \text{clip}(s_i(\theta), 1 - \epsilon, 1 + \epsilon) \cdot A_i\big)\Big) \\
A_i &= \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G \})}{\text{std}(\{r_1, r_2, \dots, r_G \})}
\end{align} $$

**Strategy**: instead of clipping per-token $r_{i,t}$, clip the **sequence-level** $s_i$ — a single scalar that aggregates information across all tokens. By treating the entire response as one action, the clip on $s_i$ becomes a more meaningful proxy for keeping $\pi_\theta$ close to $\pi_\beta$.

### Gradient Comparison: GSPO vs GRPO

Dropping the clip mechanism to see the structural difference:

$$ \begin{align}
\nabla_{\theta}\mathcal{J}_{GSPO}(\theta)
&= \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G}\Big(\frac{\pi_\theta(o_i | q)}{\pi_{\theta_{old}} (o_i|q)}\Big)^{\frac{1}{|o_i|}} A_i \cdot  \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} 
\nabla_{\theta} \log \pi_\theta(o_{i,t} | q, o_{i<t})\Big] \\
\nabla_{\theta}\mathcal{J}_{GRPO}(\theta) 
&= \mathbb{E}\Big[\frac{1}{G}\sum_{i=1}^{G} A_{i} \frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
    \frac{\pi_\theta(o_{i,t} | q, o_{i<t})}{\pi_{\theta_{old}}(o_{i,t} | q, o_{i<t})}
    \nabla_{\theta} \log \pi_\theta(o_{i,t} | q, o_{i<t})\Big]
\end{align} $$

**Key insight**: GSPO applies a **uniform** sequence-level weight to all tokens, while GRPO applies **per-token** weights that inject high-variance noise. GSPO treats all tokens in a response equally, eliminating the instability that plagues GRPO.

### Token-Level Variant: GSPO-token

$$
\begin{align}
\mathcal{J}_{GSPO\text{-}token}(\theta) &= 
\mathbb{E}_{(q,a) \sim D,\; \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)} 
\frac{1}{G} \sum^{G}_{i=1}
\frac{1}{|o_i|} \sum^{|o_i|}_{t=1}
\min\big(s_{i,t}(\theta) \cdot A_{i,t},\;
    \text{clip}(s_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon) \cdot A_{i,t}\big)
\end{align}
$$

$$
s_{i,t}(\theta) = \text{sg}[s_i(\theta)] \cdot
\frac{\pi_{\theta}(o_{i,t} | x, o_{i < t})}{\text{sg}[\pi_{\theta}(o_{i,t} | x, o_{i < t})]}
$$

where $\text{sg}[\cdot]$ denotes stop-gradient. The gradient becomes:

$$ \begin{align}
\nabla_{\theta} \mathcal{J}_{GSPO\text{-}token}(\theta) &= \mathbb{E}\Big[
    \frac{1}{G}
    \sum_{i=1}^{G}
    \Big(\frac{\pi_\theta(o_{i} | q)}{\pi_{\theta_{old}}(o_{i} | q)}\Big)^{\frac{1}{|o_{i}|}}
    \frac{1}{|o_i|}
    \sum_{t=1}^{|o_i|}
    A_{i,t}
    \nabla_{\theta} \log \pi_\theta(o_{i,t} | q, o_{i<t})
\Big]
\end{align} $$

### Takeaway

By giving all tokens in a response equal weight, GSPO achieves more stable importance sampling. If you view the action space as the entire response rather than a single token at each time step, GSPO makes a lot of sense at the sequence level.

- ✅ Aggregates information across all tokens → better proxy for the true ratio
- ✅ More stable than per-token IS

---

## 6.2 SAPO: The Smoother

> **Citation:** Gao et al. "Soft Adaptive Policy Optimization." *Qwen Team, Alibaba*, 2025.

### What GRPO Problem Does SAPO Solve?

| GRPO Problem | How SAPO Fixes It |
|-------------|-------------------|
| **Hard clip creates gradient discontinuity** | Smooth sigmoid gate — no gradient jumps |
| **GSPO suppresses ALL gradients when few tokens are off-policy** | Token-adaptive: only down-weights the offending tokens |
| **Symmetric clip has asymmetric effects** | Asymmetric temperatures: $\tau_{\text{neg}} > \tau_{\text{pos}}$ |

### Surrogate

Replace the hard clip with a smooth sigmoid gate $f$ applied to the per-token ratio:

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

### How It Works

$w_{i,t}$ peaks at $r_{i,t} = 1$ (on-policy) and decays smoothly as the ratio deviates — implementing a **continuous trust region**:
- Near on-policy → gradients preserved (encourages useful updates)
- Far off-policy → gradients attenuated (prevents instability)
- At $r_{i,t} = 1$: gradient equals the unclipped objective, regardless of $\tau$

### Sequence-Coherent + Token-Adaptive

SAPO is both **sequence-coherent** and **token-adaptive**:
- Under mild assumptions (small steps + low intra-sequence dispersion), the average token gate concentrates to a sequence-level gate $g(\log s_i) = \text{sech}^2(\frac{\tau_i}{2} \log s_i)$ — reducing to a GSPO-like formulation but with smooth boundaries
- When a few tokens are highly off-policy, GSPO suppresses **all** gradients for that sequence; SAPO selectively down-weights only the offending tokens

### Asymmetric Temperatures

$\tau_{\text{neg}} > \tau_{\text{pos}}$ — negative-token updates (which increase logits of many inappropriate tokens) decay faster, reflecting their greater instability risk.

- ✅ Smooth trust region (no gradient discontinuity) · ✅ Token-adaptive + sequence-coherent
- ⚠️ Per-token ratio, but with better gating

---

## 6.3 DPPO: The Theorist

> **Citation:** Qi et al. "Rethinking the Trust Region in LLM Reinforcement Learning." 2026.

### What GRPO Problem Does DPPO Solve?

| GRPO Problem | How DPPO Fixes It |
|-------------|-------------------|
| **Per-token ratio is a noisy MC estimate of true divergence** | Directly estimate $D_{TV}$ / $D_{KL}$ |
| **Ratio is hypersensitive to probability magnitude** | Distribution-level constraint is magnitude-invariant |
| **Clip can't distinguish safe vs unsafe updates** | Divergence correctly identifies dangerous updates |

### The Root Cause

DPPO identifies the root cause: PPO's ratio $r_t(\theta)$ is a **noisy single-sample Monte Carlo estimate** of the true policy divergence. The ratio is hypersensitive to probability magnitude:

- A rare token going from $10^{-5}$ to $10^{-3}$ produces $r_t = 100$ (triggers clip), yet the actual $D_{TV}$ change is negligible
- A dominant token dropping from $0.99$ to $0.8$ has $r_t \approx 0.81$ (no clip), yet the divergence is catastrophic

### Surrogate

Replace the ratio clip with a **direct divergence constraint** — bringing back TRPO's distribution-level guarantee using only first-order optimization:

$$
\max_\theta \; \mathbb{E}_{s_t, a_t \sim \color{red}{\pi_\beta}} \left[ \big(r_t(\theta) - 1\big) \cdot \hat{A}_t \right] \quad \text{s.t.} \quad \max_t \; D_{TV}\big(\color{red}{\pi_\beta}(\cdot|s_t) \| \color{blue}{\pi_\theta}(\cdot|s_t)\big) \le \delta
$$

Note: the surrogate uses $(r_t - 1)$ instead of $r_t$ — they differ by a constant and have the same gradient, but this form makes the connection to the performance difference identity clearer.

### Efficient Approximations

To avoid storing the full vocabulary distribution (memory-prohibitive for LLMs), DPPO introduces:

- **Binary-TV**: approximate $D_{TV}(\pi_\beta(\cdot|s_t), \pi_\theta(\cdot|s_t)) \approx |\pi_\beta(y_t|s_t) - \pi_\theta(y_t|s_t)|$ — collapse the vocabulary into "sampled token" vs "everything else"
- **Top-K**: compute divergence over the top-K most probable tokens, capturing the essential distributional shift with negligible overhead

### Key Findings

1. **Trust region is essential** even at tiny learning rates ($10^{-6}$) — without it, training-inference mismatch accumulates and collapses
2. **Anchor to rollout policy $\mu_{\theta'}$**, not recomputed $\pi_{\theta'}$ — using recomputed policy as anchor leads to instability
3. **Primary instability source**: a tiny fraction ($\le 0.5\%$) of updates on negative samples that push the policy far outside the trust region

- ✅ Distribution-level trust region (like TRPO) · ✅ First-order optimization (like PPO)
- ✅ Correctly distinguishes safe vs unsafe updates regardless of token probability

---

## 6.4 MinPRO: The Prefix Corrector

> **Citation:** Lei et al. "A Step Back: Prefix Importance Ratio Stabilizes Policy Optimization." *University of Sydney & ByteDance & NTU*, 2025.

### What GRPO Problem Does MinPRO Solve?

| GRPO Problem | How MinPRO Fixes It |
|-------------|-------------------|
| **Token-level ratio $\rho_t$ ignores prefix context** | Incorporates prefix-level IS correction |
| **Off-policy drift causes training collapse** | Suppresses gradients from unlikely prefixes |
| **Cumulative prefix ratio $\rho_{1:t}$ has explosive variance** | Uses $\min_{i<t} \rho_i$ as a stable, non-cumulative surrogate |
| **Hard clipping discards too many token gradients under off-policy** | Adopts soft clipping (CISPO-style) with prefix correction |

### Theoretical Foundation: The Prefix Importance Ratio

MinPRO starts from a rigorous theoretical observation. By the policy gradient theorem under off-policy conditions, the correct gradient is:

$$
\nabla_\theta J(\theta) = \sum_{t=1}^{T} \mathbb{E}_{(o_1,\ldots,o_t) \sim \pi_{\theta_{old}}} \left[ \rho_{1:t} \nabla_\theta \log \pi_\theta(o_t | o_{<t}) A^\pi(o_t; o_{<t}) \right]
$$

where the **prefix importance ratio** is:

$$
\rho_{1:t} = \frac{P_\theta(o_1, \ldots, o_t)}{P_{\theta_{old}}(o_1, \ldots, o_t)} = \prod_{i=1}^{t} \rho_i = \prod_{i=1}^{t} \frac{\pi_\theta(o_i | o_{<i})}{\pi_{\theta_{old}}(o_i | o_{<i})}
$$

This is the **theoretically rigorous** correction term — not the token-level ratio $\rho_t$ that GRPO, DAPO, and most other methods use. The token-level ratio is a relaxation that works reasonably well in near on-policy settings, but **fails catastrophically under large off-policy drift**.

### Why Token-Level Ratio Fails Under Off-Policy

MinPRO provides compelling empirical evidence: under off-policy training (where rollouts are delayed by 2+ global steps), GRPO, GSPO, and CISPO all exhibit severe instability:

- **GRPO** (hard clip): converges to a low reward plateau with severe oscillations
- **GSPO** (sequence-level clip): similarly unstable rewards and entropy dynamics
- **CISPO** (soft clip): even more pronounced failure — training collapses as rewards drop and entropy explodes

The root cause: in autoregressive LLMs, $\rho_{1:t} = \prod_{i=1}^{t} \rho_i$ grows multiplicatively with sequence length. The token-level approximation $\rho_t$ rapidly diverges from the true prefix ratio as off-policyness increases, especially for long rollouts (often exceeding 10,000 tokens).

### The MinPRO Objective

Directly using $\rho_{1:t}$ is impractical — the cumulative product suffers from (1) extreme values causing large variance, and (2) length bias (extreme values cluster near sequence ends, where tokens are most important for correctness).

MinPRO's key insight: replace the unstable cumulative product with a **non-cumulative surrogate** — the minimum token ratio in the prefix:

$$
\underline{\rho}_t = \min_{i < t} \rho_i
$$

The prefix ratio is then approximated as $\rho_{1:t} \approx \underline{\rho}_t \cdot \rho_t$, yielding the MinPRO objective:

$$
\mathcal{J}_{MinPRO}(\theta) = \mathbb{E}_{q \sim \mathcal{S},\, \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|q)} \left[ \frac{1}{\sum_{i=1}^{G} |o_i|} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \text{sg}\left[\text{clip}\left(\underline{\rho}^i_t \cdot \rho^i_t,\, 1-\varepsilon_{low},\, 1+\varepsilon_{high}\right)\right] \hat{A}^i_t \log \pi_\theta(o^i_t | q, o^i_{<t}) \right]
$$

where $\text{sg}[\cdot]$ denotes stop-gradient (soft clipping, CISPO-style).

### Intuition

When $\underline{\rho}_t$ is very small, it means some token in the prefix is extremely unlikely under the current policy — the entire prefix trajectory is "off-track." In this case, the gradient for token $o_t$ should be suppressed, even if $\rho_t$ itself looks normal. Prior methods ignore this prefix-level signal entirely and still apply the full gradient, leading to unstable updates.

MinPRO elegantly captures this: $\underline{\rho}_t$ acts as a "prefix health check" — if any preceding token has drifted far from the current policy, the entire downstream gradient is attenuated.

### Why $\min$ Instead of $\prod$?

The minimum prefix ratio avoids both failure modes of the raw cumulative product:

| Property | $\rho_{1:t} = \prod_{i=1}^{t} \rho_i$ | $\underline{\rho}_t = \min_{i<t} \rho_i$ |
|---|---|---|
| **Variance** | Exponential growth with $t$ | Bounded, non-cumulative |
| **Length bias** | Extreme values cluster at sequence end | No systematic position bias |
| **Prefix signal** | ✅ Full prefix information | ✅ Captures worst-case deviation |
| **Numerical stability** | ❌ Can overflow/underflow | ✅ Always in a reasonable range |

### Experimental Results

Tested on Qwen3-8B-Base, Qwen3-14B-Base, and Qwen3-30B-A3B-Base (MoE) across 7 math benchmarks under large off-policy drift:

- **Training stability**: MinPRO maintains consistently higher and more stable rewards throughout training, while GRPO, GSPO, and CISPO exhibit severe oscillations or collapse
- **pass@1**: MinPRO achieves the highest average scores (59.4 on 8B, 62.6 on 14B), outperforming all baselines including M2PO
- **pass@k**: Best overall pass@k performance across $k \in \{1, 2, 4, 8, 16, 32, 64, 128\}$
- **MoE scaling**: Superior and stable performance on Qwen3-30B-A3B, demonstrating scalability

### Discussion: Hard Clip vs Soft Clip Under Off-Policy

MinPRO's analysis reveals an important insight about the failure modes of different clipping strategies:

- **GRPO (hard clip)**: Under large off-policy drift, many token ratios fall outside the trust region and get clipped — discarding a large fraction of gradients. This leads to a low but stable reward plateau (too many useful gradients are thrown away).
- **CISPO (soft clip)**: Retains all token gradients but only constrains magnitudes. Without prefix-level correction, extreme token ratios still propagate unstable updates, causing abrupt training collapse.
- **MinPRO**: By incorporating $\underline{\rho}_t$, it provides a principled mechanism for suppressing overly influential high-ratio tokens while preserving informative gradients — achieving both stability and strong performance.

### Unsuccessful Attempts (Lessons Learned)

The MinPRO paper also documents what *didn't* work:

1. **Direct use of $\rho_{1:t}$**: Replacing $\rho_t$ with the full prefix ratio in CISPO (with soft clipping) showed no improvement — the large variance and length bias of the cumulative product cannot be ignored.
2. **Filtering by $\rho_{1:t}$**: Removing the lowest 1% of tokens ranked by $\rho_{1:t}$ caused training to stall — because extremely small $\rho_{1:t}$ values cluster at sequence ends, where tokens are crucial for producing correct final answers.

### Takeaway

MinPRO is the first method to successfully incorporate prefix-level IS correction into LLM policy optimization. Its key insight — using $\min_{i<t} \rho_i$ as a stable proxy for the cumulative prefix ratio — is both theoretically motivated and practically effective. It demonstrates that the commonly used token-level ratio is fundamentally insufficient under off-policy conditions, and that even a simple prefix-aware correction can dramatically improve training stability.

- ✅ Prefix-level IS correction (theoretically grounded) · ✅ Numerically stable (non-cumulative)
- ✅ Superior off-policy stability · ✅ Scales to large MoE models
- ⚠️ $\min$ is a conservative approximation of the full prefix ratio

---

# 7. Summary

## 7.1 The Two Orthogonal Axes

The methods in this survey split into two orthogonal axes of improvement:

```
PPO (The Foundation)
    │
    ├── Problem: Needs a value function (extra memory & params)
    ├── Problem: Designed for dense rewards, not LLM's sparse end-of-sequence rewards
    ├── Problem: Per-token clip is blind to 10^5-dim distribution shift
    │
    ▼
GRPO (The Innovation — Advantage Estimation)
    │
    ├── Insight: Drop the value function entirely
    ├── Solution: Use group-relative advantages instead
    ├── Benefit: Zero extra params, perfect for single-reward tasks
    └── Limitation: Inherits PPO's broken trust region unchanged
    │
    ├─── Axis 1: Advantage & Training ──────────────────────────
    │
    ├── DAPO (The Optimizer)
    │   ├── Remove KL, asymmetric clip, dynamic sampling
    │   ├── Token-level normalization, overlong reward shaping
    │   └── ⚠️ Still per-token trust region heuristic
    │
    ├── Entropy Mechanism (The Explorer)
    │   └── Clip high-covariance samples to preserve diversity
    │
    ├── Natural Language Feedback (The Mentor)
    │   └── Rich critiques + aggregated refinement + mixed optimization
    │
    ├─── Axis 2: Trust Region ──────────────────────────────────
    │
    ├── GSPO (The Stabilizer)
    │   └── Sequence-level importance weights (geometric mean)
    │
    ├── SAPO (The Smoother)
    │   └── Smooth sigmoid gate, token-adaptive + sequence-coherent
    │
    ├── DPPO (The Theorist)
    │   └── Direct D_TV/D_KL estimation (distribution-level, 1st-order)
    │
    └── MinPRO (The Prefix Corrector)
        └── Prefix importance ratio via min-token surrogate
```

## 7.2 Trust Region Comparison

| Method | Action ratio $\frac{\pi_\theta}{\pi_\beta}$ | Trust region mechanism | Guarantee |
|--------|---|---|---|
| **TRPO** | Full distribution | Hard KL constraint (2nd-order) | ✅ Theoretical bound |
| **PPO / GRPO** | Single sampled token | Ratio clip at one sample | ❌ Fails for $\|\mathcal{A}\| \gg 1$ |
| **DAPO** (Clip-Higher) | Single token, asymmetric bounds | Asymmetric ratio clip | ⚠️ Heuristic improvement |
| **GSPO** | Geometric mean over all tokens | Clip on sequence-level ratio | ✅ Better proxy |
| **SAPO** | Per-token with smooth gating | Sigmoid soft gate ($\tau$-controlled) | ✅ Smooth + token-adaptive |
| **DPPO** | Full distribution (approx.) | Direct $D_{TV}$/$D_{KL}$ estimate (1st-order) | ✅ Distribution-level, scalable |
| **MinPRO** | Prefix-corrected per-token | $\underline{\rho}_t \cdot \rho_t$ with soft clip | ✅ Prefix-level, stable under off-policy |

## 7.3 Quick Reference: All Methods at a Glance

| Method | Problem It Solves | Key Innovation | Axis |
|--------|-------------------|----------------|------|
| **GRPO** | Value function overhead for sparse rewards | Group-relative advantage estimation | Advantage |
| **DAPO** | KL constraint stifles growth; wasted compute | Remove KL, asymmetric clip, dynamic sampling | Both |
| **DAPO** | Overlong responses unpunished | Length-aware reward shaping | Advantage |
| **GSPO** | Token-level weights = variance noise | Sequence-level importance weight (geometric mean) | Trust Region |
| **SAPO** | Hard clip = gradient discontinuity; GSPO over-suppresses | Smooth sigmoid gate, token-adaptive | Trust Region |
| **DPPO** | Per-token ratio blind to distribution shift | Direct $D_{TV}$ estimation, distribution-level | Trust Region |
| **Entropy** | Entropy collapse kills exploration | Clip high-covariance samples | Advantage |
| **NLF** | Scalar reward too sparse; no "why" signal | Rich language critiques + aggregated refinement | Advantage |
| **MinPRO** | Token-level ratio fails under off-policy drift | Prefix IS correction via $\min_{i<t} \rho_i$ surrogate | Trust Region |

## 7.4 Key Themes

1. **Value Function? Who Needs It!** LLMs give us ONE reward at the end. PPO's value function just learns a baseline — which we can get for FREE from group comparison.

2. **Sample Efficiency is Everything.** Whether it's dynamic sampling (DAPO), sequence-level importance weights (GSPO), or selective entropy preservation, the goal is the same: squeeze maximum learning from minimum samples.

3. **Exploration vs. Exploitation in Text Space.** DAPO's asymmetric clipping lets exploration tokens breathe; Entropy Mechanism preserves diversity; NLF uses critiques to explore reasoning paths that pure rewards never signal.

4. **Trust Region Must Be Distribution-Level.** The progression from PPO → GSPO → SAPO → DPPO shows a clear trend: moving from per-token heuristics toward distribution-level guarantees, while keeping first-order optimization tractable.

5. **Prefix Context Matters.** MinPRO demonstrates that even within the token-level paradigm, incorporating prefix-level information (via the minimum prefix ratio) dramatically improves off-policy stability — a direction orthogonal to the distribution-level improvements of DPPO.

---

# 8. Outlook

## 8.1 The Prefix Importance Ratio: From Theory to Practice

A central theme across the methods surveyed is the treatment of the **state distribution ratio** $\frac{d^{\pi_\theta}(s)}{d^{\pi_\beta}(s)}$. Most methods drop it entirely, hoping that a tight action-level trust region implies $d^{\pi_\theta} \approx d^{\pi_\beta}$. MinPRO was the first to successfully incorporate prefix-level IS correction, demonstrating that this "dropped" term matters enormously under off-policy conditions.

The theoretical foundation is rooted in a special structural property of LLMs:

### LLM Transitions Are Deterministic

In autoregressive generation, the transition is **deterministic**: given state $s_t = (q, o_{<t})$ and action $a_t = o_t$, the next state is uniquely determined:

$$
P(s_{t+1} | s_t, a_t) = \mathbf{1}[s_{t+1} = (q, o_{\le t})]
$$

This means the state distribution ratio **decomposes exactly** into a product of per-token action ratios:

$$
\frac{d^{\pi_\theta}(s_t)}{d^{\pi_\beta}(s_t)} = \frac{\pi_\theta(o_1|q) \cdot \pi_\theta(o_2|q,o_1) \cdots \pi_\theta(o_{t-1}|q,o_{<t-1})}{\pi_\beta(o_1|q) \cdot \pi_\beta(o_2|q,o_1) \cdots \pi_\beta(o_{t-1}|q,o_{<t-1})} = \prod_{j=1}^{t-1} r_j(\theta)
$$

This identity is **exact** — not an approximation. It does not hold in general MDPs (where stochastic transitions break the product structure), but it holds in LLMs because the environment is fully determined by the agent's own actions.

### The True Objective Becomes Computable

Substituting back into the full importance weight:

$$
w(s_t, a_t) = \underbrace{\prod_{j=1}^{t-1} r_j(\theta)}_{\text{state correction (exact)}} \cdot \underbrace{r_t(\theta)}_{\text{action correction}} = \prod_{j=1}^{t} r_j(\theta)
$$

The true off-policy surrogate is therefore:

$$
L^{\text{true}}(\theta) = \mathbb{E}_{o \sim \pi_\beta} \left[ \sum_{t=1}^{T} \left(\prod_{j=1}^{t} r_j(\theta)\right) \cdot \hat{A}_t \right]
$$

This is no longer intractable — every $r_j(\theta)$ is already computed during training. The state ratio, which most methods drop, is **free** in the LLM setting.

### The Variance Problem: What MinPRO Taught Us

The obvious challenge: $\prod_{j=1}^{t} r_j(\theta)$ suffers from **exponential variance growth** — the classic curse of trajectory-level importance sampling (Precup et al., 2000). MinPRO's experience confirms this:

1. **Direct use of $\rho_{1:t}$** failed — the large variance and length bias cannot be ignored
2. **Filtering by $\rho_{1:t}$** failed — extreme values cluster at sequence ends where tokens matter most
3. **$\min_{i<t} \rho_i$ succeeded** — a non-cumulative surrogate that preserves prefix signal without variance explosion

This suggests a spectrum of prefix correction strategies, from conservative to exact:

| Strategy | Formula | Bias | Variance | Status |
|---|---|---|---|---|
| Token-level (GRPO) | $\rho_t$ | High | Low | ❌ Fails off-policy |
| MinPRO | $\underline{\rho}_t \cdot \rho_t$ | Moderate | Low | ✅ Proven effective |
| Truncated window $k$ | $\prod_{j=t-k}^{t} \rho_j$ | Moderate | Moderate | 🔬 Unexplored |
| Geometric mean | $(\prod_{j=1}^{t} \rho_j)^{1/t}$ | Moderate | Low | 🔬 Unexplored as state correction |
| Full prefix | $\prod_{j=1}^{t} \rho_j$ | Zero | Explosive | ❌ Impractical |

The truncated window approach — $w_t^{(k)}(\theta) = \prod_{j=\max(1,\, t-k)}^{t} r_j(\theta)$ — remains unexplored and could offer a middle ground between MinPRO's conservative $\min$ and the full prefix product. It can be further stabilized by combining with existing trust region mechanisms — e.g., applying SAPO's soft gate or using DPPO's divergence constraint alongside the prefix correction.

### Connection to Existing Methods

- **MinPRO** uses $\min_{i<t} \rho_i$ as a stable proxy for the prefix ratio — the first successful incorporation of prefix-level IS correction in LLM RL.
- **GSPO** computes the geometric mean $s_i = (\prod_{t=1}^{|o_i|} r_{i,t})^{1/|o_i|}$ as a **sequence-level clip target**. Note that this is a fundamentally different mathematical object from the prefix ratio $\rho_{1:t} = \prod_{j=1}^{t} r_j$: GSPO's $s_i$ is a single scalar summarizing the *entire* sequence (used to replace per-token clipping), while $\rho_{1:t}$ is a *position-dependent* cumulative product (used for state distribution correction at each step $t$). They serve orthogonal purposes — GSPO addresses the action-level trust region, while the prefix ratio addresses the state-level distribution mismatch.
- **V-trace** (Espeholt et al., 2018) uses truncated cumulative IS products $\prod_{j=s}^{t-1} \min(\bar{c}, r_j)$ — but for **value estimation**, not policy gradient correction, and under stochastic transitions where the product is only an approximation.
- **Per-decision IS** (Precup et al., 2000) established the theoretical foundation for step-wise IS correction in RL, but was never applied to the LLM setting where deterministic transitions make it exact.

### Why This Direction Has More to Give

MinPRO has validated the core thesis — prefix-level correction matters — but the design space is far from exhausted. The central challenge remains: **how to approximate $\rho_{1:t} = \prod_{j=1}^{t} \rho_j$ with low variance while preserving meaningful prefix signal?**

#### Variance Reduction Strategies for the Prefix Product

Beyond MinPRO's $\min$ surrogate, several promising approaches exist:

**1. Truncated IS (V-trace style)** — Clip each factor *before* multiplying:

$$\bar{\rho}_{1:t} = \prod_{j=1}^{t} \min(\bar{c},\, \rho_j)$$

With $\bar{c} \leq 1$, the product is monotonically non-increasing, bounding variance growth. V-trace uses this for value estimation in general MDPs; in LLMs where the prefix product is the *exact* state ratio, truncated IS has a cleaner theoretical interpretation. The truncation threshold $\bar{c}$ provides a direct bias-variance knob.

**2. Log-space operations** — The prefix product becomes a sum in log-space:

$$\log \rho_{1:t} = \sum_{j=1}^{t} \log \rho_j$$

Sums grow with *linear* variance (not exponential). Several log-space strategies are possible:
- **Clipped log-sum**: $\exp\left(\text{clip}\left(\sum_{j=1}^{t} \log \rho_j,\, -c,\, c\right)\right)$ — directly bounds the prefix product's range
- **Exponential moving average (EMA)**: $\bar{l}_t = \alpha \bar{l}_{t-1} + (1-\alpha) \log \rho_t$, then use $e^{\bar{l}_t}$ as the surrogate. Smoother than $\min$, captures gradual drift rather than only worst-case deviation

The $\exp(\cdot)$ back-transformation introduces bias, but in practice this may be acceptable.

**3. Self-normalized IS** — Normalize weights across the group:

$$\tilde{w}_t^{(i)} = \frac{\rho_{1:t}^{(i)}}{\sum_{i=1}^{G} \rho_{1:t}^{(i)}}$$

Weights sum to 1 by construction, preventing explosion. This naturally fits GRPO's group structure ($G$ rollouts per prompt). The bias is $O(1/G)$, which may be acceptable for typical group sizes ($G = 8 \sim 16$).

**4. Truncated window** — Only look back $k$ steps:

$$w_t^{(k)} = \prod_{j=\max(1,\, t-k)}^{t} \rho_j$$

A direct analogue of n-step returns. The hyperparameter $k$ controls the bias-variance tradeoff: $k=0$ recovers GRPO, $k=T$ recovers the full prefix product.

**5. Segmented products** — Reset the product at natural boundaries (e.g., reasoning steps, sentence breaks):

$$w_t = \prod_{j=\text{seg\_start}(t)}^{t} \rho_j$$

LLM reasoning is often structured ("Step 1: ... Step 2: ..."). Cross-step prefix products may be less meaningful than within-step products. Requires a segmentation strategy (newlines, special tokens, or attention patterns).

**6. Control variates** — Reduce variance by subtracting a correlated baseline:

$$\rho_{1:t}^{\text{cv}} = \rho_{1:t} - \alpha(b_t - \mathbb{E}[b_t])$$

A natural choice is $b_t = \rho_t$ (token-level ratio, with $\mathbb{E}_{\pi_\beta}[\rho_t] = 1$), using the token-level signal to "denoise" the prefix product.

#### The Full Spectrum

```
Low Variance, High Bias                              Low Bias, High Variance
◄─────────────────────────────────────────────────────────────────────────►

  ρ_t        min ρ_i      Log EMA     Truncated    V-trace     Self-norm    Full ∏ρ_j
 (GRPO)     (MinPRO)                  window k     clipped      IS          (exact)
  │            │            │            │            │            │            │
  ▼            ▼            ▼            ▼            ▼            ▼            ▼
 忽略前缀    最坏情况     平滑追踪     近期窗口    截断因子     组内归一化   完整乘积
```

All strategies share a key property: **zero computational overhead** — every $r_j(\theta)$ is already computed during training; any prefix-based correction is essentially a cumulative operation over existing values.

#### Other Open Directions

1. **Combining with distribution-level trust regions** — MinPRO operates within the token-ratio paradigm (with soft clipping). Combining prefix correction with DPPO's direct divergence estimation could yield both prefix-aware and distribution-level guarantees.
2. **Adaptive prefix windows** — Instead of using all preceding tokens ($\min_{i<t}$), adaptively selecting the relevant prefix window based on local off-policyness could reduce unnecessary conservatism.
3. **Theoretical analysis** — What is the optimal bias-variance tradeoff for prefix correction in the LLM setting? Can we derive finite-sample bounds that account for the deterministic transition structure?

## 8.2 Beyond Prefix Correction: Open Directions

Several other promising directions emerge from the analysis across all methods:

1. **Unifying trust region and advantage estimation**: Current methods treat these as separate problems. SAPO's insight — that token-adaptive gating can be both sequence-coherent and locally responsive — hints at a unified framework where the trust region mechanism itself is advantage-aware.

2. **Adaptive trust region geometry**: DPPO shows that direct divergence estimation outperforms ratio-based proxies. But the choice of divergence ($D_{TV}$ vs $D_{KL}$ vs others) is fixed. An adaptive scheme that selects the divergence measure based on the local geometry of the policy manifold could yield tighter bounds.

3. **Structured exploration via feedback**: NLF demonstrates that natural language critiques can bootstrap exploration far more effectively than scalar rewards. Combining NLF's feedback-driven exploration with DPPO's distribution-level trust region could yield a system that explores intelligently while remaining provably stable.

4. **Token-level advantage estimation**: All current methods assign the same advantage to every token in a response ($A_{i,t} = A_i$). But not all tokens contribute equally to correctness. Developing token-level credit assignment — perhaps through attention-based attribution or causal intervention — could dramatically improve sample efficiency by focusing gradient signal on the tokens that actually matter.

5. **Off-policy as a first-class citizen**: MinPRO and M2PO demonstrate that off-policy training is not just a necessary evil but a regime that demands dedicated algorithmic design. As asynchronous RL frameworks (AREAL, Laminar) push off-policyness even further, methods that are explicitly designed for large policy lag — rather than adapted from on-policy algorithms — will become increasingly important.
