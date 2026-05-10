# State-Corrected GRPO：从原始 PPO/GRPO 到前缀状态比修正

> 本文解释 `state_corrected_loss.py` 中实现的 `state_corrected_grpo` 损失是怎么来的、跟 verl 里原始 PPO / GRPO 损失有什么本质区别，以及脚本 `scripts/training/run_state_corrected.sh` 为什么要那样配置。
>
> 相关文件：
> - 损失实现：[`state_corrected_loss.py`](../state_corrected_loss.py)
> - 训练脚本：[`scripts/training/run_state_corrected.sh`](../scripts/training/run_state_corrected.sh)

---

## 一、先把 loss 的 "血脉" 讲清楚

所有 policy gradient 方法本质都是同一个式子：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim \pi_\theta}\Big[\sum_t A_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)\Big]
$$

意思是：**在当前策略 π_θ 下采样**得到轨迹 τ，用优势 $A_t$ 加权 score function。这是"**同策略 (on-policy)**"的 REINFORCE。

但真实训练中：

- 不是每采一条数据就更新一次，而是采完一 batch（rollout），**用同一批数据反复更新 K 次**（PPO 的 `ppo_epochs`）。
- 第二次更新开始，采样策略 **π_β**（也叫 π_old）和当前策略 **π_θ** 已经不一样了。
- 于是变成"**离策略 (off-policy)**"：数据是 π_β 采的，梯度却要估 π_θ 的。

**离策略的严格形式需要重要性采样 (IS) 修正**：

$$
\nabla_\theta J \;=\; \mathbb{E}_{\tau\sim \pi_\beta}\Big[\sum_t \underbrace{\tfrac{d^{\pi_\theta}(s_t)}{d^{\pi_\beta}(s_t)}}_{\text{状态比 }w_t} \cdot \underbrace{\tfrac{\pi_\theta(a_t|s_t)}{\pi_\beta(a_t|s_t)}}_{\text{动作比 }r_t} \cdot A_t \cdot \nabla_\theta \log\pi_\theta(a_t|s_t)\Big]
$$

这里有 **两个比值**：

1. **状态访问比** $w_t$：在 π_θ 下到达状态 $s_t$ 的概率 / 在 π_β 下到达 $s_t$ 的概率。
2. **动作比** $r_t$：给定 $s_t$，π_θ 选 $a_t$ 的概率 / π_β 选 $a_t$ 的概率。

---

## 二、原始 PPO / GRPO 的 loss —— **只修正动作比，扔掉状态比**

verl 里（以及几乎所有 LLM 上的 PPO / GRPO 实现）用的是这种简化形式：

$$
L_{\text{PPO}}(\theta) = -\mathbb{E}\Big[\sum_t \min\big(r_t \cdot \hat A_t,\; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot \hat A_t\big)\Big]
$$

伪代码：

```python
ratio = exp(log_prob - old_log_prob)           # r_t = π_θ / π_β
surr1 = ratio * advantages
surr2 = clip(ratio, 1-ε, 1+ε) * advantages
loss  = -min(surr1, surr2)                     # 逐 token 取 min，再平均
```

**特点：**

| 维度 | 做法 | 隐含假设 |
|---|---|---|
| 状态比 $w_t$ | **默认当 1**（直接忽略） | π_θ 和 π_β 差不多，$w_t \approx 1$ |
| 动作比 $r_t$ | 保留，并用 clip 把它压在 $[1-\varepsilon, 1+\varepsilon]$ 内 | 防止大比值引爆梯度 |
| clip 的作用 | 当 $r_t$ 超出 band 且方向会让目标变大时，**截断**使梯度变 0 | 信赖域 |

GRPO 相对 PPO 只是把 $A_t$ 换成 **同一个 prompt 下 G 条 rollout 内部归一化的相对回报**（没有 critic），**loss 结构完全一样**。所以 PPO 和 GRPO 都继承同一个毛病：

> **把状态比 $w_t = d^{\pi_\theta}(s_t)/d^{\pi_\beta}(s_t)$ 省略了。**

在传统 RL（迷宫、机器人等）状态比没法显式算，只能忽略，这是妥协。但 **LLM 有个独特性质**：

> 给定前缀 token $o_{<t}$，状态 $s_t$ 就完全确定了（状态 = context）。**转移是确定性的**。

所以状态比可以 **严格展开** 成前缀动作比的连乘：

$$
w_t \;=\; \frac{d^{\pi_\theta}(s_t)}{d^{\pi_\beta}(s_t)} \;=\; \prod_{j=0}^{t-1} \frac{\pi_\theta(o_j|s_j)}{\pi_\beta(o_j|s_j)} \;=\; \prod_{j=0}^{t-1} r_j
$$

—— 这就是 state-corrected loss 要加回来的东西。

---

## 三、新 loss：state-corrected GRPO

### 3.1 理论形式

$$
\boxed{\;L = -\sum_t \underbrace{\Big(\prod_{j<t} r_j\Big)}_{w_t\text{ 状态比，detach}} \cdot \underbrace{r_t}_{\text{动作比，detach}} \cdot \log\pi_\theta(a_t|s_t) \cdot A_t \cdot m_t\;}
$$

对比原始 PPO / GRPO 的最大不同：**每个 token 的 loss 前面多乘了一个前缀连乘** $w_t = \prod_{j<t} r_j$。

这个式子在理论上是 **真正无偏** 的离策略梯度估计。

### 3.2 实现关键片段（`state_corrected_loss.py` 第 470–580 行）

```python
log_ratio = log_prob - old_log_prob            # log r_t
ratio     = exp(log_ratio)                     # r_t

# w_t = 前缀连乘，但连乘会方差爆炸 → 用策略降方差：
log_state_weight = strategy_fn(log_ratio, ...) # log w_t
state_weight     = exp(log_state_weight).detach()

# 两道闸门（combined_mask = 两个都通过才为 1）：
not_clamped = (log_min ≤ log w_t ≤ log_max)    # SC 闸：状态比偏得太离谱 → 关
not_clipped = (1-ε ≤ r_t ≤ 1+ε)                # PPO 闸：动作比偏得太离谱 → 关
combined_mask = not_clamped * not_clipped

pg_losses = - state_weight * ratio.detach() * log_prob * advantages * combined_mask
```

### 3.3 和原始 loss 的逐项对比

| 维度 | **原始 PPO / GRPO** | **State-Corrected GRPO** |
|---|---|---|
| 状态比 $w_t$ | **丢弃**（隐式当 1） | 显式重构 $\prod_{j<t} r_j$，并做降方差 |
| 动作比 $r_t$ | 留下，乘进 loss | 留下，乘进 loss（但 `detach`） |
| 形式 | 比值形式：`ratio * A` | REINFORCE 形式：`w · r · log π · A` |
| 梯度从哪里流 | 通过 `ratio = exp(log_prob − old_log_prob)` 流回 log_prob | 通过 `log_prob` 直接流（w, r 都 detach） |
| 动作比越界时 | `min(surr1, surr2)` 起效 → 对应方向梯度变 0（**clip-and-continue**） | `combined_mask = 0` → **整个 token 的梯度丢掉** |
| SC 闸（状态比越界） | 不存在 | 独立一道：$\log w_t \notin [\log w_{\min}, \log w_{\max}]$ → 丢 |
| 对 `old_log_prob` 的要求 | 不敏感，π_β 用 actor 重算也行 | **严格要求 π_β = 采样时 rollout policy**，否则 $w_t$ 无意义 |

### 3.4 为什么需要 `bypass_mode`

在 verl 的默认流程里：

1. rollout → 得到 tokens
2. **更新之前**，用 actor 再前向一次重算 `old_log_prob` ← `_compute_old_log_prob()`
3. 多轮 `ppo_epochs`，每轮 actor 参数会变，但 `old_log_prob` 固定不变

**问题**：第 2 步的 `old_log_prob` 并不严格等于 **采样时 rollout engine (vLLM)** 的 log_prob：

- rollout 用 vLLM（可能 fp16、bf16、不同 kernel）
- 重算用 actor（FSDP、flash-attention、bf16）
- 两者数值上有漂移，甚至 tokenizer / pad 对齐都可能差一点

对普通 PPO 没事（clip 一兜底就吃掉）；但对 **state correction 是致命的**：$w_t$ 是前缀连乘，小误差 × T 个 token 就爆。

所以脚本里同时打开：

```bash
actor_rollout_ref.rollout.calculate_log_probs=True   # vLLM 边采样边返回 log_prob
algorithm.rollout_correction.bypass_mode=True        # 用 rollout log_prob 直接当 old_log_prob
```

`state_corrected_loss.py` 还 monkey-patch 了 `apply_bypass_mode`，避免 bypass 把 `loss_mode` 覆盖成内置的 `bypass_mode`。

这样就保证了：**`old_log_prob` 逐 token 精确等于采样时 vLLM 的 π_β** → 状态比的连乘在数学上是对的 → 还顺便省了一次 actor forward。

### 3.5 为什么要 8 种策略（方差问题）

理论上 $w_t = \prod_{j<t} r_j$ 最干净，但 **方差随 $t$ 指数爆炸**：假设每个 $r_j$ 标准差 $\sigma$，$T$ 个 token 后乘积的相对标准差约 $e^{T\sigma}$。对 $T = 2048$ 的 response，直接用等于自杀。

于是 8 个策略其实是 **"如何在不破坏太多无偏性的前提下压方差"** 的 8 种思路：

| 策略 | 做了什么 | 偏差 / 方差 trade-off |
|---|---|---|
| `identity` | $w_t = 1$ | 完全退化成 REINFORCE；**和原始 loss 的状态部分一致**，作对照 |
| `none` | 真·连乘 | **无偏**，方差爆炸 |
| `truncated_window`（默认 k=5） | 只连乘最近 $k$ 个 | 有偏，方差可控 |
| `vtrace` | 每项先 clip 到 $\bar c$ 再乘 | 有偏，方差硬上界 |
| `log_ema` | log 空间做 EMA | 平滑，丢时间信息 |
| `min_prefix` | 取前缀 min | 悲观下界，MinPRO 风格 |
| `self_normalized` | batch 里归一化 | 消 drift，偏差渐近 0 |
| `baseline_corrected` | 组内几何均值做控制变量 | **无偏**，专治 timestep drift $E[\log w_t] \approx -t \cdot \text{KL}$ |

### 3.6 "两道闸门" 为什么这样设计

原始 PPO 的 clip 逻辑很绕：`min(r·A, clip(r)·A)`，意思是只在 "策略更新方向会让目标更大" 时才截。state-corrected loss 采用更直白的 **Plan A 统一门控**：

- 状态比越界 → 这个 token 完全不可信 → 梯度关掉
- 动作比越界 → 这个 token 完全不可信 → 梯度关掉
- **两道闸任一不过就丢**，survived token 才贡献梯度

metrics 里的 `joint_active_frac` / `excluded_sc_only_frac` / `excluded_ppo_only_frac` 就是用来监控两道闸各自切掉了多少 token。

---

## 四、一张图总结

```
                原始 PPO / GRPO                       State-Corrected GRPO
           ┌──────────────────────┐             ┌────────────────────────────────┐
           │   r_t · A_t          │             │   w_t · r_t · log π_θ · A_t    │
 token t   │   (clip r_t 起兜底)   │             │   (两道闸 combined_mask 起兜底) │
           └──────────────────────┘             └────────────────────────────────┘
状态比 w_t:   丢弃 ≡ 1                           ∏_{j<t} r_j，多策略降方差，detach
动作比 r_t:   clip-and-continue                 detach；越界 → mask=0 丢掉整 token
old_lp 来源:  actor 重算（可容忍漂移）            rollout vLLM 原样（bypass_mode，强要求）
梯度流:       通过 ratio                        通过 log π_θ  (REINFORCE 风格)
```

**一句话总结**：原始 PPO / GRPO 在 LLM 上只修正了动作比、忽略状态比；这个新 loss 利用 LLM 转移确定性这一特性，把被丢弃的状态比 $\prod_{j<t} r_j$ 显式加回来，并提供多种降方差策略和"双闸门 mask"来让它在数值上能训起来。

---

## 五、快速查询表（跑实验用）

**切换策略 / 超参（环境变量）：**

| 变量 | 默认 | 作用 |
|---|---|---|
| `SC_STRATEGY` | `truncated_window` | 选择降方差策略 |
| `SC_LOOKBACK_K` | `5` | `truncated_window` 的窗口长度；`-1` = 全前缀；`0` = 无修正 |
| `SC_VTRACE_C` | `1.0` | `vtrace` 的截断阈值 $\bar c$ |
| `SC_EMA_ALPHA` | `0.9` | `log_ema` 的平滑系数 |
| `SC_GROUP_SIZE` | `n_resp_per_prompt` | `baseline_corrected` 组大小，必须等于 `rollout.n` |
| `SC_MAX_STATE_WEIGHT` | `2.0` | $w_t$ 上限（脚本中设置） |
| `SC_MIN_STATE_WEIGHT` | `0.5` | $w_t$ 下限（脚本中设置） |
| `SC_PPO_CLIP_EPSILON` | `0.2` | 对 $r_t$ 的 PPO band $\varepsilon$ |

**典型调用：**

```bash
# 默认 truncated_window, k=5
bash scripts/training/run_state_corrected.sh

# V-trace
SC_STRATEGY=vtrace VTRACE_C=1.0 bash scripts/training/run_state_corrected.sh

# 纯 REINFORCE 消融底线
SC_STRATEGY=identity bash scripts/training/run_state_corrected.sh

# 无偏但高方差基线
SC_STRATEGY=none bash scripts/training/run_state_corrected.sh

# 扫 k（通常由 run_ablation_k.sh 批量做）
for k in 0 1 3 5 10 -1; do
  SC_STRATEGY=truncated_window LOOKBACK_K=$k \
  bash scripts/training/run_state_corrected.sh
done
```

**重点监控的 W&B metric：**

- `actor/state_weight_{mean,max,std}`：$w_t$ 的数值情况，看方差是否可控
- `actor/state_weight_clamp_{lower,upper}_frac`：被 SC 闸关掉的 token 比例
- `actor/ppo_clip_{lower,upper}_frac`：被 PPO 闸关掉的 token 比例
- `actor/joint_active_frac`：**真正贡献梯度的 token 比例**（两道闸都通过的）
- `actor/excluded_{sc_only,ppo_only,both}_frac`：两道闸各自切掉多少（互斥分解）
- `actor/log_state_weight_{mean,std}`：log 空间诊断，`baseline_corrected` 下应接近 0 且方差小
