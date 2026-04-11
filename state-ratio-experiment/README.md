# State Ratio Experiment - Qwen2.5-1.5B 版本

## 项目概述

本项目基于 Qwen2.5-1.5B-Instruct 模型，对比三种强化学习算法在数学推理任务上的表现：

1. **GRPO Baseline** - 标准的组相对策略优化（Group-Relative Policy Optimization）
2. **GSPO Baseline** - 序列级几何平均比率裁剪（Geometric Mean Ratio Clip）
3. **SC-GRPO (State-Corrected GRPO)** - 我们提出的状态校正方法，通过截断前缀重要性采样比率校正状态分布不匹配

### 实验设计

- **模型**: Qwen2.5-1.5B-Instruct (针对小模型优化)
- **硬件**: 8× H100 80GB (单节点)
- **数据**: GSM8K + MATH 数学推理数据集
- **算法**: GRPO (组相对归一化)
- **训练轮数**: 15 轮

### 1.5B 模型优化特点

相比 7B 版本，1.5B 模型配置进行了以下优化：

- **更大的批次大小**: `train_batch_size=1024` (原 512)
- **更长的序列长度**: `max_prompt_length=2048`, `max_response_length=4096`
- **更高的学习率**: `lr=2e-6` (原 1e-6)
- **更高的GPU利用率**: `gpu_memory_utilization=0.8` (原 0.6)
- **更大的微批次**: `ppo_micro_batch_size_per_gpu=8` (原 4)

## 环境要求

### 硬件要求
- GPU: 8× H100 80GB 或同等性能的GPU
- 内存: 建议至少 256GB 系统内存
- 存储: 至少 500GB 可用空间用于模型和数据缓存

### 软件要求
- Docker Engine 20.10+
- NVIDIA Container Toolkit
- 支持 CUDA 的 GPU 驱动

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd state-ratio-experiment
```

### 2. 设置环境变量（可选）

```bash
# 自定义模型、数据和输出目录（可选）
export MODEL_DIR=/path/to/models
export DATA_DIR=/path/to/data
export OUTPUT_DIR=/path/to/output
```

### 3. 启动 Docker 容器

```bash
# 进入交互式 shell
bash docker-run.sh

# 或者直接运行所有实验
bash docker-run.sh bash run_all.sh
```

### 4. 在容器内运行实验

```bash
# 首次运行：下载模型和数据
bash setup.sh

# 运行所有实验（GRPO + GSPO + SC-GRPO）
bash run_all.sh

# 或者分别运行单个实验
bash run_all.sh grpo    # 仅运行 GRPO
bash run_all.sh gspo    # 仅运行 GSPO  
bash run_all.sh sc      # 仅运行 State-Corrected GRPO
bash run_all.sh ablation # 运行 k 值消融实验
```

## 实验详情

### 实验 1: GRPO Baseline
- **文件**: `scripts/run_grpo_baseline.sh`
- **描述**: 标准的 GRPO 算法，使用逐令牌裁剪
- **关键参数**: `clip_ratio=0.2`

### 实验 2: GSPO Baseline  
- **文件**: `scripts/run_gspo_baseline.sh`
- **描述**: 序列级几何平均比率裁剪
- **关键参数**: `clip_ratio_low=0.0003`, `clip_ratio_high=0.0004`

### 实验 3: State-Corrected GRPO
- **文件**: `scripts/run_state_corrected.sh`
- **描述**: GRPO + 截断前缀状态校正
- **关键参数**: 
  - `state_correction_lookback_k=5` (默认)
  - `state_correction_max_weight=5.0`
  - `state_correction_min_weight=0.2`

### 消融实验
- **文件**: `scripts/run_ablation_k.sh`
- **描述**: 测试不同 k 值对性能的影响
- **k 值**: 0, 3, 5, 10, -1

## 自定义配置

### 修改模型路径
```bash
# 在运行前设置环境变量
export MODEL_PATH=/path/to/your/model
```

### 调整超参数
每个实验脚本都包含详细的超参数配置，可以根据需要修改：
- 批次大小和序列长度
- 学习率和训练轮数
- 裁剪比率和状态校正参数

### 使用不同的数据集
修改脚本中的 `train_files` 和 `val_files` 路径指向您的数据集。

## 结果监控

实验使用 `verl` 框架的日志系统，可以通过以下方式监控进度：

1. **控制台输出**: 实时显示训练指标
2. **检查点**: 每 20 步保存一次模型检查点
3. **验证**: 每 5 步进行一次验证评估

## 故障排除

### 常见问题

1. **GPU 内存不足**
   - 减小 `train_batch_size` 或 `ppo_micro_batch_size_per_gpu`
   - 降低 `gpu_memory_utilization`

2. **模型下载失败**
   - 检查网络连接
   - 设置 `HF_HUB_OFFLINE=1` 使用离线模式

3. **Docker 权限问题**
   - 确保用户有权限访问 GPU
   - 检查 NVIDIA Container Toolkit 安装

### 性能优化建议

- 使用 `--shm-size=256g` 确保足够的共享内存
- 启用 `flash_attention_2` 加速注意力计算
- 使用梯度检查点节省内存

## 文件结构

```
state-ratio-experiment/
├── Dockerfile                 # Docker 镜像配置
├── docker-run.sh              # Docker 启动脚本
├── run_all.sh                 # 主运行脚本
├── setup.sh                   # 环境设置脚本
├── scripts/                   # 实验脚本目录
│   ├── run_grpo_baseline.sh   # GRPO 实验
│   ├── run_gspo_baseline.sh   # GSPO 实验
│   ├── run_state_corrected.sh # SC-GRPO 实验
│   ├── run_ablation_k.sh      # 消融实验
│   └── prepare_data.sh        # 数据预处理
├── state_corrected_loss.py    # 自定义损失函数
└── tests/                     # 测试文件
```

## 引用

如果您使用本项目的代码或方法，请引用相关论文：

```bibtex
@article{state_ratio_2024,
  title={State Ratio Correction for Policy Optimization},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## 许可证

本项目基于 MIT 许可证开源。