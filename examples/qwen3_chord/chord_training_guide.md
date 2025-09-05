# Trinity-RFT CHORD混合训练完整指南

## 目录
- [CHORD算法介绍](#chord算法介绍)
- [配置文件说明](#配置文件说明) 
- [数据准备和预处理](#数据准备和预处理)
- [训练配置流程](#训练配置流程)
- [运行步骤](#运行步骤)
- [参数调优指南](#参数调优指南)
- [故障排除](#故障排除)

## CHORD算法介绍

### 什么是CHORD？
CHORD（**Ch**oosing **o**ptimal **r**atio for **d**ynamic weighting）是一种**动态混合SFT和强化学习**的训练算法，其核心思想是：

1. **双数据源训练**：同时使用专家数据（高质量SFT数据）和强化学习数据
2. **动态权重调节**：使用μ参数动态控制SFT损失和RL损失的权重比例
3. **余弦调度策略**：μ参数遵循warmup→peak→decay的调度曲线

### CHORD训练流程

```
初始阶段 (Warmup)     高峰阶段 (Peak)      衰减阶段 (Decay)
μ: 0 → 0.6           μ: 0.6              μ: 0.6 → 0.05
主要是RL训练           SFT和RL平衡          主要是RL训练
```

### CHORD的优势
- **稳定性**：SFT损失提供稳定的学习信号
- **性能**：RL损失优化特定任务表现  
- **鲁棒性**：动态权重避免灾难性遗忘
- **效率**：比纯SFT+纯RL的两阶段训练更高效

## 配置文件说明

### 生成的配置文件概览

| 文件名 | 用途 | 适用场景 |
|--------|------|----------|
| `qwen3_chord_config.yaml` | 完整配置 | 生产环境，最佳性能 |
| `qwen3_chord_train_config.yaml` | 详细训练参数 | 配合主配置使用 |
| `qwen3_chord_simple.yaml` | 简化配置 | 测试环境，快速验证 |
| `qwen3_chord_simple_train.yaml` | 简化训练参数 | 配合简化配置使用 |

### 关键参数解析

#### 1. **μ调度参数**
```yaml
policy_loss_fn_args:
  mu_warmup_steps: 100    # μ从0增长到peak的步数
  mu_decay_steps: 300     # μ从peak衰减到valley的步数  
  mu_peak: 0.6           # μ的最高值（SFT权重最大）
  mu_valley: 0.05        # μ的最低值（后期以RL为主）
```

#### 2. **数据比例参数**
```yaml
sample_strategy_args:
  expert_data_ratio: 0.25  # 专家数据占总训练数据的25%
```

#### 3. **CHORD变体选择**
```yaml
policy_loss_fn_args:
  enable_phi_function: true  # true=CHORD-φ, false=CHORD-μ
```

## 数据准备和预处理

### 步骤1：数据格式检查
确保你的`sft_data`目录包含以下格式之一的数据：

#### 格式A：简单问答格式 (推荐)
```json
{"prompt": "什么是机器学习？", "response": "机器学习是人工智能的一个分支..."}
{"prompt": "如何学习Python？", "response": "学习Python可以从以下几个方面开始..."}
```

#### 格式B：对话格式
```json
{"messages": [{"role": "user", "content": "什么是机器学习？"}, {"role": "assistant", "content": "机器学习是..."}]}
```

#### 格式C：其他字段名
```json
{"question": "什么是机器学习？", "answer": "机器学习是..."}
{"input": "如何学习Python？", "output": "学习Python可以..."}
```

### 步骤2：运行数据预处理脚本
```bash
# 基础预处理（使用默认参数）
python prepare_chord_data.py

# 自定义参数预处理
python prepare_chord_data.py \
    --input_dir sft_data \
    --output_dir chord_processed_data \
    --expert_ratio 0.3 \
    --train_ratio 0.85 \
    --seed 42
```

**参数说明：**
- `--input_dir`: 原始数据目录
- `--output_dir`: 处理后数据输出目录
- `--expert_ratio`: 专家数据比例（用于SFT训练）
- `--train_ratio`: 训练数据比例（剩余为测试数据）
- `--seed`: 随机种子（保证可重复性）

### 步骤3：验证预处理结果
预处理完成后会生成：
```
chord_processed_data/
├── expert_data.jsonl     # SFT专家数据
├── rl_data.jsonl        # 强化学习数据  
├── test_data.jsonl      # 测试评估数据
└── all_train_data.jsonl # 所有训练数据
```

## 训练配置流程

### 阶段1：选择适合的配置文件

#### 新手用户 → 使用简化配置
```bash
# 适合：初次使用、硬件资源有限、快速验证
cp qwen3_chord_simple.yaml my_chord_config.yaml
cp qwen3_chord_simple_train.yaml my_chord_train_config.yaml
```

#### 高级用户 → 使用完整配置
```bash  
# 适合：生产环境、追求最佳性能、有充足硬件
cp qwen3_chord_config.yaml my_chord_config.yaml
cp qwen3_chord_train_config.yaml my_chord_train_config.yaml
```

### 阶段2：根据硬件调整配置

#### GPU数量调整
```yaml
cluster:
  gpu_per_node: 4  # 改为你的实际GPU数量

# 相应调整其他参数
explorer:
  rollout_model:
    tensor_parallel_size: 2  # 通常设为 gpu_per_node 的一半
    
policy_loss_fn_args:
  ngpus_trainer: 4  # 设为训练使用的GPU数量
```

#### 显存优化（如果显存不足）
```yaml
# 主配置文件
buffer:
  batch_size: 4        # 减小批次大小
  train_batch_size: 32 # 减小训练批次

# 训练配置文件  
actor:
  ppo_micro_batch_size_per_gpu: 2  # 减小微批次
  ppo_max_token_len_per_gpu: 4000  # 减小token长度
  fsdp_config:
    param_offload: True      # 启用参数offload
    optimizer_offload: True  # 启用优化器offload
```

### 阶段3：数据路径配置

#### 方法1：使用预处理后的数据（推荐）
```yaml
buffer:
  explorer_input:
    taskset:
      path: "chord_processed_data/rl_data.jsonl"  # RL数据
  trainer_input:
    sft_warmup_dataset:
      path: "chord_processed_data/expert_data.jsonl"  # 专家数据
```

#### 方法2：使用原始数据
```yaml
buffer:
  explorer_input:
    taskset:
      path: "sft_data"  # 原始数据目录
  trainer_input:
    sft_warmup_dataset:
      path: "sft_data"  # 同一数据源，算法自动分配
```

## 运行步骤

### 步骤1：环境准备
```bash
# 1. 启动Ray集群
ray stop  # 停止已有的Ray进程
ray start --head

# 2. 设置监控（可选）
export WANDB_API_KEY=your_wandb_api_key
wandb login

# 3. 验证环境
python -c "import trinity; print('Trinity-RFT安装正确')"
```

### 步骤2：数据预处理（如果还未完成）
```bash
python prepare_chord_data.py --input_dir sft_data --output_dir chord_data
```

### 步骤3：配置验证
```bash
# 检查配置文件语法
trinity validate --config my_chord_config.yaml
```

### 步骤4：启动训练

#### 简化版训练
```bash
trinity run --config qwen3_chord_simple.yaml
```

#### 完整版训练
```bash
trinity run --config qwen3_chord_config.yaml
```

### 步骤5：监控训练进展
```bash
# 查看实时日志
tail -f logs/trinity.log

# 或使用WandB网页界面
# 访问 https://wandb.ai/your-username/your-project
```

## 参数调优指南

### 核心参数调优优先级

#### 1. **μ调度参数** (最重要)
```yaml
# 保守策略（适合大多数场景）
mu_warmup_steps: 100   
mu_decay_steps: 300    
mu_peak: 0.4          # 较低的peak，RL为主
mu_valley: 0.05       

# 激进策略（SFT数据质量很高时）
mu_warmup_steps: 50    
mu_decay_steps: 200    
mu_peak: 0.8          # 较高的peak，SFT为主
mu_valley: 0.1        
```

#### 2. **数据比例调整**
```yaml
# 专家数据质量很高 → 增加比例
expert_data_ratio: 0.4

# 专家数据质量一般 → 减少比例  
expert_data_ratio: 0.15
```

#### 3. **学习率调整**
```yaml
# 4B模型推荐学习率范围
optim:
  lr: 1e-6  # 保守，稳定
  lr: 3e-6  # 平衡
  lr: 5e-6  # 激进，收敛快但可能不稳定
```

### 性能调优指南

#### 提升训练速度
```yaml
# 增加并行度
explorer:
  runner_num: 16        # 增加并发任务
  rollout_model:
    engine_num: 4       # 增加推理引擎

# 优化批次大小
buffer:
  batch_size: 32        # 在显存允许下尽量大
```

#### 提升训练稳定性
```yaml
# 使用较小的学习率
optim:
  lr: 1e-6
  lr_warmup_steps_ratio: 0.2  # 增加热身比例

# 使用梯度裁剪
actor:
  grad_clip: 0.5  # 较严格的梯度裁剪
```

#### 节省显存
```yaml
# 启用offload
fsdp_config:
  param_offload: True
  optimizer_offload: True

# 减小token长度
ppo_max_token_len_per_gpu: 4000

# 启用梯度检查点  
enable_gradient_checkpointing: True
```

## 故障排除

### 常见问题及解决方案

#### 1. **显存不足 (CUDA Out of Memory)**
```bash
# 解决方案A：减小批次大小
batch_size: 4
ppo_micro_batch_size_per_gpu: 1

# 解决方案B：启用offload
param_offload: True
optimizer_offload: True

# 解决方案C：减小模型并行度
tensor_parallel_size: 1
```

#### 2. **训练不收敛**
```bash
# 解决方案A：调整μ参数
mu_peak: 0.2  # 减小SFT权重
mu_warmup_steps: 200  # 增加热身步数

# 解决方案B：调整学习率
lr: 1e-6  # 使用更小的学习率
lr_warmup_steps_ratio: 0.3  # 增加热身比例

# 解决方案C：检查数据质量
python prepare_chord_data.py --expert_ratio 0.5  # 增加专家数据比例
```

#### 3. **数据加载错误**
```bash
# 检查数据格式
head -n 5 sft_data/*.jsonl

# 重新预处理数据
python prepare_chord_data.py --input_dir sft_data --output_dir new_chord_data

# 验证数据格式
python -c "
import json
with open('chord_data/expert_data.jsonl', 'r') as f:
    sample = json.loads(f.readline())
    print('数据格式:', list(sample.keys()))
"
```

#### 4. **Ray集群连接问题**
```bash
# 重启Ray集群
ray stop
sleep 5
ray start --head

# 检查Ray状态
ray status

# 如果仍有问题，清理Ray缓存
ray stop
rm -rf /tmp/ray
ray start --head
```

#### 5. **模型保存/加载问题**
```bash
# 检查检查点目录权限
ls -la checkpoints/

# 手动创建检查点目录
mkdir -p checkpoints/qwen3_chord/

# 验证模型路径
ls -la cache/models/modelscope/hub/models/qwen/Qwen3-4B/
```

### 性能监控和调试

#### 1. **监控关键指标**
- `mu`: μ参数值，观察调度是否正确
- `loss`: 总损失，应该逐渐下降
- `expert/sft_loss`: SFT损失
- `usual/pg_loss`: 策略损失  
- `ppo_kl`: KL散度，过高说明训练过快

#### 2. **调试工具**
```bash
# 启用详细日志
export TRINITY_LOG_LEVEL=DEBUG

# 使用TensorBoard（替代WandB）
monitor:
  monitor_type: tensorboard

# 检查数据流
python -c "
from trinity.buffer import get_buffer_reader
# ... 调试代码
"
```

## 最佳实践建议

### 1. **分阶段训练策略**
```bash
# 阶段1：使用简化配置验证流程
trinity run --config qwen3_chord_simple.yaml

# 阶段2：调优参数并使用完整配置
trinity run --config qwen3_chord_config.yaml
```

### 2. **数据质量优化**
- **专家数据**：选择高质量、多样化的SFT数据
- **RL数据**：确保数据涵盖目标任务分布
- **数据清洗**：移除格式错误、内容重复的数据

### 3. **超参数调优流程**
1. 首先固定μ调度参数，调优批次大小和学习率
2. 然后固定训练超参数，调优μ调度曲线
3. 最后微调专家数据比例和其他高级参数

### 4. **实验管理**
```bash
# 使用有意义的实验名称
name: "qwen3-4b-chord-v1.0-peak0.6-lr2e6"

# 记录重要配置变更
# 保存不同版本的配置文件
```

通过以上完整指南，你应该能够成功使用CHORD算法训练Qwen3-4B模型。如有其他问题，请参考Trinity-RFT官方文档或提交Issue。
