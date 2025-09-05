# Trinity-RFT CHORD混合训练配置包

基于Trinity-RFT框架和examples/mix_chord分析，为Qwen3-4B模型和sft_data数据定制的CHORD混合训练完整解决方案。

## 📦 包含文件

### 🎯 配置文件（主要）
| 文件名 | 用途 | 推荐场景 |
|--------|------|----------|
| `qwen3_chord_config.yaml` | 完整CHORD配置 | 生产环境，追求最佳性能 |
| `qwen3_chord_train_config.yaml` | 完整训练参数 | 配合主配置使用 |
| `qwen3_chord_simple.yaml` | 简化CHORD配置 | 测试环境，快速验证 |
| `qwen3_chord_simple_train.yaml` | 简化训练参数 | 配合简化配置使用 |

### 🛠️ 工具脚本
| 文件名 | 用途 |
|--------|------|
| `prepare_chord_data.py` | 数据预处理脚本，自动分割SFT和RL数据 |
| `run_chord_training.sh` | 一键启动脚本，自动化完整训练流程 |

### 📚 文档
| 文件名 | 内容 |
|--------|------|
| `chord_training_guide.md` | 详细训练指南，包含参数调优和故障排除 |
| `README_CHORD.md` | 本文件，快速使用说明 |

## 🚀 三种使用方式

### 方式1：一键启动（最简单）
```bash
# 基础启动（使用默认参数）
bash run_chord_training.sh

# 自定义参数启动
bash run_chord_training.sh \
    --input-dir sft_data \
    --expert-ratio 0.3 \
    --config-type simple
```

### 方式2：手动步骤（可控性强）
```bash
# 1. 数据预处理
python prepare_chord_data.py --input_dir sft_data --expert_ratio 0.25

# 2. 启动Ray集群  
ray start --head

# 3. 启动训练
trinity run --config qwen3_chord_simple.yaml
```

### 方式3：完全自定义（高级用户）
```bash
# 1. 复制配置文件模板
cp qwen3_chord_config.yaml my_config.yaml

# 2. 根据需要修改配置参数
# 3. 手动执行训练流程
```

## 🎛️ CHORD算法核心参数

### μ参数调度（最重要）
```yaml
policy_loss_fn_args:
  mu_warmup_steps: 100    # μ增长阶段步数
  mu_decay_steps: 300     # μ衰减阶段步数
  mu_peak: 0.6           # μ峰值（SFT权重最大）
  mu_valley: 0.05        # μ谷值（后期RL为主）
```

### 数据配比控制
```yaml
sample_strategy_args:
  expert_data_ratio: 0.25  # 专家数据占比25%
```

### CHORD变体选择
```yaml
policy_loss_fn_args:
  enable_phi_function: true  # true=CHORD-φ, false=CHORD-μ
```

## 🔧 常见配置调整

### 显存不足时
```yaml
# 减小批次大小
buffer:
  batch_size: 8
  train_batch_size: 32

# 启用offload
fsdp_config:
  param_offload: True
  optimizer_offload: True
```

### GPU数量调整
```yaml
cluster:
  gpu_per_node: 4  # 你的GPU数量

explorer:
  rollout_model:
    tensor_parallel_size: 2  # 通常是GPU数量的一半
```

### 训练速度优化
```yaml
explorer:
  runner_num: 16      # 增加并发任务数
  rollout_model:
    engine_num: 4     # 增加推理引擎数
```

## 📊 数据格式支持

### 推荐格式：简单问答
```json
{"prompt": "什么是机器学习？", "response": "机器学习是..."}
```

### 对话格式
```json
{"messages": [{"role": "user", "content": "问题"}, {"role": "assistant", "content": "答案"}]}
```

### 其他字段名（自动识别）
```json
{"question": "问题", "answer": "答案"}
{"input": "输入", "output": "输出"}
```

## 📈 训练监控

### 关键指标
- `mu`: μ参数值，观察调度曲线
- `loss`: 总损失，应逐渐下降
- `expert/sft_loss`: SFT损失（专家数据）
- `usual/pg_loss`: GRPO损失（RL数据）

### 监控方法
```bash
# 实时日志
tail -f logs/trinity.log

# WandB（如果配置）
export WANDB_API_KEY=your_key
# 然后查看网页界面

# Ray集群状态
ray status
```

## 🔍 故障排除

### 显存不足
```bash
# 错误: CUDA Out of Memory
# 解决: 减小batch_size和micro_batch_size
```

### 数据加载错误  
```bash
# 重新预处理数据
python prepare_chord_data.py --input_dir sft_data
```

### Ray连接问题
```bash
# 重启Ray集群
ray stop && ray start --head
```

### 训练不收敛
```bash
# 调整学习率和μ参数
lr: 1e-6  # 更小的学习率
mu_peak: 0.4  # 更小的SFT权重
```

## 🎯 最佳实践

### 1. 分阶段验证
1. 先用`simple`配置验证流程
2. 再用完整配置追求性能

### 2. 数据质量优先
- 确保专家数据质量高
- 移除重复和错误数据
- 保证数据格式一致性

### 3. 超参数调优顺序
1. 首先调优批次大小和学习率
2. 然后调优μ调度参数  
3. 最后微调数据比例

### 4. 实验管理
```bash
# 使用有意义的实验名称
name: "qwen3-4b-chord-v1.2-peak0.6-expert0.3"

# 保存重要配置版本
cp my_config.yaml configs/experiment_v1.2.yaml
```

## 📞 获取帮助

1. **详细指南**: 查看 `chord_training_guide.md`
2. **Trinity-RFT文档**: https://modelscope.github.io/Trinity-RFT/
3. **CHORD论文**: https://arxiv.org/abs/2508.11408
4. **GitHub Issues**: https://github.com/modelscope/Trinity-RFT/issues

## 🏆 预期效果

使用CHORD混合训练，相比纯SFT或纯RL训练，你可以期待：
- ✅ 更稳定的训练过程
- ✅ 更好的任务性能  
- ✅ 更强的生成质量
- ✅ 更少的灾难性遗忘

开始你的CHORD混合训练之旅吧！🚀
