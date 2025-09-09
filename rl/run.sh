#!/bin/bash

# 简单运行 Trinity-RFT CHORD 算法的脚本
# 基于 README_zh.md 中的步骤

set -e

echo "=================================="
echo "Trinity-RFT CHORD 训练启动"
echo "时间: $(date)"
echo "=================================="

# 进入项目根目录
cd /home/haoyang/workspace/Trinity-RFT

# 设置使用GPU 0,1,2,3 (4张GPU卡)
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "设置GPU: 使用4张GPU卡 (GPU 0,1,2,3)"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 第一步：检查并启动 Ray 集群
echo "检查 Ray 集群状态..."
if ray status > /dev/null 2>&1; then
    echo "✅ Ray 集群已在运行，跳过启动步骤"
else
    echo "启动 Ray 集群..."
    ray start --head
fi

# 第二步：（可选）设置 WandB
if [[ -n "${WANDB_API_KEY}" ]]; then
    echo "设置 WandB..."
    wandb login
fi

# 第三步：运行 CHORD 训练
echo "开始运行 CHORD 算法训练..."
echo "配置文件: rl/mix_chord.yaml"

trinity run --config rl/mix_chord.yaml

echo "训练完成！"
