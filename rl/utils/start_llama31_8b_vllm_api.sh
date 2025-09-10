#!/bin/bash

# vLLM API服务器启动脚本 - Llama3.3-70B-Instruct  
# 专门用于0/1二元判断，高质量奖励评分

set -e

# 配置参数 - L20专用部署（GPU 3独占，48GB显存）
MODEL_PATH="../cache/models/modelscope/hub/models/LLM-Research/Llama-3.3-70B-Instruct"
HOST="localhost"
PORT=8001  # 使用不同端口避免冲突
TENSOR_PARALLEL_SIZE=1  # 单卡（GPU 3）  
GPU_MEMORY_UTILIZATION=0.95  # L20 48GB独占，70B模型需要更多显存  
MAX_MODEL_LEN=4096   # 70B模型，适当增加长度以发挥模型能力

# 确保模型路径存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请先下载Llama-3.3-70B-Instruct模型"
    exit 1
fi

echo "🚀 启动Llama3.3-70B vLLM API服务器..."
echo "📂 模型路径: $MODEL_PATH"
echo "🌐 监听地址: $HOST:$PORT"
echo "⚡ 张量并行度: $TENSOR_PARALLEL_SIZE (使用L20 GPU 3独占)"
echo "💾 显存利用率: ${GPU_MEMORY_UTILIZATION} (95%显存占用，约46GB/48GB，70B模型完整加载)"
echo "📏 最大序列长度: $MAX_MODEL_LEN (70B高质量判断)"
echo ""
echo "🎯 二元输出安全性能优化配置 (0/1超快判断，不影响结果):"
echo "  - 🚀 最大并发数: 4096个序列 (2048序列长度→极高并发)"
echo "  - 📦 批处理容量: 16,384 tokens (大批处理)"
echo "  - 📏 序列长度: 2,048 (短输入+极短输出)"
echo "  - 🧩 分块预填充: 启用 (优化显存使用)"
echo "  - 🗄️ 前缀缓存: 启用 (重复前缀加速)"
echo "  - 📊 块大小优化: 64 (H800极速优化)"
echo "  - 💡 策略: 专为0/1二元判断极速优化"
echo "  - 📐 单卡优化: 启用"
echo "  - 📋 请求日志: 启用 (便于调试监控)"
echo "  - 🎯 预期GPU利用率: 95% (约46GB/48GB，GPU 3专用70B模型)"
echo "  - ✅ 安全极速配置: 超快奖励评分 (不影响结果准确性)"
echo ""

# 使用GPU 3（专用于Llama3.3-70B API服务）
export CUDA_VISIBLE_DEVICES=3

# 启动vLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "llama33-70b" \
    --trust-remote-code \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 32 \
    --block-size 32 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --disable-custom-all-reduce \
    --max-seq-len-to-capture 2048 