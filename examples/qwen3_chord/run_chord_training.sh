#!/bin/bash

# Trinity-RFT CHORD训练快速启动脚本
# 自动化处理数据预处理、环境检查和训练启动

set -e  # 遇到错误立即退出

echo "=== Trinity-RFT CHORD训练快速启动脚本 ==="
echo "时间: $(date)"
echo

# 配置参数（可修改）
INPUT_DATA_DIR="sft_data"
OUTPUT_DATA_DIR="chord_data" 
EXPERT_RATIO="0.25"
CONFIG_TYPE="simple"  # 可选: simple, full
MODEL_PATH="cache/models/modelscope/hub/models/qwen/Qwen3-4B"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DATA_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DATA_DIR="$2" 
            shift 2
            ;;
        --expert-ratio)
            EXPERT_RATIO="$2"
            shift 2
            ;;
        --config-type)
            CONFIG_TYPE="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --help)
            echo "使用方法: $0 [选项]"
            echo "选项:"
            echo "  --input-dir DIR      输入数据目录 (默认: sft_data)"
            echo "  --output-dir DIR     输出数据目录 (默认: chord_data)"
            echo "  --expert-ratio RATIO 专家数据比例 (默认: 0.25)"
            echo "  --config-type TYPE   配置类型: simple|full (默认: simple)"
            echo "  --model-path PATH    模型路径 (默认: cache/models/...)"
            echo "  --help              显示帮助信息"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "配置参数:"
echo "  输入数据目录: $INPUT_DATA_DIR"
echo "  输出数据目录: $OUTPUT_DATA_DIR"  
echo "  专家数据比例: $EXPERT_RATIO"
echo "  配置类型: $CONFIG_TYPE"
echo "  模型路径: $MODEL_PATH"
echo

# 函数：检查命令是否存在
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "错误: 命令 '$1' 未找到，请确保已正确安装"
        exit 1
    fi
}

# 函数：检查文件是否存在
check_file() {
    if [[ ! -f "$1" ]]; then
        echo "错误: 文件 '$1' 不存在"
        exit 1
    fi
}

# 函数：检查目录是否存在
check_directory() {
    if [[ ! -d "$1" ]]; then
        echo "错误: 目录 '$1' 不存在"
        exit 1
    fi
}

# 步骤1: 环境检查
echo "🔍 步骤1: 检查环境依赖..."
check_command "python"
check_command "ray"
check_command "trinity"

echo "✅ 环境检查通过"
echo

# 步骤2: 检查数据和模型
echo "🔍 步骤2: 检查数据和模型..."
check_directory "$INPUT_DATA_DIR"
check_directory "$MODEL_PATH"

# 检查数据文件
data_files_count=$(find "$INPUT_DATA_DIR" -name "*.json*" | wc -l)
if [[ $data_files_count -eq 0 ]]; then
    echo "错误: 在 $INPUT_DATA_DIR 中没有找到 .json 或 .jsonl 文件"
    exit 1
fi

echo "✅ 找到 $data_files_count 个数据文件"
echo "✅ 模型路径检查通过"
echo

# 步骤3: 停止现有Ray进程并启动新的
echo "🚀 步骤3: 准备Ray集群..."
echo "停止现有Ray进程..."
ray stop 2>/dev/null || true
sleep 2

echo "启动新的Ray集群..."
ray start --head
sleep 3

# 验证Ray状态
if ! ray status &>/dev/null; then
    echo "错误: Ray集群启动失败"
    exit 1
fi

echo "✅ Ray集群启动成功"
echo

# 步骤4: 数据预处理
echo "📊 步骤4: 数据预处理..."
if [[ ! -d "$OUTPUT_DATA_DIR" ]] || [[ -z "$(ls -A "$OUTPUT_DATA_DIR" 2>/dev/null)" ]]; then
    echo "运行数据预处理脚本..."
    python prepare_chord_data.py \
        --input_dir "$INPUT_DATA_DIR" \
        --output_dir "$OUTPUT_DATA_DIR" \
        --expert_ratio "$EXPERT_RATIO" \
        --seed 42
    
    echo "✅ 数据预处理完成"
else
    echo "✅ 检测到已有预处理数据，跳过预处理步骤"
fi
echo

# 步骤5: 选择和准备配置文件
echo "⚙️ 步骤5: 准备配置文件..."

if [[ "$CONFIG_TYPE" == "simple" ]]; then
    MAIN_CONFIG="qwen3_chord_simple.yaml"
    TRAIN_CONFIG="qwen3_chord_simple_train.yaml"
    echo "使用简化配置"
elif [[ "$CONFIG_TYPE" == "full" ]]; then
    MAIN_CONFIG="qwen3_chord_config.yaml"  
    TRAIN_CONFIG="qwen3_chord_train_config.yaml"
    echo "使用完整配置"
else
    echo "错误: 不支持的配置类型 '$CONFIG_TYPE'"
    exit 1
fi

# 检查配置文件
check_file "$MAIN_CONFIG"
check_file "$TRAIN_CONFIG"

# 创建运行时配置副本（避免修改原文件）
RUNTIME_CONFIG="runtime_chord_config.yaml"
cp "$MAIN_CONFIG" "$RUNTIME_CONFIG"

# 更新配置文件中的路径
echo "更新配置文件路径..."
if command -v sed &> /dev/null; then
    # 更新数据路径
    sed -i.bak "s|path: \"sft_data\"|path: \"$OUTPUT_DATA_DIR/rl_data.jsonl\"|g" "$RUNTIME_CONFIG"
    sed -i.bak "s|path: 'sft_data'|path: '$OUTPUT_DATA_DIR/expert_data.jsonl'|g" "$RUNTIME_CONFIG"
    
    # 更新模型路径
    sed -i.bak "s|cache/models/modelscope/hub/models/qwen/Qwen3-4B|$MODEL_PATH|g" "$RUNTIME_CONFIG"
    
    # 删除备份文件
    rm -f "${RUNTIME_CONFIG}.bak"
    
    echo "✅ 配置文件路径更新完成"
else
    echo "⚠️  警告: sed命令不可用，请手动检查配置文件中的路径"
fi
echo

# 步骤6: 显示训练前摘要
echo "📋 步骤6: 训练配置摘要"
echo "----------------------------------------"
echo "配置文件: $RUNTIME_CONFIG"
echo "训练配置: $TRAIN_CONFIG"
echo "专家数据: $OUTPUT_DATA_DIR/expert_data.jsonl"
echo "RL数据: $OUTPUT_DATA_DIR/rl_data.jsonl"
echo "模型路径: $MODEL_PATH"
echo "检查点目录: ./checkpoints/"
echo "----------------------------------------"
echo

# 步骤7: 确认启动
echo "🎯 步骤7: 准备启动训练..."
echo "即将启动CHORD混合训练，这可能需要几个小时到几天的时间"
echo "你可以通过以下方式监控训练进度:"
echo "  - 查看日志: tail -f logs/trinity.log"
echo "  - 使用WandB: 如果配置了WANDB_API_KEY"
echo "  - 检查检查点: ls -la checkpoints/"
echo

# 提供选择是否继续
read -p "是否继续启动训练? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "训练被取消"
    exit 0
fi

# 步骤8: 启动训练
echo "🚀 步骤8: 启动CHORD训练..."
echo "训练开始时间: $(date)"
echo

# 启动训练（后台运行并记录PID）
trinity run --config "$RUNTIME_CONFIG" &
TRINITY_PID=$!

echo "✅ 训练已在后台启动 (PID: $TRINITY_PID)"
echo "训练日志将保存在 logs/ 目录中"
echo

# 等待几秒钟检查进程是否成功启动
sleep 5
if kill -0 $TRINITY_PID 2>/dev/null; then
    echo "✅ 训练进程运行正常"
    echo
    echo "📖 有用的命令:"
    echo "  查看实时日志:    tail -f logs/trinity.log"
    echo "  检查训练状态:    ps aux | grep trinity"  
    echo "  停止训练:        kill $TRINITY_PID"
    echo "  检查检查点:      ls -la checkpoints/"
    echo "  Ray集群状态:     ray status"
    echo
    echo "🎉 CHORD训练启动成功！"
    echo "进程将在后台继续运行，你可以安全地关闭此终端"
else
    echo "❌ 训练进程启动失败，请检查日志文件"
    exit 1
fi

# 可选：等待用户选择是否持续监控
echo
read -p "是否要持续监控训练日志? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始监控训练日志 (Ctrl+C 停止监控):"
    echo "----------------------------------------"
    tail -f logs/trinity.log 2>/dev/null || echo "日志文件暂未创建，请稍后再试"
fi
