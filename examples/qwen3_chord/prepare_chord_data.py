#!/usr/bin/env python3
"""
数据预处理脚本 - 为CHORD训练准备数据
将sft_data目录中的数据处理成Trinity-RFT CHORD训练所需的格式
"""

import json
import os
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse


def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
    return data


def load_json_file(filepath: str) -> List[Dict[str, Any]]:
    """加载JSON文件"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
            return data
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return []


def normalize_data_format(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """标准化数据格式为 {prompt, response} 格式"""
    normalized = []
    
    for item in data:
        prompt = None
        response = None
        
        # 尝试不同的字段名组合
        prompt_keys = ['prompt', 'question', 'input', 'text', 'query', 'problem']
        response_keys = ['response', 'answer', 'output', 'target', 'solution', 'completion']
        
        # 查找prompt字段
        for key in prompt_keys:
            if key in item and item[key]:
                prompt = str(item[key]).strip()
                break
                
        # 查找response字段  
        for key in response_keys:
            if key in item and item[key]:
                response = str(item[key]).strip()
                break
        
        # 处理messages格式（对话数据）
        if 'messages' in item and isinstance(item['messages'], list):
            messages = item['messages']
            user_msg = None
            assistant_msg = None
            
            for msg in messages:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    if msg['role'] == 'user' and not user_msg:
                        user_msg = msg['content']
                    elif msg['role'] == 'assistant' and not assistant_msg:
                        assistant_msg = msg['content']
            
            if user_msg and assistant_msg:
                prompt = user_msg.strip()
                response = assistant_msg.strip()
        
        # 只有当prompt和response都存在时才添加
        if prompt and response:
            normalized.append({
                'prompt': prompt,
                'response': response
            })
    
    return normalized


def split_data_for_chord(data: List[Dict[str, str]], expert_ratio: float = 0.25, 
                        train_ratio: float = 0.8) -> Dict[str, List[Dict[str, str]]]:
    """
    为CHORD训练分割数据
    
    Args:
        data: 标准化后的数据
        expert_ratio: 专家数据比例（用于SFT）
        train_ratio: 训练数据比例
    
    Returns:
        包含不同用途数据的字典
    """
    random.shuffle(data)
    
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    
    # 分割训练和测试数据
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    # 从训练数据中分离专家数据
    expert_size = int(len(train_data) * expert_ratio)
    expert_data = train_data[:expert_size]
    rl_data = train_data[expert_size:]
    
    return {
        'expert_data': expert_data,    # SFT专家数据
        'rl_data': rl_data,           # 强化学习数据  
        'test_data': test_data,       # 测试/评估数据
        'all_train_data': train_data  # 所有训练数据
    }


def save_data_splits(data_splits: Dict[str, List[Dict[str, str]]], output_dir: str):
    """保存分割后的数据"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in data_splits.items():
        if split_data:  # 只保存非空数据
            output_file = output_path / f"{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in split_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')
            print(f"保存 {len(split_data)} 条数据到 {output_file}")


def main():
    parser = argparse.ArgumentParser(description='为CHORD训练准备数据')
    parser.add_argument('--input_dir', type=str, default='sft_data', 
                       help='输入数据目录')
    parser.add_argument('--output_dir', type=str, default='chord_data',
                       help='输出数据目录')  
    parser.add_argument('--expert_ratio', type=float, default=0.25,
                       help='专家数据比例')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练数据比例')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    print(f"开始处理数据目录: {args.input_dir}")
    print(f"专家数据比例: {args.expert_ratio}")
    print(f"训练数据比例: {args.train_ratio}")
    
    # 加载所有数据文件
    input_path = Path(args.input_dir)
    all_data = []
    
    if not input_path.exists():
        print(f"错误: 输入目录 {args.input_dir} 不存在")
        return
    
    # 支持的文件格式
    for file_pattern in ['*.jsonl', '*.json']:
        for filepath in input_path.glob(file_pattern):
            print(f"加载文件: {filepath}")
            
            if filepath.suffix == '.jsonl':
                file_data = load_jsonl_file(str(filepath))
            else:
                file_data = load_json_file(str(filepath))
            
            if file_data:
                all_data.extend(file_data)
                print(f"  加载了 {len(file_data)} 条数据")
    
    if not all_data:
        print("错误: 没有找到有效的数据文件")
        return
    
    print(f"总共加载了 {len(all_data)} 条原始数据")
    
    # 标准化数据格式
    normalized_data = normalize_data_format(all_data)
    print(f"标准化后有效数据: {len(normalized_data)} 条")
    
    if len(normalized_data) == 0:
        print("错误: 没有有效的prompt-response数据对")
        return
    
    # 分割数据
    data_splits = split_data_for_chord(
        normalized_data, 
        expert_ratio=args.expert_ratio,
        train_ratio=args.train_ratio
    )
    
    # 打印数据统计
    print("\n数据分割统计:")
    for split_name, split_data in data_splits.items():
        print(f"  {split_name}: {len(split_data)} 条")
    
    # 保存分割后的数据
    save_data_splits(data_splits, args.output_dir)
    
    # 生成配置建议
    print(f"\n建议的CHORD配置参数:")
    print(f"  expert_data_ratio: {args.expert_ratio}")
    print(f"  专家数据路径: {args.output_dir}/expert_data.jsonl")
    print(f"  RL数据路径: {args.output_dir}/rl_data.jsonl")
    print(f"  评估数据路径: {args.output_dir}/test_data.jsonl")
    
    print(f"\n数据预处理完成! 输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
