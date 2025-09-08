#!/usr/bin/env python3
"""
数据集验证脚本
检查训练数据集中是否能够正确提取原始用户问题
"""

import json
import os
import sys
from typing import List, Tuple

def extract_original_question(prompt_text: str) -> str:
    """从完整的prompt中提取原始用户问题"""
    # 查找 "User query:" 标记后的内容
    if "User query:" not in prompt_text:
        raise ValueError(f"❌ 在prompt中未找到'User query:'标记")
    
    start_idx = prompt_text.find("User query:") + len("User query:")
    
    # 找到"Graph evidence:"的位置
    end_idx = prompt_text.find("Graph evidence:", start_idx)
    if end_idx == -1:
        raise ValueError(f"❌ 在prompt中未找到'Graph evidence:'标记")
    
    question = prompt_text[start_idx:end_idx].strip()
    
    if not question:
        raise ValueError(f"❌ 提取的问题为空")
    
    return question

def validate_dataset(dataset_path: str) -> Tuple[int, int, List[dict]]:
    """
    验证数据集
    返回: (成功数量, 总数量, 错误列表)
    """
    success_count = 0
    total_count = 0
    errors = []
    
    print(f"🔍 开始验证数据集: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集文件不存在: {dataset_path}")
        return 0, 0, [{"error": "文件不存在", "path": dataset_path}]
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_count += 1
            
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 检查必要字段
                if 'prompt' not in data:
                    errors.append({
                        "line": line_num,
                        "error": "缺少'prompt'字段",
                        "data_keys": list(data.keys())
                    })
                    continue
                
                # 尝试提取原始问题
                original_question = extract_original_question(data['prompt'])
                
                # 验证提取结果
                if len(original_question) < 5:
                    errors.append({
                        "line": line_num,
                        "error": "提取的问题过短",
                        "extracted": original_question
                    })
                    continue
                
                if len(original_question) > 500:
                    errors.append({
                        "line": line_num,
                        "error": "提取的问题过长",
                        "extracted_length": len(original_question),
                        "extracted_preview": original_question[:100] + "..."
                    })
                    continue
                
                success_count += 1
                
                # 每1000行打印一次进度
                if line_num % 1000 == 0:
                    print(f"✅ 已验证 {line_num} 行，成功率: {success_count/line_num*100:.1f}%")
                
            except json.JSONDecodeError as e:
                errors.append({
                    "line": line_num,
                    "error": f"JSON解析错误: {e}",
                    "line_content": line[:100] + "..." if len(line) > 100 else line
                })
            except ValueError as e:
                errors.append({
                    "line": line_num,
                    "error": str(e),
                    "prompt_preview": data.get('prompt', '')[:200] + "..." if len(data.get('prompt', '')) > 200 else data.get('prompt', '')
                })
            except Exception as e:
                errors.append({
                    "line": line_num,
                    "error": f"未知错误: {e}",
                    "data_preview": str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                })
    
    return success_count, total_count, errors

def main():
    """主函数"""
    print("🚀 Trinity-RFT 数据集验证器")
    print("=" * 50)
    
    # 待验证的数据集路径
    dataset_paths = [
        "rl/datasets/rl_train/rl_train.jsonl",
        "rl/datasets/sft_train/sft_train.jsonl",
    ]
    
    total_success = 0
    total_count = 0
    all_errors = []
    
    for dataset_path in dataset_paths:
        print(f"\n📁 验证数据集: {dataset_path}")
        print("-" * 40)
        
        success, count, errors = validate_dataset(dataset_path)
        total_success += success
        total_count += count
        
        print(f"📊 验证结果:")
        print(f"   ✅ 成功: {success}/{count} ({success/count*100:.1f}%)")
        print(f"   ❌ 失败: {len(errors)}")
        
        if errors:
            print(f"\n🔍 错误详情 (显示前5个):")
            for i, error in enumerate(errors[:5]):
                print(f"   {i+1}. 行 {error.get('line', 'N/A')}: {error.get('error', 'Unknown')}")
                if 'extracted' in error:
                    print(f"      提取结果: {error['extracted']}")
                if 'prompt_preview' in error:
                    print(f"      Prompt预览: {error['prompt_preview']}")
                print()
        
        all_errors.extend(errors)
    
    print("\n" + "=" * 50)
    print(f"📈 总体验证结果:")
    print(f"   ✅ 总成功: {total_success}/{total_count} ({total_success/total_count*100:.1f}%)")
    print(f"   ❌ 总失败: {len(all_errors)}")
    
    if len(all_errors) == 0:
        print(f"🎉 所有数据都能正确提取原始问题！可以安全运行训练。")
        return 0
    else:
        print(f"⚠️  发现 {len(all_errors)} 个问题，建议修复后再运行训练。")
        
        # 保存错误报告
        error_report_path = "rl/dataset_validation_errors.json"
        with open(error_report_path, 'w', encoding='utf-8') as f:
            json.dump(all_errors, f, ensure_ascii=False, indent=2)
        print(f"📄 详细错误报告已保存到: {error_report_path}")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())
