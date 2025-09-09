#!/usr/bin/env python3

import re
import yaml
import sys
import logging
import os
from pathlib import Path
from datetime import datetime

# 导入当前项目的评估工具
sys.path.append(str(Path(__file__).parent))
from utils.eval import Eval

# 导入Trinity-RFT奖励函数基类
try:
    from trinity.common.rewards.reward_fn import RewardFn, REWARD_FUNCTIONS
    print("✅ 成功导入Trinity-RFT奖励函数基类")
except ImportError:
    print("警告: 无法导入Trinity-RFT奖励函数基类，将使用函数接口")
    RewardFn = None
    REWARD_FUNCTIONS = None

def load_config(config_path: str = None):
    """加载配置文件"""
    if config_path is None:
        # 从当前文件所在目录加载config.yaml
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def setup_loggers():
    """设置两个独立的日志记录器"""
    # 创建logs目录 - 使用autodl-tmp目录
    current_dir = Path(__file__).parent
    logs_dir = current_dir / ".." / ".." / "autodl-tmp" / "outputs" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 设置第一个日志记录器 - 完整输出
    logger1 = logging.getLogger('reward_full')
    logger1.setLevel(logging.INFO)
    # 清除现有的handlers
    logger1.handlers.clear()
    
    handler1 = logging.FileHandler(logs_dir / f"reward_full_{timestamp}.log", encoding='utf-8')
    formatter1 = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler1.setFormatter(formatter1)
    logger1.addHandler(handler1)
    logger1.propagate = False  # 防止重复输出
    
    # 设置第二个日志记录器 - 简化输出
    logger2 = logging.getLogger('reward_simple')
    logger2.setLevel(logging.INFO)
    # 清除现有的handlers
    logger2.handlers.clear()
    
    handler2 = logging.FileHandler(logs_dir / f"reward_simple_{timestamp}.log", encoding='utf-8')
    formatter2 = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler2.setFormatter(formatter2)
    logger2.addHandler(handler2)
    logger2.propagate = False  # 防止重复输出
    
    return logger1, logger2

# 全局日志记录器
_logger1, _logger2 = setup_loggers()

def extract_content(text: str, expected_type: str = None) -> tuple[str, str]:
    """从文本中提取标签内容"""
    patterns = {
        "answer": r'<answer>(.*?)</answer>',
        "query": r'<query>(.*?)</query>',
        "label": r'<label>(.*?)</label>',
        "think": r'<think>(.*?)</think>'
    }
    
    matches = {tag: re.search(pattern, text, re.DOTALL) for tag, pattern in patterns.items()}
    
    if expected_type is None:
        # 提取任何找到的内容
        for tag, match in matches.items():
            if match:
                return tag, match.group(1).strip()
        return "none", ""
    
    # 只提取指定类型的内容
    match = matches.get(expected_type)
    if match:
        return expected_type, match.group(1).strip()
    return "none", ""

def format_reward(completion: str) -> tuple[float, str]:
    """检查输出格式并返回奖励分数"""
    completion = completion.strip()
    
    # 查找所有标签
    matches = {
        'think': re.search(r'<think>(.*?)</think>', completion, re.DOTALL),
        'label': re.search(r'<label>\s*(able|unable)\s*</label>', completion, re.IGNORECASE),
        'answer': re.search(r'<answer>(.*?)</answer>', completion, re.DOTALL),
        'query': re.search(r'<query>(.*?)</query>', completion, re.DOTALL)
    }
    
    # 基础检查
    if not matches['think']: 
        return 0.0, "缺少<think>标签"
    elif not matches['think'].group(1).strip(): 
        return 0.0, "think标签为空"
    elif not matches['label']: 
        return 0.0, "缺少<label>标签"
    else:
        label_content = matches['label'].group(1).strip().lower()
        if label_content not in ['able', 'unable']: 
            return 0.0, f"label内容无效: {label_content}"
        # 检查标签顺序
        elif matches['think'].start() >= matches['label'].start():
            return 0.0, "<think>标签应该在<label>标签之前"
        # 根据label内容检查相应的标签
        elif label_content == "able":
            if not matches['answer']: 
                return 0.0, "label为'able'时缺少<answer>标签"
            elif not matches['answer'].group(1).strip(): 
                return 0.0, "answer标签为空"
            elif matches['query']: 
                return 0.0, "label为'able'时不应该有<query>标签"
            elif matches['label'].start() >= matches['answer'].start():
                return 0.0, "<label>标签应该在<answer>标签之前"
            else:
                # 检查是否有标签外的额外文本
                clean_text = completion
                for tag in ['think', 'label', 'answer', 'query']:
                    pattern = f'<{tag}>.*?</{tag}>'
                    clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
                
                if clean_text.strip():
                    return 0.0, f"包含标签外的额外文本: {clean_text.strip()[:50]}..."
                else:
                    return 1.0, "格式正确"
        elif label_content == "unable":
            if not matches['query']: 
                return 0.0, "label为'unable'时缺少<query>标签"
            elif not matches['query'].group(1).strip(): 
                return 0.0, "query标签为空"
            elif matches['answer']: 
                return 0.0, "label为'unable'时不应该有<answer>标签"
            elif matches['label'].start() >= matches['query'].start():
                return 0.0, "<label>标签应该在<query>标签之前"
            else:
                # 检查是否有标签外的额外文本
                clean_text = completion
                for tag in ['think', 'label', 'answer', 'query']:
                    pattern = f'<{tag}>.*?</{tag}>'
                    clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
                
                if clean_text.strip():
                    return 0.0, f"包含标签外的额外文本: {clean_text.strip()[:50]}..."
                else:
                    return 1.0, "格式正确"

def answer_reward(completion: str, question: str, gold_answer: str) -> tuple[float, str, str]:
    """评估LLM输出的答案质量，支持answer和query两种情况
    
    Returns:
        tuple[float, str, str]: (评分, 评估信息, llama3.1原始输出)
    """
    try:
        config = load_config()
        api_config = config.get("api", {})
        evaluator = Eval(
            llm_api_key=api_config.get("key", "EMPTY"),
            base_url=api_config.get("base_url", "http://localhost:8000/v1"),
            model=api_config.get("model", "qwen3-4b"),
            temperature=api_config.get("temperature", 0.3)
        )
        
        # 从gold_answer中提取标准内容
        gold_content_type, gold_content = extract_content(gold_answer)
        
        # 如果gold_answer格式有问题（无法提取到有效内容），直接报错
        if gold_content_type == "none":
            return 0.0, f"gold_answer格式错误", ""
        
        # 根据gold_answer的格式提取LLM输出中的对应内容
        content_type, content = extract_content(completion, gold_content_type)
        
        # 处理评估逻辑
        if content_type == "answer" and gold_content_type == "answer":
            eval_result = evaluator.eval_answer(question, gold_content, content)
            eval_type = "答案"
        elif content_type == "query" and gold_content_type == "query":
            eval_result = evaluator.eval_query(question, gold_content, content)
            eval_type = "继续查询"
        else:
            return 0.0, "格式不匹配", ""
        
        # 解析评估结果
        try:
            reward_value = float(eval_result.strip())
            if reward_value in [0.0, 1.0]:
                return reward_value, eval_type, eval_result  # 添加原始输出
            else:
                return 0.0, f"评估结果无效: {reward_value}", eval_result
        except ValueError:
            return 0.0, f"无法解析评估结果: {eval_result}", eval_result
            
    except Exception as e:
        return 0.0, f"评估过程出错: {str(e)}", ""

def label_reward(completion: str, gold_answer: str) -> tuple[float, str, str]:
    """
    检查LLM输出中的<label>标签是否与gold_answer中的<label>匹配
    """
    try:
        # 从gold_answer中提取<label>标签内容
        gold_label_type, gold_label = extract_content(gold_answer, "label")
        
        # 如果gold_answer中没有<label>标签，直接报错
        if gold_label_type == "none":
            return 0.0, "未知", "gold_answer中没有找到<label>标签"
        
        # 检查gold_label是否为"able"或"unable"
        gold_label_clean = gold_label.lower().strip()
        if gold_label_clean not in ["able", "unable"]:
            return 0.0, gold_label_clean, f"gold_answer中的标签无效: {gold_label}"
        
        # 从completion中提取<label>标签内容
        completion_label_type, completion_label = extract_content(completion, "label")
        
        if completion_label_type == "label":
            # 比较两个标签内容
            completion_label_clean = completion_label.lower().strip()
            
            if completion_label_clean == gold_label_clean:
                return 1.0, gold_label_clean, "匹配"
            else:
                return 0.0, gold_label_clean, f"不匹配({completion_label_clean})"
        else:
            # completion中没有找到<label>标签
            return 0.0, gold_label_clean, "缺失标签"
            
    except Exception as e:
        return 0.0, "未知", f"标签评估出错: {str(e)}"

def all_reward(data_source, solution_str, ground_truth, extra_info=None):
    """
    verl兼容的奖励函数
    
    Args:
        data_source: 数据源标识
        solution_str: 模型生成的回答
        ground_truth: 标准答案（包含question和gold_answer的JSON字符串）
        extra_info: 额外信息
    
    Returns:
        float: 总奖励分数 (0.0-3.0)
    """
    try:
        import json
        # 解析ground_truth中的信息
        if isinstance(ground_truth, str):
            try:
                gt_data = json.loads(ground_truth)
            except json.JSONDecodeError:
                # 如果不是JSON，假设它就是gold_answer
                gt_data = {"gold_answer": ground_truth, "question": ""}
        else:
            gt_data = ground_truth
        
        question = gt_data.get("question", "")
        gold_answer = gt_data.get("gold_answer", "")
        
        if not gold_answer:
            print(f"警告: 缺少gold_answer信息")
            return 0.0
        
        # 1. 格式评估 (0-1分)
        format_score, format_msg = format_reward(solution_str)
        
        # 2. 标签评估 (0-1分)
        label_score, gold_label, label_msg = label_reward(solution_str, gold_answer)
        
        # 3. 答案评估 (0-1分)
        answer_score, answer_msg, reward_model_output = answer_reward(solution_str, question, gold_answer)
        
        # 计算总分
        total_score = format_score + label_score + answer_score
        
        # 准备详细信息
        format_info = f"格式({format_score}:{format_msg[:20]})" if format_score < 1.0 else f"格式({format_score})"
        label_info = f"标签({label_score}:{gold_label}-{label_msg[:20]})" if label_score < 1.0 else f"标签({label_score}:{gold_label})"
        answer_info = f"内容({answer_score}:{answer_msg[:30]})" if answer_score < 1.0 else f"内容({answer_score})"
        
        # 显示完整的模型输出，转换换行符为空格
        output_full = solution_str.replace('\n', ' ').strip()
            
        # 一行输出所有信息：奖励评估、llama3.1输出、回答模型输出
        combined_output1 = f"🎯 {total_score:.1f}/3.0 | {format_info} {label_info} {answer_info} | 奖励模型输出:{reward_model_output.strip()} | 回答模型输出:{output_full}"
        combined_output2 = f"🎯 {total_score:.1f}/3.0 | {format_info} {label_info} {answer_info} | 奖励模型输出:{reward_model_output.strip()}"
        
        # 记录到两个独立的日志文件
        _logger1.info(combined_output1)  # 完整日志（包含回答模型输出）
        _logger2.info(combined_output2)  # 简化日志（不包含回答模型输出）
        
        # 控制台输出简化版本
        # print(combined_output2)
        
        return total_score
        
    except Exception as e:
        print(f"奖励计算出错: {str(e)}")
        return 0.0 


# Trinity-RFT兼容的奖励函数类
if RewardFn and REWARD_FUNCTIONS:
    @REWARD_FUNCTIONS.register_module("all_reward_class")
    class AllRewardFn(RewardFn):
        """综合奖励函数类，继承自RewardFn基类，兼容Trinity-RFT框架"""
        
        def __init__(self, **kwargs):
            """初始化奖励函数"""
            pass
            
        def __call__(self, experience, messages, **kwargs):
            """
            Trinity-RFT框架调用接口
            
            Args:
                experience: 经验对象，包含任务信息
                messages: 消息列表，包含模型输出
                **kwargs: 其他参数
            
            Returns:
                Dict[str, float]: 奖励分数字典
            """
            try:
                print(f"🔍 AllRewardFn调用:")
                print(f"  experience类型: {type(experience)}")
                print(f"  experience属性: {[attr for attr in dir(experience) if not attr.startswith('_')]}")
                print(f"  messages数量: {len(messages)}")
                
                # 从experience中提取任务信息
                if hasattr(experience, 'task'):
                    task = experience.task
                    print(f"  task: {task}")
                    
                    # 从task中获取原始数据
                    if hasattr(task, 'data'):
                        task_data = task.data
                        print(f"  task.data: {task_data}")
                    else:
                        print(f"  task属性: {[attr for attr in dir(task) if not attr.startswith('_')]}")
                        task_data = {}
                
                # 从messages中获取模型输出
                if messages and len(messages) > 0:
                    # 通常最后一条消息是模型的回复
                    model_response = messages[-1].get('content', '') if isinstance(messages[-1], dict) else str(messages[-1])
                    print(f"  模型输出: {model_response[:200]}...")
                else:
                    model_response = ""
                    print(f"  无模型输出")
                
                # 构建ground_truth
                if task_data and isinstance(task_data, dict):
                    if 'answer' in task_data:
                        # 从answer字段解析ground truth
                        ground_truth = {
                            "question": task_data.get('prompt', ''),
                            "gold_answer": task_data['answer']
                        }
                        print(f"  ground_truth: {ground_truth}")
                    else:
                        raise ValueError(f"task_data中缺少answer字段: {list(task_data.keys())}")
                else:
                    raise ValueError(f"无法获取task_data: experience属性={[attr for attr in dir(experience) if not attr.startswith('_')]}")
                
                # 调用原始的all_reward函数
                reward_score = all_reward(
                    data_source="trinity_rft",
                    solution_str=model_response,
                    ground_truth=ground_truth,
                    extra_info=kwargs
                )
                print(f"  计算得到奖励: {reward_score}")
                
                # Trinity-RFT期望返回字典格式
                return {"reward": reward_score}
                
            except Exception as e:
                print(f"AllRewardFn调用出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return {"reward": 0.0}
else:
    # 如果没有成功导入RewardFn，创建一个空的类作为占位符
    class AllRewardFn:
        """占位符类，当无法导入Trinity-RFT基类时使用"""
        def __call__(self, experience, messages, **kwargs):
            return {"reward": 0.0}
    print("警告: 使用占位符AllRewardFn类")