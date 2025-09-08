# -*- coding: utf-8 -*-
"""Custom Workflow for Trinity-RFT project - exclusively uses all_reward function."""

import json
from typing import List, Optional

import openai

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows.workflow import WORKFLOWS, SimpleWorkflow, Task


class CustomWorkflow(SimpleWorkflow):
    """A custom workflow that exclusively uses the all_reward function from rl.reward module."""

    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[openai.OpenAI]] = None,
    ):
        self.reset(task)
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    def reset(self, task: Task):
        """Reset the workflow with a new task."""
        self.format_args = task.format_args
        self.system_prompt = task.format_args.system_prompt
        self.reply_prefix = task.format_args.reply_prefix

        self.raw_task = task.raw_task
        self.task_desc = task.task_desc
        self.truth = task.truth

        self.workflow_args = task.workflow_args
        self.reward_fn_args = task.reward_fn_args

        # 始终使用all_reward函数
        from rl.reward import all_reward
        self.reward_fn = all_reward
        
    def _extract_original_question(self, prompt_text: str) -> str:
        """从完整的prompt中提取原始用户问题"""
        # 查找 "User query:" 标记后的内容
        if "User query:" not in prompt_text:
            raise ValueError(f"❌ 在prompt中未找到'User query:'标记。Prompt开头: {prompt_text[:200]}...")
        
        start_idx = prompt_text.find("User query:") + len("User query:")
        
        # 找到"Graph evidence:"的位置
        end_idx = prompt_text.find("Graph evidence:", start_idx)
        if end_idx == -1:
            raise ValueError(f"❌ 在prompt中未找到'Graph evidence:'标记。User query后内容: {prompt_text[start_idx:start_idx+200]}...")
        
        question = prompt_text[start_idx:end_idx].strip()
        
        if not question:
            raise ValueError(f"❌ 提取的问题为空。Start: {start_idx}, End: {end_idx}")
        
        return question

    def run(self) -> List[Experience]:
        """Run the workflow and return experiences."""
        messages = self.format_messages()

        self.logger.debug("start chat")
        responses = self.model.chat(messages, **self.rollout_args)
        
        experiences = []
        for i, response in enumerate(responses):
            # 准备ground_truth数据格式，适配all_reward函数
            if isinstance(self.truth, dict):
                # 如果truth已经是字典格式
                ground_truth = json.dumps(self.truth)
            else:
                # 否则构造包含question和gold_answer的格式
                # 提取原始问题而不是使用完整的prompt
                original_question = self._extract_original_question(self.task_desc)
                ground_truth_dict = {
                    "question": original_question,
                    "gold_answer": self.truth
                }
                ground_truth = json.dumps(ground_truth_dict)

            # 调用all_reward函数计算奖励
            try:
                reward_score = self.reward_fn(
                    data_source="custom_workflow",
                    solution_str=response.response_text,
                    ground_truth=ground_truth,
                    extra_info=None
                )
                
                # 将奖励分数设置到response对象
                response.reward = float(reward_score)
                
                # 设置metrics (只包含数值类型，避免字符串导致求和错误)
                if response.metrics is None:
                    response.metrics = {}
                response.metrics.update({
                    "total_reward": float(reward_score)
                })
                
            except Exception as e:
                self.logger.error(f"奖励计算失败: {e}")
                # 设置默认奖励为0
                response.reward = 0.0
                if response.metrics is None:
                    response.metrics = {}
                response.metrics.update({
                    "total_reward": 0.0
                })

            response.eid.run = i + self.run_id_base

            self.logger.debug(
                f"task_desc: {self.task_desc}, "
                f"response: {response.response_text[:100]}..., "
                f"reward: {response.reward}"
            )

            # 将response添加到experiences列表（model.chat已经返回Experience对象）
            experiences.append(response)

        return experiences

# 安全注册workflow，避免重复注册错误
try:
    if "custom_workflow" not in WORKFLOWS.modules:
        WORKFLOWS.register_module("custom_workflow")(CustomWorkflow)
        print("✅ 成功注册CustomWorkflow")
    else:
        print("⚠️ CustomWorkflow已注册，跳过重复注册")
except Exception as e:
    print(f"⚠️ CustomWorkflow注册错误: {e}")
