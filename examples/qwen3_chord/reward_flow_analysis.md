# Trinity-RFT中强化学习奖励信息传入机制详解

## 概述

Trinity-RFT中的奖励信息传入是一个精心设计的五层架构，从奖励函数定义到最终的模型参数更新，形成了完整的数据流。

## 1. 奖励函数定义层

### 基础接口：`RewardFn`
```python
class RewardFn(ABC):
    """奖励函数基类"""
    
    @abstractmethod
    def __call__(self, **kwargs) -> Dict[str, float]:
        """计算奖励，返回奖励字典"""
        pass
```

### 内置奖励函数类型

#### 1.1 数学任务奖励 (`MathRewardFn`)
```python
@REWARD_FUNCTIONS.register_module("math_reward")
class MathRewardFn(RewardFn):
    def __call__(self, response: str, truth: Optional[str] = None) -> dict[str, float]:
        # 计算准确性奖励
        accuracy_score = self.accuracy_reward(response, truth)
        # 计算格式奖励
        format_score = self.format_reward(response)
        
        return {**accuracy_score, **format_score}
```

#### 1.2 准确性奖励 (`AccuracyReward`)
```python
@REWARD_FUNCTIONS.register_module("accuracy_reward")
class AccuracyReward(RewardFn):
    def __call__(self, response: str, truth: str) -> Dict[str, float]:
        # 解析答案并比较
        predicted = self.answer_parser(response)
        ground_truth = self.answer_parser(truth)
        
        reward = 1.0 if predicted == ground_truth else 0.0
        return {"accuracy": reward}
```

#### 1.3 格式奖励 (`FormatReward`)
```python
@REWARD_FUNCTIONS.register_module("format_reward")  
class FormatReward(RewardFn):
    def __call__(self, response: str) -> Dict[str, float]:
        # 检查响应是否符合期望格式
        if re.match(self.pattern, response):
            return {"format_score": 0.0}  # 格式正确
        else:
            return {"format_score": -0.1}  # 格式错误，扣分
```

## 2. 工作流奖励计算层

### 2.1 简单工作流 (`SimpleWorkflow`)
```python
@WORKFLOWS.register_module("simple_workflow")
class SimpleWorkflow(Workflow):
    def run(self) -> List[Experience]:
        # 1. 格式化消息
        messages = self.format_messages()
        
        # 2. 模型生成响应
        responses = self.model.chat(messages, **self.rollout_args)
        
        # 3. 为每个响应计算奖励
        for i, response in enumerate(responses):
            # 调用奖励函数
            reward_dict = self.reward_fn(
                response=response.response_text,
                truth=self.truth,
            )
            
            # 更新响应的metrics
            if response.metrics is None:
                response.metrics = {}
            response.metrics.update(reward_dict)
            
            # 计算总奖励（所有分项奖励的总和）
            reward = sum(reward_dict.values())
            
            # 将奖励存储到Experience对象中
            response.reward = reward
            response.eid.run = i + self.run_id_base
            
        return responses
```

### 2.2 分步奖励工作流 (`StepWiseRewardWorkflow`)
```python
class StepWiseRewardWorkflow(Workflow):
    def run(self) -> list[Experience]:
        experiences = []
        
        for step in range(self.max_step_num):
            # 执行单步操作
            continue_run = self.step(step_num=step)
            
            # 从模型历史中提取经验
            exps = self.model.extract_experience_from_history()
            
            # 计算当前步骤的奖励
            reward = self.reward(exps, step_num=step)
            
            # 为每个经验设置奖励和步骤信息
            for exp in exps:
                exp.reward = reward
                exp.eid.step = step
                
            experiences.extend(exps)
            
            if not continue_run:
                break
                
        return experiences
```

### 2.3 奖励传播工作流 (`RewardPropagationWorkflow`)
```python
class RewardPropagationWorkflow(Workflow):
    def run(self) -> list[Experience]:
        experiences = []
        
        # 执行所有步骤
        for step in range(self.max_step_num):
            continue_run = self.step(step_num=step)
            exps = self.model.extract_experience_from_history()
            
            for exp in exps:
                exp.eid.step = step
            experiences.extend(exps)
            
            if not continue_run:
                break
        
        # 在所有步骤完成后计算整体奖励
        reward = self.reward(experiences)
        
        # 将奖励传播给所有经验
        for exp in experiences:
            exp.reward = reward
            
        return experiences
```

## 3. 经验数据存储层

### 3.1 Experience对象结构
```python
@dataclass
class Experience:
    eid: EID = field(default_factory=EID)  # 经验唯一标识
    tokens: Optional[Tensor] = None        # 令牌序列
    prompt_length: int = 1                 # 提示长度
    logprobs: Optional[Tensor] = None      # 对数概率
    
    # 核心奖励信息
    reward: Optional[float] = None         # 总奖励分数
    advantages: Optional[Tensor] = None    # 优势函数值
    returns: Optional[Tensor] = None       # 回报值
    
    # 详细奖励信息
    metrics: dict[str, float] = field(     # 详细奖励指标
        default_factory=dict
    )
    
    # 文本信息
    response_text: Optional[str] = None    # 响应文本
    prompt_text: Optional[str] = None      # 提示文本
    
    # 其他训练相关字段...
```

### 3.2 奖励信息的设置过程
```python
# 在工作流中设置奖励
def set_reward_in_workflow():
    # 1. 计算分项奖励
    reward_dict = reward_fn(response=response_text, truth=ground_truth)
    # 例: {"accuracy": 1.0, "format_score": 0.0}
    
    # 2. 更新metrics字段（保存详细信息）
    experience.metrics.update(reward_dict)
    
    # 3. 计算总奖励
    total_reward = sum(reward_dict.values())  # 1.0 + 0.0 = 1.0
    
    # 4. 设置总奖励
    experience.reward = total_reward
```

## 4. 批量数据处理层

### 4.1 Experiences容器
```python
@dataclass
class Experiences:
    """批量经验数据容器"""
    eids: List[EID]                        # 经验ID列表
    tokens: Tensor                         # [batch_size, seq_length]
    rewards: Tensor                        # [batch_size] - 关键奖励张量
    advantages: Optional[Tensor]           # [batch_size, response_length]
    returns: Optional[Tensor]              # [batch_size, response_length]
    attention_masks: Tensor                # 注意力掩码
    action_masks: Optional[Tensor]         # 动作掩码
    
    @classmethod
    def gather_experiences(cls, experiences: list[Experience]) -> Experiences:
        """将单个经验聚合成批量数据"""
        rewards = torch.tensor([exp.reward for exp in experiences])
        # ... 其他字段处理
        return cls(rewards=rewards, ...)
```

### 4.2 批量处理示例
```python
# 从单个Experience到批量Experiences
individual_experiences = [
    Experience(reward=1.0, tokens=[1, 2, 3, 4]),
    Experience(reward=0.5, tokens=[1, 2, 5, 6]),  
    Experience(reward=0.0, tokens=[1, 2, 7, 8]),
]

# 聚合成批量数据
batch_experiences = Experiences.gather_experiences(individual_experiences)

# 结果:
# batch_experiences.rewards = tensor([1.0, 0.5, 0.0])
# batch_experiences.tokens = tensor([[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 7, 8]])
```

## 5. 训练器消费层

### 5.1 训练器读取和处理
```python
class Trainer:
    def train_step(self, experiences: Experiences):
        # 1. 读取奖励信息
        rewards = experiences.rewards  # [batch_size]
        
        # 2. 计算优势函数
        advantages = self.advantage_fn(
            rewards=rewards,
            values=values,  # 来自价值网络
            # ... 其他参数
        )
        
        # 3. 计算策略损失
        policy_loss = self.policy_loss_fn(
            logprobs=logprobs,
            old_logprobs=old_logprobs,
            advantages=advantages,  # 使用处理后的奖励信息
        )
        
        # 4. 反向传播和参数更新
        policy_loss.backward()
        self.optimizer.step()
```

### 5.2 优势函数计算
```python
class AdvantageFunction:
    def compute_advantages(self, rewards: Tensor, values: Tensor) -> Tensor:
        """使用奖励计算优势函数"""
        returns = self.compute_returns(rewards)
        advantages = returns - values
        return advantages
        
    def compute_returns(self, rewards: Tensor) -> Tensor:
        """计算回报值"""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
            
        return returns
```

## 配置中的奖励函数设置

### YAML配置示例
```yaml
buffer:
  explorer_input:
    # 默认奖励函数配置
    default_reward_fn_type: 'math_reward'  # 指定奖励函数类型
    
    # 任务级别的奖励函数配置
    taskset:
      reward_fn_args:
        answer_parser: simple_answer_parser
        pattern: ".*?<answer>.*?</answer>.*?"
        
# 工作流级别的奖励配置
workflows:
  math_workflow:
    reward_fn_type: 'math_boxed_reward'
    reward_fn_args:
      with_think: true
      format_score_coef: 0.1
```

### 奖励函数的动态加载
```python
# 在工作流初始化时加载奖励函数
def load_reward_fn(reward_fn_type: str, reward_fn_args: dict) -> RewardFn:
    reward_fn_class = REWARD_FUNCTIONS.get(reward_fn_type)
    return reward_fn_class(**reward_fn_args)

# 使用示例
reward_fn = load_reward_fn('math_reward', {'pattern': r'<answer>.*?</answer>'})
```

## 自定义奖励函数

### 创建自定义奖励函数
```python
@REWARD_FUNCTIONS.register_module("custom_reward")
class CustomRewardFn(RewardFn):
    def __init__(self, weight_accuracy: float = 1.0, weight_length: float = 0.1):
        self.weight_accuracy = weight_accuracy
        self.weight_length = weight_length
    
    def __call__(self, response: str, truth: str, **kwargs) -> Dict[str, float]:
        # 1. 准确性奖励
        accuracy = 1.0 if response.strip() == truth.strip() else 0.0
        
        # 2. 长度奖励（鼓励简洁回答）
        length_penalty = -0.01 * len(response.split())
        
        # 3. 组合奖励
        total_accuracy = self.weight_accuracy * accuracy
        total_length = self.weight_length * length_penalty
        
        return {
            "accuracy": total_accuracy,
            "length_penalty": total_length,
        }
```

### 在配置中使用自定义奖励
```yaml
buffer:
  explorer_input:
    default_reward_fn_type: 'custom_reward'
    reward_fn_args:
      weight_accuracy: 2.0
      weight_length: 0.05
```

## 总结

Trinity-RFT的奖励信息传入机制具有以下特点：

1. **模块化设计**: 奖励函数可以独立定义和组合
2. **灵活配置**: 支持YAML配置和代码级别的自定义
3. **多级缓存**: 从单个Experience到批量Experiences的高效处理
4. **完整追踪**: 从奖励计算到模型更新的全链路可追踪
5. **扩展性强**: 易于添加新的奖励函数类型

这种设计让研究者可以专注于奖励函数的设计，而无需关心底层的数据流和训练逻辑。
