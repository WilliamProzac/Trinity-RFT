#!/usr/bin/env python3
"""
初始化自定义组件，确保Trinity-RFT能够加载自定义奖励函数和工作流
"""

import sys
from pathlib import Path

# 确保当前目录在Python路径中
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

# 导入自定义组件以触发注册
try:
    from . import custom_workflow
    from . import reward  # 导入奖励函数模块
    print("✅ 自定义奖励函数和工作流加载成功")
    print("   - all_reward: 三重奖励函数")
    print("   - custom_workflow: 自定义工作流")
except Exception as e:
    print(f"❌ 加载自定义组件时出错: {e}")

__all__ = ['reward', 'custom_workflow']
