# -*- coding: utf-8 -*-
"""Workflow module"""
from .customized_math_workflows import MathBoxedWorkflow
from .customized_toolcall_workflows import ToolCallWorkflow
from .envs.agentscope.agentscope_react_workflow import AgentScopeReactV2MathWorkflow
from .envs.alfworld.alfworld_workflow import AlfworldWorkflow, StepWiseAlfworldWorkflow
from .envs.alfworld.RAFT_alfworld_workflow import RAFTAlfworldWorkflow
from .envs.alfworld.RAFT_reflect_alfworld_workflow import RAFTReflectAlfworldWorkflow
from .envs.email_searcher.workflow import EmailSearchWorkflow
from .envs.sciworld.sciworld_workflow import SciWorldWorkflow
from .envs.webshop.webshop_workflow import WebShopWorkflow
from .eval_workflow import MathEvalWorkflow
from .math_rm_workflow import MathRMWorkflow
from .math_ruler_workflow import MathRULERWorkflow
from .simple_mm_workflow import SimpleMMWorkflow
from .workflow import WORKFLOWS, MathWorkflow, SimpleWorkflow, Task, Workflow
# Import custom workflow from rl directory
import sys
from pathlib import Path
rl_path = Path(__file__).parent.parent.parent.parent / "rl"
if str(rl_path) not in sys.path:
    sys.path.append(str(rl_path))
try:
    from custom_workflow import CustomWorkflow
except ImportError as e:
    print(f"Warning: Could not import CustomWorkflow: {e}")
    CustomWorkflow = None
except Exception as e:
    # 捕获重复注册等其他错误
    print(f"Warning: CustomWorkflow import/registration error: {e}")
    try:
        # 尝试直接获取已注册的workflow
        from trinity.common.workflows.workflow import WORKFLOWS
        CustomWorkflow = WORKFLOWS.get('custom_workflow')
        if CustomWorkflow is not None:
            print("✅ 成功获取已注册的CustomWorkflow")
    except:
        CustomWorkflow = None

__all__ = [
    "Task",
    "Workflow",
    "WORKFLOWS",
    "SimpleWorkflow",
    "MathWorkflow",
    "CustomWorkflow",
    "WebShopWorkflow",
    "AlfworldWorkflow",
    "StepWiseAlfworldWorkflow",
    "RAFTAlfworldWorkflow",
    "RAFTReflectAlfworldWorkflow",
    "SciWorldWorkflow",
    "MathBoxedWorkflow",
    "MathRMWorkflow",
    "ToolCallWorkflow",
    "MathEvalWorkflow",
    "AgentScopeReactV2MathWorkflow",
    "EmailSearchWorkflow",
    "MathRULERWorkflow",
    "SimpleMMWorkflow",
]
