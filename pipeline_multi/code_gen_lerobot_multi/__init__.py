"""
Multi-Robot Code Generation Module for LeRobot

This module provides code generation capabilities for controlling multiple robots
simultaneously. It extends the single-robot code_gen_lerobot module with:
- Multi-robot coordination and synchronization
- Per-robot code generation with shared detection results
- Task-specific prompts (including towel folding)

Structure:
- forward_execution/: Forward task code generation
- reset_execution/: Reset task code generation
- prompt/: Common templates and task-specific prompts
  - common_template_prompt.py: Base classes and templates
  - task_prompts/: Individual task prompt files
"""

__version__ = "1.0.0"

# Re-export from existing code_gen_lerobot for LLM utilities
from code_gen_lerobot.llm import llm_response, detect_provider
from code_gen_lerobot.llm_utils.openai_utils import chatgpt_response

# Multi-robot specific exports
from .forward_execution.code_gen import multi_robot_code_gen, single_robot_code_gen
from .forward_execution.spec_gen import multi_robot_spec_gen
from .reset_execution.code_gen import multi_robot_reset_code_gen

# Prompt templates and task prompts
from .prompt import (
    TaskPrompt,
    ForwardTemplatePrompt,
    ResetTemplatePrompt,
    TowelFoldingPrompt,
    PickAndPlacePrompt,
    StackingPrompt,
    DoorHingeAssemblyPrompt,
)

__all__ = [
    # LLM utilities
    "llm_response",
    "detect_provider",
    "chatgpt_response",
    # Forward execution
    "multi_robot_code_gen",
    "single_robot_code_gen",
    "multi_robot_spec_gen",
    # Reset execution
    "multi_robot_reset_code_gen",
    # Prompt templates
    "TaskPrompt",
    "ForwardTemplatePrompt",
    "ResetTemplatePrompt",
    # Task prompts
    "TowelFoldingPrompt",
    "PickAndPlacePrompt",
    "StackingPrompt",
    "DoorHingeAssemblyPrompt",
]
