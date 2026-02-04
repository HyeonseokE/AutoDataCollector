"""
Task-Specific Prompts Module

Contains:
- common_template_prompt: Base classes and template generators
- task_prompts/: Individual task prompt files
"""

from .common_template_prompt import (
    TaskPrompt,
    ForwardTemplatePrompt,
    ResetTemplatePrompt,
)

from .task_prompts import (
    TowelFoldingPrompt,
    PickAndPlacePrompt,
    StackingPrompt,
    DoorHingeAssemblyPrompt,
)

__all__ = [
    # Base classes
    "TaskPrompt",
    "ForwardTemplatePrompt",
    "ResetTemplatePrompt",
    # Task prompts
    "TowelFoldingPrompt",
    "PickAndPlacePrompt",
    "StackingPrompt",
    "DoorHingeAssemblyPrompt",
]
