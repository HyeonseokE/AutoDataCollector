"""
Task-Specific Prompts for Multi-Robot Code Generation

Each task has its own prompt file with forward/reset context and example codes.
"""

from .towel_folding_prompt import TowelFoldingPrompt
from .pick_and_place_prompt import PickAndPlacePrompt
from .stacking_prompt import StackingPrompt
from .door_hinge_assembly_prompt import DoorHingeAssemblyPrompt

__all__ = [
    "TowelFoldingPrompt",
    "PickAndPlacePrompt",
    "StackingPrompt",
    "DoorHingeAssemblyPrompt",
]
