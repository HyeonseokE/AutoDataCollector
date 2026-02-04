"""
Reset Execution Module

Reset execution pipeline for CaP-based robot control.
Uses object detection to find current positions and restores objects to original positions.

Flow:
1. Load original_positions from saved execution context
2. Use detection to get current_positions (where objects are NOW)
3. Generate reset code: pick from current → place at original/random
4. Execute reset to restore pre-task state or shuffle to new positions

Reset Modes:
- "original": Restore objects to their initial positions (default)
- "random": Shuffle objects to random positions within workspace
"""

from .code_gen import (
    lerobot_reset_code_gen,
    lerobot_reset_code_gen_from_context,
    extract_code_from_response,
)
from .prompt import (
    lerobot_reset_code_gen_prompt,
    lerobot_reset_spec_gen_prompt,
)
from .workspace import (
    ResetWorkspace,
    is_grippable,
    classify_objects,
    generate_random_positions,
    GRIPPER_MAX_OPEN_WIDTH,
)

__all__ = [
    # Code generation
    "lerobot_reset_code_gen",
    "lerobot_reset_code_gen_from_context",
    "extract_code_from_response",
    # Prompts
    "lerobot_reset_code_gen_prompt",
    "lerobot_reset_spec_gen_prompt",
    # Workspace
    "ResetWorkspace",
    "is_grippable",
    "classify_objects",
    "generate_random_positions",
    "GRIPPER_MAX_OPEN_WIDTH",
]
