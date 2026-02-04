"""
Stacking Task Prompt

Stack one object on top of another.
"""

from ..common_template_prompt import TaskPrompt


class StackingPrompt(TaskPrompt):
    """물건 쌓기 태스크"""

    def __init__(self):
        super().__init__(
            name="stacking",
            description="Stack one object on top of another",
            keywords=["stack", "top", "on", "쌓", "위에"],
            skill_sequence=[
                "move_to_initial_state",
                "gripper_open",
                "move_to_position (approach top object)",
                "execute_pick_object (top object)",
                "move_to_position (lift)",
                "move_to_position (approach bottom object)",
                "execute_place_object (is_table=False)",
                "move_to_position (retract)",
                "move_to_initial_state",
                "move_to_free_state",
            ],
        )

    def get_forward_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        return """
**Stacking Task Guidelines:**

Place one object on top of another.

**Key Difference from Pick-and-Place:**
- Use `is_table=False` when placing on another object
- The place height = target object's height (z-coordinate)

**Sequence:**
1. Pick the object to be stacked
2. Move above the base object
3. Place with is_table=False (descends to base object's height)
"""

    def get_reset_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        return """
**Stacking Reset Guidelines:**

To reset (unstack):
1. Pick the stacked object from on top of the base
2. Place it back at its original position on the table
3. Use is_table=True since the original position is on the table
"""

    def get_forward_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        return '''
# Stacking Task
# Robot ID: {robot_id}
from skills.skills_lerobot import LeRobotSkills
import numpy as np

def execute_task():
    """Stack one object on top of another."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
    )
    skills.connect()

    try:
        # ============================================================
        # Object Positions from Detection
        # ============================================================
        # Object to be picked and stacked
        top_obj = positions["top_object_name"]  # Replace with actual object name
        top_pos = top_obj["position"]
        top_offset = top_obj.get("gripper_offset", 0.0)

        # Base object (stack target)
        base_obj = positions["base_object_name"]  # Replace with actual target name
        base_pos = base_obj["position"]

        approach_height = 0.15  # 15cm above objects

        # ============================================================
        # Execute Stacking
        # ============================================================
        skills.move_to_initial_state()
        skills.gripper_open()

        # Approach top object
        skills.move_to_position([top_pos[0], top_pos[1], approach_height], gripper_offset=top_offset)

        # Pick top object
        skills.execute_pick_object(top_pos, gripper_offset=top_offset)

        # Lift
        skills.move_to_position([top_pos[0], top_pos[1], approach_height], gripper_offset=top_offset)

        # Approach base object (stack target)
        skills.move_to_position([base_pos[0], base_pos[1], approach_height], gripper_offset=top_offset)

        # Place on top of base object (is_table=False)
        # Place position uses base object's x,y and base object's height as z
        stack_pos = [base_pos[0], base_pos[1], base_pos[2]]
        skills.execute_place_object(stack_pos, gripper_offset=top_offset, is_table=False)

        # Retract
        skills.move_to_position([base_pos[0], base_pos[1], approach_height], gripper_offset=top_offset)

        # Cleanup
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
'''

    def get_reset_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        return '''
# Stacking Reset Task (Unstack)
# Robot ID: {robot_id}
from skills.skills_lerobot import LeRobotSkills
import numpy as np

def execute_reset_task():
    """Reset: unstack object and return to original position."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
    )
    skills.connect()

    try:
        # ============================================================
        # Object Positions
        # current_positions: where objects are now (stacked)
        # target_positions: where objects should be (original unstacked)
        # ============================================================
        # The stacked object is now on top of base
        stacked_obj = current_positions["top_object_name"]  # Replace with actual name
        stacked_pos = stacked_obj["position"]
        gripper_offset = stacked_obj.get("gripper_offset", 0.0)

        # Target is the original position on the table
        target_pos = target_positions["top_object_name"]

        approach_height = 0.15  # 15cm above objects

        # ============================================================
        # Execute Unstacking (pick from stack, place on table)
        # ============================================================
        skills.move_to_initial_state()
        skills.gripper_open()

        # Approach stacked position
        skills.move_to_position([stacked_pos[0], stacked_pos[1], approach_height], gripper_offset=gripper_offset)

        # Pick from stack (is_table=False since it's on another object)
        skills.execute_pick_object(stacked_pos, gripper_offset=gripper_offset)

        # Lift
        skills.move_to_position([stacked_pos[0], stacked_pos[1], approach_height], gripper_offset=gripper_offset)

        # Approach original position
        skills.move_to_position([target_pos[0], target_pos[1], approach_height], gripper_offset=gripper_offset)

        # Place on table at original position
        skills.execute_place_object(target_pos, gripper_offset=gripper_offset, is_table=True)

        # Retract
        skills.move_to_position([target_pos[0], target_pos[1], approach_height], gripper_offset=gripper_offset)

        # Cleanup
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_reset_task()
'''
