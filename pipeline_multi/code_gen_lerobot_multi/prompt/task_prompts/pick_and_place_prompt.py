"""
Pick and Place Task Prompt

Standard pick-and-place manipulation task.
"""

from ..common_template_prompt import TaskPrompt


class PickAndPlacePrompt(TaskPrompt):
    """물건 집어서 옮기기 태스크"""

    def __init__(self):
        super().__init__(
            name="pick_and_place",
            description="Pick up an object and place it at a target location",
            keywords=["pick", "place", "put", "move", "집", "놓", "옮"],
            skill_sequence=[
                "move_to_initial_state",
                "gripper_open",
                "move_to_position (approach pick)",
                "execute_pick_object",
                "move_to_position (lift)",
                "move_to_position (approach place)",
                "execute_place_object",
                "move_to_position (retract)",
                "move_to_initial_state",
                "move_to_free_state",
            ],
        )

    def get_forward_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        return """
**Pick and Place Task Guidelines:**

Standard pick-and-place manipulation task.

**Sequence:**
1. Open gripper
2. Move to approach position above pick object
3. Execute pick (descend, grip, save pitch)
4. Lift to approach height
5. Move to approach position above place target
6. Execute place (descend with saved pitch, release)
7. Retract

**Key Parameters:**
- approach_height: 0.15m (15cm above objects)
- gripper_offset: from detection result
- is_table: True if placing on table, False if stacking
"""

    def get_reset_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        return """
**Pick and Place Reset Guidelines:**

To reset, reverse the operation:
1. Pick the object from its current (placed) position
2. Place it back at its original position
"""

    def get_forward_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        return '''
# Pick and Place Task
# Robot ID: {robot_id}
from skills.skills_lerobot import LeRobotSkills
import numpy as np

def execute_task():
    """Pick object and place at target location."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
    )
    skills.connect()

    try:
        # ============================================================
        # Object Positions from Detection
        # ============================================================
        pick_obj = positions["object_name"]  # Replace with actual object name
        pick_pos = pick_obj["position"]
        pick_offset = pick_obj.get("gripper_offset", 0.0)

        place_obj = positions["target_name"]  # Replace with actual target name
        place_pos = place_obj["position"]

        approach_height = 0.15  # 15cm above objects

        # ============================================================
        # Execute Pick and Place
        # ============================================================
        skills.move_to_initial_state()
        skills.gripper_open()

        # Approach pick position
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=pick_offset)

        # Pick object
        skills.execute_pick_object(pick_pos, gripper_offset=pick_offset)

        # Lift
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=pick_offset)

        # Approach place position
        skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=pick_offset)

        # Place object
        skills.execute_place_object(place_pos, gripper_offset=pick_offset, is_table=True)

        # Retract
        skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=pick_offset)

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
# Pick and Place Reset Task
# Robot ID: {robot_id}
from skills.skills_lerobot import LeRobotSkills
import numpy as np

def execute_reset_task():
    """Reset: move object from current position back to original position."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
    )
    skills.connect()

    try:
        # ============================================================
        # Object Positions
        # current_positions: where objects are now (after forward execution)
        # target_positions: where objects should be (original positions)
        # ============================================================
        obj = current_positions["object_name"]  # Replace with actual object name
        current_pos = obj["position"]
        gripper_offset = obj.get("gripper_offset", 0.0)

        # Target is the original position
        target_pos = target_positions["object_name"]

        approach_height = 0.15  # 15cm above objects

        # ============================================================
        # Execute Reset (reverse pick and place)
        # ============================================================
        skills.move_to_initial_state()
        skills.gripper_open()

        # Approach current position
        skills.move_to_position([current_pos[0], current_pos[1], approach_height], gripper_offset=gripper_offset)

        # Pick from current position
        skills.execute_pick_object(current_pos, gripper_offset=gripper_offset)

        # Lift
        skills.move_to_position([current_pos[0], current_pos[1], approach_height], gripper_offset=gripper_offset)

        # Approach original position
        skills.move_to_position([target_pos[0], target_pos[1], approach_height], gripper_offset=gripper_offset)

        # Place at original position
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
