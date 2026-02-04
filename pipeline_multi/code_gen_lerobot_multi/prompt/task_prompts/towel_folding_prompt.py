"""
Towel Folding Task Prompt

Two-robot coordinated towel folding task.
"""

from ..common_template_prompt import TaskPrompt


class TowelFoldingPrompt(TaskPrompt):
    """
    수건 접기 태스크 전용 프롬프트 (Two-Robot Coordinated)

    두 로봇이 협력하여 수건을 접는 태스크:
    - 각 로봇은 자신의 base에 가까운 상호작용점을 자동 선택
    - Pick and Place 패턴으로 수건 접기 구현
    """

    def __init__(self):
        super().__init__(
            name="towel_folding",
            description="Fold a towel using two-robot coordination",
            keywords=["towel", "fold", "접", "수건"],
            skill_sequence=[
                "move_to_initial_state",
                "rotate_90degree (Robot 3 only)",
                "execute_pause_for_sync('rotate_90degree')",
                "gripper_open",
                "move_to_position (approach pick position)",
                "execute_multi_pick_object (pick front edge, sync inside)",
                "move_to_position (lift)",
                "execute_pause_for_sync('move_to_position:lift')",
                "move_to_position (approach place position)",
                "execute_multi_place_object (place at back - fold, sync inside)",
                "move_to_position (lift)",
                "move_to_initial_state",
                "move_to_free_state",
            ],
        )

    def get_forward_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        return """
**Two-Robot Coordinated Towel Folding:**

This task requires two robots working together to fold a towel.
Each robot handles the interaction point CLOSER to its base position.

**Robot Base Positions (World Frame):**
```python
ROBOT_BASES = {
    2: [0.0395, 0.2374, 0.0049],   # Robot 2: y ≈ +0.24m (positive Y side)
    3: [0.0508, -0.2515, 0.0000],  # Robot 3: y ≈ -0.25m (negative Y side)
}
```

**Interaction Points Calculation:**
Given towel center at [cx, cy, cz]:
- x_offset = 0.13m (13cm, constant offset)
- Point A: [cx + 0.13, cy - 0.07, 0]  (y = center - 7cm, z = 0, table surface)
- Point B: [cx + 0.13, cy + 0.07, 0]  (y = center + 7cm, z = 0, table surface)

**Note**: Use `0` for pick/place z-coordinate (table surface level).

**Automatic Point Assignment (IMPORTANT):**
Each robot must calculate which point is closer to its base and use that point.

```python
import numpy as np

# Robot base positions (world frame, in meters)
ROBOT_BASES = {
    2: np.array([0.0395, 0.2374, 0.0049]),
    3: np.array([0.0508, -0.2515, 0.0000]),
}

# Constant offset for pick/place positions
x_offset = 0.13  # 13cm (constant)

def get_y_offset(robot_id, cx, cy):
    '''Determine y_offset based on distance from robot base to interaction points.'''
    robot_base = ROBOT_BASES[robot_id]

    # Two candidate interaction points (XY only for distance calculation)
    point_A = np.array([cx + x_offset, cy - 0.07])  # y_offset = -0.07
    point_B = np.array([cx + x_offset, cy + 0.07])  # y_offset = +0.07

    # Calculate XY distance to each point
    dist_A = np.linalg.norm(point_A - robot_base[:2])
    dist_B = np.linalg.norm(point_B - robot_base[:2])

    # Return y_offset for the CLOSER point
    return -0.07 if dist_A < dist_B else +0.07
```

**Pick and Place Pattern for Folding:**
- **Pick Position** (front edge): [cx + 0.13, cy + y_offset, -0.01]  (z = -1cm, below table surface)
- **Place Position** (back - fold): [cx - 0.13, cy + y_offset, 0.02]  (z = +2cm, above table surface)

**Fold Motion Sequence:**
1. Move to initial state
2. Move to approach height above pick position
3. Open gripper
4. Execute pick at front edge (descend to 0, grip)
5. Lift to approach height
6. Move directly to place position (back edge)
7. Execute place (descend to 0, release) - creates fold
8. Lift and return to initial state

**Key Parameters:**
- approach_height = 0.15m (15cm above towel)
- x_offset = 0.13m (13cm, constant offset from center)
- y_offset = ±0.07m (7cm left/right from center, auto-selected)
- pick_z = -0.01m (-1cm, below table surface for picking)
- place_z = 0.02m (+2cm, above table surface for placing)
- gripper_offset = 0.0 (towel is thin)
- is_table = True (placing on table surface)

**Robot-Specific Setup (CRITICAL - Synchronization):**

⚠️ **USE execute_multi_pick_object and execute_multi_place_object** ⚠️

These multi-robot functions have sync_barrier.wait() calls INSIDE them,
ensuring both robots close/open grippers at exactly the same moment.

**Sync points using execute_pause_for_sync:**
- **rotate_90degree**: After Robot 3 completes rotation (Robot 2 waits here)
- **move_to_position:lift**: After pick lift, before place approach
- **pick_ready**: Handled INSIDE execute_multi_pick_object (before gripper_close)
- **place_ready**: Handled INSIDE execute_multi_place_object (before gripper_open)

**Robot Setup:**
- **Robot 3**: rotate_90degree(-1) TWICE, then execute_pause_for_sync
- **Robot 2**: move_to_initial_state, then execute_pause_for_sync (wait for Robot 3)
- Use `execute_multi_pick_object(skills, sync_barrier, pick_pos, gripper_offset=0.0)`
- Use `execute_multi_place_object(skills, sync_barrier, place_pos, gripper_offset=0.0, is_table=True, gripper_open_ratio=0.3)`
- Use `execute_pause_for_sync(skills, sync_barrier, "skill_name")` for explicit sync
- The `sync_barrier` object is pre-injected and available globally
"""

    def get_reset_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        return """
**Towel Reset (Unfold) Guidelines:**

After folding, the towel needs to be unfolded to reset.
This is the REVERSE of the fold operation.

**Reset Motion (with x_offset = 0.13m constant):**
- **Pick Position** (folded edge): [cx - 0.13, cy + y_offset, 0]  (z = 0, table surface)
- **Place Position** (original edge): [cx + 0.13, cy + y_offset, 0]  (z = 0, table surface)

**Automatic y_offset Selection:**
Use the same distance-based calculation as forward execution:
```python
ROBOT_BASES = {
    2: np.array([0.0395, 0.2374, 0.0049]),
    3: np.array([0.0508, -0.2515, 0.0000]),
}

# Constant offset for pick/place positions
x_offset = 0.13  # 13cm (constant)

def get_y_offset(robot_id, cx, cy):
    robot_base = ROBOT_BASES[robot_id]
    point_A = np.array([cx - x_offset, cy - 0.07])
    point_B = np.array([cx - x_offset, cy + 0.07])
    dist_A = np.linalg.norm(point_A - robot_base[:2])
    dist_B = np.linalg.norm(point_B - robot_base[:2])
    return -0.07 if dist_A < dist_B else +0.07
```

**Reset Sequence:**
1. Move to initial state
2. Move to approach height above folded edge (back)
3. Open gripper
4. Execute pick at folded edge
5. Lift to approach height
6. Move to original edge position (front)
7. Execute place (unfolds the towel)
8. Return to initial state
"""

    def get_forward_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        return '''
# Two-Robot Coordinated Towel Folding
# Robot ID: {robot_id}
from skills.skills_lerobot import LeRobotSkills
from pipeline_multi.multi_skills import execute_multi_pick_object, execute_multi_place_object, execute_pause_for_sync
import numpy as np

def execute_task():
    """Fold the towel by picking front edge and placing at back."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
    )
    skills.connect()

    try:
        # ============================================================
        # Robot Base Positions (World Frame)
        # ============================================================
        ROBOT_BASES = {
            2: np.array([0.0395, 0.2374, 0.0049]),   # y ≈ +0.24m
            3: np.array([0.0508, -0.2515, 0.0000]),  # y ≈ -0.25m
        }

        # ============================================================
        # Towel Center from Detection
        # ============================================================
        towel = positions["green towel"]
        cx, cy, cz = towel["position"]
        approach_height = 0.15

        # Constant offset for pick/place positions
        x_offset = 0.13  # 13cm (constant)

        # ============================================================
        # Auto-select y_offset based on distance to robot base
        # ============================================================
        robot_id = {robot_id}
        robot_base = ROBOT_BASES[robot_id]

        # Two candidate interaction points (front edge)
        point_A = np.array([cx + x_offset, cy - 0.07])  # y_offset = -0.07
        point_B = np.array([cx + x_offset, cy + 0.07])  # y_offset = +0.07

        # Calculate distance and choose closer point
        dist_A = np.linalg.norm(point_A - robot_base[:2])
        dist_B = np.linalg.norm(point_B - robot_base[:2])
        y_offset = -0.07 if dist_A < dist_B else +0.07

        # ============================================================
        # Calculate Pick and Place Positions
        # ============================================================
        # Pick: front edge of towel (z = -1cm, below table)
        pick_pos = [cx + x_offset, cy + y_offset, -0.01]

        # Place: back edge for folding (z = 2*cz, twice towel height)
        place_pos = [cx - x_offset, cy + y_offset, 2*cz]

        # ============================================================
        # Execute Fold Motion
        # rotate_90degree: After rotation (Robot 2 waits for Robot 3)
        # move_to_position:lift: After pick lift, before place approach
        # pick_ready/place_ready: Handled INSIDE multi functions
        # ============================================================
        skills.move_to_initial_state()

        # Robot 3 ONLY: Rotate first, then sync
        if robot_id == 3:
            skills.rotate_90degree(-1)
            skills.rotate_90degree(-1)

        execute_pause_for_sync(skills, sync_barrier, "rotate_90degree")  # After rotation
        skills.gripper_open()
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height])
        execute_multi_pick_object(skills, sync_barrier, pick_pos, gripper_offset=0.0)
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height])
        execute_pause_for_sync(skills, sync_barrier, "move_to_position:lift")  # After pick lift
        skills.move_to_position([place_pos[0], place_pos[1], approach_height])
        execute_multi_place_object(skills, sync_barrier, place_pos, gripper_offset=0.0, is_table=True, gripper_open_ratio=0.3)
        skills.move_to_position([place_pos[0], place_pos[1], approach_height])

        # === Cleanup ===
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
'''

    def get_reset_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        return '''
# Two-Robot Coordinated Towel Reset (Unfold)
# Robot ID: {robot_id}
from skills.skills_lerobot import LeRobotSkills
import numpy as np

def execute_reset_task():
    """Unfold the towel by picking folded edge and placing at original position."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
    )
    skills.connect()

    try:
        # ============================================================
        # Robot Base Positions (World Frame)
        # ============================================================
        ROBOT_BASES = {
            2: np.array([0.0395, 0.2374, 0.0049]),   # y ≈ +0.24m
            3: np.array([0.0508, -0.2515, 0.0000]),  # y ≈ -0.25m
        }

        # ============================================================
        # Towel Center from Detection (current folded state)
        # ============================================================
        towel = current_positions["green towel"]
        cx, cy, cz = towel["position"]
        approach_height = 0.15

        # Constant offset for pick/place positions
        x_offset = 0.13  # 13cm (constant)

        # ============================================================
        # Auto-select y_offset based on distance to robot base
        # ============================================================
        robot_id = {robot_id}
        robot_base = ROBOT_BASES[robot_id]

        # Two candidate interaction points (at folded edge position)
        point_A = np.array([cx - x_offset, cy - 0.08])  # y_offset = -0.08
        point_B = np.array([cx - x_offset, cy + 0.08])  # y_offset = +0.08

        # Calculate distance and choose closer point
        dist_A = np.linalg.norm(point_A - robot_base[:2])
        dist_B = np.linalg.norm(point_B - robot_base[:2])
        y_offset = -0.08 if dist_A < dist_B else +0.08

        # ============================================================
        # Calculate Pick and Place Positions for Reset
        # ============================================================
        # Pick: folded edge (back) at z = 0 (table surface)
        pick_pos = [cx - x_offset, cy + y_offset, 0]

        # Place: original edge (front) at z = 0 (table surface)
        place_pos = [cx + x_offset, cy + y_offset, 0]

        # ============================================================
        # Execute Unfold Motion
        # ============================================================
        skills.move_to_initial_state()
        skills.gripper_open()
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height])
        skills.execute_pick_object(pick_pos, gripper_offset=0.0)
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height])
        skills.move_to_position([place_pos[0], place_pos[1], approach_height])
        skills.execute_place_object(place_pos, gripper_offset=0.0, is_table=True)
        skills.move_to_position([place_pos[0], place_pos[1], approach_height])

        # === Cleanup ===
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_reset_task()
'''
