"""
Door Hinge Assembly Task Prompt

Two-robot coordinated door hinge assembly task.
"""

from ..common_template_prompt import TaskPrompt


class DoorHingeAssemblyPrompt(TaskPrompt):
    """
    Door Hinge 조립 태스크 (Two-Robot Coordinated)

    두 로봇이 협력하여 두 개의 door hinge를 조립하는 태스크:
    - 각 로봇은 자신의 base에 가까운 hinge를 자동 선택
    - 한 로봇이 hinge를 고정하고, 다른 로봇이 끼움
    - 동기화하여 정밀한 조립 수행
    """

    def __init__(self):
        super().__init__(
            name="door_hinge_assembly",
            description="Assemble two door hinges using two-robot coordination",
            keywords=["hinge", "assembly", "assemble", "door", "조립", "힌지", "경첩"],
            skill_sequence=[
                "move_to_initial_state",
                "gripper_open",
                "move_to_position (approach hinge)",
                "execute_multi_pick_object (pick hinge, sync inside)",
                "move_to_position (lift)",
                "move_to_position (approach assembly point)",
                "execute_pause_for_sync('move_to_position:assembly')",
                "# Holder robot: stay fixed",
                "# Inserter robot: move to insert",
                "execute_pause_for_sync('move_to_position:insert_approach')",
                "execute_place_object (release)",
                "execute_pause_for_sync('execute_place_object')",
                "move_to_position (retract)",
                "move_to_initial_state",
                "move_to_free_state",
            ],
        )

    def get_tcp_offset(self):
        """Return task-specific TCP offset for door hinge assembly (2cm)"""
        return [-0.02, 0.0, 0.0]

    def get_forward_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        role_info = ""
        if assigned_role:
            role_info = f"""
**⚠️ YOUR ASSIGNED ROLE: {assigned_role.upper()}**
You are Robot {robot_id} and your role is **{assigned_role.upper()}**.
"""
        return f"""{role_info}
**Two-Robot Coordinated Door Hinge Assembly:**

This task requires two robots working together to assemble two door hinges.
Each robot handles the hinge CLOSER to its base position.
One robot holds the hinge fixed (holder), the other inserts (inserter).

**Robot Base Positions (World Frame):**
```python
ROBOT_BASES = {{
    2: [0.0395, 0.2374, 0.0049],   # Robot 2: y ≈ +0.24m (positive Y side)
    3: [0.0508, -0.2515, 0.0000],  # Robot 3: y ≈ -0.25m (negative Y side)
}}
```

**Object Detection & Role Assignment (COLOR-BASED):**
- **"red door hinge"**: **HOLDER** - This hinge is held fixed in place. The robot gripping this hinge must stay stationary during assembly.
- **"gray circle of the pink part"**: **INSERTER** - Pick at gray circle (pin) position for precise alignment. The robot gripping here moves toward the holder.

**⚠️ CRITICAL: INSERTER picks at "gray circle of the pink part" for precise pin alignment!**

**Hinge Assignment:**
- Your role (holder/inserter) is already assigned - check the ASSIGNED ROLE section above
- Red hinge → HOLDER role (pick at center)
- Pink hinge → INSERTER role (pick at **gray circle** position for pin alignment)

**Role Behavior:**
- **Holder (red hinge)**: Pick hinge → Lift → **MOVE TO MIDPOINT** → Wait for insertion → Release
- **Inserter (gray circle)**: Pick at gray circle → Lift → **DETECT HOLE POSITION** → Move to approach → Descend → Insert → Release

**Assembly Point Definition:**
- **Holder**: Uses pre-calculated MIDPOINT between both hinges
- **Inserter**: Dynamically detects "hole of the red part" after holder is in position

```python
# HOLDER: Calculate assembly point as midpoint
assembly_point = [
    (red_pos[0] + pink_pos[0]) / 2,
    (red_pos[1] + pink_pos[1]) / 2,
    approach_height
]

# INSERTER: Detect hole position dynamically after sync
hole_detection = skills.detect_objects(queries=["hole of the red part"], timeout=5.0)
hole_pos = hole_detection["hole of the red part"]["position"]
assembly_point = [hole_pos[0], hole_pos[1], hole_pos[2]]  # 실제 hole 위치
assembly_approach = [hole_pos[0], hole_pos[1], approach_height + 0.10]  # approach 높이 + 10cm
```

**Assembly Motion Sequence:**

1. **Both robots**: Move to initial state
2. **Both robots**: gripper_open() ← ⚠️ REQUIRED before pick!
3. **Holder**: Pick red hinge at center / **Inserter**: Pick pink hinge at **gray circle** position
4. **Both robots**: Lift to approach_height
5. **Holder robot**: Move to midpoint (assembly_point)
6. **Sync**: assembly_ready (holder is now in position)
7. **Inserter robot**: Detect "hole of the red part" → Move to assembly_approach (approach_height + 10cm)
8. **Sync**: insert_approach_ready
9. **Inserter robot**: execute_place_object (descend to hole_pos[2] + release)
10. **Holder robot**: STAY FIXED until execute_place_object sync, then gripper_open
11. **Both robots**: Retract and return

**Key Parameters:**
- approach_height (holder) = 0.05m (5cm - lower for stable holding)
- approach_height (inserter) = 0.15m (15cm - higher for approach)
- assembly_point = [hole_pos[0], hole_pos[1], hole_pos[2]] (실제 hole 위치)
- assembly_approach = [hole_pos[0], hole_pos[1], approach_height + 0.10] (approach_height + 10cm)
- gripper_offset = from detection (hinge thickness)

**Synchronization Points (using execute_pause_for_sync):**
- **pick_ready**: Inside execute_multi_pick_object (before gripper close)
- **move_to_position:assembly**: Both lifted, holder at assembly point
- **move_to_position:insert_approach**: Inserter at approach position (approach_height + 10cm)
- **execute_place_object**: Inserter finished, holder can release

**Robot-Specific Behavior:**

⚠️ **IMPORTANT**: Your role is pre-assigned. Do NOT use `if role ==` checks. Just execute your role's code directly!

**HOLDER Role** (approach_height = 0.05):
```python
skills.move_to_initial_state()
skills.gripper_open()
skills.move_to_position([pick_pos[0], pick_pos[1], 0.05], gripper_offset=...)
execute_multi_pick_object(skills, sync_barrier, pick_pos, gripper_offset=...)
skills.move_to_position([pick_pos[0], pick_pos[1], 0.05], gripper_offset=...)
# Move to midpoint
assembly_point = [(red_pos[0]+pink_pos[0])/2, (red_pos[1]+pink_pos[1])/2, 0.05]
skills.move_to_position(assembly_point, gripper_offset=...)
execute_pause_for_sync(skills, sync_barrier, "move_to_position:assembly")
execute_pause_for_sync(skills, sync_barrier, "move_to_position:insert_approach")  # STAY FIXED
execute_pause_for_sync(skills, sync_barrier, "execute_place_object")  # STAY FIXED
skills.gripper_open()  # Release after insertion
```

**INSERTER Role** (approach_height = 0.15):
```python
skills.move_to_initial_state()
skills.gripper_open()
skills.move_to_position([pick_pos[0], pick_pos[1], 0.15], gripper_offset=...)
execute_multi_pick_object(skills, sync_barrier, pick_pos, gripper_offset=...)
skills.move_to_position([pick_pos[0], pick_pos[1], 0.15], gripper_offset=...)
execute_pause_for_sync(skills, sync_barrier, "move_to_position:assembly")  # Wait for holder
# DYNAMICALLY DETECT hole position
hole_detection = skills.detect_objects(queries=["hole of the red part"], timeout=5.0)
hole_pos = hole_detection["hole of the red part"]["position"]
assembly_point = [hole_pos[0], hole_pos[1], hole_pos[2]]
assembly_approach = [hole_pos[0], hole_pos[1], 0.15]
skills.move_to_position(assembly_approach, gripper_offset=...)
execute_pause_for_sync(skills, sync_barrier, "move_to_position:insert_approach")
skills.execute_place_object(assembly_point, gripper_offset=..., is_table=False)
execute_pause_for_sync(skills, sync_barrier, "execute_place_object")
```
"""

    def get_reset_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        return """
**Door Hinge Assembly Reset (Disassemble) Guidelines:**

To reset, the assembled hinges need to be separated and returned to original positions.

**Reset Motion:**
1. Both robots pick the assembled hinge (at assembly point)
2. Inserter robot pulls back to separate
3. Both robots return hinges to original positions
4. Release on table

**Original Position Tracking:**
The original positions of each hinge should be saved during forward execution
or re-detected before reset.

**Reset Sequence:**
1. Move to initial state
2. Move to assembly point
3. Both pick the assembly
4. Sync and separate (inserter pulls back)
5. Each robot returns hinge to original position
6. Place on table
7. Return to initial state
"""

    def get_forward_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        """
        Return role-specific example code.

        Args:
            robot_id: Robot ID (2 or 3)
            assigned_role: "holder" or "inserter"
        """
        if assigned_role == "holder":
            return self._get_holder_example_code()
        elif assigned_role == "inserter":
            return self._get_inserter_example_code()
        else:
            # Fallback: return generic code with runtime role detection
            return self._get_generic_example_code()

    def _get_holder_example_code(self) -> str:
        """Example code for HOLDER role (red hinge - stays fixed)"""
        return '''
# Door Hinge Assembly - HOLDER Role (Red Hinge)
# Robot ID: {robot_id}
# Role: HOLDER - Pick red hinge, move to assembly point, STAY FIXED
from skills.skills_lerobot import LeRobotSkills
from pipeline_multi.multi_skills import execute_multi_pick_object, execute_pause_for_sync
import numpy as np

def execute_task():
    """HOLDER: Pick red hinge, hold fixed at assembly point while inserter inserts."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
        tcp_offset=[-0.02, 0.0, 0.0],
    )
    skills.connect()

    try:
        # ============================================================
        # ⚠️ HOLDER_APPROACH_HEIGHT = 0.05 (5cm) - DO NOT USE 0.15!
        # ============================================================
        HOLDER_APPROACH_HEIGHT = 0.05  # MUST be 0.05, NOT 0.15!

        # Object Positions
        red_part = positions["red part"]
        pink_part = positions["pink part"]
        red_pos = red_part["position"]
        pink_pos = pink_part["position"]

        # HOLDER picks RED PART
        my_hinge_pos = red_pos
        my_gripper_offset = red_part.get("gripper_offset", 0.0)

        # Assembly point at LOW height (0.05m)
        assembly_point = [
            (red_pos[0] + pink_pos[0]) / 2,
            (red_pos[1] + pink_pos[1]) / 2,
            HOLDER_APPROACH_HEIGHT  # 0.05m
        ]

        # Execute
        skills.move_to_initial_state()
        skills.gripper_open()
        skills.move_to_position([my_hinge_pos[0], my_hinge_pos[1], HOLDER_APPROACH_HEIGHT], gripper_offset=my_gripper_offset)
        execute_multi_pick_object(skills, sync_barrier, my_hinge_pos, gripper_offset=my_gripper_offset)
        skills.move_to_position([my_hinge_pos[0], my_hinge_pos[1], HOLDER_APPROACH_HEIGHT], gripper_offset=my_gripper_offset)

        # Move to assembly point and sync
        skills.move_to_position(assembly_point, gripper_offset=my_gripper_offset)
        execute_pause_for_sync(skills, sync_barrier, "move_to_position:assembly")

        # STAY FIXED while inserter approaches
        execute_pause_for_sync(skills, sync_barrier, "move_to_position:insert_approach")

        # STAY FIXED until insertion complete
        execute_pause_for_sync(skills, sync_barrier, "execute_place_object")

        # Release
        skills.gripper_open()

        # Retract
        skills.move_to_position([assembly_point[0], assembly_point[1], HOLDER_APPROACH_HEIGHT + 0.05], gripper_offset=my_gripper_offset)
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
'''

    def _get_inserter_example_code(self) -> str:
        """Example code for INSERTER role (pink hinge - moves toward holder)"""
        return '''
# Door Hinge Assembly - INSERTER Role (Pink Hinge)
# Robot ID: {robot_id}
# Role: INSERTER - Pick at pink_part CENTER, use tcp_offset_override for pin alignment
from skills.skills_lerobot import LeRobotSkills
from pipeline_multi.multi_skills import execute_multi_pick_object, execute_pause_for_sync, compute_tcp_offset_for_pin
import numpy as np

def execute_task():
    """INSERTER: Pick at pink_part CENTER (NOT gray_circle), use tcp_offset_override."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
        tcp_offset=[-0.02, 0.0, 0.0],
    )
    skills.connect()

    try:
        INSERTER_APPROACH_HEIGHT = 0.15  # Inserter uses 0.15m

        # ============================================================
        # ⚠️ CRITICAL: Pick at pink_part["position"], NOT gray_circle!
        # gray_circle is ONLY used for calculating pin_offset
        # ============================================================
        pink_part = positions["pink part"]
        gray_circle = positions["gray circle of the pink part"]

        # Pick position = pink_part CENTER (stable grip)
        PICK_POSITION = pink_part["position"]  # ⚠️ Use pink_part, NOT gray_circle!
        PIN_POSITION = gray_circle["position"]  # For tcp_offset calculation

        my_gripper_offset = pink_part.get("gripper_offset", 0.045)

        # Execute
        skills.move_to_initial_state()
        skills.gripper_open()

        # ⚠️ Pick at PICK_POSITION (pink_part center), NOT gray_circle!
        skills.move_to_position([PICK_POSITION[0], PICK_POSITION[1], INSERTER_APPROACH_HEIGHT], gripper_offset=my_gripper_offset)
        execute_multi_pick_object(skills, sync_barrier, PICK_POSITION, gripper_offset=my_gripper_offset)
        skills.move_to_position([PICK_POSITION[0], PICK_POSITION[1], INSERTER_APPROACH_HEIGHT], gripper_offset=my_gripper_offset)

        # ⚠️ Compute tcp_offset for PIN alignment (World→Local conversion)
        # Must be called AFTER execute_multi_pick_object (uses saved gripper rotation)
        assembly_tcp_offset = compute_tcp_offset_for_pin(
            skills,
            pin_position_world=PIN_POSITION,
            pick_position_world=PICK_POSITION,
        )

        # Sync with holder
        execute_pause_for_sync(skills, sync_barrier, "move_to_position:assembly")

        # Detect hole position
        hole_detection = skills.detect_objects(
            queries=["hole of the red part"],
            timeout=15.0,
            visualize=False
        )
        hole_pos = hole_detection["hole of the red part"]["position"]

        assembly_point = [hole_pos[0], hole_pos[1], hole_pos[2]]
        assembly_approach = [hole_pos[0], hole_pos[1], INSERTER_APPROACH_HEIGHT]

        # Move with tcp_offset_override (pin aligns with hole), maintaining pitch from pick
        skills.move_to_position(
            assembly_approach,
            gripper_offset=my_gripper_offset,
            tcp_offset_override=assembly_tcp_offset,
            target_pitch=skills._saved_pitch  # Pitch saved by execute_multi_pick_object()
        )
        execute_pause_for_sync(skills, sync_barrier, "move_to_position:insert_approach")

        # Descend and release
        skills.execute_place_object(
            assembly_point,
            gripper_offset=my_gripper_offset,
            is_table=False,
            tcp_offset_override=assembly_tcp_offset
        )

        execute_pause_for_sync(skills, sync_barrier, "execute_place_object")

        # Retract
        skills.move_to_position([assembly_point[0], assembly_point[1], INSERTER_APPROACH_HEIGHT + 0.05], gripper_offset=my_gripper_offset)
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
'''

    def _get_generic_example_code(self) -> str:
        """Generic example code with runtime role detection (fallback)"""
        return '''
# Two-Robot Coordinated Door Hinge Assembly
# Robot ID: {robot_id}
from skills.skills_lerobot import LeRobotSkills
from pipeline_multi.multi_skills import execute_multi_pick_object, execute_pause_for_sync
import numpy as np

def execute_task():
    """Assemble door hinges - role determined at runtime based on distance."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
        tcp_offset=[-0.02, 0.0, 0.0],
    )
    skills.connect()

    try:
        # ⚠️ Different heights for each role!
        HOLDER_HEIGHT = 0.05   # Holder: 5cm (LOW)
        INSERTER_HEIGHT = 0.15 # Inserter: 15cm (HIGH)

        ROBOT_BASES = {
            2: np.array([0.0395, 0.2374, 0.0049]),
            3: np.array([0.0508, -0.2515, 0.0000]),
        }

        red_part = positions["red part"]
        pink_part = positions["pink part"]
        gray_circle = positions["gray circle of the pink part"]
        red_pos = red_part["position"]
        pink_center_pos = pink_part["position"]
        pin_pos = gray_circle["position"]

        robot_id = {robot_id}
        robot_base = ROBOT_BASES[robot_id]
        dist_to_red = np.linalg.norm(np.array(red_pos[:2]) - robot_base[:2])
        dist_to_pink = np.linalg.norm(np.array(pink_center_pos[:2]) - robot_base[:2])

        if dist_to_red < dist_to_pink:
            # HOLDER
            my_pick_pos = red_pos
            my_gripper_offset = red_part.get("gripper_offset", 0.0)
            role = "holder"
            approach_height = HOLDER_HEIGHT  # 0.05m
            assembly_tcp_offset = None
        else:
            # INSERTER: pick at pink_part CENTER, NOT gray_circle!
            my_pick_pos = pink_center_pos  # ⚠️ pink_part, NOT gray_circle!
            my_gripper_offset = pink_part.get("gripper_offset", 0.045)
            role = "inserter"
            approach_height = INSERTER_HEIGHT  # 0.15m

            # Calculate pin_offset
            pin_offset_x = pin_pos[0] - pink_center_pos[0]
            pin_offset_y = pin_pos[1] - pink_center_pos[1]
            pin_offset_z = pin_pos[2] - pink_center_pos[2]

            assembly_tcp_offset = [
                -0.02 + pin_offset_x,
                0.0 + pin_offset_y,
                0.0 + pin_offset_z
            ]

        skills.move_to_initial_state()
        skills.gripper_open()
        skills.move_to_position([my_pick_pos[0], my_pick_pos[1], approach_height], gripper_offset=my_gripper_offset)
        execute_multi_pick_object(skills, sync_barrier, my_pick_pos, gripper_offset=my_gripper_offset)
        skills.move_to_position([my_pick_pos[0], my_pick_pos[1], approach_height], gripper_offset=my_gripper_offset)

        if role == "holder":
            assembly_point = [
                (red_pos[0] + pink_center_pos[0]) / 2,
                (red_pos[1] + pink_center_pos[1]) / 2,
                HOLDER_HEIGHT  # 0.05m
            ]
            skills.move_to_position(assembly_point, gripper_offset=my_gripper_offset)
            execute_pause_for_sync(skills, sync_barrier, "move_to_position:assembly")
            execute_pause_for_sync(skills, sync_barrier, "move_to_position:insert_approach")
            execute_pause_for_sync(skills, sync_barrier, "execute_place_object")
            skills.gripper_open()
        else:
            execute_pause_for_sync(skills, sync_barrier, "move_to_position:assembly")
            hole_detection = skills.detect_objects(
                queries=["hole of the red part"],
                timeout=15.0,
                visualize=False
            )
            hole_pos = hole_detection["hole of the red part"]["position"]
            assembly_point = [hole_pos[0], hole_pos[1], hole_pos[2]]
            assembly_approach = [hole_pos[0], hole_pos[1], INSERTER_HEIGHT + 0.10]

            skills.move_to_position(
                assembly_approach,
                gripper_offset=my_gripper_offset,
                tcp_offset_override=assembly_tcp_offset
            )
            execute_pause_for_sync(skills, sync_barrier, "move_to_position:insert_approach")

            skills.execute_place_object(
                assembly_point,
                gripper_offset=my_gripper_offset,
                is_table=False,
                tcp_offset_override=assembly_tcp_offset
            )
            execute_pause_for_sync(skills, sync_barrier, "execute_place_object")

        skills.move_to_position([assembly_point[0], assembly_point[1], approach_height + 0.05], gripper_offset=my_gripper_offset)
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
'''

    def get_reset_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        return '''
# Two-Robot Coordinated Door Hinge Disassembly (Reset)
# Robot ID: {robot_id}
from skills.skills_lerobot import LeRobotSkills
from pipeline_multi.multi_skills import execute_multi_pick_object, execute_pause_for_sync
import numpy as np

def execute_reset_task():
    """Disassemble hinges and return to original positions."""

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot{robot_id}.yaml",
        frame="world",
        tcp_offset=[-0.02, 0.0, 0.0],  # 2cm TCP offset for assembly task
    )
    skills.connect()

    try:
        # ============================================================
        # Robot Base Positions (World Frame)
        # ============================================================
        ROBOT_BASES = {
            2: np.array([0.0395, 0.2374, 0.0049]),
            3: np.array([0.0508, -0.2515, 0.0000]),
        }

        # ============================================================
        # Assembled Hinge Position (current) and Original Positions (target)
        # ============================================================
        # Current: assembled at assembly point
        assembled_hinge = current_positions["assembled hinge"]
        assembly_pos = assembled_hinge["position"]
        gripper_offset = assembled_hinge.get("gripper_offset", 0.0)

        # Target: original positions for each hinge
        red_target = target_positions["red door hinge"]
        pink_target = target_positions["pink door hinge"]

        approach_height = 0.15
        assembly_height = 0.08

        # ============================================================
        # Determine which hinge this robot handles
        # ============================================================
        robot_id = {robot_id}
        robot_base = ROBOT_BASES[robot_id]

        dist_to_red = np.linalg.norm(np.array(red_target[:2]) - robot_base[:2])
        dist_to_pink = np.linalg.norm(np.array(pink_target[:2]) - robot_base[:2])

        if dist_to_red < dist_to_pink:
            my_hinge_name = "red door hinge"
            my_target_pos = red_target
            role = "holder"  # Red hinge was the holder
        else:
            my_hinge_name = "pink door hinge"
            my_target_pos = pink_target
            role = "inserter"  # Pink hinge was the inserter

        # ============================================================
        # Role reminder (from forward execution):
        # - Red hinge = HOLDER (was fixed during assembly)
        # - Pink hinge = INSERTER (was inserted into holder)
        # ============================================================

        # ============================================================
        # Execute Disassembly Motion
        # ============================================================
        skills.move_to_initial_state()

        # Move to assembly point
        skills.gripper_open()
        skills.move_to_position([assembly_pos[0], assembly_pos[1], approach_height], gripper_offset=gripper_offset)

        # Pick from assembled position (both robots grip the assembly)
        execute_multi_pick_object(skills, sync_barrier, assembly_pos, gripper_offset=gripper_offset)
        skills.move_to_position([assembly_pos[0], assembly_pos[1], approach_height], gripper_offset=gripper_offset)

        execute_pause_for_sync(skills, sync_barrier, "move_to_position:disassembly")

        # Separate: each robot pulls to its original position
        skills.move_to_position([my_target_pos[0], my_target_pos[1], approach_height], gripper_offset=gripper_offset)

        execute_pause_for_sync(skills, sync_barrier, "move_to_position:separated")

        # Place at original position
        skills.execute_place_object(my_target_pos, gripper_offset=gripper_offset, is_table=True)
        skills.move_to_position([my_target_pos[0], my_target_pos[1], approach_height], gripper_offset=gripper_offset)

        # Cleanup
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_reset_task()
'''
