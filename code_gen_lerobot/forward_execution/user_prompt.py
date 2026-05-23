"""
Forward Execution Prompts

LLM prompts for forward task execution (code generation and spec generation).
"""

from pathlib import Path
from typing import Dict, List, Optional

from .skill_api_doc import ROBOT_API_DOC


def _get_frame_for_robot(robot_id: int) -> str:
    """pix2robot 캘리브레이션이 있으면 base_link, 없으면 world."""
    pix2robot_path = (
        Path(__file__).parent.parent.parent
        / "robot_configs" / "pix2robot_matrices"
        / f"robot{robot_id}_pix2robot_data.npz"
    )
    return "base_link" if pix2robot_path.exists() else "world"


def lerobot_code_gen_prompt(
    instruction: str,
    object_positions: Dict,
    spec: str = None,
    robot_id: int = 3,
) -> str:
    """
    LeRobot SO-101용 코드 생성 프롬프트

    Franka realworld_code_gen_prompt 양식을 기반으로 LeRobot에 맞게 단순화

    Args:
        instruction: 자연어 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        object_positions: 객체별 정보 딕셔너리
                         Extended format: {name: {"position": [x,y,z]}}
        spec: 코드 생성 가이드라인/스펙 (선택)
        robot_id: 로봇 번호 (2 또는 3)

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    # 로봇 설정 파일 경로
    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"
    frame = _get_frame_for_robot(robot_id)

    # 객체 위치 포맷팅 (extended format)
    positions_lines = []
    for name, info in object_positions.items():
        if info is not None:
            pos = info["position"]
            positions_lines.append(
                f'    "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
            )
    positions_str = "\n".join(positions_lines)

    # 검출 실패한 객체
    not_found = [name for name, info in object_positions.items() if info is None]
    not_found_str = ", ".join(not_found) if not_found else "None"

    prompt = f"""You are an AI assistant tasked with generating executable Python code that controls a LeRobot SO-101 robot arm.
    The goal is to ensure the robot can achieve the specified objective by executing a sequence of skill actions.

### **Input Details:**

    1. **Task instruction**:
       The goal specifies what the robot must accomplish.

       Provided Task instruction:
       ```
       {instruction}
       ```

    2. **Initial Image**:
       The overhead camera image showing the current workspace state.
       (See attached image)

    3. **Object Positions (World Coordinate Frame, unit: meters)**:
       The detected target object positions in the environment.
       Object position z-coordinate = object height (table surface height is z=0).
       ```python
       positions = {{
      {positions_str}
       }}
       ```
       **CRITICAL**: The `positions` dictionary keys are the ONLY valid keys. You MUST use these exact key names (e.g., `positions["microphone"]`, NOT `positions["power_button"]`). Each key corresponds to a detected object, and its position may represent a specific interaction point (e.g., a button) on that object.
       Objects not found: {not_found_str}

    4. **Specification (Code Generation Guidelines)**:
       The specification outlines the exact requirements and constraints for generating the executable Python code.
       ```
       {spec}
       ```

    5. **Available Skills**:
       The LeRobotSkills class provides these methods:

       | Method | Description | Key Parameters |
       |--------|-------------|----------------|
       | `connect()` | Connect to robot | - |
       | `disconnect()` | Disconnect from robot | - |
       | `gripper_open()` | Open gripper standalone (use only when arm stays still; prefer `move_to_position(..., gripper_action="open")`) | - |
       | `gripper_close()` | Close gripper standalone (use only when arm stays still; prefer `move_to_position(..., gripper_action="close")`) | - |
       | `move_to_initial_state()` | Move to initial/home position | - |
       | `move_to_free_state()` | Move to safe parking position | - |
       | `move_to_position(position, ...)` | Move end-effector to [x,y,z]. Supports integrated gripper transition. | position, target_name, gripper_action, gripper_start_fraction |
       | `rotate_90degree(direction)` | Rotate gripper 90° | direction: 1 (CW) or -1 (CCW) |
       | `execute_pick_object(object_position, ...)` | Descend to pick position (2.5cm from top), close gripper, save pitch | object_position, object_name |
       | `execute_place_object(place_position, ...)` | Descend to place position with saved pitch, open gripper 70% | place_position, is_table, gripper_open_ratio, target_name |
       | `execute_place_lid(place_position, ...)` | Lid-specific place: descend → drag -x 2cm → release. Use INSTEAD OF execute_place_object when seating a lid on a container | place_position, pull_distance, gripper_open_ratio, target_name |
       | `execute_press(position, ...)` | 2-phase press: descend to contact, then press with torque limit | position, press_depth, contact_height, hold_time, target_name |
       | `execute_pull(start_position, distance, ...)` | **OPENING** drawer/door — pulls -x by `distance` m. Grasp → drag → release → retreat-with-close | start_position, distance, object_name |
       | `execute_push(start_position, distance, ...)` | **CLOSING** drawer/door — pushes +x by `distance + 3cm` (internal margin). Push (gripper open) → retreat | start_position, distance, object_name |

       **execute_pick_object**: Call from pick_approach position. Moves TCP to pick height (internally 2.5cm below object top), closes gripper, and **saves current pitch**.
         - **IMPORTANT**: Pass the object position as-is from the positions dictionary. The function internally handles the grasp offset.
         - **object_name**: Pass the object name for subgoal labeling (e.g., "yellow dice")
       **execute_place_object**: Call from place_approach position. Moves TCP to place height **with saved pitch restored**, then opens gripper.
         - **IMPORTANT**: Pass the target surface position as-is. The function internally calculates the correct release height.
         - is_table=True: place on table (pass any position, z is ignored), is_table=False: place on another object (pass the target object's position)
         - gripper_open_ratio=0.7: opens gripper to 70% (ALWAYS use 0.7)
         - Pitch is automatically restored from the saved value at pick time
         - **target_name**: Pass the target name for subgoal labeling (e.g., "blue dish")
       **execute_place_lid**: Lid-specific variant of execute_place_object. Use whenever the held object is a lid (any object whose name contains "lid") being seated on top of a container.
         - Same descent semantics as execute_place_object(is_table=False): place_position is the container's top-surface position, release height = surface_z + saved pick_z.
         - After descent, drags the lid by `pull_distance` (default 2cm) in the -x direction (toward the robot base) with the gripper still closed, then opens the gripper. This compensates for systematic +x landing offset so the lid sits centered on the rim.
         - Pass the container's position as-is (e.g. `positions["pot"]["position"]`). Leave `pull_distance` at default unless instructed otherwise. Use `gripper_open_ratio=0.7`.
         - Do NOT use execute_place_object for lids — use execute_place_lid.

       **execute_press**: Call from approach position with gripper closed. 2-phase descent: normal speed to contact surface, then slow press with torque limit (400/1000).

         - `contact_height`: surface height of the button/switch (meters, e.g., object's z value)
         - `press_depth`: how far to push below contact surface (meters, default 0.01 = 1cm)
         - `hold_time`: seconds to hold pressed state (default 0.3)
         - Close gripper BEFORE calling. Approach and retreat handled by LLM code.
       **execute_push**: Call from approach position above start with gripper closed. Internally handles everything: descends to pre-contact (3cm behind start in opposite push direction), moves linearly through start to end, then retreats to approach_height. **No need for a separate retreat move after calling.**
         - `start_position`: contact point [x, y, z] — the interaction point where the gripper first touches the object (e.g., object's left edge for a left-to-right push)
         - `end_position`: where to end pushing [x, y, z]. If no target position exists, compute from start_position + direction * distance.
         - `push_height`: EE height during push (meters, default 0.01 = 1cm). Set to ~1/3 of object height.
         - Moves in a straight line (not an arc). Close gripper BEFORE calling.
         - **Push distance guide**: 3–5cm is usually sufficient. Do NOT use large distances (e.g., 10cm+) unless explicitly instructed.
         - **World frame directions**: +x = forward (away from robot), -x = backward (toward robot), +y = right, -y = left.

       **execute_pull** — for **OPENING** drawer/door/articulated handle.
         - Args: `start_position` (handle grasp point), `distance` (meters; -x direction is fixed).
         - **Extract `distance` from the user instruction**. Examples:
           - "Open the drawer 5cm" → `distance=0.05`
           - "Open the top drawer 10cm" → `distance=0.10`
           - "Pull the cabinet door open 8cm" → `distance=0.08`
           - If no distance specified in the instruction, use a reasonable default (drawer: 0.10, cabinet door: 0.12).
         - Pre: approach above start with `gripper_action="open"` (gripper open during approach).
         - Internal: descend → grasp → linear pull -x `distance` → release → retreat (gripper closing during last 30%).

       **execute_push** — for **CLOSING** drawer/door — open jaws push handle from inside.
         - Args: `start_position` (handle current open position), `distance` (meters; +x direction is fixed).
         - The caller passes the **same distance the drawer was opened by**. The skill internally pushes `distance + 3cm` to ensure full closure.
         - **Extract `distance` from the user instruction OR use the same value as the open instruction**. Example: if forward opened with 10cm → close with `distance=0.10` (skill pushes 12cm internally).
         - Pre: approach above start with `gripper_action="open"` (gripper STAYS open during push — acts as paddle).
         - Internal: descend → linear push +x `distance + 3cm` → retreat-with-close.

       **For both**: DO NOT use `execute_pick_object` + `execute_place_object` for drawers/doors. `execute_pick_object` auto-lifts the handle, and `execute_place_object(is_table=True)` descends to z=0 (the floor). Always use `execute_pull` (open) and `execute_push` (close).

       **Pitch Handling**: Pitch is automatically saved at pick and restored at place. No need for maintain_pitch during movement.

    6. **Skill Composition Patterns**:

       ```
       # PICK pattern (approach WHILE opening gripper, then descend to pick):
       pick_obj = positions["object_name"]
       pick_pos = pick_obj["position"]
       approach_height = 0.12

       skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], target_name="object_name", gripper_action="open", gripper_start_fraction=0.3, skill_description="Approach object_name and open gripper", verification_question="Is the gripper above object_name and open?")
       skills.execute_pick_object(pick_pos, object_name="object_name", skill_description="Pick up object_name", verification_question="Is object_name grasped by the gripper?")
       skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], target_name="object_name", skill_description="Lift object_name", verification_question="Is object_name lifted off the table?")

       # PLACE on OBJECT pattern (retreat WHILE closing gripper at 20% of retreat motion):
       place_obj = positions["target_object"]
       place_pos = place_obj["position"]

       skills.move_to_position([place_pos[0], place_pos[1], approach_height], target_name="target_object", skill_description="Move object_name above target_object", verification_question="Is object_name above target_object?")
       skills.execute_place_object(place_pos, is_table=False, gripper_open_ratio=0.7, target_name="target_object", skill_description="Place object_name on target_object", verification_question="Is object_name placed on target_object?")
       skills.move_to_position([place_pos[0], place_pos[1], approach_height], target_name="target_object", gripper_action="close", gripper_start_fraction=0.7, skill_description="Retreat from target_object and close gripper", verification_question="Is the gripper clear of target_object and closed?")

       # PLACE LID pattern (lid-specific variant: descends, drags -x 2cm, releases — compensates +x landing bias):
       # Use this whenever the held object is a lid (object name contains "lid").
       # Container z must be the rim/top surface (already provided in positions dict).
       lid_target = positions["pot"]   # or any container whose top is the lid's resting surface
       lid_target_pos = lid_target["position"]

       skills.move_to_position([lid_target_pos[0], lid_target_pos[1], approach_height], target_name="pot", skill_description="Move lid above pot", verification_question="Is the lid above the pot?")
       skills.execute_place_lid(lid_target_pos, gripper_open_ratio=0.7, target_name="pot", skill_description="Place lid on pot and seat it", verification_question="Is the lid centered and seated on the pot?")
       skills.move_to_position([lid_target_pos[0] - 0.02, lid_target_pos[1], approach_height], target_name="pot", gripper_action="close", gripper_start_fraction=0.2, skill_description="Retreat from pot and close gripper", verification_question="Is the gripper clear of the pot and closed?")

       # LATERAL PICK pattern (approach from side at object height WHILE opening gripper, then slide in):
       # Use when the object is thin/tall and top-down approach is not suitable (e.g., gooseneck, handle, lever).
       # Determine offset direction from scene analysis — approach from the obstacle-free side.
       lat_obj = positions["object_name"]
       lat_pos = lat_obj["position"]

       skills.move_to_position([lat_pos[0] + offset_x, lat_pos[1] + offset_y, lat_pos[2]], target_name="object_name", gripper_action="open", gripper_start_fraction=0.3)
       skills.execute_pick_object(lat_pos, object_name="object_name")
       skills.move_to_position([lat_pos[0], lat_pos[1], approach_height], target_name="object_name")

       # PUSH pattern (close gripper, approach above contact point, execute_push handles the rest):
       # push_height ≈ 1/3 of object height. Push distance: 3–5cm is usually sufficient.
       # World frame: +x=forward, -x=backward, +y=right, -y=left.
       # execute_push internally: descends to pre-contact (3cm behind start), pushes linearly to end, retreats.
       push_obj = positions["object_name"]
       push_start = push_obj["points"]["<contact_label>"]  # select appropriate contact point for push direction
       push_end = [push_start[0], push_start[1] + 0.05, push_start[2]]  # e.g., 5cm to the right (+y)

       skills.gripper_close()
       skills.move_to_position([push_start[0], push_start[1], approach_height], target_name="object_name")
       skills.execute_push(push_start, push_end, push_height=push_start[2] * 0.3, object_name="object_name")

       # ─── OPEN drawer/door — execute_pull(start, distance) ───
       # `distance` extracted from the user instruction (e.g., "Open the drawer 10cm" → 0.10).
       # Direction fixed -x. Skill descends + grasps + drags -x distance + releases + retreats.
       OPEN_DISTANCE = 0.10   # ← from "Open the top drawer 10cm" instruction
       handle = positions["top drawer handle"]["points"]["grasp center"]

       skills.move_to_position(
           [handle[0], handle[1], approach_height],
           target_name="top drawer handle",
           gripper_action="open", gripper_start_fraction=0.3,
           skill_description="Approach drawer handle and open gripper",
           verification_question="Is gripper above handle and open?",
       )
       skills.execute_pull(
           handle, distance=OPEN_DISTANCE,
           object_name="top drawer",
           skill_description="Pull the drawer open",
           verification_question="Is the drawer pulled open?",
       )

       # ─── CLOSE drawer/door — execute_push(start, distance) ───
       # `distance` should match the open distance. Skill internally pushes distance + 3cm
       # to ensure full closure. Gripper stays OPEN (set by approach), acts as paddle.
       CLOSE_DISTANCE = OPEN_DISTANCE   # match the opening; skill adds +3cm internally
       open_handle = positions["top drawer handle"]["points"]["grasp center"]   # current (open) position

       skills.move_to_position(
           [open_handle[0], open_handle[1], approach_height],
           target_name="top drawer handle",
           gripper_action="open", gripper_start_fraction=0.3,
           skill_description="Approach drawer handle (gripper open to push)",
           verification_question="Is gripper above handle and open?",
       )
       skills.execute_push(
           open_handle, distance=CLOSE_DISTANCE,
           object_name="top drawer",
           skill_description="Push the drawer closed",
           verification_question="Is the drawer fully closed?",
       )

       # PRESS pattern (close gripper first, approach, press, retreat):
       press_obj = positions["object_with_button"]
       press_pos = press_obj["position"]  # button/switch position

       skills.gripper_close()
       skills.move_to_position([press_pos[0], press_pos[1], approach_height], target_name="object_with_button")
       skills.execute_press(press_pos, contact_height=press_pos[2], press_depth=0.01, hold_time=0.3, target_name="object_with_button")
       skills.move_to_position([press_pos[0], press_pos[1], approach_height], target_name="object_with_button")
       ```

    7. **Executable Code Skeleton**:

       ```python
# pick object_name and place on target_name
from skills.skills_lerobot import LeRobotSkills

def execute_task():
    '''Execute the robot task based on the goal.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="{frame}",
    )
    skills.connect()

    try:
        approach_height = 0.12  # 12cm above objects

        skills.move_to_initial_state()

        # === Object Positions ===
        pick_obj = positions["object_name"]
        pick_pos = pick_obj["position"]

        place_obj = positions["target_name"]
        place_pos = place_obj["position"]

        # === PICK === (approach opens gripper in-motion)
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], target_name="object_name", gripper_action="open", gripper_start_fraction=0.3)
        skills.execute_pick_object(pick_pos, object_name="object_name")
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], target_name="object_name")

        # === PLACE on object === (retreat closes gripper at 20% of retreat)
        skills.move_to_position([place_pos[0], place_pos[1], approach_height], target_name="target_name")
        skills.execute_place_object(place_pos, is_table=False, gripper_open_ratio=0.7, target_name="target_name")
        skills.move_to_position([place_pos[0], place_pos[1], approach_height], target_name="target_name", gripper_action="close", gripper_start_fraction=0.7)

        # === Cleanup ===
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
       ```

### **Task Requirements:**

1. **Complete the Executable Code:**
   - Use `execute_pick_object` and `execute_place_object` for pick/place operations
   - Ensure the code is executable as-is

2. **Guidelines for Implementation:**
   - Always start with `move_to_initial_state()`. DO NOT call `gripper_open()` or `gripper_close()` right after — gripper transitions happen inside `move_to_position(...)` via `gripper_action`.
   - Always end with `move_to_free_state()`
   - Use `approach_height = 0.12` (12cm) for approach/lift movements
   - **ALWAYS pass object/target positions as-is** to execute_pick_object and execute_place_object (the functions handle grasp offset internally)
   - Use `is_table=True` when placing on table, `is_table=False` when placing on another object
   - **Stacking**: When placing on a stack, compute the accumulated stack height. For the place position, use the stack location's XY and set z = sum of all stacked objects' heights (from their original detected positions). Example: to place C on top of A→B stack, use `[A_pos[0], A_pos[1], A_pos[2] + B_pos[2]]`.
   - **ALWAYS use `gripper_open_ratio=0.7`** in `execute_place_object()` to open gripper 70%
   - **Integrated gripper motion (IMPORTANT)**:
     - Pick approach → `move_to_position(..., gripper_action="open", gripper_start_fraction=0.3)` (opens during last 70% of approach)
     - Place retreat → `move_to_position(..., gripper_action="close", gripper_start_fraction=0.7)` (closes during last 30% of retreat)
     - Do NOT call standalone `gripper_open()` / `gripper_close()` for pick/place. Use them only when the arm must stay still (rare).
     - When `gripper_action` is used, write `skill_description` as a compound sentence: `"Approach <obj> and open gripper"`, `"Retreat from <target> and close gripper"`.
   - Always include try/finally for proper cleanup
   - **ALWAYS pass `target_name` or `object_name` parameters** for subgoal labeling in dataset recording
   - **ALWAYS pass `skill_description` and `verification_question`** for every skill call:
     - `skill_description`: concise sentence describing what this action does (e.g., "Move gripper above chocolate_pie_1 to prepare for picking")
     - `verification_question`: Yes/No question to visually verify the outcome (e.g., "Is the gripper positioned above chocolate_pie_1?")

3. **Output Format:**
   - Do not use code blocks in your final answer
   - Provide the generated code in plain text format
   - Include all imports and the complete execute_task() function

**Generate the complete executable Python code:**
"""

    return prompt


def turn3_code_gen_prompt(
    instruction: str,
    robot_id: int = 3,
    all_points: list = None,
) -> str:
    """
    Turn 3: 코드 생성 프롬프트 (multi-turn용)

    Turn 1-2의 컨텍스트가 이미 chat session에 있으므로,
    positions는 Turn 2에서 확정된 world 좌표를 참조합니다.
    중복 설명을 최소화하고 코드 생성에 집중합니다.

    Args:
        instruction: 자연어 목표
        robot_id: 로봇 번호 (2 또는 3)
        all_points: Turn 2에서 검출된 모든 critical points (grasp + interaction)

    Returns:
        Turn 3 user prompt 문자열
    """

    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"
    frame = _get_frame_for_robot(robot_id)
    robot_api_doc = ROBOT_API_DOC

    # all_points를 자연어로 포맷팅
    points_desc = ""
    if all_points:
        # object별로 그룹핑
        from collections import defaultdict
        by_object = defaultdict(list)
        for pt in all_points:
            by_object[pt["object_label"]].append(pt)

        lines = []
        for obj, pts in by_object.items():
            lines.append(f"  {obj}:")
            for pt in pts:
                label = pt.get("label", "unknown")
                role = pt.get("role", "unknown")
                reasoning = pt.get("reasoning", "")
                lines.append(f'    - "{label}" [{role}]: {reasoning}')
        points_desc = "\n".join(lines)

    points_section = ""
    if points_desc:
        points_section = f"""
**Detected Critical Points** (from scene analysis):
Choose the most appropriate point for the task. Access via `positions["object"]["points"]["label"]`.
{points_desc}
"""

    prompt = f"""### Your Job (Turn 3 — Code Generation)

Now generate executable Python code to complete the task using the LeRobot SO-101 robot arm.
Use the scene understanding, detected objects, and grasp/place points from our previous conversation turns.

**Grasp Guidelines**:
- The gripper has symmetric fingers (both actuated), max opening 0.07m.
- Always open the gripper before approaching the grasp pose.
- Ensure the target position has enough clearance to avoid collisions.
{points_section}
**The `positions` dictionary** will be provided at runtime as a global variable with this structure:
```python
positions = {{
    "object_name": {{
        "position": [x, y, z],          # default point (grasp center)
        "points": {{                      # all detected critical points
            "<label_1>": [x, y, z],
            "<label_2>": [x, y, z],
            ...
        }}
    }},
    ...
}}
```
- `position`: default grasp point for this object.
- `points`: all detected critical points for this object. Choose the best point for the task.
- Coordinate values are managed internally — do NOT inspect or use the raw numbers.
- **CRITICAL**: You MUST use ONLY the exact key names from the `positions` dictionary provided earlier in the conversation. Do NOT invent new key names.

**Available Robot API Skills**:

```python
{robot_api_doc}
```

**Skill Composition Patterns**:

```python
# START — always first. NOTE: DO NOT call gripper_open()/gripper_close() right after
# move_to_initial_state(). Gripper transitions are integrated into move_to_position
# via gripper_action="open"/"close".
approach_height = 0.12
skills.move_to_initial_state()

# PICK — approach opens gripper during the motion (starts at 30% of motion)
pick_obj = positions["object_name"]
pick_pos = pick_obj["position"]
skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], target_name="object_name", gripper_action="open", gripper_start_fraction=0.3)
skills.execute_pick_object(pick_pos, object_name="object_name")
skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], target_name="object_name")

# PLACE ON OBJECT (is_table=False) — retreat closes gripper starting at 20% of retreat motion
place_obj = positions["target_object"]
place_pos = place_obj["position"]
skills.move_to_position([place_pos[0], place_pos[1], approach_height], target_name="target_object")
skills.execute_place_object(place_pos, is_table=False, gripper_open_ratio=0.7, target_name="target_object")
skills.move_to_position([place_pos[0], place_pos[1], approach_height], target_name="target_object", gripper_action="close", gripper_start_fraction=0.7)

# PLACE ON TABLE — same as above but is_table=True

# PLACE LID — lid-specific variant: descend → drag -x 2cm → release. USE THIS (not execute_place_object) when the held object's name contains "lid".
# Container z must already be the rim/top surface (positions dict provides this for "pot" etc.).
lid_target = positions["pot"]   # container the lid sits on
lid_target_pos = lid_target["position"]
skills.move_to_position([lid_target_pos[0], lid_target_pos[1], approach_height], target_name="pot")
skills.execute_place_lid(lid_target_pos, gripper_open_ratio=0.7, target_name="pot")
skills.move_to_position([lid_target_pos[0] - 0.02, lid_target_pos[1], approach_height], target_name="pot", gripper_action="close", gripper_start_fraction=0.2)

# LATERAL PICK — approach from side at object height (for thin/tall objects like gooseneck, handle, lever)
# Determine offset direction from scene analysis — approach from obstacle-free side
lat_obj = positions["object_name"]
lat_pos = lat_obj["position"]
skills.move_to_position([lat_pos[0] + offset_x, lat_pos[1] + offset_y, lat_pos[2]], target_name="object_name", gripper_action="open", gripper_start_fraction=0.3)
skills.execute_pick_object(lat_pos, object_name="object_name")
skills.move_to_position([lat_pos[0], lat_pos[1], approach_height], target_name="object_name")

# PUSH — close gripper, approach above contact point, execute_push handles descent + push + retreat
# execute_push internally: descends to pre-contact (3cm behind start), pushes linearly to end, retreats to approach_height.
# Push distance: 3–5cm is usually sufficient. World frame: +x=forward, -x=backward, +y=right, -y=left.
push_obj = positions["object_name"]
push_start = push_obj["points"]["<contact_label>"]  # select contact point for push direction
push_end = [push_start[0], push_start[1] + 0.05, push_start[2]]  # e.g., 5cm to the right (+y)
skills.gripper_close()
skills.move_to_position([push_start[0], push_start[1], approach_height], target_name="object_name")
skills.execute_push(push_start, push_end, push_height=push_start[2] * 0.3, object_name="object_name")

# PRESS — close gripper first, approach, press, retreat
press_obj = positions["object_with_button"]
press_pos = press_obj["position"]
skills.gripper_close()
skills.move_to_position([press_pos[0], press_pos[1], approach_height], target_name="object_with_button")
skills.execute_press(press_pos, contact_height=press_pos[2], press_depth=0.01, hold_time=0.3, target_name="object_with_button")
skills.move_to_position([press_pos[0], press_pos[1], approach_height], target_name="object_with_button")

# END — always last
skills.move_to_free_state()
```

**Code Skeleton** (MUST follow exactly — do NOT add imports or parameters):
```python
def execute_task():
    # `skills` and `positions` are pre-injected global variables.
    # Do NOT import anything. Do NOT instantiate LeRobotSkills yourself.
    skills.connect()
    try:
        # START → PICK → PLACE → END
        pass
    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

**Guidelines**:
1. Always START with `move_to_initial_state()` and END with `move_to_initial_state()` then `move_to_free_state()`.
   - **DO NOT** call `gripper_open()` / `gripper_close()` right after `move_to_initial_state()`. Gripper transitions are integrated into `move_to_position(..., gripper_action=...)`.
2. `approach_height = 0.12` (12cm) for all approach/lift.
3. **ALWAYS** pass positions as-is to execute_pick_object and execute_place_object (grasp offset handled internally).
   - **CRITICAL**: This is a SINGLE-ARM robot. Do NOT pass `arm=`, `left_arm=`, `right_arm=` to any skill function. All functions operate on the single connected arm automatically.
4. `is_table=True` on table, `is_table=False` on another object.
5. **Subtask pattern**: Wrap each logical unit of work with `set_subtask()` before and `clear_subtask()` after.
   - CRITICAL: At both `set_subtask()` and `clear_subtask()`, ALL grippers must be empty (no object held). A subtask boundary is defined by the gripper-empty condition. If an arm is holding an object, the subtask is not yet complete — do NOT call `clear_subtask()` until all grippers have released.
   - **CRITICAL (identical objects → SINGLE subtask)**: When the task involves visually identical objects being moved to similar-pattern targets (e.g., distributing N chocolate pies to N plates, stacking M identical blocks), wrap **ALL of them in ONE subtask** (e.g., `set_subtask("distribute all chocolate pies to plates")` → pick-place pie_1 → pick-place pie_2 → … → `clear_subtask()`). Do NOT call `set_subtask`/`clear_subtask` between identical-object pick-places. Use the **INITIAL** detection positions for every iteration — do NOT re-detect between them (VLM cannot distinguish identical objects and will swap labels, picking the just-placed one again). The single-subtask rule applies only to *identical* objects; when objects are *distinct* (e.g., red block + blue cube + green sphere) each gets its own subtask with re-detection in between.
6. **Re-detection (MANDATORY between subtasks, FORBIDDEN before the first)**: The `positions` dict is pre-populated by the codegen pipeline's initial multi-turn detection (Turn 0 → Turn 1 → Turn 2). The first subtask MUST use this initial `positions` data directly — do NOT call `skills.detect_objects([...])` before Subtask 1. Adding "Re-detect to ensure accuracy" before the first pick is WRONG: it duplicates the initial detection, wastes ~15s and ~5k tokens per episode, and risks introducing label inconsistencies. ONLY call `detect_objects` BEFORE Subtasks 2, 3, …, where the previous subtask may have changed the scene (opening a lid uncovers contents, settling shifts an object, lighting/shadow shifts after arm motion). Pattern:
   ```python
   skills.move_to_initial_state()  # clear arm from camera view first
   # detect_objects() automatically merges fresh detections INTO the global
   # `positions` dict in-place AND preserves Turn 2 point labels (e.g.
   # "top placement point", "grasp center"). Objects that fail re-detection
   # keep their previous values. No `updated = ...` / conditional update.
   skills.detect_objects(["next_pick_target", "next_place_target"])
   pick_pos = positions["next_pick_target"]["position"]      # re-extract
   place_pos = positions["next_place_target"]["points"]["top placement point"]
   ```
   - **CRITICAL (identical objects)**: NEVER re-detect objects that look identical to already-placed objects. The VLM cannot distinguish them and will confuse labels. For those, keep the initial positions instead.
   - **CRITICAL**: After `detect_objects`, you MUST **re-extract ALL local variables** from the `positions` dict. The dict entries are updated in-place, but previously extracted variables still reference the OLD values.
   - **NOTE**: `detect_objects` takes ONLY one positional argument (a list of object names). Do NOT pass `skill_description` or other kwargs.
7. **ALWAYS** `gripper_open_ratio=0.7` in `execute_place_object()`.
8. **Integrated gripper motion**:
   - Pick approach: `move_to_position(..., gripper_action="open", gripper_start_fraction=0.3)` — opens during last 70% of approach.
   - Place retreat: `move_to_position(..., gripper_action="close", gripper_start_fraction=0.7)` — closes during last 30% of retreat.
   - When `gripper_action` is used, write `skill_description` as compound: e.g., `"Approach <obj> and open gripper"`, `"Retreat from <target> and close gripper"`.
9. Wrap with `try/finally` → `disconnect()`.
10. **Pitch Handling**: Pitch is automatically saved at pick and restored at place. No need for maintain_pitch during movement.

**Output**: Complete executable Python code (no code blocks, plain text).

**Generate the complete executable Python code now:**
"""

    return prompt


def context_summary_prompt() -> str:
    """Session 1 마지막에 컨텍스트 요약을 요청하는 프롬프트"""
    return """### Context Handoff Summary

Before we move to code generation, summarize ONLY the factual observations concisely (under 300 words).
Do NOT plan any strategy, sequence, or approach — just describe what you see.

1. **Scene Layout**: Table setup, object positions, workspace boundaries
2. **Object Properties**: Each object's size, shape, color, graspability, fragility
3. **Spatial Relationships**: Which objects are near/far, above/below, stacking order if any
4. **Arm Reachability**: For each object, state which arm(s) can reach it based on the workspace image (bright area = reachable). Example: "yellow block is in the left arm's bright area only" or "red cup is in the overlap zone, reachable by both arms"
5. **Physical Constraints**: Observable risks such as tight clearance between objects, objects near edges, unstable stacking — factual observations only, no action recommendations

IMPORTANT: Do NOT include any task strategy, action sequence, or movement plan. The code generation session will decide the approach based on the API and workspace constraints.

Output as a structured summary. This will be passed to a fresh code generation session."""


def codegen_with_context_prompt(
    instruction: str,
    robot_id: int = 3,
    all_points: list = None,
    context_summary: str = "",
    positions: dict = None,
) -> str:
    """
    새로운 chat session에서 컨텍스트 요약과 함께 코드를 생성하는 프롬프트.
    Session 1의 누적 토큰 없이 깨끗한 세션에서 코드 생성.
    """

    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"
    frame = _get_frame_for_robot(robot_id)
    robot_api_doc = ROBOT_API_DOC

    # all_points를 자연어로 포맷팅
    points_desc = ""
    if all_points:
        from collections import defaultdict
        by_object = defaultdict(list)
        for pt in all_points:
            by_object[pt["object_label"]].append(pt)

        lines = []
        for obj, pts in by_object.items():
            lines.append(f"  {obj}:")
            for pt in pts:
                label = pt.get("label", "unknown")
                role = pt.get("role", "unknown")
                reasoning = pt.get("reasoning", "")
                lines.append(f'    - "{label}" [{role}]: {reasoning}')
        points_desc = "\n".join(lines)

    points_section = ""
    if points_desc:
        points_section = f"""
**Detected Critical Points** (from scene analysis):
Choose the most appropriate point for the task. Access via `positions["object"]["points"]["label"]`.
{points_desc}
"""

    # positions dict의 key 구조만 포맷팅 (좌표값은 숨김)
    positions_keys = ""
    if positions:
        key_lines = []
        for name, info in positions.items():
            if isinstance(info, dict) and "position" in info:
                pts = info.get("points", {})
                pt_labels = ", ".join(f'"{k}"' for k in pts.keys()) if pts else ""
                key_lines.append(f'  - "{name}": position, points: [{pt_labels}]')
        positions_keys = "\n".join(key_lines)

    prompt = f"""### Scene Context (from prior analysis session)

{context_summary}

### Code Generation Task

Generate executable Python code to complete the following task using the robot.

**Task**: "{instruction}"

**Grasp Guidelines**:
- The gripper has symmetric fingers (both actuated), max opening 0.07m.
- Always open the gripper before approaching the grasp pose.
- Ensure the target position has enough clearance to avoid collisions.
{points_section}
**The `positions` dictionary** is provided at runtime as a global variable with the following keys:
{positions_keys}
- Access via `positions["object_name"]["position"]` or `positions["object_name"]["points"]["label"]`.
- Coordinate values are managed internally — do NOT inspect or use the raw numbers.
- **CRITICAL**: You MUST use ONLY the exact key names shown above. Do NOT invent new key names.
- **CRITICAL**: Do NOT redefine or hardcode the `positions` dictionary in your code. It is already available as a global variable at runtime.
- **CRITICAL**: Do NOT hardcode any coordinate values (e.g., `[0.15, -0.25, 0.0]`). For locations not in the dictionary, use `move_to_pixel([y, x])` or `execute_place_at_pixel([y, x])` with normalized 0–1000 image coordinates.

**Available Robot API Skills**:

```python
{robot_api_doc}
```

**Skill Composition Patterns**:

```python
# START — always first
approach_height = 0.12
skills.move_to_initial_state(skill_description="Move to initial position", verification_question="Is the robot at initial position?")

# PICK — approach opens gripper in-motion (start at 30% of approach)
pick_obj = positions["object_name"]
pick_pos = pick_obj["position"]
skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], target_name="object_name", gripper_action="open", gripper_start_fraction=0.3, skill_description="Approach object_name and open gripper", verification_question="Is the gripper above object_name and open?")
skills.execute_pick_object(pick_pos, object_name="object_name", skill_description="Pick up object_name", verification_question="Is object_name grasped?")
skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], target_name="object_name", skill_description="Lift object_name", verification_question="Is object_name lifted?")

# PLACE ON OBJECT (is_table=False) — retreat closes gripper starting at 20% of retreat
place_obj = positions["target_object"]
place_pos = place_obj["position"]
skills.move_to_position([place_pos[0], place_pos[1], approach_height], target_name="target_object", skill_description="Move object_name above target_object", verification_question="Is object_name above target_object?")
skills.execute_place_object(place_pos, is_table=False, gripper_open_ratio=0.7, target_name="target_object", skill_description="Place object_name on target_object", verification_question="Is object_name on target_object?")
skills.move_to_position([place_pos[0], place_pos[1], approach_height], target_name="target_object", gripper_action="close", gripper_start_fraction=0.7, skill_description="Retreat from target_object and close gripper", verification_question="Is the gripper clear of target_object and closed?")

# PLACE LID — lid-specific variant: descend → drag -x 2cm → release. USE THIS (not execute_place_object) whenever the held object's name contains "lid".
# Container z is already the rim/top surface in positions dict (e.g. positions["pot"]).
lid_target = positions["pot"]
lid_target_pos = lid_target["position"]
skills.move_to_position([lid_target_pos[0], lid_target_pos[1], approach_height], target_name="pot", skill_description="Move lid above pot", verification_question="Is the lid above the pot?")
skills.execute_place_lid(lid_target_pos, gripper_open_ratio=0.7, target_name="pot", skill_description="Place lid on pot and seat it", verification_question="Is the lid centered and seated on the pot?")
skills.move_to_position([lid_target_pos[0] - 0.02, lid_target_pos[1], approach_height], target_name="pot", gripper_action="close", gripper_start_fraction=0.2, skill_description="Retreat from pot and close gripper", verification_question="Is the gripper clear of the pot and closed?")

# PLACE AT PIXEL (is_table=True) — target is NOT in positions dict (e.g., empty spot on table)
# Specify [y, x] in normalized 0–1000 coordinates from the top-view image.
# ⚠ REACHABILITY CHECK: Before using any pixel coordinate, verify on the workspace image
#   that [y, x] falls inside the BRIGHT area (cyan arc). If not, shift inward.
# Example: place at the left side of table → [500, 200] (verified: inside bright area)
skills.move_to_pixel([500, 200], target_name="left side", skill_description="Move object_name above left side of table", verification_question="Is object_name above the left side?")
skills.execute_place_at_pixel([500, 200], is_table=True, gripper_open_ratio=0.7, target_name="left side", skill_description="Place object_name at left side of table", verification_question="Is object_name placed at the left side?")
# NOTE: move_to_pixel does not yet support gripper_action; close gripper with standalone call if needed.
skills.move_to_pixel([500, 200], target_name="left side", skill_description="Retract from left side", verification_question="Is the gripper clear of the left side?")
skills.gripper_close(skill_description="Close gripper after release", verification_question="Is the gripper closed?")

# SUBTASK + RE-DETECTION PATTERN (MANDATORY for multi-object tasks)
# Each pick-place of one object = one subtask.
# Wrap with set_subtask() before and clear_subtask() after.
# Re-detect ONLY between subtasks (BEFORE 2, BEFORE 3, …) — NEVER before Subtask 1.
# The initial `positions` dict from codegen multi-turn detection is already current
# at the start; calling detect_objects() before Subtask 1 is redundant.

# Example: stack B on A, then C on B

# Subtask 1: 1st pick-place — uses initial detection (no prior subtask to invalidate state).
# NOTE: For brevity this example omits explicit approach/retreat move_to_position calls.
# In practice include them with gripper_action="open"/"close" as shown above.
skills.set_subtask("pick A and place at target")
skills.execute_pick_object(a_pos, object_name="A", skill_description="Pick A", verification_question="Is A grasped?")
skills.execute_place_object(target_pos, is_table=True, gripper_open_ratio=0.7, target_name="target", skill_description="Place A", verification_question="Is A placed?")
skills.clear_subtask()

# Re-detect Subtask 2's targets — pick=B, place=A (A is the just-placed surface to stack on).
# detect_objects() merges results into the global `positions` dict in-place
# AND preserves Turn 2 point labels. NOTE: takes ONLY one positional arg
# (list of names) — do NOT pass skill_description or other kwargs.
skills.move_to_initial_state()  # clear arm from camera view first
skills.detect_objects(["A", "B"])
a_pos = positions["A"]["position"]   # re-extract after auto-merge
b_pos = positions["B"]["position"]

# Subtask 2: 2nd object — pick → place
skills.set_subtask("pick B and place on A")
skills.execute_pick_object(b_pos, object_name="B", skill_description="Pick B", verification_question="Is B grasped?")
skills.execute_place_object(a_pos, is_table=False, gripper_open_ratio=0.7, target_name="A", skill_description="Place B on A", verification_question="Is B on A?")
skills.clear_subtask()

# Re-detect Subtask 3's targets — pick=C, place=B (newly placed; need updated position).
skills.move_to_initial_state()
skills.detect_objects(["B", "C"])
b_pos = positions["B"]["position"]
c_pos = positions["C"]["position"]

# Subtask 3: 3rd object — pick → place
skills.set_subtask("pick C and place on B")
skills.execute_pick_object(c_pos, object_name="C", skill_description="Pick C", verification_question="Is C grasped?")
skills.execute_place_object(b_pos, is_table=False, gripper_open_ratio=0.7, target_name="B", skill_description="Place C on B", verification_question="Is C on B?")
skills.clear_subtask()

# DISTRIBUTE PATTERN — identical objects → SINGLE subtask, NO re-detection between them.
# Example: distribute 2 visually identical chocolate pies (pie_1, pie_2) onto 2 plates (plate_1, plate_2).
# Both pies look the same → VLM cannot reliably tell pie_1 from pie_2 after one has moved →
# wrap all pies in ONE subtask, use INITIAL positions for every pie.
#
# ── PER-PnP PATTERN (6 steps, repeated for every object) ──────────────────────
#   1. APPROACH (pick)  : move_to_position([obj_xy, h], gripper_action="open")    — above object
#   2. PICK             : execute_pick_object(obj_pos)                            — descend + grasp
#   3. LIFT             : move_to_position([obj_xy, h])                           — vertical lift back to carry height
#   4. APPROACH (place) : move_to_position([target_xy, h])                        — translate to above target
#   5. PLACE            : execute_place_object(target_pos)                        — descend + release
#   6. RETREAT          : move_to_position([target_xy, h], gripper_action="close")— retreat + gripper close
# The two APPROACH steps (1 and 4) are BOTH required — one above pick object, one above place target.
# Do not skip step 4 (don't rely on execute_place_object's internal xy_lead_descent for cross-table transit).
# ───────────────────────────────────────────────────────────────────────────────

skills.set_subtask("distribute all chocolate pies to plates")

# ─── PnP 1: pie_1 → plate_1 (use initial detection) ───────────────────────────
pie1_pos = positions["chocolate_pie_1"]["position"]
plate1_pos = positions["red_plate_1"]["position"]
# 1. APPROACH (pick)
skills.move_to_position([pie1_pos[0], pie1_pos[1], approach_height], target_name="chocolate_pie_1", gripper_action="open", gripper_start_fraction=0.3, skill_description="Approach chocolate_pie_1 and open gripper", verification_question="Is gripper above chocolate_pie_1 and open?")
# 2. PICK
skills.execute_pick_object(pie1_pos, object_name="chocolate_pie_1", skill_description="Pick chocolate_pie_1", verification_question="Is chocolate_pie_1 grasped?")
# 3. LIFT
skills.move_to_position([pie1_pos[0], pie1_pos[1], approach_height], target_name="chocolate_pie_1", skill_description="Lift chocolate_pie_1", verification_question="Is chocolate_pie_1 lifted?")
# 4. APPROACH (place)
skills.move_to_position([plate1_pos[0], plate1_pos[1], approach_height], target_name="red_plate_1", skill_description="Move chocolate_pie_1 above red_plate_1", verification_question="Is chocolate_pie_1 above red_plate_1?")
# 5. PLACE
skills.execute_place_object(plate1_pos, is_table=True, gripper_open_ratio=0.7, target_name="red_plate_1", skill_description="Place chocolate_pie_1 on red_plate_1", verification_question="Is chocolate_pie_1 on red_plate_1?")
# 6. RETREAT
skills.move_to_position([plate1_pos[0], plate1_pos[1], approach_height], target_name="red_plate_1", gripper_action="close", gripper_start_fraction=0.7, skill_description="Retreat from red_plate_1 and close gripper", verification_question="Is gripper clear and closed?")

# ─── PnP 2: pie_2 → plate_2 ───────────────────────────────────────────────────
# Stay inside the SAME subtask: NO clear_subtask, NO set_subtask, NO detect_objects.
# Reuse INITIAL positions for pie_2 / plate_2 (no re-detection — VLM swaps identical labels).
pie2_pos = positions["chocolate_pie_2"]["position"]
plate2_pos = positions["red_plate_2"]["position"]
# 1. APPROACH (pick)
skills.move_to_position([pie2_pos[0], pie2_pos[1], approach_height], target_name="chocolate_pie_2", gripper_action="open", gripper_start_fraction=0.3, skill_description="Approach chocolate_pie_2 and open gripper", verification_question="Is gripper above chocolate_pie_2 and open?")
# 2. PICK
skills.execute_pick_object(pie2_pos, object_name="chocolate_pie_2", skill_description="Pick chocolate_pie_2", verification_question="Is chocolate_pie_2 grasped?")
# 3. LIFT
skills.move_to_position([pie2_pos[0], pie2_pos[1], approach_height], target_name="chocolate_pie_2", skill_description="Lift chocolate_pie_2", verification_question="Is chocolate_pie_2 lifted?")
# 4. APPROACH (place)
skills.move_to_position([plate2_pos[0], plate2_pos[1], approach_height], target_name="red_plate_2", skill_description="Move chocolate_pie_2 above red_plate_2", verification_question="Is chocolate_pie_2 above red_plate_2?")
# 5. PLACE
skills.execute_place_object(plate2_pos, is_table=True, gripper_open_ratio=0.7, target_name="red_plate_2", skill_description="Place chocolate_pie_2 on red_plate_2", verification_question="Is chocolate_pie_2 on red_plate_2?")
# 6. RETREAT
skills.move_to_position([plate2_pos[0], plate2_pos[1], approach_height], target_name="red_plate_2", gripper_action="close", gripper_start_fraction=0.7, skill_description="Retreat from red_plate_2 and close gripper", verification_question="Is gripper clear and closed?")

skills.clear_subtask()  # close the SINGLE distribute subtask AFTER all pies are placed.

# END — always last
skills.move_to_initial_state()
skills.move_to_free_state(skill_description="Move to safe position", verification_question="Is the robot at safe position?")
```

**Code Skeleton** (MUST follow exactly — do NOT add imports or parameters):
```python
def execute_task():
    # `skills` and `positions` are pre-injected global variables.
    # Do NOT import anything. Do NOT instantiate LeRobotSkills yourself.
    skills.connect()
    try:
        # START → PICK → PLACE → END
        pass
    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

**Guidelines**:
1. Always START with `move_to_initial_state()` and END with `move_to_initial_state()` then `move_to_free_state()`.
   - **DO NOT** call `gripper_open()` / `gripper_close()` right after `move_to_initial_state()`. Gripper transitions are integrated into `move_to_position(..., gripper_action=...)`.
2. `approach_height = 0.12` (12cm) for all approach/lift.
3. **ALWAYS** pass positions as-is to execute_pick_object and execute_place_object (grasp offset handled internally).
   - **CRITICAL**: This is a SINGLE-ARM robot. Do NOT pass `arm=`, `left_arm=`, `right_arm=` to any skill function. All functions operate on the single connected arm automatically.
4. `is_table=True` on table, `is_table=False` on another object.
5. **Subtask pattern**: Wrap each logical unit of work with `set_subtask()` before and `clear_subtask()` after.
   - CRITICAL: At both `set_subtask()` and `clear_subtask()`, ALL grippers must be empty (no object held). A subtask boundary is defined by the gripper-empty condition. If an arm is holding an object, the subtask is not yet complete — do NOT call `clear_subtask()` until all grippers have released.
   - **CRITICAL (identical objects → SINGLE subtask)**: When the task involves visually identical objects being moved to similar-pattern targets (e.g., distributing N chocolate pies to N plates, stacking M identical blocks), wrap **ALL of them in ONE subtask** (e.g., `set_subtask("distribute all chocolate pies to plates")` → pick-place pie_1 → pick-place pie_2 → … → `clear_subtask()`). Do NOT call `set_subtask`/`clear_subtask` between identical-object pick-places. Use the **INITIAL** detection positions for every iteration — do NOT re-detect between them (VLM cannot distinguish identical objects and will swap labels, picking the just-placed one again). The single-subtask rule applies only to *identical* objects; when objects are *distinct* (e.g., red block + blue cube + green sphere) each gets its own subtask with re-detection in between.
6. **Re-detection (MANDATORY between subtasks, FORBIDDEN before the first)**: The `positions` dict is pre-populated by the codegen pipeline's initial multi-turn detection (Turn 0 → Turn 1 → Turn 2). The first subtask MUST use this initial `positions` data directly — do NOT call `skills.detect_objects([...])` before Subtask 1. Adding "Re-detect to ensure accuracy" before the first pick is WRONG: it duplicates the initial detection, wastes ~15s and ~5k tokens per episode, and risks introducing label inconsistencies. ONLY call `detect_objects` BEFORE Subtasks 2, 3, …, where the previous subtask may have changed the scene (opening a lid uncovers contents, settling shifts an object, lighting/shadow shifts after arm motion). Pattern:
   ```python
   skills.move_to_initial_state()  # clear arm from camera view first
   # detect_objects() automatically merges fresh detections INTO the global
   # `positions` dict in-place AND preserves Turn 2 point labels (e.g.
   # "top placement point", "grasp center"). Objects that fail re-detection
   # keep their previous values. No `updated = ...` / conditional update.
   skills.detect_objects(["next_pick_target", "next_place_target"])
   pick_pos = positions["next_pick_target"]["position"]      # re-extract
   place_pos = positions["next_place_target"]["points"]["top placement point"]
   ```
   - **CRITICAL (identical objects)**: NEVER re-detect objects that look identical to already-placed objects. The VLM cannot distinguish them and will confuse labels. For those, keep the initial positions instead.
   - **CRITICAL**: After `detect_objects`, you MUST **re-extract ALL local variables** from the `positions` dict. The dict entries are updated in-place, but previously extracted variables still reference the OLD values.
   - **NOTE**: `detect_objects` takes ONLY one positional argument (a list of object names). Do NOT pass `skill_description` or other kwargs.
7. **ALWAYS** `gripper_open_ratio=0.7` in `execute_place_object()`.
8. **Integrated gripper motion**:
   - Pick approach: `move_to_position(..., gripper_action="open", gripper_start_fraction=0.3)` — opens during last 70% of approach.
   - Place retreat: `move_to_position(..., gripper_action="close", gripper_start_fraction=0.7)` — closes during last 30% of retreat.
   - When `gripper_action` is used, write `skill_description` as compound: e.g., `"Approach <obj> and open gripper"`, `"Retreat from <target> and close gripper"`.
9. Wrap with `try/finally` → `disconnect()`.
10. **Pitch Handling**: Pitch is automatically saved at pick and restored at place. No need for maintain_pitch during movement.
9. **ALWAYS** pass `skill_description` and `verification_question` for every skill call:
   - `skill_description`: concise sentence describing the action (e.g., "Move gripper above chocolate_pie_1")
   - `verification_question`: Yes/No question to verify the outcome (e.g., "Is the gripper above chocolate_pie_1?")
10. **NEVER hardcode coordinate values** (e.g., `[0.15, -0.25, 0.0]`). Use `positions` dict for detected objects, and `move_to_pixel([y, x])` / `execute_place_at_pixel([y, x])` with normalized 0–1000 coordinates for any location not in the dict.
11. **MANDATORY — Pixel Coordinate Reachability Check**:
    Whenever you write a hardcoded pixel coordinate (e.g., `[500, 200]`), you MUST follow this checklist BEFORE using it in code:
      (a) **Locate on image**: Find the [y, x] point on the workspace image.
      (b) **Check brightness**: Is the point inside the **BRIGHT area** (within the cyan arc)? The bright area is the robot's reachable donut-shaped zone.
      (c) **If YES** → use the coordinate as-is.
      (d) **If NO** (point is in the darkened/unreachable area) → shift the coordinate inward to the nearest point that IS inside the bright area. Typical adjustments: move away from image edges, move toward the center of the cyan arc.
      (e) **Document in a code comment**: Write a comment on the same line explaining WHY you chose that coordinate.
          Example: `target_pixel = [450, 300]  # left-center of bright area, task says "left side of table"`
    NEVER guess extreme values like [900, 900] or [100, 100] — these are almost always in the dark (unreachable) area.

**Output**: Complete executable Python code (no code blocks, plain text).

**Generate the complete executable Python code now:**
"""

    return prompt


def lerobot_spec_gen_prompt(
    instruction: str,
    object_positions: Dict[str, List[float]],
) -> str:
    """
    LeRobot SO-101용 태스크 스펙 생성 프롬프트

    자연어 목표를 구조화된 스펙(단계별 액션)으로 변환합니다.
    이 스펙은 이후 code_gen에서 실제 Python 코드로 변환됩니다.

    Args:
        instruction: 자연어 목표 (예: "빨간 컵을 파란 상자에 놓아라")
        object_positions: 객체별 위치 딕셔너리 {name: [x, y, z]}

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    # 객체 위치 포맷팅
    positions_str = "\n".join([
        f'    - {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
        for name, pos in object_positions.items()
        if pos is not None
    ])

    # 검출 실패한 객체
    not_found = [name for name, pos in object_positions.items() if pos is None]
    not_found_str = ", ".join(not_found) if not_found else "None"

    # 객체 이름 리스트
    object_names = list(object_positions.keys())

    prompt = f"""
You are an AI assistant that generates structured task specifications for a LeRobot SO-101 robot arm.
Your task is to analyze the goal and break it down into a sequence of skill actions.

### **Input Details:**

1. **Goal (Natural Language)**:
   What the robot must accomplish.
   ```
   {instruction}
   ```

2. **Detected Objects and Positions (World Frame, meters)**:
   Object position z-coordinate = object height (table surface is z=0).
   ```
    {positions_str}
   ```

3. **Available Skills**:
   | Skill | Description | Parameters |
   |-------|-------------|------------|
   | `move_to_initial_state` | Move to home position | - |
   | `move_to_free_state` | Move to safe parking position | - |
   | `rotate_90degree` | Rotate gripper 90° | direction: "cw" or "ccw" |
   | `gripper_open` | Open the gripper standalone (rare; prefer `move_to_position` with `gripper_action="open"`) | - |
   | `gripper_close` | Close the gripper standalone (rare; prefer `move_to_position` with `gripper_action="close"`) | - |
   | `move_to_position` | Move end-effector to position; supports integrated gripper open/close during motion | object: str, approach_height: float, gripper_action: "open" or "close", gripper_start_fraction: float |
   | `execute_pick_object` | Descend (2.5cm from top) + gripper_close + save pitch | object: str |
   | `execute_place_object` | Descend with saved pitch + gripper_open | target: str, is_table: bool |
   | `execute_press` | 2-phase press with torque limit | position, contact_height, press_depth, hold_time, target_name |
   | `execute_push` | Descend → run-up → linear push → retreat (all-in-one) | start_position, end_position, push_height, object_name |

   **execute_pick_object**: Call after moving to pick_approach. Descends to 2.5cm below object top, closes gripper, and **saves current pitch**.
   **execute_place_object**: Call after moving to place_approach. Descends to place height **with saved pitch restored**, then opens gripper.
     - is_table=true: place on table
     - is_table=false: place on another object
     - Pitch is automatically restored from pick time
   **execute_press**: Call after closing gripper and moving to approach. Descends to contact surface, then presses with torque limit.
     - contact_height: button surface height (use object's z value)
     - press_depth: push depth below contact (default 0.01m)
     - hold_time: hold duration (default 0.3s)
   **execute_push**: Call after closing gripper and moving to approach above start. Internally: descends to pre-contact (3cm behind start), pushes linearly to end, retreats. No separate retreat needed.
     - start_position: contact point (interaction point, e.g., object edge)
     - end_position: push end position. If no target, compute from start + direction * distance (3–5cm is usually sufficient).
     - push_height: EE height during push (default 0.01m, ~1/3 of object height)
     - World frame: +x=forward, -x=backward, +y=right, -y=left

4. **Available Object Names**:
   {object_names}

### **Output Format**

Specification: {{"required_skills": [...], "steps": [...]}}

### **Examples**

**Example 1**: "Pick up the red cup and place it on the carrier"
```
Specification: {{
  "required_skills": ["move_to_initial_state", "move_to_position", "execute_pick_object", "execute_place_object", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "move_to_position", "object": "red cup", "approach_height": 0.20, "gripper_action": "open", "gripper_start_fraction": 0.3}},
    {{"step": 3, "action": "execute_pick_object", "object": "red cup"}},
    {{"step": 4, "action": "move_to_position", "object": "red cup", "approach_height": 0.20}},
    {{"step": 5, "action": "move_to_position", "target": "carrier", "approach_height": 0.20}},
    {{"step": 6, "action": "execute_place_object", "target": "carrier", "is_table": true}},
    {{"step": 7, "action": "move_to_position", "target": "carrier", "approach_height": 0.20, "gripper_action": "close", "gripper_start_fraction": 0.2}},
    {{"step": 8, "action": "move_to_free_state"}}
  ]
}}
```

**Example 2**: "Stack the red dice on top of the blue box"
```
Specification: {{
  "required_skills": ["move_to_initial_state", "move_to_position", "execute_pick_object", "execute_place_object", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "move_to_position", "object": "red dice", "approach_height": 0.20, "gripper_action": "open", "gripper_start_fraction": 0.3}},
    {{"step": 3, "action": "execute_pick_object", "object": "red dice"}},
    {{"step": 4, "action": "move_to_position", "object": "red dice", "approach_height": 0.20}},
    {{"step": 5, "action": "move_to_position", "target": "blue box", "approach_height": 0.20}},
    {{"step": 6, "action": "execute_place_object", "target": "blue box", "is_table": false}},
    {{"step": 7, "action": "move_to_position", "target": "blue box", "approach_height": 0.20, "gripper_action": "close", "gripper_start_fraction": 0.2}},
    {{"step": 8, "action": "move_to_free_state"}}
  ]
}}
```

### **Guidelines**

1. Use `execute_pick_object` and `execute_place_object` for pick/place operations
2. Always start with `move_to_initial_state` (do NOT follow with `gripper_open`/`gripper_close`)
3. Always end with `move_to_free_state`
4. Use `approach_height: 0.10` for approach/retract movements (10cm above target)
5. Use `is_table: true` when placing on table, `is_table: false` when stacking on another object
6. Use ONLY the object names from the detected objects list
7. **Integrated gripper**: pick approach uses `"gripper_action": "open", "gripper_start_fraction": 0.3`; place retreat uses `"gripper_action": "close", "gripper_start_fraction": 0.2`.

### **Generate Specification:**
"""

    return prompt
