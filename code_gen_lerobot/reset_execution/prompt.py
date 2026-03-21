"""
Reset Execution Prompts

LLM prompts for reset/reverse task execution.
Uses forward execution context (spec, code) to understand what was done,
then generates code to restore the environment to its initial state.

Includes:
- Single-turn prompts: lerobot_reset_spec_gen_prompt, lerobot_reset_code_gen_prompt
- Multi-turn prompts: turn0_reset_scene_understanding_prompt, turn_codegen_reset_prompt
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _get_frame_for_robot(robot_id: int) -> str:
    """pix2robot 캘리브레이션이 있으면 base_link, 없으면 world."""
    pix2robot_path = (
        Path(__file__).parent.parent.parent
        / "robot_configs" / "pix2robot_matrices"
        / f"robot{robot_id}_pix2robot_data.npz"
    )
    return "base_link" if pix2robot_path.exists() else "world"


def lerobot_reset_spec_gen_prompt(
    original_instruction: str,
    original_positions: Dict[str, List[float]],
    current_positions: Dict[str, List[float]],
    forward_spec: Dict = None,
) -> str:
    """
    LeRobot SO-101용 리셋 태스크 스펙 생성 프롬프트

    Args:
        original_instruction: 원래 태스크 목표
        original_positions: 객체별 원래 위치 (복귀 목표)
        current_positions: 객체별 현재 위치 (detection으로 획득)
        forward_spec: Forward execution에서 생성된 spec

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    # Helper to extract position from extended or legacy format
    def get_position(info):
        if info is None:
            return None
        elif isinstance(info, dict) and "position" in info:
            return info["position"]
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            return list(info[:3])
        return None

    # 원래 위치 포맷팅 (extended format 지원)
    original_lines = []
    for name, info in original_positions.items():
        pos = get_position(info)
        if pos is not None:
            original_lines.append(f'    - {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]')
    original_str = "\n".join(original_lines)

    # 현재 위치 포맷팅 (extended format 지원)
    current_lines = []
    for name, info in current_positions.items():
        pos = get_position(info)
        if pos is not None:
            current_lines.append(f'    - {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]')
    current_str = "\n".join(current_lines)

    # Forward spec 포맷팅
    forward_spec_str = json.dumps(forward_spec, indent=2) if forward_spec else "Not available"

    prompt = f"""
You are an AI assistant that generates structured task specifications for a LeRobot SO-101 robot arm.
Your task is to generate a RESET specification that restores the environment to its initial state.

### **Context: Forward Execution (What Was Done)**

1. **Forward Task Goal**:
   ```
   {original_instruction}
   ```

2. **Forward Spec (Step-by-Step Plan)**:
   ```json
   {forward_spec_str}
   ```

3. **Environment States**:
   Object position z-coordinate = object height (table surface is z=0).
   - Initial State (Before Forward):
     ```
     {original_str}
     ```
   - Current State (After Forward):
     ```
     {current_str}
     ```

### **Your Task: Generate RESET Specification**

Generate a specification that restores the environment to its initial state
(before "{original_instruction}" was executed).

### **Available Skills**:
| Skill | Description | Parameters |
|-------|-------------|------------|
| `move_to_initial_state` | Move to home position | - |
| `move_to_free_state` | Move to safe parking position | - |
| `rotate_90degree` | Rotate gripper 90 deg | direction: "cw" or "ccw" |
| `gripper_open` | Open the gripper | - |
| `move_to_position` | Move end-effector to position | object: str, offset_z: float |
| `execute_pick_object` | Descend (3cm from top) + gripper_close + save pitch | object: str |
| `execute_place_object` | Descend with saved pitch + gripper_open | target: str, is_table: bool |

**execute_pick_object**: Call after moving to pick_approach. Descends to 3cm below object top, closes gripper, and **saves current pitch**.
**execute_place_object**: Call after moving to place_approach. Descends to place height **with saved pitch restored**, then opens gripper.
  - is_table=true: place on table
  - is_table=false: place on another object
  - Pitch is automatically restored from pick time

### **Output Format**

Specification: {{"required_skills": [...], "steps": [...]}}

### **Guidelines**

1. Analyze the forward spec to understand what was done
2. Generate steps that restore each moved object to its initial position
3. Use `execute_pick_object` and `execute_place_object` for pick/place operations
4. Always start with `move_to_initial_state`
5. Always end with `move_to_initial_state` and `move_to_free_state`
6. Use `offset_z: 0.15` for approach/retract movements
7. Use `is_table: true` when placing on table

### **Generate Reset Specification:**
"""

    return prompt


def lerobot_reset_code_gen_prompt(
    original_instruction: str,
    target_positions: Dict[str, List[float]],
    current_positions: Dict[str, List[float]],
    forward_spec: Dict = None,
    forward_code: str = None,
    robot_id: int = 3,
    is_random_reset: bool = False,
) -> str:
    """
    LeRobot SO-101용 리셋 코드 생성 프롬프트

    Forward execution context (spec, code)를 참조하여
    환경을 초기 상태로 되돌리거나 랜덤 위치로 shuffle하는 코드를 생성합니다.

    Args:
        original_instruction: 원래 태스크 목표
        target_positions: 객체별 목표 위치 (original 또는 random)
        current_positions: 객체별 현재 위치 (detection으로 획득)
        forward_spec: Forward execution에서 생성된 spec
        forward_code: Forward execution에서 생성된 Python 코드
        robot_id: 로봇 번호 (2 또는 3)
        is_random_reset: True면 랜덤 위치로 shuffle, False면 초기 위치로 복귀

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    # 로봇 설정 파일 경로
    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"
    frame = _get_frame_for_robot(robot_id)

    # Helper to extract position from extended or legacy format
    def get_position(info):
        if info is None:
            return None
        elif isinstance(info, dict) and "position" in info:
            return info["position"]
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            return list(info[:3])
        return None

    # 타겟 위치 포맷팅
    target_lines = []
    for name, info in target_positions.items():
        pos = get_position(info)
        if pos is not None:
            target_lines.append(
                f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
            )
    target_str = "\n".join(target_lines)

    # 현재 위치 포맷팅
    current_lines = []
    for name, info in current_positions.items():
        pos = get_position(info)
        if pos is not None:
            current_lines.append(f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},')
    current_str = "\n".join(current_lines)

    # Forward spec 포맷팅
    forward_spec_str = json.dumps(forward_spec, indent=2) if forward_spec else "Not available"

    # Forward code 포맷팅 (너무 길면 요약)
    if forward_code:
        # 코드가 너무 길면 핵심 부분만 추출
        if len(forward_code) > 2000:
            forward_code_str = forward_code[:2000] + "\n... (truncated)"
        else:
            forward_code_str = forward_code
    else:
        forward_code_str = "Not available"

    # 모드에 따른 task description
    if is_random_reset:
        task_title = "RANDOM RESET (Shuffle Objects)"
        task_description = f"""Generate executable Python code that **moves each object to a NEW RANDOM position**.

This is NOT restoring to initial state - this is shuffling objects to new locations
for the next training episode. The target positions below are randomly generated
within the robot's workspace."""
        target_label = "Random Target Positions (Generated)"
    else:
        task_title = "RESET to Initial Environment"
        task_description = f"""Generate executable Python code that **restores the environment to its initial state**
(before "{original_instruction}" was executed).

Use the forward spec and code above as reference to understand what was done,
then generate code that returns the environment to its initial state."""
        target_label = "Initial State (Before Forward Execution)"

    prompt = f"""You are an AI assistant tasked with generating executable Python code for a LeRobot SO-101 robot arm.
Your task is to generate reset code that moves objects from their current positions to target positions.

### **Context: Forward Execution (What Was Done)**

1. **Forward Task Goal**:
   ```
   {original_instruction}
   ```

2. **Forward Spec (Step-by-Step Plan)**:
   ```json
   {forward_spec_str}
   ```

3. **Forward Generated Code**:
   ```python
   {forward_code_str}
   ```

4. **Environment States**:
   Object position z-coordinate = object height (table surface is z=0).
   - **{target_label}**:
     ```python
     target_positions = {{
{target_str}
     }}
     ```
   - **Current State** (Detected by Camera):
     ```python
     current_positions = {{
{current_str}
     }}
     ```

### **Your Task: {task_title}**

{task_description}

### **Available Skills**:
| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `connect()` | Connect to robot | - |
| `disconnect()` | Disconnect from robot | - |
| `gripper_open(skill_description=None)` | Open gripper | skill_description: str |
| `move_to_initial_state(skill_description=None)` | Move to initial/home position | skill_description: str |
| `move_to_free_state(skill_description=None)` | Move to safe parking position | skill_description: str |
| `move_to_position(position, ..., skill_description=None)` | Move end-effector to [x,y,z] | position, target_name, skill_description: str |
| `rotate_90degree(direction, skill_description=None)` | Rotate gripper 90° | direction: 1 (CW) or -1 (CCW), skill_description: str |
| `execute_pick_object(object_position, ..., skill_description=None)` | Descend to pick position (3cm from top), close gripper, save pitch | object_position, **object_name**: str, skill_description: str |
| `execute_place_object(place_position, ..., skill_description=None)` | Descend to place position with saved pitch, open gripper 70% | place_position, is_table, gripper_open_ratio, **target_name**: str, skill_description: str |

**execute_pick_object**: Call from pick_approach position. Moves TCP to grasp height, closes gripper, and **saves current pitch**.
  - **IMPORTANT**: Pass the object position as-is. The function internally handles the grasp height offset.
  - **object_name**: Pass the object name for subgoal labeling (e.g., "red cup")
**execute_place_object**: Call from place_approach position. Moves TCP to place height **with saved pitch restored**, then opens gripper.
  - **IMPORTANT**: Pass the target position as-is. The function internally calculates the correct release height.
  - is_table=True: place on table (z=0), is_table=False: place on another object
  - **ALWAYS use `gripper_open_ratio=0.7`** to open gripper 70%
  - **target_name**: Pass the target name for subgoal labeling (e.g., "table", "original position")
  - **NOTE**: `execute_place_object` does NOT have `object_name` parameter. Use `target_name` instead.
  - Pitch is automatically restored from pick time

**skill_description (REQUIRED)**: A natural language sentence describing the semantic intent of each skill call.
  - This is recorded as `skill.natural_language` in the dataset for robot policy learning.
  - Describe WHY the robot is performing this action in context of the reset task.
  - Each call MUST have a unique, descriptive `skill_description`.
  - Use "initial position" (not "home position") and "free position" (not "parking position") to match skill names.

### **Skill Composition Patterns** (MUST follow exactly)

```python
# PICK pattern — ALWAYS open gripper BEFORE approaching the object
cx, cy, cz = <current object position x, y, z>
approach_height = 0.20

skills.gripper_open(skill_description="open gripper to prepare for picking the <object_name>")
skills.move_to_position([cx, cy, approach_height], target_name="<object_name>", skill_description="approach above the <object_name> for grasping")
skills.execute_pick_object([cx, cy, cz], object_name="<object_name>", skill_description="descend and grasp the <object_name>")
skills.move_to_position([cx, cy, approach_height], target_name="<object_name>", skill_description="lift the <object_name> to safe height after grasping")

# PLACE ON TABLE pattern — is_table=True
tx, ty, tz = <target position x, y, z>

skills.move_to_position([tx, ty, approach_height], target_name="original position", skill_description="move above the target to place the <object_name>")
skills.execute_place_object([tx, ty, tz], is_table=True, gripper_open_ratio=0.7, target_name="original position", skill_description="lower the <object_name> onto the target position")
skills.move_to_position([tx, ty, approach_height], target_name="original position", skill_description="retreat upward after placing the <object_name>")
```

### **Code Template**:

```python
from skills.skills_lerobot import LeRobotSkills

def execute_task():
    '''Move objects from current positions to target positions.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="{frame}",
    )
    skills.connect()

    try:
        approach_height = 0.20  # 20cm above objects

        # Target positions (where to place objects)
        target_positions = {{
{target_str}
        }}

        # Current positions (from detection)
        current_positions = {{
{current_str}
        }}

        skills.move_to_initial_state(skill_description="move to initial position to start the reset task")

        # === RESET: Move objects to target positions ===
        # ALWAYS reference current_positions["name"]["position"] and target_positions["name"]["position"]
        # Do NOT hardcode any coordinate values

        # ... (your reset logic referencing the dicts) ...

        skills.move_to_initial_state(skill_description="return to initial position after completing the reset")
        skills.move_to_free_state(skill_description="move to free position for safe parking")

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

### **Example: Forward Task → Reset Code**

**Forward Task**: "pick up the red cup and place it on the blue box"

**Environment States**:
- Initial: red cup at [0.15, 0.05, 0.02], blue box at [0.20, -0.05, 0.03]
- Current: red cup at [0.20, -0.05, 0.05] (on blue box), blue box at [0.20, -0.05, 0.03]

**Reset Code** (moves red cup back to its original position on table):
```python
from skills.skills_lerobot import LeRobotSkills

def execute_task():
    '''Move objects to target positions.'''

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot3.yaml",
        frame="{frame}",
    )
    skills.connect()

    try:
        approach_height = 0.20

        # Current position (from detection) - red cup is on blue box
        current_pos = [0.20, -0.05, 0.05]

        # Target position (where to place) - original position on table
        target_pos = [0.15, 0.05, 0.02]

        skills.move_to_initial_state(skill_description="move to initial position to start the reset task")

        # === PICK red cup from current position ===
        skills.gripper_open(skill_description="open gripper to prepare for picking the red cup")
        skills.move_to_position([current_pos[0], current_pos[1], approach_height], target_name="red cup", skill_description="approach above the red cup for grasping")
        skills.execute_pick_object(current_pos, object_name="red cup", skill_description="descend and grasp the red cup")
        skills.move_to_position([current_pos[0], current_pos[1], approach_height], target_name="red cup", skill_description="lift the red cup to safe height after grasping")

        # === PLACE red cup at target position (on table) ===
        skills.move_to_position([target_pos[0], target_pos[1], approach_height], target_name="original position", skill_description="move above the original position to place the red cup back")
        skills.execute_place_object(target_pos, is_table=True, gripper_open_ratio=0.7, target_name="original position", skill_description="lower the red cup onto its original position on the table")
        skills.move_to_position([target_pos[0], target_pos[1], approach_height], target_name="original position", skill_description="retreat upward after placing the red cup")

        skills.move_to_initial_state(skill_description="return to initial position after completing the reset")
        skills.move_to_free_state(skill_description="move to free position for safe parking")

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

### **Guidelines**

1. Analyze the forward spec and code to understand what was moved
2. Generate code that moves each object from current to target position
3. **Follow the Skill Composition Patterns above exactly** — especially `gripper_open()` BEFORE every pick approach
4. Use `execute_pick_object` (with `object_name=`) and `execute_place_object` (with `target_name=`) for pick/place operations
5. **NEVER pass `object_name` to `execute_place_object`** — it only accepts `target_name`
6. **ALWAYS reference `current_positions` and `target_positions` dicts** — e.g. `current_positions["name"]["position"]` and `target_positions["name"]["position"]`
7. Do NOT hardcode coordinate values — the dicts are injected as global variables at runtime and may change between episodes
8. **ALWAYS pass object/target positions as-is** to execute_pick_object and execute_place_object (grasp offset handled internally)
9. Use `approach_height = 0.20` (20cm) for all approach/lift movements
10. **Pitch Handling**: Pitch is automatically saved at pick and restored at place. No need for maintain_pitch during movement
11. **ALWAYS use `gripper_open_ratio=0.7`** in execute_place_object
12. Use `is_table=True` when placing on table
13. Always include try/finally for proper cleanup
14. Always start with `move_to_initial_state()`, end with `move_to_initial_state()` and `move_to_free_state()`
15. **ALWAYS pass unique `skill_description`** for EVERY skill call

### **Output Format**
- Provide complete executable Python code
- Do not use code blocks in your final answer
- Include all imports and the complete execute_task() function

**Generate the complete executable RESET code:**
"""

    return prompt


# ============================================================
# Multi-Turn Reset Prompts
# ============================================================

def turn0_reset_scene_understanding_prompt(
    original_instruction: str,
    reset_mode: str,
    workspace_bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None,
    original_object_labels: List[str] = None,
) -> str:
    """
    Turn 0: Reset 장면 이해 프롬프트 (VLM multi-turn 용).

    Image 1 = 현재 상태 (workspace 시각화 포함)
    Image 2 = 초기 상태 (forward 실행 전)

    Args:
        original_instruction: 원래 forward 태스크 명령
        reset_mode: "original" | "random"
        workspace_bounds: (legacy, 미사용)
        original_object_labels: Forward에서 검출된 원래 물체 라벨 리스트
    """
    workspace_desc = """\
In Image 1, the robot's reachable workspace is visually marked:
- The BRIGHT area shows the robot's reachable donut-shaped workspace (between min and max reach from the robot base)
- The DARKENED areas are UNREACHABLE by the robot (too close to robot base, or too far away)
- CYAN DOTTED ARCS show the inner (min reach) and outer (max reach) boundaries
- GREEN RECTANGLE shows the camera FOV safe margin (30px inset) — objects must stay within this rectangle
All object placements MUST be within the intersection of the bright donut area AND the green rectangle."""

    if reset_mode == "original":
        mode_desc = f"""\
**RESET MODE: ORIGINAL (restore to initial state)**
- Image 2 shows the INITIAL state (before "{original_instruction}" was executed).
- Your goal is to identify which objects were moved by the forward task, and determine how to return them to their initial positions shown in Image 2.
- Compare Image 1 (current) vs Image 2 (initial) to find which objects have changed position."""
    else:
        mode_desc = f"""\
**RESET MODE: RANDOM (shuffle to new positions)**
- Image 2 shows the INITIAL state (before "{original_instruction}" was executed).
- Your goal is to identify which objects were manipulated during the forward task.
- These objects will be moved to NEW random positions (computed programmatically).
- Compare Image 1 (current) vs Image 2 (initial) to identify which objects were manipulated."""

    # 원래 물체 라벨 안내
    if original_object_labels:
        labels_str = ", ".join(f'"{l}"' for l in original_object_labels)
        label_guidance = f"""
**IMPORTANT — Object Labels**:
During the forward task, the following objects were detected: {labels_str}.
When referring to objects in your analysis, use these EXACT labels to maintain consistency.
If the same physical object type appears multiple times (e.g., two cups), use the labels above
and map them to the objects you see in the current image."""
    else:
        label_guidance = ""

    return f"""\
You are given **two images**:
1. **Image 1 (Current state)** — the workspace AFTER the forward task "{original_instruction}" was executed.
2. **Image 2 (Initial state)** — the workspace BEFORE the forward task was executed.

{workspace_desc}

{mode_desc}
{label_guidance}

Analyze the scene and describe:
1. **Object identification**: What objects are visible in both images? Describe each object's color, shape, and approximate size.
2. **Change analysis**: Compare Image 1 vs Image 2. Which objects changed position? Where were they before, and where are they now?
3. **Reset plan**: Which objects need to be moved for the reset? In what order should they be moved?

**Important**:
- This is a scene understanding step ONLY.
- Do NOT generate any code, numeric coordinates, or step-by-step execution plans.
- Focus purely on visual observation and spatial analysis."""


def turn1_reset_bbox_detection_prompt(
    original_object_labels: List[str] = None,
) -> str:
    """
    Turn 1 (Reset 전용): Forward에서 검출된 라벨을 강제 사용하여 bbox 검출.

    Forward의 turn1_prompt와 달리, VLM이 라벨을 자유롭게 생성하지 않고
    Forward에서 사용한 정확한 라벨을 그대로 사용하도록 강제합니다.

    Args:
        original_object_labels: Forward에서 검출된 물체 라벨 리스트 (필수)
    """
    if original_object_labels:
        labels_json = ", ".join(f'"{l}"' for l in original_object_labels)
        label_instruction = f"""
**CRITICAL — You MUST use these EXACT labels**: [{labels_json}]
These are the object labels from the forward task detection. Each label corresponds to a specific physical object.
- Do NOT rename, rephrase, or modify these labels in any way.
- Do NOT use spaces instead of underscores or vice versa — copy the labels exactly as given.
- Match each label to the corresponding object you identified in the scene analysis above."""
    else:
        label_instruction = """
**Every label must be unique.** If multiple objects of the same type exist, append a numeric suffix to distinguish them (e.g., `"egg_1"`, `"egg_2"`, `"red_plate_1"`, `"red_plate_2"`)."""

    return f"""\
Now, for each task-relevant object you identified above, detect their bounding boxes in the given image (overhead camera image).

For each object, provide:
1. **box_2d**: Bounding box as `[ymin, xmin, ymax, xmax]` — exactly 4 integers, each normalized to 0–1000 (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of this image).
2. **label**: The object label.
{label_instruction}

### Output Format
Return a JSON array:
```json
[
  {{"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}},
  {{"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}}
]
```

**Important**:
- Only include the main task-relevant objects (not sub-parts).
- Focus on providing accurate bounding box coordinates.
- **Carefully match each bounding box to the correct label** by comparing the visual appearance of each detected object with your analysis from above. Do NOT swap labels between objects."""


def turn_codegen_reset_prompt(
    target_positions: Dict,
    current_positions: Dict,
    robot_id: int,
    is_random_reset: bool,
    workspace_bounds=None,
    all_points: Dict = None,
) -> str:
    """
    CodeGen Turn: Reset 코드 생성 프롬프트 (VLM multi-turn 용).

    Args:
        target_positions: 목표 위치 dict (extended format)
        current_positions: 현재 위치 dict (extended format)
        robot_id: 로봇 번호 (2 or 3)
        is_random_reset: True면 random mode, False면 original mode
        workspace_bounds: (legacy, 미사용)
        all_points: VLM Turn 2+에서 검출된 모든 포인트 (optional, for context)
    """
    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"
    frame = _get_frame_for_robot(robot_id)

    # Helper to extract position
    def get_pos(info):
        if info is None:
            return None
        elif isinstance(info, dict) and "position" in info:
            return info["position"]
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            return list(info[:3])
        return None

    # Format target positions
    target_lines = []
    for name, info in target_positions.items():
        pos = get_pos(info)
        if pos is not None:
            target_lines.append(
                f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
            )
    target_str = "\n".join(target_lines)

    # Format current positions
    current_lines = []
    for name, info in current_positions.items():
        pos = get_pos(info)
        if pos is not None:
            current_lines.append(
                f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
            )
    current_str = "\n".join(current_lines)

    # Mode description
    if is_random_reset:
        task_title = "RANDOM RESET (Shuffle Objects)"
        task_desc = "Move each object to its NEW RANDOM target position."
        target_label = "Random Target Positions (Generated)"
    else:
        task_title = "RESET to Initial Environment"
        task_desc = "Restore each object to its initial position (before the forward task)."
        target_label = "Initial Positions (Before Forward Execution)"

    return f"""\
Now generate executable Python code for the reset task.

### **Task: {task_title}**
{task_desc}

### **WORKSPACE CONSTRAINT**
The robot's reachable workspace is the intersection of: (1) a donut-shaped reach area, and (2) the camera FOV safe margin (green rectangle).
All target positions are pre-validated to be within this area.
Do NOT modify the provided target positions.

### **Environment States**
Object position z-coordinate = object height (table surface is z=0).

- **{target_label}**:
  ```python
  target_positions = {{
{target_str}
  }}
  ```
- **Current State** (Detected by VLM + pix2world):
  ```python
  current_positions = {{
{current_str}
  }}
  ```

### **Available Skills**
| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `connect()` | Connect to robot | - |
| `disconnect()` | Disconnect from robot | - |
| `gripper_open(skill_description=None)` | Open gripper | skill_description: str |
| `move_to_initial_state(skill_description=None)` | Move to initial/home position | skill_description: str |
| `move_to_free_state(skill_description=None)` | Move to safe parking position | skill_description: str |
| `move_to_position(position, ..., skill_description=None)` | Move end-effector to [x,y,z] | position, target_name, skill_description: str |
| `rotate_90degree(direction, skill_description=None)` | Rotate gripper 90° | direction: 1 (CW) or -1 (CCW), skill_description: str |
| `execute_pick_object(object_position, ..., skill_description=None)` | Descend to pick position (3cm from top), close gripper, save pitch | object_position, **object_name**: str, skill_description: str |
| `execute_place_object(place_position, ..., skill_description=None)` | Descend to place position with saved pitch, open gripper 70% | place_position, is_table, gripper_open_ratio, **target_name**: str, skill_description: str |

**execute_pick_object**: Call from pick_approach position. Moves TCP to grasp height, closes gripper, and **saves current pitch**.
  - **IMPORTANT**: Pass the object position as-is. The function internally handles the grasp height offset.
  - **object_name**: Pass the object name for subgoal labeling (e.g., "red cup")
**execute_place_object**: Call from place_approach position. Moves TCP to place height **with saved pitch restored**, then opens gripper.
  - **IMPORTANT**: Pass the target position as-is. The function internally calculates the correct release height.
  - is_table=True: place on table (z=0), is_table=False: place on another object
  - **ALWAYS use `gripper_open_ratio=0.7`** to open gripper 70%
  - **target_name**: Pass the target name for subgoal labeling (e.g., "table", "original position")
  - **NOTE**: `execute_place_object` does NOT have `object_name` parameter. Use `target_name` instead.
  - Pitch is automatically restored from pick time

**skill_description (REQUIRED)**: A natural language sentence describing the semantic intent of each skill call.
  - Describe WHY the robot is performing this action in context of the reset task.
  - Each call MUST have a unique, descriptive `skill_description`.
  - Use "initial position" (not "home position") and "free position" (not "parking position") to match skill names.

### **Skill Composition Patterns** (MUST follow exactly)

```python
# PICK pattern — ALWAYS open gripper BEFORE approaching the object
cx, cy, cz = <current object position x, y, z>
approach_height = 0.20

skills.gripper_open(skill_description="open gripper to prepare for picking the <object_name>")
skills.move_to_position([cx, cy, approach_height], target_name="<object_name>", skill_description="approach above the <object_name> for grasping")
skills.execute_pick_object([cx, cy, cz], object_name="<object_name>", skill_description="descend and grasp the <object_name>")
skills.move_to_position([cx, cy, approach_height], target_name="<object_name>", skill_description="lift the <object_name> to safe height after grasping")

# PLACE ON TABLE pattern — is_table=True
tx, ty, tz = <target position x, y, z>

skills.move_to_position([tx, ty, approach_height], target_name="original position", skill_description="move above the target to place the <object_name>")
skills.execute_place_object([tx, ty, tz], is_table=True, gripper_open_ratio=0.7, target_name="original position", skill_description="lower the <object_name> onto the target position")
skills.move_to_position([tx, ty, approach_height], target_name="original position", skill_description="retreat upward after placing the <object_name>")
```

### **Code Template**

```python
from skills.skills_lerobot import LeRobotSkills

def execute_task():
    '''Move objects from current positions to target positions.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="{frame}",
    )
    skills.connect()

    try:
        approach_height = 0.20  # 20cm above objects

        # Target positions (where to place objects)
        target_positions = {{
{target_str}
        }}

        # Current positions (from detection)
        current_positions = {{
{current_str}
        }}

        skills.move_to_initial_state(skill_description="move to initial position to start the reset task")

        # === RESET: Move objects to target positions ===
        # ALWAYS reference current_positions["name"]["position"] and target_positions["name"]["position"]
        # Do NOT hardcode any coordinate values

        # ... (your reset logic referencing the dicts) ...

        skills.move_to_initial_state(skill_description="return to initial position after completing the reset")
        skills.move_to_free_state(skill_description="move to free position for safe parking")

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

### **Guidelines**

1. Analyze the scene understanding from Turn 0 and the detected positions above
2. Generate code that moves each object from current to target position
3. **Follow the Skill Composition Patterns above exactly** — especially `gripper_open()` BEFORE every pick approach
4. Use `execute_pick_object` (with `object_name=`) and `execute_place_object` (with `target_name=`) for pick/place operations
5. **NEVER pass `object_name` to `execute_place_object`** — it only accepts `target_name`
6. **ALWAYS reference `current_positions` and `target_positions` dicts** — e.g. `current_positions["name"]["position"]` and `target_positions["name"]["position"]`
7. Do NOT hardcode coordinate values — the dicts are injected as global variables at runtime and may change between episodes
8. **ALWAYS pass object/target positions as-is** to execute_pick_object and execute_place_object (grasp offset handled internally)
9. Use `approach_height = 0.20` (20cm) for all approach/lift movements
10. **Pitch Handling**: Pitch is automatically saved at pick and restored at place. No need for maintain_pitch during movement
11. **ALWAYS use `gripper_open_ratio=0.7`** in execute_place_object
12. Use `is_table=True` when placing on table
13. Always include try/finally for proper cleanup
14. Always start with `move_to_initial_state()`, end with `move_to_initial_state()` and `move_to_free_state()`
15. **ALWAYS pass unique `skill_description`** for EVERY skill call

### **Output Format**
- Provide complete executable Python code
- Do not include markdown code blocks
- Include all imports and the complete execute_task() function

**Generate the complete executable RESET code:**"""


def reset_context_summary_prompt() -> str:
    """Session 1 마지막에 reset 컨텍스트 요약을 요청하는 프롬프트"""
    return """### Context Handoff Summary (Reset)

Before we move to reset code generation, summarize your understanding concisely (under 300 words):

1. **Current Scene**: Where each object is now (after the forward task)
2. **Object Properties**: Each object's size, shape, graspability
3. **Spatial Relationships**: Which objects are near each other, stacking if any
4. **Reset Strategy**: Which objects to move first, any ordering constraints (e.g., unstacking)
5. **Potential Risks**: Objects too large to grip, collision risks, workspace boundary issues

Output as a structured summary. This will be passed to a fresh code generation session."""


def codegen_reset_with_context_prompt(
    context_summary: str,
    target_positions: Dict,
    current_positions: Dict,
    robot_id: int,
    is_random_reset: bool,
    workspace_bounds=None,
    all_points: Dict = None,
) -> str:
    """
    새로운 chat session에서 컨텍스트 요약과 함께 reset 코드를 생성하는 프롬프트.
    Session 1의 누적 토큰 없이 깨끗한 세션에서 코드 생성.
    """
    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"
    frame = _get_frame_for_robot(robot_id)

    def get_pos(info):
        if info is None:
            return None
        elif isinstance(info, dict) and "position" in info:
            return info["position"]
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            return list(info[:3])
        return None

    target_lines = []
    for name, info in target_positions.items():
        pos = get_pos(info)
        if pos is not None:
            target_lines.append(
                f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
            )
    target_str = "\n".join(target_lines)

    current_lines = []
    for name, info in current_positions.items():
        pos = get_pos(info)
        if pos is not None:
            current_lines.append(
                f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
            )
    current_str = "\n".join(current_lines)

    if is_random_reset:
        task_title = "RANDOM RESET (Shuffle Objects)"
        task_desc = "Move each object to its NEW RANDOM target position."
        target_label = "Random Target Positions (Generated)"
    else:
        task_title = "RESET to Initial Environment"
        task_desc = "Restore each object to its initial position (before the forward task)."
        target_label = "Initial Positions (Before Forward Execution)"

    return f"""\
### Scene Context (from prior analysis session)

{context_summary}

### Code Generation Task: {task_title}
{task_desc}

### **WORKSPACE CONSTRAINT**
The robot's reachable workspace is the intersection of: (1) a donut-shaped reach area, and (2) the camera FOV safe margin (green rectangle).
All target positions are pre-validated to be within this area.
Do NOT modify the provided target positions.

### **Environment States**
Object position z-coordinate = object height (table surface is z=0).

- **{target_label}**:
  ```python
  target_positions = {{
{target_str}
  }}
  ```
- **Current State** (Detected by VLM + pix2world):
  ```python
  current_positions = {{
{current_str}
  }}
  ```

### **Available Skills**
| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `connect()` | Connect to robot | - |
| `disconnect()` | Disconnect from robot | - |
| `gripper_open()` | Open gripper | - |
| `move_to_initial_state()` | Move to initial/home position | - |
| `move_to_free_state()` | Move to safe parking position | - |
| `move_to_position(position, ...)` | Move end-effector to [x,y,z] | position, target_name |
| `rotate_90degree(direction)` | Rotate gripper 90° | direction: 1 (CW) or -1 (CCW) |
| `execute_pick_object(object_position, ...)` | Descend to pick, close gripper, save pitch | object_position, object_name |
| `execute_place_object(place_position, ...)` | Descend to place with saved pitch, open gripper | place_position, is_table, gripper_open_ratio, target_name |

**execute_pick_object**: Pass the object position as-is. The function internally handles the grasp height offset.
**execute_place_object**: Pass the target position as-is. The function internally calculates the correct release height.
  - is_table=True: place on table, is_table=False: place on another object
  - **ALWAYS use `gripper_open_ratio=0.7`**

### **Skill Composition Patterns** (MUST follow exactly)

```python
# PICK from current position — ALWAYS reference current_positions dict
cur = current_positions["object_name"]["position"]
approach_height = 0.20

skills.gripper_open()
skills.move_to_position([cur[0], cur[1], approach_height], target_name="object_name")
skills.execute_pick_object(cur, object_name="object_name")
skills.move_to_position([cur[0], cur[1], approach_height], target_name="object_name")

# PLACE at target position — ALWAYS reference target_positions dict
tgt = target_positions["object_name"]["position"]

skills.move_to_position([tgt[0], tgt[1], approach_height], target_name="object_name target")
skills.execute_place_object(tgt, is_table=True, gripper_open_ratio=0.7, target_name="object_name target")
skills.move_to_position([tgt[0], tgt[1], approach_height], target_name="object_name target")
```

### **Code Template**

**CRITICAL**: You MUST reference `current_positions` and `target_positions` dicts in skill calls. Do NOT hardcode coordinate values directly. The dicts are provided as global variables at runtime.

```python
from skills.skills_lerobot import LeRobotSkills

def execute_task():
    '''Move objects from current positions to target positions.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="{frame}",
    )
    skills.connect()

    try:
        approach_height = 0.20

        skills.move_to_initial_state()

        # === RESET: Move each object from current to target ===
        # ALWAYS use: current_positions["name"]["position"] and target_positions["name"]["position"]
        # Do NOT hardcode any coordinate values

        # ... (your reset logic referencing the dicts) ...

        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

### **Unstacking (Disassembling a Stack)**

When objects are stacked (one object sitting on top of another), you MUST unstack from **top to bottom**.

- **How to detect stacking from z-values**:
  Each object's z-value = the height of its top surface from the table (z=0).
  When objects are on the table, z ≈ the object's own height.
  When stacked, z = sum of heights below + own height, so z is much higher than its original z.
  **The object with the highest z is the topmost** — always pick it first.
  Example with 3 blocks stacked (A bottom, B middle, C top), each block ~2cm tall:
    - A (on table): z ≈ 0.02 (its own height ~2cm)
    - B (on A): z ≈ 0.06 (A's height + B's height ≈ 6cm)
    - C (on B): z ≈ 0.08 (A's height + B's height + C's height ≈ 8cm)
  → Pick order: C (z=0.08) → B (z=0.06) → A (z=0.02)

- **Order**: Always pick the topmost object first (highest z). Never pick a lower object while something is on top.
- **Pick from stack**: Pass the current (elevated) position as-is to `execute_pick_object` — the function handles grasp height internally.
- **Place on table**: Use `is_table=True` when placing the unstacked object back to its target (table-level) position.

```python
# UNSTACK pattern — always pick highest-z object first
# Example: 3 blocks stacked — C(top, z=0.08), B(middle, z=0.06), A(bottom, z=0.02)

# Step 1: Pick the TOP object C (highest z = 0.08)
skills.gripper_open()
skills.move_to_position([cx, cy, approach_height], target_name="C")
skills.execute_pick_object([cx, cy, 0.08], object_name="C")
skills.move_to_position([cx, cy, approach_height], target_name="C")

skills.move_to_position([c_tx, c_ty, approach_height], target_name="original position")
skills.execute_place_object([c_tx, c_ty, c_tz], is_table=True, gripper_open_ratio=0.7, target_name="original position")
skills.move_to_position([c_tx, c_ty, approach_height], target_name="original position")

# Step 2: Pick the MIDDLE object B (z = 0.06) — safe because C is removed
skills.gripper_open()
skills.move_to_position([bx, by, approach_height], target_name="B")
skills.execute_pick_object([bx, by, 0.06], object_name="B")
skills.move_to_position([bx, by, approach_height], target_name="B")

skills.move_to_position([b_tx, b_ty, approach_height], target_name="original position")
skills.execute_place_object([b_tx, b_ty, b_tz], is_table=True, gripper_open_ratio=0.7, target_name="original position")
skills.move_to_position([b_tx, b_ty, approach_height], target_name="original position")

# Step 3: Pick the BOTTOM object A (z = 0.02) — safe because B and C are removed
skills.gripper_open()
skills.move_to_position([ax, ay, approach_height], target_name="A")
skills.execute_pick_object([ax, ay, 0.02], object_name="A")
skills.move_to_position([ax, ay, approach_height], target_name="A")

skills.move_to_position([a_tx, a_ty, approach_height], target_name="original position")
skills.execute_place_object([a_tx, a_ty, a_tz], is_table=True, gripper_open_ratio=0.7, target_name="original position")
skills.move_to_position([a_tx, a_ty, approach_height], target_name="original position")
```

### **Guidelines**

1. Generate code that moves each object from current to target position
2. **Follow the Skill Composition Patterns above exactly** — especially `gripper_open()` BEFORE every pick approach
3. **Hardcode actual coordinate values** from current_positions and target_positions directly in the code
4. Do NOT reference `current_positions` or `target_positions` as variables - extract and use the actual [x,y,z] values
5. **ALWAYS pass object/target positions as-is** to execute_pick_object and execute_place_object
6. Use `approach_height = 0.20` (20cm) for all approach/lift movements
7. **Pitch Handling**: Pitch is automatically saved at pick and restored at place
8. **ALWAYS use `gripper_open_ratio=0.7`** in execute_place_object
9. Use `is_table=True` when placing on table
10. Always include try/finally for proper cleanup
11. Always start with `move_to_initial_state()`, end with `move_to_initial_state()` and `move_to_free_state()`
12. **Unstacking**: If objects are stacked, always unstack from top to bottom before moving them

### **Output Format**
- Provide complete executable Python code
- Do not include markdown code blocks

**Generate the complete executable RESET code:**"""
