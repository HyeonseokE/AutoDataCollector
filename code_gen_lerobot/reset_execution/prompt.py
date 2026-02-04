"""
Reset Execution Prompts

LLM prompts for reset/reverse task execution.
Uses forward execution context (spec, code) to understand what was done,
then generates code to restore the environment to its initial state.
"""

import json
from typing import Dict, List, Optional


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
| `move_to_position` | Move end-effector to position | object: str, offset_z: float, apply_gripper_offset: bool |
| `execute_pick_object` | Descend (3cm from top) + gripper_close + save pitch | object: str, apply_gripper_offset: bool |
| `execute_place_object` | Descend with saved pitch + gripper_open | target: str, apply_gripper_offset: bool, is_table: bool |

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

    # Helper to extract position and gripper_offset from extended or legacy format
    def get_position_and_offset(info):
        if info is None:
            return None, 0.0
        elif isinstance(info, dict) and "position" in info:
            pos = info["position"]
            offset = info.get("gripper_offset", 0.02)
            return pos, offset
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            return list(info[:3]), 0.02  # default offset
        return None, 0.0

    # 타겟 위치 포맷팅 (simple list - target은 gripper_offset 불필요)
    target_lines = []
    for name, info in target_positions.items():
        pos, _ = get_position_and_offset(info)
        if pos is not None:
            target_lines.append(f'        "{name}": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}],')
    target_str = "\n".join(target_lines)

    # 현재 위치 포맷팅 (extended format - gripper_offset 포함)
    current_lines = []
    for name, info in current_positions.items():
        pos, offset = get_position_and_offset(info)
        if pos is not None:
            current_lines.append(f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], "gripper_offset": {offset:.4f}}},')
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
| Method | Description | Parameters |
|--------|-------------|------------|
| `connect()` | Connect to robot | - |
| `disconnect()` | Disconnect from robot | - |
| `gripper_open()` | Open gripper | - |
| `move_to_initial_state()` | Move to initial/home position | - |
| `move_to_free_state()` | Move to safe parking position | - |
| `move_to_position(position, gripper_offset=0.0)` | Move end-effector to [x,y,z] | position: List[float], gripper_offset: float |
| `rotate_90degree(direction)` | Rotate gripper 90 deg | direction: 1 (CW) or -1 (CCW) |
| `execute_pick_object(object_position, gripper_offset=0.0)` | Descend (3cm from top) + gripper_close + save pitch | object_position: [x,y,z], gripper_offset: float |
| `execute_place_object(place_position, gripper_offset=0.0, is_table=True)` | Descend with saved pitch + gripper_open | place_position: [x,y,z], gripper_offset: float, is_table: bool |

**execute_pick_object**: Call from pick_approach position. Moves TCP to 3cm below object top, closes gripper, and **saves current pitch**.
**execute_place_object**: Call from place_approach position. Moves TCP to place height **with saved pitch restored**, then opens gripper.
  - is_table=True: place on table (z=0)
  - is_table=False: place on another object
  - Pitch is automatically restored from pick time

### **Code Template**:

```python
from skills.skills_lerobot import LeRobotSkills

def execute_reset_task():
    '''Move objects from current positions to target positions.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="world",
    )
    skills.connect()

    try:
        approach_height = 0.15  # 15cm above objects

        # Target positions (where to place objects)
        target_positions = {{
{target_str}
        }}

        # Current positions (from detection)
        current_positions = {{
{current_str}
        }}

        skills.move_to_initial_state()

        # === RESET: Move objects to target positions ===
        # Extract actual [x,y,z] values and hardcode them below
        # Example:
        #   obj_current = [0.20, -0.05, 0.05]  # from current_positions
        #   obj_target = [0.15, 0.05, 0.02]    # from target_positions
        #   offset = 0.02

        # ... (your reset logic with hardcoded values) ...

        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_reset_task()
```

### **Example: Forward Task → Reset Code**

**Forward Task**: "pick up the red cup and place it on the blue box"

**Environment States**:
- Initial: red cup at [0.15, 0.05, 0.02], blue box at [0.20, -0.05, 0.03]
- Current: red cup at [0.20, -0.05, 0.05] (on blue box), blue box at [0.20, -0.05, 0.03]

**Reset Code** (moves red cup to target position on table):
```python
from skills.skills_lerobot import LeRobotSkills

def execute_reset_task():
    '''Move objects to target positions.'''

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot3.yaml",
        frame="world",
    )
    skills.connect()

    try:
        approach_height = 0.15

        # Current position (from detection) - red cup is on blue box
        current_pos = [0.20, -0.05, 0.05]
        offset = 0.02  # gripper_offset

        # Target position (where to place) - original position on table
        target_pos = [0.15, 0.05, 0.02]

        skills.move_to_initial_state()

        # === PICK red cup from current position ===
        skills.gripper_open()
        skills.move_to_position([current_pos[0], current_pos[1], current_pos[2] + approach_height], gripper_offset=offset)
        skills.execute_pick_object(current_pos, gripper_offset=offset)
        skills.move_to_position([current_pos[0], current_pos[1], current_pos[2] + approach_height])

        # === PLACE red cup at target position (on table) ===
        skills.move_to_position([target_pos[0], target_pos[1], approach_height], gripper_offset=offset)
        skills.execute_place_object(target_pos, gripper_offset=offset, is_table=True)
        skills.move_to_position([target_pos[0], target_pos[1], approach_height], gripper_offset=offset)

        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_reset_task()
```

### **Guidelines**

1. Analyze the forward spec and code to understand what was moved
2. Generate code that moves each object from current to target position
3. Use `execute_pick_object` and `execute_place_object` for pick/place operations
4. **Hardcode actual coordinate values** from current_positions and target_positions directly in the code
5. Do NOT reference `current_positions` or `target_positions` as variables - extract and use the actual [x,y,z] values
6. Use `is_table=True` when placing on table
7. Always include try/finally for proper cleanup
8. Always start with `move_to_initial_state()`
9. Always end with `move_to_initial_state()` and `move_to_free_state()`

### **Output Format**
- Provide complete executable Python code
- Do not use code blocks in your final answer
- Include all imports and the complete execute_reset_task() function

**Generate the complete executable RESET code:**
"""

    return prompt
