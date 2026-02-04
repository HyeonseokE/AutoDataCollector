"""
Forward Execution Prompts

LLM prompts for forward task execution (code generation and spec generation).
"""

from typing import Dict, List, Optional


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
                         Extended format: {name: {"position": [x,y,z], "gripper_offset": float}}
        spec: 코드 생성 가이드라인/스펙 (선택)
        robot_id: 로봇 번호 (2 또는 3)

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    # 로봇 설정 파일 경로
    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"

    # 객체 위치 포맷팅 (extended format with gripper_offset)
    positions_lines = []
    for name, info in object_positions.items():
        if info is not None:
            pos = info["position"]
            offset = info.get("gripper_offset", 0.0)
            positions_lines.append(
                f'    "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], "gripper_offset": {offset:.3f}}},'
            )
    positions_str = "\n".join(positions_lines)

    # 검출 실패한 객체
    not_found = [name for name, info in object_positions.items() if info is None]
    not_found_str = ", ".join(not_found) if not_found else "None"

    prompt = f"""You are an AI assistant tasked with generating executable Python code that controls a LeRobot SO-101 robot arm.
    The goal is to ensure the robot can achieve the specified objective by executing a sequence of skill actions.

### **Input Details:**

    1. **Goal (Natural Language Description)**:
       The goal specifies what the robot must accomplish, described in plain language.

       Provided Goal Description:
       ```
       {instruction}
       ```

    2. **Object Positions (World Coordinate Frame, unit: meters)**:
       The detected target object positions in the environment.
       Object position z-coordinate = object height (table surface height is z=0).
       ```python
       positions = {{
      {positions_str}
       }}
       ```
       Objects not found: {not_found_str}

    3. **Specification (Code Generation Guidelines)**:
       The specification outlines the exact requirements and constraints for generating the executable Python code.
       ```
       {spec}
       ```

    4. **Available Skills**:
       The LeRobotSkills class provides these methods:

       | Method | Description | Parameters |
       |--------|-------------|------------|
       | `connect()` | Connect to robot | - |
       | `disconnect()` | Disconnect from robot | - |
       | `gripper_open()` | Open gripper | - |
       | `move_to_initial_state()` | Move to initial/home position | - |
       | `move_to_free_state()` | Move to safe parking position | - |
       | `move_to_position(position, gripper_offset=0.0)` | Move end-effector to [x,y,z] | position: List[float], gripper_offset: float |
       | `rotate_90degree(direction)` | Rotate gripper 90° | direction: 1 (CW) or -1 (CCW) |
       | `execute_pick_object(object_position, gripper_offset=0.0)` | Descend to pick position (3cm from top), close gripper, save pitch | object_position: [x,y,z], gripper_offset: float |
       | `execute_place_object(place_position, gripper_offset=0.0, is_table=True, gripper_open_ratio=0.7)` | Descend to place position with saved pitch, open gripper 70% | place_position: [x,y,z], gripper_offset: float, is_table: bool, gripper_open_ratio: float |

       **execute_pick_object**: Call from pick_approach position. Moves TCP to pick height, closes gripper, and **saves current pitch**.
         - **IMPORTANT**: Pass position with z = object_height / 2 (cz/2) to grasp at middle of object
       **execute_place_object**: Call from place_approach position. Moves TCP to place height **with saved pitch restored**, then opens gripper.
         - **IMPORTANT**: Pass position with z = picked_object_height / 2 (pick_pos[2]/2) to release at middle height
         - is_table=True: place on table, is_table=False: place on another object
         - gripper_open_ratio=0.7: opens gripper to 70% (ALWAYS use 0.7)
         - Pitch is automatically restored from the saved value at pick time

       **Gripper Offset**: Use `gripper_offset=positions["object"]["gripper_offset"]` for asymmetric gripper collision avoidance.
       **Pitch Handling**: Pitch is automatically saved at pick and restored at place. No need for maintain_pitch during movement.

    5. **Skill Composition Patterns**:

       ```
       # PICK pattern:
       pick_obj = positions["object_name"]
       pick_pos = pick_obj["position"]  # [x, y, z] where z = object height
       offset = pick_obj["gripper_offset"]
       approach_height = 0.15  # 15cm above object

       skills.gripper_open()
       skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset)  # pick_approach
       skills.execute_pick_object([pick_pos[0], pick_pos[1], pick_pos[2]/2], gripper_offset=offset)  # grasp at half height (cz/2)
       skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset)  # lift

       # PLACE on OBJECT pattern (e.g., place or stack on another object):
       place_obj = positions["target_object"]
       place_pos = place_obj["position"]  # target x, y position

       skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset) # place_approach
       skills.execute_place_object([place_pos[0], place_pos[1], pick_pos[2]/2], gripper_offset=offset, is_table=False, gripper_open_ratio=0.7)  # release at picked object's half height (cz/2)
       skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset) # lift
       ```

    6. **Executable Code Skeleton**:

       ```python
# pick object_name and place on target_name
from skills.skills_lerobot import LeRobotSkills

def execute_task():
    '''Execute the robot task based on the goal.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="world",
    )
    skills.connect()

    try:
        approach_height = 0.15  # 15cm above objects

        skills.move_to_initial_state()
        skills.rotate_90degree(-1)  # Rotate 90° CCW after initial state

        # === Object Positions ===
        pick_obj = positions["object_name"]
        pick_pos = pick_obj["position"]
        offset = pick_obj["gripper_offset"]

        place_obj = positions["target_name"]
        place_pos = place_obj["position"]

        # === PICK ===
        skills.gripper_open()
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset)
        skills.execute_pick_object([pick_pos[0], pick_pos[1], pick_pos[2]/2], gripper_offset=offset)  # grasp at half height (cz/2)
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height])  # lift

        # === PLACE on object ===
        skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset)  # place_approach
        skills.execute_place_object([place_pos[0], place_pos[1], pick_pos[2]/2], gripper_offset=offset, is_table=False, gripper_open_ratio=0.7)  # release at picked object's half height (cz/2)
        skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset) # lift

        # === Cleanup ===
        skills.move_to_initial_state()
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
   - Always start with `move_to_initial_state()` followed by `rotate_90degree(-1)` (CCW rotation)
   - Always end with `move_to_initial_state()` then `move_to_free_state()`
   - Use `approach_height = 0.15` (15cm) for approach/lift movements
   - **ALWAYS use z = pick_pos[2]/2 (half height) for both execute_pick_object and execute_place_object** to grasp/release at middle of object
   - Use `is_table=True` when placing on table, `is_table=False` when placing on another object
   - **ALWAYS use `gripper_open_ratio=0.7`** in `execute_place_object()` to open gripper 70%
   - Always include try/finally for proper cleanup

3. **Output Format:**
   - Do not use code blocks in your final answer
   - Provide the generated code in plain text format
   - Include all imports and the complete execute_task() function

**Generate the complete executable Python code:**
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
   | `gripper_open` | Open the gripper | - |
   | `move_to_position` | Move end-effector to position | object: str, approach_height: float, gripper_offset: bool |
   | `execute_pick_object` | Descend (3cm from top) + gripper_close + save pitch | object: str, gripper_offset: bool |
   | `execute_place_object` | Descend with saved pitch + gripper_open | target: str, gripper_offset: bool, is_table: bool |

   **execute_pick_object**: Call after moving to pick_approach. Descends to 3cm below object top, closes gripper, and **saves current pitch**.
   **execute_place_object**: Call after moving to place_approach. Descends to place height **with saved pitch restored**, then opens gripper.
     - is_table=true: place on table
     - is_table=false: place on another object
     - Pitch is automatically restored from pick time

4. **Available Object Names**:
   {object_names}

### **Output Format**

Specification: {{"required_skills": [...], "steps": [...]}}

### **Examples**

**Example 1**: "Pick up the red cup and place it on the carrier"
```
Specification: {{
  "required_skills": ["move_to_initial_state", "rotate_90degree", "gripper_open", "move_to_position", "execute_pick_object", "execute_place_object", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "rotate_90degree", "direction": "ccw"}},
    {{"step": 3, "action": "gripper_open"}},
    {{"step": 4, "action": "move_to_position", "object": "red cup", "approach_height": 0.15, "gripper_offset": true}},
    {{"step": 5, "action": "execute_pick_object", "object": "red cup", "gripper_offset": true}},
    {{"step": 6, "action": "move_to_position", "object": "red cup", "approach_height": 0.15}},
    {{"step": 7, "action": "move_to_position", "target": "carrier", "approach_height": 0.15, "gripper_offset": true}},
    {{"step": 8, "action": "execute_place_object", "target": "carrier", "gripper_offset": true, "is_table": true}},
    {{"step": 9, "action": "move_to_position", "target": "carrier", "approach_height": 0.15, "gripper_offset": true}},
    {{"step": 10, "action": "move_to_initial_state"}},
    {{"step": 11, "action": "move_to_free_state"}}
  ]
}}
```

**Example 2**: "Stack the red dice on top of the blue box"
```
Specification: {{
  "required_skills": ["move_to_initial_state", "rotate_90degree", "gripper_open", "move_to_position", "execute_pick_object", "execute_place_object", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "rotate_90degree", "direction": "ccw"}},
    {{"step": 3, "action": "gripper_open"}},
    {{"step": 4, "action": "move_to_position", "object": "red dice", "approach_height": 0.15, "gripper_offset": true}},
    {{"step": 5, "action": "execute_pick_object", "object": "red dice", "gripper_offset": true}},
    {{"step": 6, "action": "move_to_position", "object": "red dice", "approach_height": 0.15}},
    {{"step": 7, "action": "move_to_position", "target": "blue box", "approach_height": 0.15, "gripper_offset": true}},
    {{"step": 8, "action": "execute_place_object", "target": "blue box", "gripper_offset": true, "is_table": false}},
    {{"step": 9, "action": "move_to_position", "target": "blue box", "approach_height": 0.15, "gripper_offset": true}},
    {{"step": 10, "action": "move_to_initial_state"}},
    {{"step": 11, "action": "move_to_free_state"}}
  ]
}}
```

### **Guidelines**

1. Use `execute_pick_object` and `execute_place_object` for pick/place operations
2. Always start with `move_to_initial_state` followed by `rotate_90degree` (direction: "ccw")
3. Always end with `move_to_initial_state` then `move_to_free_state`
4. Use `approach_height: 0.15` for approach/retract movements (15cm above target)
5. Use `is_table: true` when placing on table, `is_table: false` when stacking on another object
6. Use ONLY the object names from the detected objects list

### **Generate Specification:**
"""

    return prompt
