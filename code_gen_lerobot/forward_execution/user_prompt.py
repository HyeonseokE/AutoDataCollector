"""
Forward Execution Prompts

LLM prompts for forward task execution (code generation and spec generation).
"""

from typing import Dict, List, Optional

from .skill_api_doc import ROBOT_API_DOC


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
       | `gripper_open(skill_description=None)` | Open gripper | skill_description: str |
       | `move_to_initial_state(skill_description=None)` | Move to initial/home position | skill_description: str |
       | `move_to_free_state(skill_description=None)` | Move to safe parking position | skill_description: str |
       | `move_to_position(position, ..., skill_description=None)` | Move end-effector to [x,y,z] | position, gripper_offset, target_name, skill_description: str |
       | `rotate_90degree(direction, skill_description=None)` | Rotate gripper 90° | direction: 1 (CW) or -1 (CCW), skill_description: str |
       | `execute_pick_object(object_position, ..., skill_description=None)` | Descend to pick position (3cm from top), close gripper, save pitch | object_position, gripper_offset, object_name, skill_description: str |
       | `execute_place_object(place_position, ..., skill_description=None)` | Descend to place position with saved pitch, open gripper 70% | place_position, gripper_offset, is_table, gripper_open_ratio, target_name, skill_description: str |

       **execute_pick_object**: Call from pick_approach position. Moves TCP to pick height, closes gripper, and **saves current pitch**.
         - **IMPORTANT**: Pass position with z = object_height - 0.02 (2cm below object top) to grasp near the top of the object
         - **object_name**: Pass the object name for subgoal labeling (e.g., "yellow dice")
       **execute_place_object**: Call from place_approach position. Moves TCP to place height **with saved pitch restored**, then opens gripper.
         - **IMPORTANT**: Pass position with z = picked_object_height - 0.02 (pick_pos[2] - 0.02) to release near the top
         - is_table=True: place on table, is_table=False: place on another object
         - gripper_open_ratio=0.7: opens gripper to 70% (ALWAYS use 0.7)
         - Pitch is automatically restored from the saved value at pick time
         - **target_name**: Pass the target name for subgoal labeling (e.g., "blue dish")

       **Gripper Offset**: Use `gripper_offset=positions["object"]["gripper_offset"]` for asymmetric gripper collision avoidance.
       **Pitch Handling**: Pitch is automatically saved at pick and restored at place. No need for maintain_pitch during movement.
       **skill_description (REQUIRED)**: A natural language sentence describing the semantic intent of each skill call.
         - This is recorded as `skill.natural_language` in the dataset for robot policy learning.
         - Describe WHY the robot is performing this action, not just WHAT it does.
         - Include the object name, the spatial context (e.g., "above", "onto"), and the role in the overall task.
         - Each call MUST have a unique, descriptive `skill_description` that distinguishes it from other calls.
         - Examples: "move to initial position to start the task", "open gripper to prepare for picking the red block",
           "approach above the red block for grasping", "descend and grasp the red block at its center",
           "lift the red block to safe height after grasping", "move above the blue dish to place the red block",
           "lower the red block onto the blue dish", "return to initial position after completing the task",
           "move to free position for safe parking"
         - Use "initial position" (not "home position") and "free position" (not "parking position") to match skill names.

    6. **Skill Composition Patterns**:

       ```
       # PICK pattern:
       pick_obj = positions["object_name"]
       pick_pos = pick_obj["position"]  # [x, y, z] where z = object height
       offset = pick_obj["gripper_offset"]
       approach_height = 0.20  # 20cm above object

       skills.gripper_open(skill_description="open gripper to prepare for picking the object_name")
       skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset, target_name="object_name", skill_description="approach above the object_name for grasping")
       skills.execute_pick_object([pick_pos[0], pick_pos[1], pick_pos[2] - 0.02], gripper_offset=offset, object_name="object_name", skill_description="descend and grasp the object_name at its center")  # grasp 2cm below top
       skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset, target_name="object_name", skill_description="lift the object_name to safe height after grasping")

       # PLACE on OBJECT pattern (e.g., place or stack on another object):
       place_obj = positions["target_object"]
       place_pos = place_obj["position"]  # target x, y position

       skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset, target_name="target_object", skill_description="move above the target_object to place the object_name")
       skills.execute_place_object([place_pos[0], place_pos[1], pick_pos[2] - 0.02], gripper_offset=offset, is_table=False, gripper_open_ratio=0.7, target_name="target_object", skill_description="lower the object_name onto the target_object")  # release 2cm below top
       skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset, target_name="target_object", skill_description="retreat upward after placing the object_name on the target_object")
       ```

    7. **Executable Code Skeleton**:

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
        approach_height = 0.20  # 20cm above objects

        skills.move_to_initial_state(skill_description="move to initial position to start the task")

        # === Object Positions ===
        pick_obj = positions["object_name"]
        pick_pos = pick_obj["position"]
        offset = pick_obj["gripper_offset"]

        place_obj = positions["target_name"]
        place_pos = place_obj["position"]

        # === PICK ===
        skills.gripper_open(skill_description="open gripper to prepare for picking the object_name")
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset, target_name="object_name", skill_description="approach above the object_name for grasping")
        skills.execute_pick_object([pick_pos[0], pick_pos[1], pick_pos[2] - 0.02], gripper_offset=offset, object_name="object_name", skill_description="descend and grasp the object_name at its center")  # grasp 2cm below top
        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset, target_name="object_name", skill_description="lift the object_name to safe height after grasping")  # lift

        # === PLACE on object ===
        skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset, target_name="target_name", skill_description="move above the target_name to place the object_name")  # place_approach
        skills.execute_place_object([place_pos[0], place_pos[1], pick_pos[2] - 0.02], gripper_offset=offset, is_table=False, gripper_open_ratio=0.7, target_name="target_name", skill_description="lower the object_name onto the target_name")  # release 2cm below top
        skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset, target_name="target_name", skill_description="retreat upward after placing the object_name on the target_name") # lift

        # === Cleanup ===
        skills.move_to_initial_state(skill_description="return to initial position after completing the task")
        skills.move_to_free_state(skill_description="move to free position for safe parking")

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
   - Always start with `move_to_initial_state()`
   - Always end with `move_to_initial_state()` then `move_to_free_state()`
   - Use `approach_height = 0.20` (20cm) for approach/lift movements
   - **ALWAYS use z = pick_pos[2] - 0.02 (2cm below object top) for both execute_pick_object and execute_place_object**
   - Use `is_table=True` when placing on table, `is_table=False` when placing on another object
   - **ALWAYS use `gripper_open_ratio=0.7`** in `execute_place_object()` to open gripper 70%
   - Always include try/finally for proper cleanup
   - **ALWAYS pass `target_name` or `object_name` parameters** for subgoal labeling in dataset recording
   - **ALWAYS pass `skill_description`** for EVERY skill call. Write a natural language sentence that describes the semantic intent (why the robot is doing this action in context of the overall task). Each description must be unique and distinguishable from other skill calls.

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
) -> str:
    """
    Turn 3: 코드 생성 프롬프트 (multi-turn용)

    Turn 1-2의 컨텍스트가 이미 chat session에 있으므로,
    positions는 Turn 2에서 확정된 world 좌표를 참조합니다.
    중복 설명을 최소화하고 코드 생성에 집중합니다.

    Args:
        instruction: 자연어 목표
        robot_id: 로봇 번호 (2 또는 3)

    Returns:
        Turn 3 user prompt 문자열
    """

    robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"
    robot_api_doc = ROBOT_API_DOC

    prompt = f"""### Your Job (Turn 3 — Code Generation)

Now generate executable Python code to complete the task using the LeRobot SO-101 robot arm.
Use the scene understanding, detected objects, and grasp/place points from our previous conversation turns.

**Grasp Guidelines**:
- The gripper has asymmetric fingers (left fixed, right actuated), max opening 0.07m.
- Always open the gripper before approaching the grasp pose.
- Ensure the target position is reachable within the workspace and has enough clearance to avoid collisions.

**The `positions` dictionary** will be provided at runtime as a global variable with this structure:
```python
positions = {{
    "object_name": {{"position": [x, y, z], "gripper_offset": float}},
    ...
}}
```
- `position`: world coordinates in meters (x=front/back, y=left/right, z=height from table).
- `gripper_offset`: asymmetric gripper collision avoidance offset in meters.

**Available Robot API Skills**:

```python
{robot_api_doc}
```

**Skill Composition Patterns**:

```python
# START — always first
approach_height = 0.20
skills.move_to_initial_state(skill_description="move to initial position to start the task")

# PICK — open gripper → approach → grasp at z - 0.02 → lift
pick_obj = positions["object_name"]
pick_pos = pick_obj["position"]
offset = pick_obj["gripper_offset"]
skills.gripper_open(skill_description="open gripper to prepare for picking the object_name")
skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset, target_name="object_name", skill_description="approach above the object_name for grasping")
skills.execute_pick_object([pick_pos[0], pick_pos[1], pick_pos[2] - 0.02], gripper_offset=offset, object_name="object_name", skill_description="descend and grasp the object_name at its center")
skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset, target_name="object_name", skill_description="lift the object_name to safe height after grasping")

# PLACE ON OBJECT — approach target → release at pick_pos[2] - 0.02 → retreat  (is_table=False)
place_obj = positions["target_object"]
place_pos = place_obj["position"]
skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset, target_name="target_object", skill_description="move above the target_object to place the object_name")
skills.execute_place_object([place_pos[0], place_pos[1], pick_pos[2] - 0.02], gripper_offset=offset, is_table=False, gripper_open_ratio=0.7, target_name="target_object", skill_description="lower the object_name onto the target_object")
skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset, target_name="target_object", skill_description="retreat upward after placing the object_name on the target_object")

# PLACE ON TABLE — same as above but is_table=True

# END — always last
skills.move_to_initial_state(skill_description="return to initial position after completing the task")
skills.move_to_free_state(skill_description="move to free position for safe parking")
```

**Code Skeleton**:
```python
from skills.skills_lerobot import LeRobotSkills

def execute_task():
    skills = LeRobotSkills(robot_config="{robot_config}", frame="world")
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
1. Always START at the beginning and END at the end.
2. `approach_height = 0.20` (20cm) for all approach/lift.
3. **ALWAYS** `z = pick_pos[2] - 0.02` (2cm below object top) for both pick and place.
4. `is_table=True` on table, `is_table=False` on another object.
5. **ALWAYS** `gripper_open_ratio=0.7` in `execute_place_object()`.
6. Wrap with `try/finally` → `disconnect()`.
7. **ALWAYS** pass unique `skill_description` for EVERY skill call.

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
  "required_skills": ["move_to_initial_state", "gripper_open", "move_to_position", "execute_pick_object", "execute_place_object", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "gripper_open"}},
    {{"step": 3, "action": "move_to_position", "object": "red cup", "approach_height": 0.20, "gripper_offset": true}},
    {{"step": 4, "action": "execute_pick_object", "object": "red cup", "gripper_offset": true}},
    {{"step": 5, "action": "move_to_position", "object": "red cup", "approach_height": 0.20}},
    {{"step": 6, "action": "move_to_position", "target": "carrier", "approach_height": 0.20, "gripper_offset": true}},
    {{"step": 7, "action": "execute_place_object", "target": "carrier", "gripper_offset": true, "is_table": true}},
    {{"step": 8, "action": "move_to_position", "target": "carrier", "approach_height": 0.20, "gripper_offset": true}},
    {{"step": 9, "action": "move_to_initial_state"}},
    {{"step": 10, "action": "move_to_free_state"}}
  ]
}}
```

**Example 2**: "Stack the red dice on top of the blue box"
```
Specification: {{
  "required_skills": ["move_to_initial_state", "gripper_open", "move_to_position", "execute_pick_object", "execute_place_object", "move_to_free_state"],
  "steps": [
    {{"step": 1, "action": "move_to_initial_state"}},
    {{"step": 2, "action": "gripper_open"}},
    {{"step": 3, "action": "move_to_position", "object": "red dice", "approach_height": 0.20, "gripper_offset": true}},
    {{"step": 4, "action": "execute_pick_object", "object": "red dice", "gripper_offset": true}},
    {{"step": 5, "action": "move_to_position", "object": "red dice", "approach_height": 0.20}},
    {{"step": 6, "action": "move_to_position", "target": "blue box", "approach_height": 0.20, "gripper_offset": true}},
    {{"step": 7, "action": "execute_place_object", "target": "blue box", "gripper_offset": true, "is_table": false}},
    {{"step": 8, "action": "move_to_position", "target": "blue box", "approach_height": 0.20, "gripper_offset": true}},
    {{"step": 9, "action": "move_to_initial_state"}},
    {{"step": 10, "action": "move_to_free_state"}}
  ]
}}
```

### **Guidelines**

1. Use `execute_pick_object` and `execute_place_object` for pick/place operations
2. Always start with `move_to_initial_state`
3. Always end with `move_to_initial_state` then `move_to_free_state`
4. Use `approach_height: 0.20` for approach/retract movements (20cm above target)
5. Use `is_table: true` when placing on table, `is_table: false` when stacking on another object
6. Use ONLY the object names from the detected objects list

### **Generate Specification:**
"""

    return prompt
