"""
Common Template Prompts for Multi-Robot Code Generation

Contains:
- TaskPrompt: Base class for task-specific prompts
- ForwardTemplatePrompt: Template for forward execution prompts
- ResetTemplatePrompt: Template for reset execution prompts
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TaskPrompt:
    """Base class for task-specific prompts"""
    name: str
    description: str
    keywords: List[str]
    skill_sequence: List[str] = field(default_factory=list)

    def get_forward_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        """Forward execution context - override in subclass"""
        raise NotImplementedError

    def get_reset_context(self, robot_id: int = None, assigned_role: str = None) -> str:
        """Reset execution context - override in subclass"""
        raise NotImplementedError

    def get_forward_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        """
        Forward execution example code - override in subclass

        Args:
            robot_id: Robot ID (2 or 3)
            assigned_role: Pre-assigned role for this robot (e.g., "holder", "inserter")
        """
        raise NotImplementedError

    def get_reset_example_code(self, robot_id: int = None, assigned_role: str = None) -> str:
        """Reset execution example code - override in subclass"""
        raise NotImplementedError

    def get_tcp_offset(self) -> Optional[List[float]]:
        """
        Get task-specific TCP offset. Override in subclass if needed.

        Returns:
            TCP offset [x, y, z] in meters, or None to use default (-4cm)
        """
        return None  # Use default TCP offset

    def matches(self, instruction: str) -> bool:
        """Check if instruction matches this task type"""
        instruction_lower = instruction.lower()
        return any(kw in instruction_lower for kw in self.keywords)


class ForwardTemplatePrompt:
    """Template for forward execution prompts"""

    @staticmethod
    def generate(
        instruction: str,
        object_positions: Dict,
        robot_id: int,
        task_prompt: TaskPrompt = None,
        spec: str = None,
        assigned_object: str = None,
        assigned_role: str = None,
    ) -> str:
        """
        Generate forward execution prompt

        Args:
            instruction: Natural language goal
            object_positions: Object positions dict {name: {"position": [x,y,z], "gripper_offset": float}}
            robot_id: Robot ID (2 or 3)
            task_prompt: Task-specific prompt instance
            spec: Additional specification
            assigned_object: Assigned object name for this robot
            assigned_role: Pre-assigned role for this robot (e.g., "holder", "inserter")

        Returns:
            Complete prompt string for LLM
        """
        robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"

        # Format object positions
        positions_lines = []
        for name, info in object_positions.items():
            if info is not None:
                pos = info["position"]
                offset = info.get("gripper_offset", 0.0)
                positions_lines.append(
                    f'    "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], "gripper_offset": {offset:.3f}}},'
                )
        positions_str = "\n".join(positions_lines)

        # Not found objects
        not_found = [name for name, info in object_positions.items() if info is None]
        not_found_str = ", ".join(not_found) if not_found else "None"

        # Task context section
        task_context_section = ""
        if task_prompt:
            context = task_prompt.get_forward_context(robot_id=robot_id, assigned_role=assigned_role)
            example = task_prompt.get_forward_example_code(robot_id=robot_id, assigned_role=assigned_role)
            # Replace {robot_id} placeholder with actual robot_id
            example = example.replace("{robot_id}", str(robot_id))
            task_context_section = f"""
    7. **Task-Specific Context**:
       {context}

    8. **Example Code**:
       ```python
       {example}
       ```
"""

        # Assigned object section
        assigned_object_section = ""
        if assigned_object:
            assigned_object_section = f"""
    ⚠️ **CRITICAL - ASSIGNED OBJECT**:
       This robot (Robot {robot_id}) is assigned to handle: **"{assigned_object}"**

       - You MUST use the position of "{assigned_object}" from the positions dictionary
       - Do NOT use any other object's position
       - The other robot will handle the other object(s)
"""

        # Assigned role section
        assigned_role_section = ""
        if assigned_role:
            assigned_role_section = f"""
    ⚠️ **CRITICAL - ASSIGNED ROLE**:
       This robot (Robot {robot_id}) is assigned role: **"{assigned_role.upper()}"**

       - You MUST implement the behavior for the "{assigned_role}" role
       - Do NOT implement the other robot's role behavior
       - Follow the role-specific instructions in the Task-Specific Context
"""

        # TCP offset section - get from task_prompt if available
        tcp_offset = None
        tcp_offset_line = ""
        if task_prompt:
            tcp_offset = task_prompt.get_tcp_offset()
        if tcp_offset:
            tcp_offset_line = f"        tcp_offset={tcp_offset},  # Task-specific TCP offset\n"

        prompt = f"""You are an AI assistant generating executable Python code for a LeRobot SO-101 robot arm.
The robot is part of a multi-robot system where each robot executes independently with its own coordinate frame.

### **Input Details:**

    1. **Goal (Natural Language Description)**:
       ```
       {instruction}
       ```

    2. **Robot Configuration**:
       - Robot ID: {robot_id}
       - Config File: {robot_config}
       - Uses frame="world" (LeRobotSkills handles world → base_link transformation internally)
{assigned_object_section}{assigned_role_section}

    3. **Object Positions (World Frame, unit: meters)**:
       These positions are in the world coordinate frame. LeRobotSkills will automatically transform them.
       ```python
       positions = {{
      {positions_str}
       }}
       ```
       Objects not found: {not_found_str}

    4. **Specification (Code Generation Guidelines)**:
       ```
       {spec if spec else "Generate appropriate pick-and-place sequence based on the goal."}
       ```

    5. **Available Skills**:
       | Method | Description | Parameters |
       |--------|-------------|------------|
       | `connect()` | Connect to robot | - |
       | `disconnect()` | Disconnect from robot | - |
       | `gripper_open()` | Open gripper | - |
       | `move_to_initial_state()` | Move to initial/home position | - |
       | `move_to_free_state()` | Move to safe parking position | - |
       | `move_to_position(position, gripper_offset=0.0)` | Move end-effector to [x,y,z] | position: List[float], gripper_offset: float |
       | `rotate_90degree(direction)` | Rotate gripper 90deg | direction: 1 (CW) or -1 (CCW) |
       | `execute_multi_pick_object(skills, sync_barrier, pos, gripper_offset=0.0)` | **Multi-robot pick** - sync before gripper close | pos: [x,y,z] |
       | `execute_multi_place_object(skills, sync_barrier, pos, gripper_offset=0.0, is_table=True, gripper_open_ratio=0.3)` | **Multi-robot place** - sync before gripper open | pos: [x,y,z] |
       | `detect_objects(queries, timeout=5.0, visualize=False)` | **Real-time object detection** - get current object positions | queries: List[str], returns Dict |

    6. **Skill Composition Patterns (Multi-Robot)**:

       ```python
       from pipeline_multi.multi_skills import execute_multi_pick_object, execute_multi_place_object, execute_pause_for_sync, compute_tcp_offset_for_pin

       # IMPORTANT: 'positions' and 'sync_barrier' are pre-injected globals
       # Do NOT redefine them! Just use them directly:
       pick_obj = positions["object_name"]  # positions is already available
       pick_pos = pick_obj["position"]
       offset = pick_obj["gripper_offset"]

       # MULTI-ROBOT PICK pattern:
       skills.gripper_open()
       skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset)
       execute_multi_pick_object(skills, sync_barrier, pick_pos, gripper_offset=offset)
       skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset)

       # MULTI-ROBOT PLACE pattern:
       skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset)
       execute_multi_place_object(skills, sync_barrier, place_pos, gripper_offset=offset, is_table=True)
       skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset)

       # REAL-TIME DETECTION pattern (for dynamic object tracking):
       # Use when you need to re-detect object positions during execution
       updated_positions = skills.detect_objects(["red part", "pink part"])
       new_pos = updated_positions["red part"]["position"]
       ```
{task_context_section}
### **Executable Code Template**:

```python
# Task: {instruction}
# Robot ID: {robot_id}
import numpy as np
from skills.skills_lerobot import LeRobotSkills
from pipeline_multi.multi_skills import execute_multi_pick_object, execute_multi_place_object, execute_pause_for_sync, compute_tcp_offset_for_pin

def execute_task():
    '''Execute the robot task based on the goal.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="world",
{tcp_offset_line}    )
    skills.connect()

    try:
        # Set approach_height based on your role (see Example Code for values)

        skills.move_to_initial_state()
        skills.gripper_open()  # ⚠️ REQUIRED: Always open gripper before pick!

        # === Object Positions (World Frame) ===
        # ... extract positions from the positions dict ...

        # === TASK EXECUTION ===
        # ... your task-specific code ...

        # === Cleanup ===
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

### **Guidelines:**

1. Always start with `move_to_initial_state()` followed by `gripper_open()` ⚠️
2. Use `execute_multi_pick_object` and `execute_multi_place_object` for multi-robot tasks
3. Always end with `move_to_initial_state()` and `move_to_free_state()`
4. Always include try/finally for proper cleanup
5. **CRITICAL: `sync_barrier` and `positions` are pre-injected global variables - do NOT redefine them!**
   - `positions` dict is already available with detected object positions
   - `sync_barrier` object is already available for synchronization
   - Just use them directly: `positions["object_name"]`, `sync_barrier.wait("phase")`

**Generate the complete executable Python code:**
"""
        return prompt


class ResetTemplatePrompt:
    """Template for reset execution prompts"""

    @staticmethod
    def generate(
        original_instruction: str,
        target_positions: Dict,
        current_positions: Dict,
        robot_id: int,
        task_prompt: TaskPrompt = None,
        forward_spec: Dict = None,
        forward_code: str = None,
        is_random_reset: bool = False,
    ) -> str:
        """
        Generate reset execution prompt

        Args:
            original_instruction: Original task goal
            target_positions: Target positions (where objects should go)
            current_positions: Current positions (where objects are now)
            robot_id: Robot ID (2 or 3)
            task_prompt: Task-specific prompt instance
            forward_spec: Forward execution spec
            forward_code: Forward execution code
            is_random_reset: True if random shuffle mode

        Returns:
            Complete prompt string for LLM
        """
        robot_config = f"robot_configs/robot/so101_robot{robot_id}.yaml"

        def get_position_and_offset(info):
            if info is None:
                return None, 0.0
            elif isinstance(info, dict) and "position" in info:
                pos = info["position"]
                offset = info.get("gripper_offset", 0.02)
                return pos, offset
            elif isinstance(info, (list, tuple)) and len(info) >= 3:
                return list(info[:3]), 0.02
            return None, 0.0

        # Format target positions
        target_lines = []
        for name, info in target_positions.items():
            pos, _ = get_position_and_offset(info)
            if pos is not None:
                target_lines.append(f'        "{name}": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}],')
        target_str = "\n".join(target_lines)

        # Format current positions
        current_lines = []
        for name, info in current_positions.items():
            pos, offset = get_position_and_offset(info)
            if pos is not None:
                current_lines.append(
                    f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], "gripper_offset": {offset:.4f}}},'
                )
        current_str = "\n".join(current_lines)

        forward_spec_str = json.dumps(forward_spec, indent=2) if forward_spec else "Not available"

        if forward_code:
            forward_code_str = forward_code[:2000] + "\n... (truncated)" if len(forward_code) > 2000 else forward_code
        else:
            forward_code_str = "Not available"

        # Mode description
        if is_random_reset:
            task_title = "RANDOM RESET (Shuffle Objects)"
            task_description = """Generate executable Python code that moves each object to a NEW RANDOM position.
This is shuffling objects to new locations for the next training episode."""
            target_label = "Random Target Positions"
        else:
            task_title = "RESET to Initial Environment"
            task_description = f"""Generate executable Python code that restores the environment to its initial state
(before "{original_instruction}" was executed)."""
            target_label = "Initial State (Before Forward Execution)"

        # Task context section
        task_context_section = ""
        if task_prompt:
            context = task_prompt.get_reset_context()
            example = task_prompt.get_reset_example_code()
            task_context_section = f"""
    8. **Task-Specific Context**:
       {context}

    9. **Example Code**:
       ```python
       {example}
       ```
"""

        prompt = f"""You are an AI assistant generating executable Python code for Robot {robot_id} in a multi-robot system.
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

4. **Robot**: Robot {robot_id} (Config: {robot_config})

5. **Environment States**:
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

6. **Your Task: {task_title}**

   {task_description}

7. **Available Skills**:
   | Method | Description | Parameters |
   |--------|-------------|------------|
   | `connect()` | Connect to robot | - |
   | `disconnect()` | Disconnect from robot | - |
   | `gripper_open()` | Open gripper | - |
   | `move_to_initial_state()` | Move to initial/home position | - |
   | `move_to_free_state()` | Move to safe parking position | - |
   | `move_to_position(position, gripper_offset=0.0)` | Move end-effector to [x,y,z] | position: List[float], gripper_offset: float |
   | `rotate_90degree(direction)` | Rotate gripper 90 deg | direction: 1 (CW) or -1 (CCW) |
   | `execute_pick_object(object_position, gripper_offset=0.0)` | Descend + gripper_close + save pitch | object_position: [x,y,z] |
   | `execute_place_object(place_position, gripper_offset=0.0, is_table=True)` | Descend with saved pitch + gripper_open | place_position: [x,y,z] |
   | `detect_objects(queries, timeout=5.0, visualize=False)` | **Real-time object detection** - get current object positions | queries: List[str], returns Dict |
{task_context_section}
### **Code Template**:

```python
# Reset Task for Robot {robot_id}
from skills.skills_lerobot import LeRobotSkills

def execute_reset_task():
    '''Move objects from current positions to target positions.'''

    skills = LeRobotSkills(
        robot_config="{robot_config}",
        frame="world",
    )
    skills.connect()

    try:
        # Set approach_height as needed for reset movements

        skills.move_to_initial_state()

        # === RESET: Move objects to target positions ===
        # Hardcode actual [x,y,z] values from current_positions and target_positions

        # ... (your reset logic) ...

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
4. **Hardcode actual coordinate values** directly in the code
5. Do NOT reference `current_positions` or `target_positions` as variables
6. Use `is_table=True` when placing on table
7. Always include try/finally for proper cleanup
8. Always start with `move_to_initial_state()`
9. Always end with `move_to_initial_state()` and `move_to_free_state()`

**Generate the complete executable RESET code for Robot {robot_id}:**
"""
        return prompt
