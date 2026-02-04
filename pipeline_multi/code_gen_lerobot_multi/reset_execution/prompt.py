"""
Reset Execution Prompts for Multi-Robot

LLM prompts for reset spec generation in multi-robot environment.
Code generation prompts are now in prompt/common_template_prompt.py (ResetTemplatePrompt).
"""

import json
from typing import Dict, List


def lerobot_reset_spec_gen_prompt_multi(
    original_instruction: str,
    original_positions: Dict[str, List[float]],
    current_positions: Dict[str, List[float]],
    forward_spec: Dict = None,
    robot_id: int = 3,
) -> str:
    """
    멀티 로봇 환경에서 리셋 스펙 생성 프롬프트

    Args:
        original_instruction: 원래 태스크 목표
        original_positions: 객체별 원래 위치
        current_positions: 객체별 현재 위치
        forward_spec: Forward execution에서 생성된 spec
        robot_id: 로봇 번호

    Returns:
        LLM에 전달할 프롬프트 문자열
    """

    def get_position(info):
        if info is None:
            return None
        elif isinstance(info, dict) and "position" in info:
            return info["position"]
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            return list(info[:3])
        return None

    # 원래 위치 포맷팅
    original_lines = []
    for name, info in original_positions.items():
        pos = get_position(info)
        if pos is not None:
            original_lines.append(f'    - {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]')
    original_str = "\n".join(original_lines)

    # 현재 위치 포맷팅
    current_lines = []
    for name, info in current_positions.items():
        pos = get_position(info)
        if pos is not None:
            current_lines.append(f'    - {name}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]')
    current_str = "\n".join(current_lines)

    forward_spec_str = json.dumps(forward_spec, indent=2) if forward_spec else "Not available"

    prompt = f"""
You are an AI assistant generating reset task specifications for Robot {robot_id} in a multi-robot system.
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

3. **Robot**: Robot {robot_id}

4. **Environment States**:
   - Initial State (Before Forward):
     ```
     {original_str}
     ```
   - Current State (After Forward):
     ```
     {current_str}
     ```

### **Your Task: Generate RESET Specification**

Generate a specification that restores the environment to its initial state.

### **Available Skills**:
| Skill | Description | Parameters |
|-------|-------------|------------|
| `move_to_initial_state` | Move to home position | - |
| `move_to_free_state` | Move to safe parking position | - |
| `rotate_90degree` | Rotate gripper 90 deg | direction: "cw" or "ccw" |
| `gripper_open` | Open the gripper | - |
| `move_to_position` | Move end-effector | object: str, approach_height: float |
| `execute_pick_object` | Descend + gripper_close + save pitch | object: str |
| `execute_place_object` | Descend with saved pitch + gripper_open | target: str, is_table: bool |

### **Output Format**

Specification: {{"robot_id": {robot_id}, "required_skills": [...], "steps": [...]}}

### **Generate Reset Specification:**
"""

    return prompt
