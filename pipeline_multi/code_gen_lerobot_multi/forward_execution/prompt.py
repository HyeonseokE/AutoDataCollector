"""
Forward Execution Prompts for Multi-Robot

LLM prompts for spec generation in multi-robot environment.
Code generation prompts are now in prompt/common_template_prompt.py (ForwardTemplatePrompt).
"""

from typing import Dict, List


def lerobot_spec_gen_prompt_multi(
    instruction: str,
    object_positions: Dict[str, List[float]],
    robot_id: int = 3,
) -> str:
    """
    멀티 로봇 환경에서 태스크 스펙 생성 프롬프트

    자연어 목표를 구조화된 스펙(단계별 액션)으로 변환합니다.

    Args:
        instruction: 자연어 목표
        object_positions: 객체별 위치 딕셔너리
        robot_id: 로봇 번호

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
You are an AI assistant generating task specifications for a LeRobot SO-101 robot arm.
This robot is Robot {robot_id} in a multi-robot system.

### **Input Details:**

1. **Goal (Natural Language)**:
   ```
   {instruction}
   ```

2. **Robot**: Robot {robot_id}

3. **Detected Objects and Positions (World Frame, meters)**:
   ```
    {positions_str}
   ```
   Objects not found: {not_found_str}

4. **Available Skills**:
   | Skill | Description | Parameters |
   |-------|-------------|------------|
   | `move_to_initial_state` | Move to home position | - |
   | `move_to_free_state` | Move to safe parking position | - |
   | `rotate_90degree` | Rotate gripper 90° | direction: "cw" or "ccw" |
   | `gripper_open` | Open the gripper | - |
   | `move_to_position` | Move end-effector to position | object: str, approach_height: float |
   | `execute_pick_object` | Descend + gripper_close + save pitch | object: str |
   | `execute_place_object` | Descend with saved pitch + gripper_open | target: str, is_table: bool |

5. **Available Object Names**: {object_names}

### **Output Format**

Specification: {{"robot_id": {robot_id}, "required_skills": [...], "steps": [...]}}

### **Guidelines**

1. Use `execute_pick_object` and `execute_place_object` for pick/place operations
2. Always start with `move_to_initial_state`
3. Always end with `move_to_initial_state` and `move_to_free_state`
4. Use `approach_height: 0.15` for approach/retract movements
5. Use ONLY the object names from the detected objects list

### **Generate Specification:**
"""

    return prompt
