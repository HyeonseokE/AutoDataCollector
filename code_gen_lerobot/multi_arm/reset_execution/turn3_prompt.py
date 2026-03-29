"""
Multi-Arm Reset Turn 3: Code Generation Prompt

Pattern: `skills` object (MultiArmSkills) is pre-created and injected.
LLM code calls skills.connect() / skills.disconnect() and uses
skills.move_to_position(), skills.pick_object(), skills.place_object(), etc.

Reuses MULTI_ARM_API_DOC from forward_execution.
"""

from typing import Dict, List, Optional

from ..multi_skill_api_doc import MULTI_ARM_API_DOC


def _format_per_arm_positions(positions: Dict, label: str) -> str:
    """Per-arm position dict를 프롬프트용 문자열로 포맷팅.

    Args:
        positions: {"left_arm": {obj: {"position": [x,y,z]}}, "right_arm": {...}}
                   또는 flat dict {obj: {"position": [x,y,z]}}
        label: 변수명 (e.g., "current_positions", "target_positions")

    Returns:
        포맷팅된 Python dict 문자열
    """
    def _get_pos(info):
        if info is None:
            return None
        elif isinstance(info, dict) and "position" in info:
            return info["position"]
        elif isinstance(info, (list, tuple)) and len(info) >= 3:
            return list(info[:3])
        return None

    # per-arm 구조 감지
    if "left_arm" in positions or "right_arm" in positions:
        lines = [f"{label} = {{"]
        for arm_key in ["left_arm", "right_arm"]:
            arm_data = positions.get(arm_key, {})
            lines.append(f'    "{arm_key}": {{')
            for name, info in arm_data.items():
                pos = _get_pos(info)
                if pos is not None:
                    lines.append(
                        f'        "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
                    )
            lines.append("    },")
        lines.append("}")
        return "\n".join(lines)
    else:
        # flat dict fallback
        lines = [f"{label} = {{"]
        for name, info in positions.items():
            pos = _get_pos(info)
            if pos is not None:
                lines.append(
                    f'    "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
                )
        lines.append("}")
        return "\n".join(lines)


def multi_arm_turn3_reset_codegen_prompt(
    current_positions: Dict = None,
    target_positions: Dict = None,
    robot_ids: List[int] = None,
    instruction: str = "move objects to their original positions",
    context_summary: str = "",
) -> str:
    """Multi-arm reset 코드 생성 프롬프트 (bi-arm API).

    Args:
        current_positions: Per-arm current positions
            {"left_arm": {obj: {"position": [...]}}, "right_arm": {...}}
        target_positions: Per-arm target positions (same structure)
        robot_ids: [left_id, right_id] e.g., [2, 3]
        instruction: Reset task description
        context_summary: VLM scene summary from Turn 0~2
    """
    if robot_ids is None:
        robot_ids = [2, 3]
    if current_positions is None:
        current_positions = {}
    if target_positions is None:
        target_positions = {}

    current_str = _format_per_arm_positions(current_positions, "current_positions")
    target_str = _format_per_arm_positions(target_positions, "target_positions")

    context_section = ""
    if context_summary:
        context_section = f"""### Scene Context (from prior analysis session)

{context_summary}

"""

    return f"""\
{context_section}### Task: Reset Environment
{instruction}
Move each object from its current position to its target position so the next task episode can begin.

**Robot Setup**:
- Two SO-101 robot arms: left arm (robot{robot_ids[0]}) handles the left side and center, right arm (robot{robot_ids[1]}) handles the right side and center.
- A pre-created `skills` object (MultiArmSkills) is provided as a global variable.
- Call `skills.connect()` at the start and `skills.disconnect()` in a finally block.

**Workspace Images** (provided as attached images):
- **Image 1**: left_arm (robot{robot_ids[0]}) workspace. The bright area inside the cyan arc is the reachable zone for left_arm.
- **Image 2**: right_arm (robot{robot_ids[1]}) workspace. The bright area inside the cyan arc is the reachable zone for right_arm.
- Assign each object to the arm whose workspace covers that object (bright area).
- If an object is reachable by both arms, prefer the arm closer to it.

### Environment States
Object position z-coordinate = object height (table surface is z=0).

The `current_positions` and `target_positions` dictionaries are provided at runtime as global variables:
```python
{current_str}

{target_str}
```

**How to use positions**:
```python
# Split positions by arm at the top of execute_reset_task()
cur_left = current_positions["left_arm"]
cur_right = current_positions["right_arm"]
tgt_left = target_positions["left_arm"]
tgt_right = target_positions["right_arm"]

# Access object positions for each arm
left_cur = cur_left["object_name"]["position"]    # use with left_arm=
left_tgt = tgt_left["object_name"]["position"]    # use with left_arm=
right_cur = cur_right["object_name"]["position"]  # use with right_arm=
right_tgt = tgt_right["object_name"]["position"]  # use with right_arm=
```
- **CRITICAL**: Do NOT redefine or hardcode the dictionaries. They are global variables.
- **CRITICAL**: Do NOT hardcode any coordinate values.

{MULTI_ARM_API_DOC}

### Skill Composition Patterns (MUST follow exactly)

```python
cur_left = current_positions["left_arm"]
cur_right = current_positions["right_arm"]
tgt_left = target_positions["left_arm"]
tgt_right = target_positions["right_arm"]
approach_height = 0.20

# === STEP 1: Move 1st object to target (no re-detection needed) ===
# Determine which arm to use based on workspace images.
# Example: object is on the left side → use left_arm.
skills.set_subtask("move object_name to target with left arm")

cur = cur_left["object_name"]["position"]
tgt = tgt_left["object_name"]["position"]

# Pick: open gripper → approach → pick → lift
skills.gripper_control(left_arm="open", right_arm="wait",
    left_skill_description="Open left gripper for object_name",
    left_verification_question="Is left gripper open?")
skills.move_to_position(
    left_arm=[cur[0], cur[1], approach_height], right_arm="wait",
    left_skill_description="Move left arm above object_name",
    left_verification_question="Is left arm above object_name?")
skills.pick_object(left_arm=cur, right_arm="wait",
    left_object_name="object_name",
    left_skill_description="Pick up object_name",
    left_verification_question="Is object_name grasped?")
skills.move_to_position(
    left_arm=[cur[0], cur[1], approach_height], right_arm="wait",
    left_skill_description="Lift object_name",
    left_verification_question="Is object_name lifted?")

# Place: approach target → place → retract
skills.move_to_position(
    left_arm=[tgt[0], tgt[1], approach_height], right_arm="wait",
    left_skill_description="Move object_name above target",
    left_verification_question="Is object_name above target?")
skills.place_object(left_arm=tgt, right_arm="wait",
    left_is_table=True,
    left_skill_description="Place object_name at target",
    left_verification_question="Is object_name placed at target?")
skills.move_to_position(
    left_arm=[tgt[0], tgt[1], approach_height], right_arm="wait",
    left_skill_description="Retract from object_name target",
    left_verification_question="Is left arm clear of object_name?")

skills.clear_subtask()

# === STEP 2: Move 2nd object (re-detect first) ===
skills.set_subtask("move next_object to target with right arm")

# Re-detection: clear arms → detect → update positions
skills.move_to_initial_state()
updated = skills.detect_objects(["object_name", "next_object"])
if "left_arm" in updated:
    cur_left.update(updated["left_arm"])
if "right_arm" in updated:
    cur_right.update(updated["right_arm"])

cur = cur_right["next_object"]["position"]
tgt = tgt_right["next_object"]["position"]

# Pick with right arm (same pattern, swap left_arm="wait" / right_arm=...)
skills.gripper_control(left_arm="wait", right_arm="open",
    right_skill_description="Open right gripper for next_object",
    right_verification_question="Is right gripper open?")
skills.move_to_position(
    left_arm="wait", right_arm=[cur[0], cur[1], approach_height],
    right_skill_description="Move right arm above next_object",
    right_verification_question="Is right arm above next_object?")
skills.pick_object(left_arm="wait", right_arm=cur,
    right_object_name="next_object",
    right_skill_description="Pick up next_object",
    right_verification_question="Is next_object grasped?")
# ... (same place pattern with right_arm) ...

skills.clear_subtask()

# === STEP 3+: repeat set_subtask → move_to_initial → detect → update → pick → place → clear_subtask ===
```

### Unstacking (Disassembling a Stack)

When objects are stacked, you MUST unstack from **top to bottom**.
- The object with the highest z is topmost — always pick it first.
- After picking each stacked object, re-detect to get updated z-heights.

### Code Skeleton

```python
def execute_reset_task():
    '''Move objects from current to target positions using two arms.'''
    skills.connect()

    try:
        cur_left = current_positions["left_arm"]
        cur_right = current_positions["right_arm"]
        tgt_left = target_positions["left_arm"]
        tgt_right = target_positions["right_arm"]
        approach_height = 0.20

        skills.move_to_initial_state()

        # Subtask 1: set_subtask → pick → place → clear_subtask
        # Re-detection: move_to_initial_state → detect_objects → update cur_left/cur_right
        # Subtask 2: set_subtask → pick → place → clear_subtask
        # ... repeat for each object ...

        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_reset_task()
```

### Task Requirements
1. Assign objects to the appropriate arm based on workspace images (bright area = reachable).
2. Use `skills.move_to_position()` / `skills.pick_object()` / `skills.place_object()` / `skills.gripper_control()` for ALL operations. Pass `left_arm="wait"` or `right_arm="wait"` for the arm that should hold position.
3. **Subtask pattern**: Each pick-place of one object = one subtask. Wrap with `set_subtask()` before and `clear_subtask()` after.
4. **Re-detection (MANDATORY)**: After each subtask (after `clear_subtask()`), call `skills.move_to_initial_state()` to clear arms from camera view, then `skills.detect_objects([...all object names...])` to update positions. Skip re-detection only after the very last subtask. The 1st object does NOT need re-detection.
   - **CRITICAL**: After `pos_left.update()` / `pos_right.update()`, you MUST **re-assign ALL local variables** extracted from the positions dict. `update()` replaces dict entries, but previously extracted variables still reference the OLD values.
5. Always start with `skills.move_to_initial_state()`, end with `skills.move_to_initial_state()` then `skills.move_to_free_state()`.
6. Use `approach_height = 0.20` for approach/retreat movements.
7. ALWAYS pass `left_skill_description`/`right_skill_description` and `left_verification_question`/`right_verification_question` for every arm that is NOT `"wait"`.
8. Always include try/finally with `skills.disconnect()` for cleanup.
9. **Unstacking**: If objects are stacked, always unstack from top to bottom (highest z first).
10. **Bimanual move**: When both arms hold the **same object** and must move together (e.g., unfolding, stretching), use `skills.bimanual_move()` instead of `skills.move_to_position()`. This guarantees synchronized progress.

### Output Format
- Provide complete executable Python code
- Do not include markdown code blocks

**Generate the complete executable RESET code:**"""
