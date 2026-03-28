"""
Multi-Arm Forward Execution User Prompts

Turn 3 code generation prompt for bi-arm setup.
Turn 0~2 are reused from single-arm forward_execution.

Pattern: Pipeline pre-creates `skills = MultiArmSkills(...)` and injects via exec_globals.
LLM code calls `skills.connect()` / `skills.disconnect()` and uses `skills.move_to_position(...)`, `skills.pick_object(...)`, etc.
"""

from typing import Dict, List, Optional

from ..multi_skill_api_doc import MULTI_ARM_API_DOC


def multi_arm_turn3_codegen_prompt(
    instruction: str,
    robot_ids: List[int] = None,
    all_points: list = None,
    context_summary: str = "",
    positions: dict = None,
) -> str:
    """
    Turn 3: Multi-arm code generation prompt.

    Args:
        instruction: Natural language task description.
        robot_ids: Robot IDs [left_id, right_id] (e.g., [2, 3]).
        all_points: Turn 2 detected critical points.
        context_summary: Scene context summary from Turn 0~2.
        positions: Detected object positions (single-arm format, for key listing).

    Returns:
        Turn 3 user prompt string.
    """
    if robot_ids is None:
        robot_ids = [2, 3]

    # Format detected points
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
Choose the most appropriate point for the task.
Access via `pos_left["object"]["points"]["label"]` or `pos_right["object"]["points"]["label"]`.
{points_desc}
"""

    # positions key listing (object names only)
    positions_keys = ""
    if positions:
        key_lines = []
        for name, info in positions.items():
            if isinstance(info, dict) and "position" in info:
                pts = info.get("points", {})
                pt_labels = ", ".join(f'"{k}"' for k in pts.keys()) if pts else ""
                key_lines.append(f'  - "{name}": position, points: [{pt_labels}]')
        positions_keys = "\n".join(key_lines)

    context_section = ""
    if context_summary:
        context_section = f"""### Scene Context (from prior analysis session)

{context_summary}

"""

    prompt = f"""{context_section}### Your Job (Turn 3 — Multi-Arm Code Generation)

Generate executable Python code to complete the task using **two** SO-101 robot arms.
Use the scene understanding, detected objects, and grasp/place points from our previous conversation turns.

**Task**: "{instruction}"

**Robot Setup**:
- Two SO-101 robot arms: left arm (robot{robot_ids[0]}) handles the left side and center, right arm (robot{robot_ids[1]}) handles the right side and center.
- A pre-created `skills` object (MultiArmSkills) is provided as a global variable.
- Call `skills.connect()` at the start and `skills.disconnect()` in a finally block.

**Workspace Images** (provided as attached images):
- **Image 1**: left_arm (robot{robot_ids[0]}) workspace. The bright area inside the cyan arc is the reachable zone for left_arm. Darkened areas are out of reach.
- **Image 2**: right_arm (robot{robot_ids[1]}) workspace. The bright area inside the cyan arc is the reachable zone for right_arm. Darkened areas are out of reach.
- Assign each object to the arm whose workspace covers that object (bright area in the corresponding image).
- If an object is reachable by both arms, prefer the arm closer to it.
- Reach range: [0.20, 0.41]m from each arm's base.

{points_section}
**The `positions` dictionary** will be provided at runtime with this structure:
```python
positions = {{
    "left_arm": {{
        "object_name": {{
            "position": [x, y, z],   # in left arm's coordinate frame
            "points": {{"<label>": [x, y, z], ...}}
        }},
        ...
    }},
    "right_arm": {{
        "object_name": {{
            "position": [x, y, z],   # in right arm's coordinate frame
            "points": {{"<label>": [x, y, z], ...}}
        }},
        ...
    }},
}}
```

**How to use positions**:
```python
# First, split positions by arm at the top of execute_task()
pos_left = positions["left_arm"]
pos_right = positions["right_arm"]

# Then access object positions for each arm
left_pick = pos_left["object_name"]["position"]    # use with skills.left_arm.*
right_pick = pos_right["object_name"]["position"]  # use with skills.right_arm.*
```

Available object keys:
{positions_keys}

- **CRITICAL**: Use ONLY exact key names from the `positions` dictionary.
- **CRITICAL**: Do NOT redefine or hardcode the `positions` dictionary in your code.
- **CRITICAL**: Do NOT hardcode any coordinate values.

{MULTI_ARM_API_DOC}

**Skill Composition Patterns**:

```python
# BOTH ARMS (parallel operation)
pos_left = positions["left_arm"]
pos_right = positions["right_arm"]
approach_height = 0.20

# Open both grippers
skills.gripper_control(left_arm="open", right_arm="open",
    left_skill_description="Open left gripper", left_verification_question="Is left gripper open?",
    right_skill_description="Open right gripper", right_verification_question="Is right gripper open?")

# Move both arms to approach positions
left_pick = pos_left["left_object"]["position"]
right_pick = pos_right["right_object"]["position"]
skills.move_to_position(
    left_arm=[left_pick[0], left_pick[1], approach_height],
    right_arm=[right_pick[0], right_pick[1], approach_height],
    left_skill_description="Move left arm above left_object",
    left_verification_question="Is left gripper above left_object?",
    right_skill_description="Move right arm above right_object",
    right_verification_question="Is right gripper above right_object?")

# Pick both
skills.pick_object(left_arm=left_pick, right_arm=right_pick,
    left_object_name="left_object", right_object_name="right_object",
    left_skill_description="Pick left_object", left_verification_question="Is left_object grasped?",
    right_skill_description="Pick right_object", right_verification_question="Is right_object grasped?")

# ONE ARM ONLY (other arm holds position with "wait")
skills.move_to_position(
    left_arm="wait",
    right_arm=[right_target[0], right_target[1], approach_height],
    right_skill_description="Move right arm above target",
    right_verification_question="Is right arm above target?")

skills.pick_object(left_arm="wait", right_arm=right_pick,
    right_object_name="right_object",
    right_skill_description="Pick right_object", right_verification_question="Is right_object grasped?")

# PIXEL-BASED PLACEMENT (for locations NOT in the positions dict, e.g., empty spot on table)
# Specify [y, x] in normalized 0–1000 coordinates from the top-view image.
target_pixel = [y, x]  # determine from the workspace image
skills.move_to_pixel(
    left_arm=target_pixel,
    right_arm="wait",
    left_skill_description="Move left arm above target location",
    left_verification_question="Is left arm above target location?")

skills.place_at_pixel(
    left_arm=target_pixel,
    right_arm="wait",
    left_is_table=True,
    left_skill_description="Place object at target location",
    left_verification_question="Is object placed at target location?")

skills.move_to_pixel(
    left_arm=target_pixel,
    right_arm="wait",
    left_skill_description="Retract from target location",
    left_verification_question="Is left arm clear of target location?")

# SUBTASK + RE-DETECTION PATTERN (MANDATORY for multi-object tasks)
# Each pick-place of one object = one subtask.
# After each subtask, re-detect all objects to update positions.

# Subtask 1
skills.set_subtask("pick A with left arm and place at center of workspace")
skills.gripper_control(left_arm="open", right_arm="wait", ...)
skills.move_to_position(left_arm=[...approach...], right_arm="wait", ...)
skills.pick_object(left_arm=pos_left["A"]["position"], right_arm="wait", ...)
skills.move_to_position(left_arm=[...above target...], right_arm="wait", ...)
skills.place_object(left_arm=target_pos, right_arm="wait", ...)
skills.clear_subtask()

# Re-detection (MANDATORY between subtasks)
skills.move_to_initial_state()  # clear arms from camera view
updated = skills.detect_objects(["A", "B"])
# Update both arm position dicts
if "left_arm" in updated:
    pos_left.update(updated["left_arm"])
if "right_arm" in updated:
    pos_right.update(updated["right_arm"])

# Subtask 2
skills.set_subtask("pick B with right arm and place on top of A")
# ... pick and place B ...
skills.clear_subtask()
```

**Code Skeleton**:

```python
def execute_task():
    '''Execute the bi-arm robot task.'''
    skills.connect()

    try:
        pos_left = positions["left_arm"]
        pos_right = positions["right_arm"]
        approach_height = 0.20

        skills.move_to_initial_state()

        # Subtask 1: set_subtask → pick → place → clear_subtask
        # Re-detection: move_to_initial_state → detect_objects → update positions
        # Subtask 2: set_subtask → pick → place → clear_subtask
        # ... repeat for each object ...

        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

**Task Requirements**:
1. Assign objects to the appropriate arm based on workspace images (bright area = reachable).
2. Use `skills.move_to_position()` / `skills.pick_object()` / `skills.place_object()` / `skills.gripper_control()` for ALL operations. Pass `left_arm="wait"` or `right_arm="wait"` for the arm that should hold position.
   - For locations NOT in the `positions` dict (e.g., empty spot on table), use `skills.move_to_pixel()` / `skills.place_at_pixel()` with [y, x] in normalized 0–1000 coordinates.
3. **Subtask pattern**: Each pick-place of one object = one subtask. Wrap with `set_subtask()` before and `clear_subtask()` after.
4. **Re-detection (MANDATORY)**: After each subtask (after `clear_subtask()`), call `skills.move_to_initial_state()` to clear arms from camera view, then `skills.detect_objects([...all object names...])` to update positions. Skip re-detection only after the very last subtask.
5. Always start with `skills.move_to_initial_state()`, end with `skills.move_to_free_state()`.
6. Use `approach_height = 0.20` for approach/retreat movements.
7. ALWAYS pass `left_skill_description`/`right_skill_description` and `left_verification_question`/`right_verification_question` for every arm that is NOT `"wait"`.
8. Always include try/finally with `skills.disconnect()` for cleanup.

**Generate the complete executable Python code:**
"""
    return prompt
