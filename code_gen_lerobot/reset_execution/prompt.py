"""
Reset Execution Prompts

LLM prompts for reset/reverse task execution (multi-turn VLM pipeline).

Includes:
- turn0_reset_scene_understanding_prompt: Scene understanding (current + initial images)
- turn1_reset_bbox_detection_prompt: BBox detection with enforced labels
- reset_context_summary_prompt: Session 1→2 context handoff
- codegen_reset_with_context_prompt: Session 2 code generation
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
    original_instruction: str = None,
    reset_instruction: str = None,
) -> str:
    """
    Turn 1 (Reset 전용): Forward에서 검출된 라벨을 강제 사용하여 bbox 검출 + 조작 전략 수립.

    Forward의 turn1_prompt와 달리, VLM이 라벨을 자유롭게 생성하지 않고
    Forward에서 사용한 정확한 라벨을 그대로 사용하도록 강제합니다.
    추가로 각 객체의 reset 조작 전략을 수립합니다.

    Args:
        original_object_labels: Forward에서 검출된 물체 라벨 리스트 (필수)
        original_instruction: Forward에서 수행한 태스크 명령 (reset 전략 수립에 활용)
        reset_instruction: 명시적 reset 태스크 명령 (주어지면 original_instruction 대신 사용)
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

    # Unfold mechanics — folded cloth has TWO edges: CREASE (where layers connect)
    # vs FREE edge (where the top layer ends, opposite of crease). Only grasping
    # the FREE edge unfolds; grasping the crease just flips the stack.
    UNFOLD_GUIDE = (
        "- For **deformable objects** (towel, cloth) that need unfolding: grasp the "
        "**FREE edge** (where the top layer ends) — NOT the **crease** (where the two "
        "layers connect). After a top-to-bottom fold, the FREE edge is at the BOTTOM "
        "of the folded stack. Use grasp labels `\"left bottom-edge grasp\"` / "
        "`\"right bottom-edge grasp\"`. Only identify grasp points on the current "
        "(deformed) state — unfold destination is provided separately."
    )

    if reset_instruction:
        forward_context = f"""
**Reset task**: {reset_instruction}
- For **rigid objects** (blocks, cups, etc.): simple pick-and-place is sufficient.
{UNFOLD_GUIDE}"""
    elif original_instruction:
        forward_context = f"""
**Forward task context**: The forward task was "{original_instruction}". Reverse it to restore each object's original state.
- For **rigid objects**: simple pick-and-place.
{UNFOLD_GUIDE}"""
    else:
        forward_context = ""

    return f"""\
Based on your Reset plan above, now for each task-relevant object, do TWO things:

**Part A — Bounding Box Detection**
Detect bounding boxes in the given image (overhead camera image).

**Part B — Reset Manipulation Strategy**
For each detected object, reason about how it should be manipulated to RESET it back to its original state.
{forward_context}

For each object, provide:
1. **box_2d**: Bounding box as `[ymin, xmin, ymax, xmax]` — exactly 4 integers, each normalized to 0–1000 (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of this image).
2. **label**: The object label.
{label_instruction}
3. **manipulation_strategy**: An object describing how to reset this object:
   - **needs_manipulation** (bool): Does this object need to be physically manipulated?
   - **arm_assignment** (`"left"`, `"right"`, or `"bimanual"`): Which arm(s) should handle this object?
   - **grasp_approach** (string): HOW to manipulate — e.g., "pick from center and place at target position" for rigid objects, "grab the BOTTOM edge of the folded stack (NOT the crease) with both arms and lift back up to the original top position" for deformable objects that were folded.
   - **expected_points** (list of strings): Point labels to identify in the crop step. Only include **grasp points** (where to grab the object) — e.g., `["grasp center"]` for simple pick-and-place, `["left bottom-edge grasp", "right bottom-edge grasp"]` for unfolding a top-to-bottom fold (the bottom-edge of the folded stack is where the originally-grasped corners now rest). Do NOT include target/destination points — the unfold destination is already known from the original pre-task positions and will be provided separately.

### Output Format
Return a JSON array:
```json
[
  {{
    "box_2d": [ymin, xmin, ymax, xmax],
    "label": "object_name",
    "manipulation_strategy": {{
      "needs_manipulation": true,
      "arm_assignment": "left",
      "grasp_approach": "Pick from center and place at target position",
      "expected_points": ["grasp center"]
    }}
  }}
]
```

**Important**:
- Only include the main task-relevant objects (not sub-parts).
- Focus on providing accurate bounding box coordinates.
- **Carefully match each bounding box to the correct label** by comparing the visual appearance of each detected object with your analysis from above. Do NOT swap labels between objects.
- **Reset strategy must consider the forward task**: if the object was folded, it needs to be unfolded; if it was simply moved, a pick-and-place is sufficient."""


def lerobot_reset_code_gen_prompt(
    original_instruction: str,
    target_positions: Dict[str, List[float]],
    current_positions: Dict[str, List[float]],
    forward_spec: Dict = None,
    forward_code: str = None,
    robot_id: int = 3,
    is_random_reset: bool = False,
) -> str:
    """Single-turn 리셋 코드 생성 프롬프트 (MULTI_TURN=false 시 사용)."""

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
                f'    "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
            )
    target_str = "\n".join(target_lines)

    current_lines = []
    for name, info in current_positions.items():
        pos = get_pos(info)
        if pos is not None:
            current_lines.append(
                f'    "{name}": {{"position": [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]}},'
            )
    current_str = "\n".join(current_lines)

    forward_spec_str = json.dumps(forward_spec, indent=2) if forward_spec else "Not available"
    forward_code_str = (forward_code[:2000] + "\n... (truncated)") if forward_code and len(forward_code) > 2000 else (forward_code or "Not available")

    if is_random_reset:
        task_title = "RANDOM RESET (Shuffle Objects)"
        task_desc = "Move each object to its NEW RANDOM target position."
    else:
        task_title = "RESET to Initial Environment"
        task_desc = f'Restore each object to its initial position (before "{original_instruction}" was executed).'

    return f"""\
You are an AI assistant tasked with generating executable Python code for a LeRobot SO-101 robot arm.
Your task is to generate reset code that moves objects from their current positions to target positions.

### **Context: Forward Execution (What Was Done)**

1. **Forward Task Goal**: {original_instruction}

2. **Forward Spec**: ```json
{forward_spec_str}
```

3. **Forward Code**: ```python
{forward_code_str}
```

### **Task: {task_title}**
{task_desc}

### **Environment States**
Object position z-coordinate = object height (table surface is z=0).

The `current_positions` and `target_positions` dictionaries are provided at runtime as global variables. Here are the actual detected values:
```python
current_positions = {{
{current_str}
}}
target_positions = {{
{target_str}
}}
```
- **CRITICAL**: Do NOT redefine or hardcode the `current_positions` or `target_positions` dictionaries in your code. They are already available as global variables at runtime. Access them directly (e.g., `current_positions["red block"]["position"]`).

### **SUBTASK + RE-DETECTION PATTERN** (MANDATORY at every subtask boundary)

**Why mandatory**: After a subtask, the scene has changed (objects moved, stacks unstacked, heights shifted). Trusting the original `current_positions` for the next subtask risks gripping empty air or wrong locations. Forward-execution enforces this same rule — reset must follow it.

**Rule**: After EVERY subtask (except the very last one), you MUST:
1. `skills.clear_subtask()` — close the just-completed subtask (gripper must be empty).
2. `skills.move_to_initial_state()` — clear the arm out of the camera view.
3. `skills.detect_objects([list of objects still to be moved])` — refresh
   their positions. The call AUTOMATICALLY merges fresh detections into
   `current_positions` in-place AND preserves Turn 2 point labels (e.g.
   "top placement point", "grasp center"). No `updated = ...` / conditional
   update boilerplate — objects whose detection failed simply keep their
   previous values.
4. **Re-assign** all local variables (`cur`, `tgt`, etc.) from
   `current_positions` (previously extracted variables still hold OLD values).
5. `skills.set_subtask(...)` — start the next subtask with FRESH coordinates.

```python
# === STEP 1: Move 1st object (object_A) to its target ===
approach_height = 0.10
cur = current_positions["object_A"]["position"]
tgt = target_positions["object_A"]["position"]
skills.set_subtask("move object_A to target")

# Step 1-1. Pick object_A — approach opens gripper in-motion (start at 30%)
skills.move_to_position([cur[0], cur[1], approach_height], target_name="object_A", gripper_action="open", gripper_start_fraction=0.3, skill_description="Approach object_A and open gripper", verification_question="Is the gripper above object_A and open?")
skills.execute_pick_object(cur, object_name="object_A", skill_description="Pick up object_A", verification_question="Is object_A grasped?")
skills.move_to_position([cur[0], cur[1], approach_height], target_name="object_A", skill_description="Lift object_A", verification_question="Is object_A lifted?")

# Step 1-2. Place object_A at target — retreat closes gripper starting at 20%
skills.move_to_position([tgt[0], tgt[1], approach_height], target_name="object_A target", skill_description="Move object_A above target position", verification_question="Is object_A above target position?")
skills.execute_place_object(tgt, is_table=True, gripper_open_ratio=0.7, target_name="object_A target", skill_description="Place object_A at target position", verification_question="Is object_A placed at target position?")
skills.move_to_position([tgt[0], tgt[1], approach_height], target_name="object_A target", gripper_action="close", gripper_start_fraction=0.7, skill_description="Retreat from target and close gripper", verification_question="Is the gripper clear of target and closed?")

# === MANDATORY SUBTASK BOUNDARY: clear → initial → re-detect → reassign ===
skills.clear_subtask()
skills.move_to_initial_state()
# Auto in-place merge into current_positions; failed detections keep prev value.
skills.detect_objects(["object_B", "object_C"])  # only objects still to be moved

# === STEP 2: Move 2nd object (object_B) — uses REFRESHED position ===
cur = current_positions["object_B"]["position"]   # ← refreshed value
tgt = target_positions["object_B"]["position"]
skills.set_subtask("move object_B to target")
# ... (same pick → place pattern as Step 1) ...

# === MANDATORY BOUNDARY again before STEP 3 ===
skills.clear_subtask()
skills.move_to_initial_state()
skills.detect_objects(["object_C"])

# === STEP 3 (LAST subtask): no re-detection needed AFTER it — final move ===
cur = current_positions["object_C"]["position"]
tgt = target_positions["object_C"]["position"]
skills.set_subtask("move object_C to target")
# ... pick → place ...
skills.clear_subtask()  # final close; no detect needed since no more subtasks
```

**CRITICAL — identical/indistinguishable objects**: If two remaining objects are visually identical (e.g., two red blocks), DO NOT include both in `detect_objects([...])` — the VLM cannot tell them apart and will swap labels. Re-detect only objects with unique appearance; for identical ones, keep using their original `current_positions` value.

### **Code Template**

```python
def execute_reset_task():
    '''Move objects from current positions to target positions.'''
    # `skills`, `current_positions`, `target_positions` are pre-injected global variables.
    # Do NOT import anything. Do NOT instantiate LeRobotSkills yourself.
    skills.connect()

    try:
        approach_height = 0.10

        skills.move_to_initial_state()

        # For each object: set_subtask → PICK → PLACE → clear_subtask
        # BETWEEN every consecutive pair of subtasks (i.e., after every subtask EXCEPT the last):
        #     clear_subtask() → move_to_initial_state() → detect_objects([remaining_objects])
        #     → re-assign cur/tgt from the updated dict → set_subtask(next).
        # This is MANDATORY — without re-detection, stale heights cause empty-air grips.
        # current_positions and target_positions are global variables (injected at runtime).

        # ... (your reset logic referencing the global dicts) ...

        skills.clear_subtask()
        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()

if __name__ == "__main__":
    execute_task()
```

### **Guidelines**

1. Generate code that moves each object from current to target position
2. **MUST call `skills.set_subtask("move object_name to target")` before each logical unit of work** — this labels the recording.
   - CRITICAL: At both `set_subtask()` and `clear_subtask()`, the gripper must be empty (no object held). A subtask boundary is defined by the gripper-empty condition. Do NOT call `clear_subtask()` until the gripper has released.
3. **Re-detection (MANDATORY at every subtask boundary)**: After every completed subtask EXCEPT the very last one, you MUST `clear_subtask()` → `move_to_initial_state()` → `detect_objects([objects_still_to_move])` → re-assign every local variable (`cur`, `tgt`, …) from the updated dict, then `set_subtask(...)` for the next step. Reason: picking objects shifts the scene (heights, stacks, occlusion) — stale `current_positions` makes the next pick grip empty air.
   - **CRITICAL (identical objects)**: NEVER include visually identical objects together in a single `detect_objects([...])` call. The VLM cannot distinguish them and will swap labels. For identical objects, keep using the original `current_positions` and re-detect only objects with unique appearance.
   - **CRITICAL**: After re-detection, you MUST **re-assign ALL local variables** (e.g., `cur`, `tgt`) from the updated dict. Previously extracted variables still reference OLD values.
   - The LAST subtask does not need a trailing re-detection (nothing left to pick).
4. **Follow the Skill Composition Patterns above exactly** — pick approach uses `move_to_position(..., gripper_action="open", gripper_start_fraction=0.3)` (no standalone `gripper_open()`); place retreat uses `move_to_position(..., gripper_action="close", gripper_start_fraction=0.7)` (no standalone `gripper_close()`).
4. **ALWAYS reference `current_positions` and `target_positions` dicts** — e.g. `current_positions["name"]["position"]` and `target_positions["name"]["position"]`
5. Do NOT redefine or hardcode coordinate values — the dicts are injected as global variables at runtime and may change between episodes
6. **ALWAYS pass object/target positions as-is** to execute_pick_object and execute_place_object
7. Use `approach_height = 0.10` for all approach/lift movements
8. **ALWAYS use `gripper_open_ratio=0.7`** in execute_place_object
9. Use `is_table=True` when placing on table
10. Always include try/finally for proper cleanup
11. Always start with `move_to_initial_state()`, end with `skills.clear_subtask()`, `move_to_initial_state()` and `move_to_free_state()`
12. **ALWAYS pass `skill_description` and `verification_question` for every skill call** — as shown in the Skill Composition Patterns above

### **Output Format**
- Provide complete executable Python code
- Do not include markdown code blocks

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

The `current_positions` and `target_positions` dictionaries are provided at runtime as global variables. Here are the actual detected values:
```python
current_positions = {{
{current_str}
}}
target_positions = {{
{target_str}
}}
```
- **CRITICAL**: Do NOT redefine or hardcode the `current_positions` or `target_positions` dictionaries in your code. They are already available as global variables at runtime. Access them directly (e.g., `current_positions["red block"]["position"]`).

### **Available Skills**
| Method | Description | Key Parameters |
|--------|-------------|----------------|
| `connect()` | Connect to robot | - |
| `disconnect()` | Disconnect from robot | - |
| `gripper_open()` | Open gripper standalone (rare; prefer `move_to_position` with `gripper_action="open"`) | - |
| `gripper_close()` | Close gripper standalone (rare; prefer `move_to_position` with `gripper_action="close"`) | - |
| `move_to_initial_state()` | Move to initial/home position | - |
| `move_to_free_state()` | Move to safe parking position | - |
| `move_to_position(position, ...)` | Move end-effector to [x,y,z] | position, target_name |
| `rotate_90degree(direction)` | Rotate gripper 90° | direction: 1 (CW) or -1 (CCW) |
| `execute_pick_object(object_position, ...)` | Descend to pick, close gripper, save pitch | object_position, object_name |
| `execute_place_object(place_position, ...)` | Descend to place with saved pitch, open gripper | place_position, is_table, gripper_open_ratio, target_name |
| `execute_pull(start, distance, ...)` | **OPENING** drawer/door — pulls -x by `distance` m. Grasp + drag + release + retreat | start_position, distance, object_name |
| `execute_push(start, distance, ...)` | **CLOSING** drawer/door — pushes +x by `distance + 3cm` (margin). Push (gripper open) + retreat | start_position, distance, object_name |

**execute_pick_object**: Pass the object position as-is. The function internally handles the grasp height offset.
**execute_place_object**: Pass the target position as-is. The function internally calculates the correct release height.
  - is_table=True: place on table, is_table=False: place on another object
  - **ALWAYS use `gripper_open_ratio=0.7`**

**execute_pull** (OPENING drawer/door): pass `start_position` (handle grasp point) and `distance` (meters, extracted from instruction). Direction fixed -x. Approach with `gripper_action="open"`. Skill: descend → grasp → drag → release → retreat.

**execute_push** (CLOSING drawer/door): pass `start_position` (current handle position, after drawer is open — re-detect) and `distance` (meters; same value as the open distance). The skill internally pushes `distance + 3cm` for full closure. Direction fixed +x. Approach with `gripper_action="open"` (open jaws push from inside). Skill: descend → linear push → retreat.

### **SUBTASK + RE-DETECTION PATTERN** (MANDATORY at every subtask boundary)

**Why mandatory**: After a subtask, the scene has changed (objects moved, stacks unstacked, heights shifted). Trusting the original `current_positions` for the next subtask risks gripping empty air or wrong locations. Forward-execution enforces this same rule — reset must follow it.

**Rule**: After EVERY subtask (except the very last one), you MUST:
1. `skills.clear_subtask()` — close the just-completed subtask (gripper must be empty).
2. `skills.move_to_initial_state()` — clear the arm out of the camera view.
3. `skills.detect_objects([list of objects still to be moved])` — refresh
   their positions. The call AUTOMATICALLY merges fresh detections into
   `current_positions` in-place AND preserves Turn 2 point labels (e.g.
   "top placement point", "grasp center"). No `updated = ...` / conditional
   update boilerplate — objects whose detection failed simply keep their
   previous values.
4. **Re-assign** all local variables (`cur`, `tgt`, etc.) from
   `current_positions` (previously extracted variables still hold OLD values).
5. `skills.set_subtask(...)` — start the next subtask with FRESH coordinates.

```python
# === STEP 1: Move 1st object (object_A) to its target ===
approach_height = 0.10
cur = current_positions["object_A"]["position"]
tgt = target_positions["object_A"]["position"]
skills.set_subtask("move object_A to target")

# Step 1-1. Pick object_A — approach opens gripper in-motion (start at 30%)
skills.move_to_position([cur[0], cur[1], approach_height], target_name="object_A", gripper_action="open", gripper_start_fraction=0.3, skill_description="Approach object_A and open gripper", verification_question="Is the gripper above object_A and open?")
skills.execute_pick_object(cur, object_name="object_A", skill_description="Pick up object_A", verification_question="Is object_A grasped?")
skills.move_to_position([cur[0], cur[1], approach_height], target_name="object_A", skill_description="Lift object_A", verification_question="Is object_A lifted?")

# Step 1-2. Place object_A at target — retreat closes gripper starting at 20%
skills.move_to_position([tgt[0], tgt[1], approach_height], target_name="object_A target", skill_description="Move object_A above target position", verification_question="Is object_A above target position?")
skills.execute_place_object(tgt, is_table=True, gripper_open_ratio=0.7, target_name="object_A target", skill_description="Place object_A at target position", verification_question="Is object_A placed at target position?")
skills.move_to_position([tgt[0], tgt[1], approach_height], target_name="object_A target", gripper_action="close", gripper_start_fraction=0.7, skill_description="Retreat from target and close gripper", verification_question="Is the gripper clear of target and closed?")

# === MANDATORY SUBTASK BOUNDARY: clear → initial → re-detect → reassign ===
skills.clear_subtask()
skills.move_to_initial_state()
# Auto in-place merge into current_positions; failed detections keep prev value.
skills.detect_objects(["object_B", "object_C"])  # only objects still to be moved

# === STEP 2: Move 2nd object (object_B) — uses REFRESHED position ===
cur = current_positions["object_B"]["position"]   # ← refreshed value
tgt = target_positions["object_B"]["position"]
skills.set_subtask("move object_B to target")
# ... (same pick → place pattern as Step 1) ...

# === MANDATORY BOUNDARY again before STEP 3 ===
skills.clear_subtask()
skills.move_to_initial_state()
skills.detect_objects(["object_C"])

# === STEP 3 (LAST subtask): no re-detection needed AFTER it — final move ===
cur = current_positions["object_C"]["position"]
tgt = target_positions["object_C"]["position"]
skills.set_subtask("move object_C to target")
# ... pick → place ...
skills.clear_subtask()  # final close; no detect needed since no more subtasks
```

**CRITICAL — identical/indistinguishable objects**: If two remaining objects are visually identical (e.g., two red blocks), DO NOT include both in `detect_objects([...])` — the VLM cannot tell them apart and will swap labels. Re-detect only objects with unique appearance; for identical ones, keep using their original `current_positions` value.

### **Code Template**

**CRITICAL**: You MUST reference `current_positions` and `target_positions` dicts in skill calls. Do NOT hardcode coordinate values directly. The dicts are provided as global variables at runtime.

```python
def execute_reset_task():
    '''Move objects from current positions to target positions.'''
    # `skills`, `current_positions`, `target_positions` are pre-injected global variables.
    # Do NOT import anything. Do NOT instantiate LeRobotSkills yourself.
    skills.connect()

    try:
        approach_height = 0.10

        skills.move_to_initial_state()

        # For each object: set_subtask → PICK → PLACE
        # MUST call skills.set_subtask() before each object's pick-place sequence.
        # ALWAYS use: current_positions["name"]["position"] and target_positions["name"]["position"]

        # ... (your reset logic referencing the dicts) ...

        skills.clear_subtask()
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
skills.move_to_position([cx, cy, approach_height], target_name="C", gripper_action="open", gripper_start_fraction=0.3)
skills.execute_pick_object([cx, cy, 0.08], object_name="C")
skills.move_to_position([cx, cy, approach_height], target_name="C")

skills.move_to_position([c_tx, c_ty, approach_height], target_name="original position")
skills.execute_place_object([c_tx, c_ty, c_tz], is_table=True, gripper_open_ratio=0.7, target_name="original position")
skills.move_to_position([c_tx, c_ty, approach_height], target_name="original position", gripper_action="close", gripper_start_fraction=0.7)

# Step 2: Pick the MIDDLE object B (z = 0.06) — safe because C is removed
skills.move_to_position([bx, by, approach_height], target_name="B", gripper_action="open", gripper_start_fraction=0.3)
skills.execute_pick_object([bx, by, 0.06], object_name="B")
skills.move_to_position([bx, by, approach_height], target_name="B")

skills.move_to_position([b_tx, b_ty, approach_height], target_name="original position")
skills.execute_place_object([b_tx, b_ty, b_tz], is_table=True, gripper_open_ratio=0.7, target_name="original position")
skills.move_to_position([b_tx, b_ty, approach_height], target_name="original position", gripper_action="close", gripper_start_fraction=0.7)

# Step 3: Pick the BOTTOM object A (z = 0.02) — safe because B and C are removed
skills.move_to_position([ax, ay, approach_height], target_name="A", gripper_action="open", gripper_start_fraction=0.3)
skills.execute_pick_object([ax, ay, 0.02], object_name="A")
skills.move_to_position([ax, ay, approach_height], target_name="A")

skills.move_to_position([a_tx, a_ty, approach_height], target_name="original position")
skills.execute_place_object([a_tx, a_ty, a_tz], is_table=True, gripper_open_ratio=0.7, target_name="original position")
skills.move_to_position([a_tx, a_ty, approach_height], target_name="original position", gripper_action="close", gripper_start_fraction=0.7)
```


### **Guidelines**

1. Generate code that moves each object from current to target position
2. **MUST call `skills.set_subtask("move object_name to target")` before each logical unit of work** — this labels the recording.
   - CRITICAL: At both `set_subtask()` and `clear_subtask()`, the gripper must be empty (no object held). A subtask boundary is defined by the gripper-empty condition. Do NOT call `clear_subtask()` until the gripper has released.
3. **Re-detection (MANDATORY at every subtask boundary)**: After every completed subtask EXCEPT the very last one, you MUST `clear_subtask()` → `move_to_initial_state()` → `detect_objects([objects_still_to_move])` → re-assign every local variable (`cur`, `tgt`, …) from the updated dict, then `set_subtask(...)` for the next step. Reason: picking objects shifts the scene (heights, stacks, occlusion) — stale `current_positions` makes the next pick grip empty air.
   - **CRITICAL (identical objects)**: NEVER include visually identical objects together in a single `detect_objects([...])` call. The VLM cannot distinguish them and will swap labels. For identical objects, keep using the original `current_positions` and re-detect only objects with unique appearance.
   - **CRITICAL**: After re-detection, you MUST **re-assign ALL local variables** (e.g., `cur`, `tgt`) from the updated dict. Previously extracted variables still reference OLD values.
   - The LAST subtask does not need a trailing re-detection (nothing left to pick).
4. **Follow the Skill Composition Patterns above exactly** — pick approach uses `move_to_position(..., gripper_action="open", gripper_start_fraction=0.3)` (no standalone `gripper_open()`); place retreat uses `move_to_position(..., gripper_action="close", gripper_start_fraction=0.7)` (no standalone `gripper_close()`).
4. **ALWAYS reference `current_positions` and `target_positions` dicts** — e.g. `current_positions["name"]["position"]` and `target_positions["name"]["position"]`
5. Do NOT redefine or hardcode coordinate values — the dicts are injected as global variables at runtime and may change between episodes
6. **ALWAYS pass object/target positions as-is** to execute_pick_object and execute_place_object
7. Use `approach_height = 0.10` (10cm) for all approach/lift movements
8. **Pitch Handling**: Pitch is automatically saved at pick and restored at place
9. **ALWAYS use `gripper_open_ratio=0.7`** in execute_place_object
10. Use `is_table=True` when placing on table
11. Always include try/finally for proper cleanup
12. Always start with `move_to_initial_state()`, end with `skills.clear_subtask()`, `move_to_initial_state()` and `move_to_free_state()`
13. **Unstacking**: If objects are stacked, always unstack from top to bottom before moving them
14. **ALWAYS pass `skill_description` and `verification_question` for every skill call** — as shown in the Skill Composition Patterns above

### **Output Format**
- Provide complete executable Python code
- Do not include markdown code blocks

**Generate the complete executable RESET code:**"""
