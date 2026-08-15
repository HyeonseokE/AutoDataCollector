


def turn1_detect_task_relevant_objects_prompt() -> str:
    """Turn 1: localize task related target object with bbox.(rough position)"""
    return """\
Based on your scene analysis above, now for each task-relevant object, do TWO things:

**Part A — Bounding Box Detection**
Detect bounding boxes in the given image (overhead camera image).

**Part B — Manipulation Strategy**
For each detected object, reason about how it should be manipulated to accomplish the task.

For each object, provide:
1. **box_2d**: Bounding box as `[ymin, xmin, ymax, xmax]` — exactly 4 integers, each normalized to 0–1000 (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of this cropped image).
2. **label**: A short, unique label matching the object name from your analysis.
3. **manipulation_strategy**: An object describing how to manipulate this object:
   - **needs_manipulation** (bool): Does this object need to be physically manipulated (grasped, pushed, folded, etc.) to complete the task?
   - **arm_assignment** (`"left"`, `"right"`, or `"bimanual"`): Which arm(s) should handle this object? Base this on the object's position in the image (left half → left arm, right half → right arm, center or large deformable → bimanual).
   - **grasp_approach** (string): Concise description of HOW to grasp/interact — e.g., "pinch the center from above", "grab the top-left and top-right edges with both arms and fold downward", "push from the left side".
   - **expected_points** (list of strings): The point labels you expect to identify in the next crop-and-point step — e.g., `["grasp center"]` for a simple pick, `["top left grasp", "top right grasp", "bottom left fold target", "bottom right fold target"]` for a bimanual fold.

### Output Format
Return a JSON array:
```json
[
  {
    "box_2d": [ymin, xmin, ymax, xmax],
    "label": "object_name",
    "manipulation_strategy": {
      "needs_manipulation": true,
      "arm_assignment": "bimanual",
      "grasp_approach": "Grab the top-left and top-right edges with both arms and fold to the bottom edge",
      "expected_points": ["top left grasp", "top right grasp", "bottom left fold target", "bottom right fold target"]
    }
  }
]
```

**Important**:
- Include **ALL visible objects** on the workspace — both task-relevant objects AND non-task objects (obstacles).
- For **placement targets** (plates, bowls, containers where other objects will be placed ON): set `needs_manipulation: false` but `expected_points: ["plate center"]`. Their center position is needed for placement.
- For **pure obstacles** (objects not mentioned in the task at all): set `needs_manipulation: false` and `expected_points: []`. These are only used for collision avoidance.
- Focus on providing accurate bounding box coordinates.
- **Carefully match each bounding box to the correct label** by comparing the visual appearance of each detected object with your analysis from above. Do NOT swap labels between objects.
- **Every label must be unique.** If multiple objects of the same type exist, append a numeric suffix to distinguish them (e.g., `"egg_1"`, `"egg_2"`, `"red plate_1"`, `"red plate_2"`).
- **Manipulation strategy must be task-aware**: consider the task instruction from Turn 0 and reason about WHAT needs to happen to each object and HOW."""
