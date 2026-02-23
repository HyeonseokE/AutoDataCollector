"""
Turn 2+ Prompt — Crop-then-Point (per object)

Predicts grasp/interaction points from per-object crop images.
(1) grasp points: optimal gripper grasp locations regarding success of the task.
(2) interaction points: None-grasping locations, only functional sub-parts for task (pin, hole, slot, etc.) that are critical for task execution.
Crops are generated from bboxes detected in Turn 1, and this prompt is called once per object.
"""


def turn2_crop_pointing_prompt(object_label: str) -> str:
    """
    Turn 2: exact location for each critical manipulation point on the cropped image of the target object.
    
    Args:
        object_label: label of the object detected in Turn 1 (e.g., "red block")

    Returns:
        prompt string corpus for turn 2
    """
    return f"""
Now I am showing you a **cropped close-up image** of the object "{object_label}" from the overhead camera.

Based on your analysis above, identify critical points on this object.

**Point types**:
1. **grasp** — optimal gripper grasp location for successful task execution.
2. **interaction** — non-grasping functional sub-part location critical for task execution (e.g., pin, hole, slot, rim, edge).

For each point, provide:
1. **point_2d**: The point location as `[y, x]` — 2 integers, each normalized to **0–1000** (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of this cropped image).
2. **label**: A short, descriptive name (e.g., "grasp center", "pin tip", "hole opening").
3. **role**: `"grasp"` or `"interaction"`.
4. **reasoning**: Why this point matters, referencing your previous analysis.

### Output Format
Return a JSON block:
```json
{{
  "critical_points": [
    {{"point_2d": [y, x], "label": "grasp center", "role": "grasp", "reasoning": "..."}},
    {{"point_2d": [y, x], "label": "pin tip", "role": "interaction", "reasoning": "..."}}
  ]
}}
```

**Important**:
- Look carefully at the cropped image and provide accurate coordinates.
- Coordinates are normalized 0–1000 relative to this cropped image.
""".strip()


# ──────────────────────────────────────────────
# [ARCHIVED] 기존 turn2 prompt
# "identify each critical manipulation point"가 모호하여
# point types 정의를 output format보다 위로 올려 명시함.
# ──────────────────────────────────────────────
# def turn2_crop_pointing_prompt(object_label: str) -> str:
#     return f"""
# Now I am showing you a **cropped close-up image** of the object "{object_label}" from the overhead camera.
#
# Based on your analysis above, identify each critical manipulation point on this object and provide their **precise locations** in the image.
#
# For each point, provide:
# 1. **point_2d**: The point location as `[y, x]` — 2 integers, each normalized to **0–1000** (where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of this cropped image).
# 2. **label**: A short, descriptive name (e.g., "grasp center", "pin tip", "hole opening").
# 3. **role**: One of:
#    - `"grasp"` — optimal gripper grasp location.
#    - `"interaction"` — functional sub-part for task (pin, hole, slot, etc.).
# 4. **reasoning**: Why this point matters, referencing your previous analysis.
#
# ### Output Format
# Return a JSON block:
# ```json
# {{
#   "critical_points": [
#     {{"point_2d": [y, x], "label": "grasp center", "role": "grasp", "reasoning": "..."}},
#     {{"point_2d": [y, x], "label": "pin tip", "role": "interaction", "reasoning": "..."}}
#   ]
# }}
# ```
#
# **Important**:
# - Look carefully at the cropped image and provide accurate coordinates.
# - Coordinates are normalized 0–1000 relative to this cropped image.
# """.strip()
