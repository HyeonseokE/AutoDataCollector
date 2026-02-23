

def turn1_detect_task_relevant_objects_prompt() -> str:
    """
    Turn 1: localize task related target object with bbox.(rough position)
    
    """

    return """\
Now, for each task-relevant object you identified above, detect their bounding boxes in the given image (overhead camera image).

For each object, provide:
1. **box_2d**: Bounding box as `[ymin, xmin, ymax, xmax]` — exactly 4 integers, each normalized to 0–1000.(where 0,0 is the top-left corner and 1000,1000 is the bottom-right corner of this cropped image.)
2. **label**: A short, unique label matching the object name from your analysis.

### Output Format
Return a JSON array:
```json
[
  {"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"},
  {"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}
]
```

**Important**:
- Only include the main task-relevant objects (not sub-parts).
- Focus on providing accurate bounding box coordinates.
- **Carefully match each bounding box to the correct label** by comparing the visual appearance of each detected object with your analysis from above. Do NOT swap labels between objects."""
