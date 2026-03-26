"""
detect_objects 스킬 전용 프롬프트

코드 실행 중 실시간 객체 re-detection을 위한 간결한 프롬프트.
코드 생성 파이프라인의 Turn 1/2와 동일한 역할이지만,
독립 호출(stateless)에 최적화되어 불필요한 컨텍스트를 제거.
"""


def detect_t1_prompt(object_names: list) -> str:
    """Turn 1: bbox detection 프롬프트.

    Args:
        object_names: 검출할 객체 이름 리스트

    Returns:
        프롬프트 문자열
    """
    labels_str = ", ".join(f'"{q}"' for q in object_names)
    return f"""Detect the following objects in the overhead camera image: [{labels_str}]

Return a JSON array with bounding boxes in normalized 0-1000 coordinates:
```json
[{{"box_2d": [ymin, xmin, ymax, xmax], "label": "object_name"}}]
```
Only include clearly visible objects. Use exact label names."""


def detect_t2_prompt(object_label: str, scene_summary: str = "") -> str:
    """Turn 2: crop 이미지에서 grasp point detection 프롬프트.

    Args:
        object_label: 객체 이름
        scene_summary: Turn 0 장면 분석 요약 (있으면 컨텍스트로 prepend)

    Returns:
        프롬프트 문자열
    """
    context = f"Scene context:\n{scene_summary}\n\n" if scene_summary else ""

    return f"""{context}This is a cropped close-up of "{object_label}" from the overhead camera.

Identify the **grasp point** — the center of the object's **top surface** as seen from above.
The robot gripper approaches from directly above, so the grasp point must be on the widest visible top face.
Do NOT pick the volumetric center of the object — pick the center of the top surface visible in this image.

Return a JSON block:
```json
{{"critical_points": [{{"point_2d": [y, x], "label": "grasp center", "role": "grasp"}}]}}
```
Coordinates are [y, x] normalized to 0-1000 relative to this cropped image."""
