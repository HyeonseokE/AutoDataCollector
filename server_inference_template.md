# Server Inference API Template

서버에 배포된 모델들의 Input/Output 명세서

---

## Model 1: CodeGen LLM

> 로봇 제어 Python 코드 생성

**Endpoint:** `POST /v1/chat/completions`

### INPUT (서버로 전송)

```json
{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{"role": "user", "content": "<프롬프트 문자열>"}],
    "max_tokens": 1500,
    "temperature": 0.0
}
```

### OUTPUT (서버에서 수신)

```json
{
    "choices": [{"message": {"content": "<생성된 Python 코드>"}}]
}
```

### 예시

**Request:**
```json
{
    "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "messages": [{
        "role": "user",
        "content": "You are an AI assistant generating Python code for a LeRobot SO-101 robot arm.\n\n### Input Details:\n1. Goal: pick up the red block and place it on the blue dish\n2. Object Positions:\n   \"red block\": {\"position\": [0.1523, 0.0547, 0.0234], \"gripper_offset\": 0.025}\n   \"blue dish\": {\"position\": [0.1847, -0.0342, 0.0152], \"gripper_offset\": 0.0}\n\nGenerate executable Python code:"
    }],
    "max_tokens": 1500,
    "temperature": 0.0
}
```

**Response:**
```json
{
    "choices": [{
        "message": {
            "content": "from skills.skills_lerobot import LeRobotSkills\n\ndef execute_task():\n    skills = LeRobotSkills(\n        robot_config=\"robot_configs/robot/so101_robot3.yaml\",\n        frame=\"world\",\n    )\n    skills.connect()\n\n    try:\n        approach_height = 0.15\n        skills.move_to_initial_state()\n\n        pick_obj = positions[\"red block\"]\n        pick_pos = pick_obj[\"position\"]\n        offset = pick_obj[\"gripper_offset\"]\n\n        place_obj = positions[\"blue dish\"]\n        place_pos = place_obj[\"position\"]\n\n        skills.gripper_open()\n        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height], gripper_offset=offset)\n        skills.execute_pick_object(pick_pos, gripper_offset=offset)\n        skills.move_to_position([pick_pos[0], pick_pos[1], approach_height])\n\n        skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset)\n        skills.execute_place_object(place_pos, gripper_offset=offset, is_table=True, gripper_open_ratio=0.7)\n        skills.move_to_position([place_pos[0], place_pos[1], approach_height], gripper_offset=offset)\n\n        skills.move_to_free_state()\n    finally:\n        skills.disconnect()\n\nif __name__ == \"__main__\":\n    execute_task()"
        }
    }]
}
```

---

## Model 2: Judge VLM

> 태스크 성공 여부 판단 (멀티모달)

**Endpoint:** `POST /v1/chat/completions`

### INPUT (서버로 전송)

```json
{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [{
        "role": "user",
        "content": [
            {"type": "text", "text": "<프롬프트 문자열>"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<이미지1>"}},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<이미지2>"}}
        ]
    }],
    "max_tokens": 1000,
    "temperature": 0.0
}
```

### OUTPUT (서버에서 수신)

```json
{
    "choices": [{"message": {"content": "<분석 결과 텍스트>"}}]
}
```

### 예시

**Request:**
```json
{
    "model": "Qwen/Qwen2-VL-2B-Instruct",
    "messages": [{
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "You are a Task Completion Judge.\n\n### Goal\npick up the red block and place it on the blue dish\n\n### Images\n- Image 1: Initial state (before execution)\n- Image 2: Final state (after execution)\n\nAnalyze the images and respond:\nPREDICTION: TRUE or FALSE or UNCERTAIN\nREASONING: <your analysis>"
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."}
            },
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD..."}
            }
        ]
    }],
    "max_tokens": 1000,
    "temperature": 0.0
}
```

**Response:**
```json
{
    "choices": [{
        "message": {
            "content": "STEP 1 - Initial Image Analysis:\n- Red block visible at center-left of image\n- Blue dish visible at center-right of image\n\nSTEP 2 - Final Image Analysis:\n- Red block now positioned on top of blue dish\n- Blue dish in same location\n\nPREDICTION: TRUE\nREASONING: The red block has been successfully picked up and placed on the blue dish as instructed."
        }
    }]
}
```

---

## Model 3: Grounding DINO

> 객체 검출 (Zero-shot Object Detection)

**Endpoint:** `POST /detect` (커스텀 API)

### INPUT (서버로 전송)

```json
{
    "image": "<base64 인코딩된 RGB 이미지>",
    "text_query": "red block",
    "box_threshold": 0.25,
    "text_threshold": 0.25
}
```

### OUTPUT (서버에서 수신)

```json
{
    "detections": [
        {
            "label": "red block",
            "confidence": 0.92,
            "bbox": [300, 220, 340, 260],
            "center": [320, 240]
        }
    ]
}
```

### 예시

**Request:**
```json
{
    "image": "/9j/4AAQSkZJRgABAQEASABIAAD/4gIoSUNDX1BST0ZJTEUAAQ...",
    "text_query": "red block",
    "box_threshold": 0.25,
    "text_threshold": 0.25
}
```

**Response:**
```json
{
    "detections": [
        {
            "label": "red block",
            "confidence": 0.92,
            "bbox": [300, 220, 340, 260],
            "center": [320, 240]
        }
    ]
}
```

> ※ Grounding DINO는 OpenAI-compatible API가 아님. 서버 구현 시 위 형식으로 래핑 필요.

---

## 이미지 Base64 변환 (Python)

```python
import cv2
import base64

# BGR 이미지를 base64로 변환
def image_to_base64(bgr_image):
    _, buffer = cv2.imencode('.jpg', bgr_image)
    return base64.b64encode(buffer).decode('utf-8')

# 사용 예시
image = cv2.imread("image.jpg")  # shape: (480, 640, 3), BGR
b64_string = image_to_base64(image)
# 결과: "/9j/4AAQSkZJRgABAQEASABIAAD..."
```

---

## 서버 엔드포인트 요약

| 모델 | 엔드포인트 | 포트 (기본) |
|------|-----------|------------|
| CodeGen LLM | `POST /v1/chat/completions` | 8001 |
| Judge VLM | `POST /v1/chat/completions` | 8002 |
| Grounding DINO | `POST /detect` | 8003 |
