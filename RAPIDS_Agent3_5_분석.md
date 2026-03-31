# RAPIDS Agent 3 & Agent 5 — 구현 분석 보고서

> **담당**: 조현석 (Agent 3, 5), 안상현 (Agent 4, 5 공동)  
> **작성 기준 코드**: `/home/lerobot/AutoDataCollector/`  
> **작성일**: 2026-03-31

---

## 목차

1. [사전제안서 vs 실제 구현 차이 분석](#1-사전제안서-vs-실제-구현-차이-분석)
2. [현재 AutoDataCollection 구현 내용 상세](#2-현재-autodatacollection-구현-내용-상세)
3. [최종보고서 작성을 위한 추가 디테일](#3-최종보고서-작성을-위한-추가-디테일)

---

## 1. 사전제안서 vs 실제 구현 차이 분석

### 1.1 Agent 3 (Data Collection Agent) 비교

| 항목 | 사전제안서 계획 | 실제 구현 | 비고 |
|---|---|---|---|
| **LLM 모델** | Claude (CodeLLM) | GPT-4o, Gemini-3-flash, DeepSeek, Qwen2.5-Coder-7B 등 다중 모델 지원 | Claude 단일 모델 → 멀티 프로바이더 라우터로 확장 |
| **코드 생성 방식** | LLM 기반 policy 코드 자동 생성 | Multi-turn VLM 파이프라인 (Turn 0~3: Scene Understanding → BBox Detection → Critical Point Detection → Code Generation) | 단일 턴 → 4단계 멀티턴으로 고도화 |
| **Perception API** | Perception API + Low-level control API 조합 | VLM crop-then-point + RealSense D435 depth → 3D 좌표 변환 | 추상적 API 명세 → 구체적 detection 파이프라인 구현 |
| **로봇 제어 API** | 명시되지 않음 | LeRobotSkills 클래스: `move_to_position`, `execute_pick_object`, `execute_place_object`, `execute_press`, `execute_push`, `rotate_90degree` 등 17종 이상의 primitive skills | 사전제안서의 "Low-level control API"를 구체화 |
| **자가 수정 메커니즘** | 에러 피드백 기반 자가 수정 | VLM Judge 결과(TRUE/FALSE/UNCERTAIN) 기반 피드백 + 코드 캐싱을 통한 반복 실행 | 코드 수준 자가 수정보다는 실행-판정-재수집 루프로 구현 |
| **환경 초기 상태 복원** | 에피소드 종료 후 리셋 코드 자동 생성 | Reset Execution 파이프라인 구현: "original" 모드 (원래 위치 복원) + "random" 모드 (랜덤 셔플) | 계획대로 구현, 추가로 랜덤 셔플 모드 |
| **멀티 에피소드** | 무중단 연속 수집 | Forward → Judge → Reset → Shuffle → Forward 반복 루프, 코드 캐싱으로 재생성 최소화 | 계획대로 구현 |
| **대상 로봇** | 명시되지 않음 (범용) | SO-101 (단일 암/양팔), Franka (reference 구현), LeKiwi 지원 | 구체적 로봇 하드웨어에 맞춤 구현 |
| **시뮬레이션** | 시뮬레이션 환경 기반 수집 | **실세계 로봇 기반 수집** (RealSense D435 카메라 + 실물 로봇) | 시뮬레이션 → 실세계로 전환 (가장 큰 차이) |
| **데이터 형식** | Open X-Embodiment 표준 | LeRobot 포맷 (HuggingFace Hub 호환) | 유사한 목적, 다른 구체적 포맷 |
| **양팔 로봇** | 명시되지 않음 | `UnifiedMultiArmPipeline` + `MultiArmSkills` 구현 | 추가 구현 |
| **레코딩 시스템** | 명시되지 않음 | `DatasetRecorder` + `SkillsRecordingWrapper` + `RecordingCallback`: 멀티 카메라 동기화, skill/subtask 라벨링, joint state 실시간 캡처 | 추가 구현 |

### 1.2 Agent 5 (Pre & Post Validator) 비교

| 항목 | 사전제안서 계획 | 실제 구현 | 비고 |
|---|---|---|---|
| **사전 검증 방법** | Neuro-Symbolic Solver (Z3 SMT Solver) 기반 정적 코드 검증 | **미구현** — Z3 SMT Solver 기반 정적 분석은 구현되지 않음 | 가장 큰 차이. LLM 생성 코드의 상징적 검증이 아닌 실행 기반 검증으로 전환 |
| **사전 검증 대안** | LLM + Symbolic Tool | Multi-turn 파이프라인에서 Turn 0~2의 perception 단계가 사전 검증 역할 수행: workspace 제약 조건 확인, 물체 도달 가능성 검증, 좌표 변환 정확성 확인 | 코드 검증 → perception 검증으로 성격 변경 |
| **사후 검증 방법** | VLM 기반 QA 형식 시각 검증 | VLM Judge (gpt-4o, Gemini-2.5-flash, Cosmos-Reason1-7B): Chain-of-Thought 5단계 평가 | 계획과 유사하게 구현 |
| **사후 검증 상세** | 관측 이미지 + 자연어 태스크 설명으로 성공/실패 판정 | 초기 이미지 + 최종 이미지 + 태스크 지시 + 물체 위치 + 실행 코드를 종합 입력, TRUE/FALSE/UNCERTAIN 3단계 판정 | 입력 정보가 더 풍부 |
| **VLM Judge 모델** | GPT-5 + Grounding SAM | gpt-4o / Gemini-2.5-flash / Cosmos-Reason1-7B (온프레미스) | GPT-5 미출시로 대체, Grounding SAM은 detection에 활용 |
| **피드백 루프** | 반복적 피드백 기반 코드 수정 루프 | Judge 결과 → 다음 에피소드에 반영, 실패 시 코드 재생성 트리거 | 자동 코드 수정이 아닌 재생성 방식 |
| **검증 기준** | ROC curve 기반 최적 threshold | VLM 자체 판단 (프롬프트 기반), 별도 threshold 학습 없음 | 학습 기반 threshold → 프롬프트 엔지니어링 |

### 1.3 핵심 차이 요약

**Agent 3 — 계획보다 확장된 부분:**
- 멀티턴 VLM 파이프라인 (4단계 인식-생성 분리)
- 다중 LLM 프로바이더 지원 (OpenAI, Gemini, DeepSeek, 로컬 vLLM)
- 양팔 로봇 지원 (`UnifiedMultiArmPipeline`)
- 레코딩 시스템 (skill/subtask 라벨링, 멀티 카메라 동기화)
- 실세계 로봇에서 직접 데이터 수집

**Agent 3 — 계획과 다른 부분:**
- 시뮬레이션 기반 → 실세계 기반으로 수집 환경 변경
- Claude 단일 모델 → 멀티 프로바이더로 변경

**Agent 5 — 구현되지 않은 부분:**
- Z3 SMT Solver 기반 Neuro-Symbolic 사전 검증
- ROC curve 기반 threshold 학습

**Agent 5 — 대안으로 구현된 부분:**
- 멀티턴 perception 파이프라인이 사전 검증 역할 (workspace 검증, 도달 가능성 등)
- VLM Chain-of-Thought 5단계 Judge가 사후 검증 (계획과 유사)

---

## 2. 현재 AutoDataCollection 구현 내용 상세

### 2.1 전체 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    사용자 입력                                │
│  instruction: "빨간 컵을 파란 상자에 넣어라"                    │
│  objects: ["red cup", "blue box"]                            │
│  --num-episodes 5 --record                                   │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│             ForwardAndResetPipeline (Orchestrator)            │
│             execution_forward_and_reset.py                    │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Episode Loop (N회 반복)                              │   │
│  │                                                       │   │
│  │  ┌─── Forward Phase ───┐   ┌─── Reset Phase ────┐   │   │
│  │  │ 1. 초기 이미지 촬영   │   │ 1. 리셋 코드 생성   │   │   │
│  │  │ 2. 물체 감지         │   │ 2. 리셋 실행        │   │   │
│  │  │ 3. 코드 생성 (LLM)   │   │ 3. 리셋 레코딩      │   │   │
│  │  │ 4. 코드 실행 (로봇)   │   │ 4. 랜덤 셔플 (옵션) │   │   │
│  │  │ 5. 최종 이미지 촬영   │   └────────────────────┘   │   │
│  │  │ 6. Judge (VLM 검증)  │                             │   │
│  │  │ 7. Forward 레코딩    │                             │   │
│  │  └─────────────────────┘                              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 코드 생성 파이프라인 (Agent 3 핵심)

#### 2.2.1 Multi-Turn VLM 파이프라인

코드 생성은 4단계 멀티턴 대화로 진행된다.

**Turn 0 — Scene Understanding (장면 이해)**
- 입력: 카메라 이미지 (top-view)
- VLM이 작업 공간의 물체, 공간 관계, 레이아웃 분석
- 코드 생성 금지 (인식 전용)
- 파일: `code_gen_lerobot/forward_execution/turn0_prompt.py`

**Turn 1 — Object BBox Detection (물체 탐지)**
- 입력: 이미지 + Turn 0 컨텍스트
- VLM이 각 물체의 bounding box 좌표 출력 (0-1000 정규화)
- 출력 형식: `[{"label": "red cup", "box_2d": [ymin, xmin, ymax, xmax]}]`
- 파일: `code_gen_lerobot/forward_execution/turn1_prompt.py`

**Turn 2 — Critical Point Detection (핵심 포인트 탐지)**
- 입력: 물체별 crop 이미지 + 컨텍스트
- 각 물체에 대해 `grasp` (잡는 위치)와 `interaction` (기능적 부분) 포인트 탐지
- crop-then-point 방식으로 정확도 향상
- 파일: `code_gen_lerobot/forward_execution/turn2_prompt.py`

**Turn 3 — Code Generation (코드 생성)**
- 입력: 축적된 컨텍스트 + 물체 3D 좌표 + 태스크 지시
- LLM이 `LeRobotSkills` API를 사용하는 Python 코드 생성
- 각 skill 호출에 `skill_description`, `verification_question` 포함 (레코딩용)
- 파일: `code_gen_lerobot/forward_execution/user_prompt.py`

#### 2.2.2 사용 모델 및 라우팅

**LLM 라우터** (`code_gen_lerobot/llm.py`):

| 프로바이더 | 모델 | 용도 |
|---|---|---|
| OpenAI | gpt-4o, gpt-4o-mini, o1, o3 | 코드 생성 |
| Google | gemini-3-flash-preview, gemini-2.0-flash | 멀티턴 VLM (이미지 입력 지원) |
| DeepSeek | deepseek-chat, deepseek-reasoner | 코드 생성 (저비용) |
| 로컬 vLLM | Qwen2.5-Coder-7B-Instruct | 온프레미스 코드 생성 |
| 로컬 vLLM | Cosmos-Reason1-7B | 온프레미스 VLM 추론 |

현재 실제 사용 설정 (`pipeline_config/paid_api_config.yaml`):
- 코드 생성: `gemini-3-flash-preview`
- Judge VLM: `gemini-2.5-flash`

#### 2.2.3 물체 감지 (Perception)
**VLM Crop-then-Point** (멀티턴 모드)
- Turn 1~2에서 VLM이 직접 물체 위치와 critical point 탐지
- BBox → crop → point detection 순서로 정확도 향상
- 이미지 좌표 0-1000 정규화 → pixel → depth → 로봇 3D 좌표

**좌표 변환** (`object_detection/localization/coordinate_transform.py`):
```
pixel (cx, cy) + depth Z → 카메라 3D → 로봇 world frame [x, y, z] (meters)
```

#### 2.2.4 로봇 제어 API (Skills)

`skills/skills_lerobot.py`의 `LeRobotSkills` 클래스:

| 카테고리 | Skill | 파라미터 | 설명 |
|---|---|---|---|
| **이동** | `move_to_position(pos, duration, target_name)` | [x,y,z] meters | Cartesian 이동 (5-DOF IK) |
| | `move_to_initial_state()` | - | 홈 포지션 |
| | `move_to_free_state()` | - | 안전 대기 위치 |
| **그리퍼** | `gripper_open(duration, ratio)` | ratio: 0.0~1.0 | 그리퍼 열기 |
| | `gripper_close(duration)` | - | 그리퍼 닫기 |
| **조합 스킬** | `execute_pick_object(pos, object_name)` | 물체 위치 | approach → descend → grasp |
| | `execute_place_object(pos, is_table, ratio, target_name)` | 목표 위치 | descend → release |
| | `execute_press(pos, depth, height, hold_time)` | 위치 + 파라미터 | 누르기 (2단계: 접촉 + 토크 제한) |
| | `execute_push(start, end, height, object_name)` | 시작/끝 위치 | 밀기 |
| **회전** | `rotate_90degree(direction, duration)` | 1=CW, -1=CCW | 손목 90도 회전 |
| **감지** | `detect_objects(object_names)` | 물체 이름 리스트 | 실시간 물체 재감지 |
| **서브태스크** | `set_subtask(description)` | 설명 문자열 | 레코딩 라벨 시작 |
| | `clear_subtask()` | - | 레코딩 라벨 종료 |

**추가 primitive skills** (`skills/` 디렉토리):
- `insert.py`, `wipe.py`, `strike.py`, `push_object.py`, `pull.py`, `stir.py`, `shake.py`, `rotate_wrist.py`

**양팔 로봇 API** (`skills/multi_arm_skills.py`):
- `MultiArmSkills` 클래스: 좌/우 암 독립 및 동기화 제어
- `move_to_position(left_arm=, right_arm=)`: "wait" 파라미터로 한 팔 대기 가능
- `pick_object(left_arm=, right_arm=)`: 양팔 동시 pick
- `bimanual_move(left_arm, right_arm)`: 동기화 이동

#### 2.2.5 Reset 실행 파이프라인

Forward 실행 후 환경 복원을 위한 별도 파이프라인:

**Reset 모드:**
- `"original"`: 물체를 초기 위치로 복원
- `"random"`: 물체를 workspace 내 랜덤 위치로 셔플

**Reset 코드 생성 프로세스:**
1. Forward `ExecutionContext` 로드 (초기 물체 위치, 실행 코드 등)
2. 현재 상태 vs 초기 상태 비교 (VLM 분석)
3. 이동할 물체 순서 결정 (스택된 경우 위부터)
4. 각 물체에 대해 pick-place 코드 생성 (중간 재감지 포함)

**핵심 패턴:**
```python
# 첫 번째 물체: 현재 감지 결과 사용
set_subtask("move object_A to target")
execute_pick_object(positions["object_A"])
execute_place_object(target_A)

# 이후 물체: 재감지 필수 (이전 이동으로 위치 변경됨)
move_to_initial_state()  # 시야에서 팔 치우기
positions = detect_objects(["object_B", "object_C"])
set_subtask("move object_B to target")
execute_pick_object(positions["object_B"])
execute_place_object(target_B)
```

#### 2.2.6 레코딩 시스템

**구성 요소:**
- `DatasetRecorder` (`record_dataset/recorder.py`): LeRobot 포맷 에피소드 기록
- `SkillsRecordingWrapper` (`record_dataset/skills_wrapper.py`): skill 호출 자동 캡처
- `RecordingCallback` (`record_dataset/callback.py`): 실시간 joint state 캡처
- `MultiArmRecorder` (`record_dataset/multi_arm_recorder.py`): 양팔 12축 기록

**기록 데이터:**
- 관측: RGB 이미지 (top, left_wrist, right_wrist), joint state (radian), ee_pos, gripper state
- 액션: 목표 joint 각도, gripper 명령
- 메타데이터: 자연어 지시, skill description, verification question, subtask label, 에피소드 성공 여부

**카메라 설정** (`pipeline_config/recording_config.yaml`):
- shared/top: RealSense D435 (640x480, 30fps)
- left_arm/wrist: OpenCV USB 카메라 (/dev/video8)
- right_arm/wrist: OpenCV USB 카메라 (/dev/video6)

**데이터 형식:** LeRobot HuggingFace Hub 호환 (30 FPS 기록)

### 2.3 Judge 시스템 (Agent 5 핵심 — 사후 검증)

#### 2.3.1 TaskJudge 클래스

파일: `judge/forward_execution/judge.py`

**입력:**
```python
result = judge.judge(
    instruction="pick red cup and place in blue box",
    initial_image=np.ndarray,      # 실행 전 이미지
    final_image=np.ndarray,        # 실행 후 이미지
    object_positions={"red cup": [x,y,z], ...},
    executed_code="skills.execute_pick_object(...)",
    image_resolution=(640, 480)
)
```

**Chain-of-Thought 5단계 평가:**

| 단계 | 내용 |
|---|---|
| Step 1 | 초기 이미지에서 태스크 관련 물체 식별 및 위치 확인 |
| Step 2 | 최종 이미지에서 동일 물체 식별 및 새 위치 확인 |
| Step 3 | 초기 → 최종 변화 분석 (무엇이 이동했는가) |
| Step 4 | 태스크 목표 달성 여부 확인 (지시와 최종 상태 비교) |
| Step 5 | 최종 판정: TRUE / FALSE / UNCERTAIN |

**출력:**
```python
{
    'prediction': 'TRUE' | 'FALSE' | 'UNCERTAIN',
    'reasoning': '5단계 추론 결과',
    'success': True | False,
    'error': None | 'error message'
}
```

**사용 모델:**
- 클라우드: gpt-4o, Gemini-2.5-flash
- 온프레미스: Cosmos-Reason1-7B, Qwen2-VL-2B-Instruct (vLLM 서버)

#### 2.3.2 사전 검증 현황

Z3 SMT Solver 기반의 정적 코드 검증은 **구현되지 않았다.**

대신, 다음 메커니즘이 사전 검증 역할을 부분적으로 수행:

1. **Workspace 제약 조건 검증** (`code_gen_lerobot/forward_execution/workspace.py`):
   - 물체가 로봇 도달 가능 영역 내에 있는지 확인
   - cyan arc (최대 도달 범위), green rectangle (안전 영역) 기반

2. **Multi-turn Perception 검증** (Turn 0~2):
   - VLM이 장면을 이해하고 물체 위치를 검증
   - Turn 2에서 grasp point와 interaction point의 물리적 타당성 확인

3. **System Prompt 내 제약 조건**:
   - 그리퍼 최대 개방 0.07m, 접근 높이 0.20m 등 하드코딩된 물리 제약
   - LLM이 코드 생성 시 이 제약을 참조

### 2.4 양팔 로봇 지원

**진입점** (`execution_forward_and_reset.py`):
```python
if len(robot_ids) == 1:
    pipeline = ForwardAndResetPipeline(...)   # 단일 암
else:
    pipeline = UnifiedMultiArmPipeline(...)   # 양팔
```

**양팔 전용 기능:**
- 좌/우 암 독립 workspace 정의 (이미지 좌반: 좌팔, 우반: 우팔, 중앙: 양팔)
- 동기화 이동 (`bimanual_move`)
- 12축 레코딩 (6 DOF × 2 + gripper × 2)
- 양팔 전용 system/user prompt (`code_gen_lerobot/multi_arm/`)

### 2.5 코드 캐싱 전략

동일 태스크의 반복 에피소드에서 코드 재생성을 최소화:

```python
# 물체 키가 동일하면 캐시된 코드 재사용
if all(key in new_positions for key in cached_forward_keys):
    code = cached_forward_code  # 재생성 없음
else:
    code = lerobot_code_gen(...)  # 새로 생성
    cached_forward_code = code
```

### 2.6 실행 컨텍스트 관리

`ExecutionContext` (`code_gen_lerobot/execution_context.py`):

Forward → Reset 간 상태 전달:
```python
@dataclass
class ExecutionContext:
    instruction: str           # 태스크 지시
    object_positions: Dict     # 물체 3D 좌표
    generated_spec: Dict       # LLM 생성 실행 계획
    generated_code: str        # 생성된 Python 코드
    execution_success: bool    # 실행 성공 여부
    timestamp: str             # 실행 시각
    robot_id: int              # 로봇 ID
    metadata: Dict             # 추가 메타데이터
```

---

## 3. 최종보고서 작성을 위한 추가 디테일

> 아래 내용은 `PRISM_Track2_최종보고서_RAPIDS.md`의 각 항목에 맞춰 작성되었다.

### 3.1 Agent 3 — 최종보고서 항목별 내용

#### (2) 사용 모델 및 도구

| 모델/도구 | 용도 |
|---|---|
| Gemini-3-flash-preview | 멀티턴 VLM 파이프라인 (Turn 0~3), 코드 생성 |
| GPT-4o / GPT-4o-mini | 코드 생성 (대안) |
| DeepSeek-chat | 코드 생성 (저비용 대안) |
| Qwen2.5-Coder-7B-Instruct | 온프레미스 코드 생성 (vLLM) |
| Intel RealSense D435 | RGB-D 카메라 (depth 기반 3D 좌표 변환) |
| LeRobot | 데이터셋 포맷 및 로봇 제어 프레임워크 |

#### (3) 구현 내용 — 보고서 서술 참고

> Agent 3는 자연어 태스크 지시를 입력받아 로봇 제어 코드를 자동 생성하고 실행하여 demonstration 데이터를 수집하는 시스템이다.

**코드 생성 파이프라인:**
LLM 기반 코드 생성은 4단계 멀티턴 VLM 파이프라인으로 구현하였다. Turn 0에서 VLM이 작업 공간의 장면을 분석하고, Turn 1에서 물체별 bounding box를 탐지하며, Turn 2에서 crop-then-point 방식으로 각 물체의 grasp point와 interaction point를 정밀 탐지한다. Turn 3에서 축적된 컨텍스트와 3D 물체 좌표를 기반으로 LLM이 LeRobotSkills API를 사용하는 실행 가능한 Python 코드를 생성한다.

**로봇 제어 API:**
`LeRobotSkills` 클래스로 17종 이상의 primitive skill을 제공한다. `move_to_position`, `execute_pick_object`, `execute_place_object` 등 조합 스킬과 `gripper_open`, `rotate_90degree` 등 기본 스킬로 구성된다. 각 스킬은 5-DOF 역기구학(IK) 기반으로 Cartesian 좌표를 관절 각도로 변환하여 실행한다.

**자가 수정 및 반복 수집:**
에피소드 실행 후 VLM Judge가 성공/실패를 판정하고, 실패 시 다음 에피소드에서 코드 재생성을 트리거한다. 성공한 코드는 캐싱하여 동일 물체 구성에서 재사용함으로써 반복 수집 효율을 높인다.

**환경 복원:**
Forward 실행 후 Reset Execution 파이프라인이 자동으로 물체를 원래 위치로 복원하거나 새로운 랜덤 위치로 셔플한다. Forward → Judge → Reset → Shuffle 루프를 반복하여 다양한 초기 조건의 demonstration 데이터를 무중단 수집한다.

**레코딩:**
`SkillsRecordingWrapper`가 각 skill 호출을 자동 캡처하여 LeRobot 포맷으로 기록한다. 멀티 카메라 (top-view + wrist) 동기화, skill/subtask 라벨링, 실시간 joint state 캡처를 지원한다.

#### (4) 핵심 기술 및 알고리즘 — 보고서 서술 참고

1. **Multi-turn VLM Code-as-Policies**: 인식과 코드 생성을 4단계로 분리하여, VLM의 장면 이해를 단계적으로 축적한 후 코드를 생성. 단일턴 대비 물체 위치 정확도와 코드 실행 성공률 향상.

2. **Crop-then-Point Detection**: 전체 이미지에서 BBox 탐지 → 물체별 crop → 정밀 point 탐지의 계층적 인식. grasp point와 interaction point를 분리하여 복잡한 조작 태스크 지원.

3. **코드 캐싱 기반 반복 수집**: 물체 키 기반 캐시로 동일 태스크 반복 시 코드 재생성 비용 제거. 랜덤 셔플 후에도 물체 구성이 동일하면 캐시 활용.

4. **Forward-Reset 자동 루프**: ExecutionContext로 Forward/Reset 간 상태를 전달하고, original/random reset 모드로 다양한 초기 조건 생성. 무인 연속 데이터 수집 실현.

5. **Skill-level Recording**: 각 primitive skill 호출에 자연어 description, verification question, subtask label을 자동 부여. VLA 학습 시 skill-conditioned policy 학습 가능.

### 3.2 Agent 5 — 최종보고서 항목별 내용

#### (2) 사용 모델 및 도구

| 구분 | 모델/도구 | 용도 |
|---|---|---|
| 사후 검증 | Gemini-2.5-flash | VLM 기반 태스크 성공/실패 판정 (주 사용) |
| 사후 검증 | gpt-4o | VLM 기반 태스크 판정 (대안) |
| 사후 검증 | Cosmos-Reason1-7B | 온프레미스 VLM 판정 (vLLM 서버) |
| 사전 검증 | Multi-turn VLM Pipeline | Workspace 제약 조건, 도달 가능성 검증 |

#### (3) 구현 내용 — 보고서 서술 참고

> Agent 5는 데이터 수집 품질을 보장하기 위한 검증 체계이다.

**사후 검증 (VLM Judge):**
태스크 실행 전후 이미지를 VLM에 입력하고, Chain-of-Thought 5단계 프로토콜로 성공/실패를 판정한다. Step 1~2에서 초기/최종 이미지의 물체를 각각 식별하고, Step 3에서 변화를 분석하며, Step 4에서 태스크 목표 달성 여부를 검증하고, Step 5에서 TRUE/FALSE/UNCERTAIN 중 하나를 출력한다. 태스크 지시, 물체 위치, 실행 코드를 추가 컨텍스트로 제공하여 판정 정확도를 높인다.

**사전 검증:**
Z3 SMT Solver 기반의 정적 코드 검증 대신, 멀티턴 VLM 파이프라인의 perception 단계(Turn 0~2)가 사전 검증 역할을 수행한다. Workspace 제약 조건(도달 가능 영역, 테이블 경계), 물체 감지 신뢰도, grasp point의 물리적 타당성을 실행 전에 검증한다. System prompt에 하드코딩된 물리 제약(그리퍼 크기, 접근 높이 등)이 LLM의 코드 생성 시 안전 제약으로 작용한다.

**검증 결과 활용:**
Judge 결과는 에피소드 메타데이터로 기록되어 데이터셋 필터링에 활용된다. 실패 판정 시 다음 에피소드에서 코드 캐시를 무효화하고 재생성을 트리거한다.

#### (4) 핵심 기술 및 알고리즘 — 보고서 서술 참고

1. **VLM Chain-of-Thought Judge**: 5단계 구조화된 추론으로 VLM 판정의 일관성과 해석 가능성 확보. 단순 yes/no가 아닌 단계별 reasoning을 통해 오판 원인 추적 가능.

2. **Multi-modal 입력 통합**: 초기/최종 이미지 + 자연어 지시 + 물체 좌표 + 실행 코드를 종합 입력하여, 시각 정보만으로는 판단 어려운 미세한 변화도 감지.

3. **Perception 기반 사전 검증**: 코드 실행 전 workspace 제약, 물체 도달 가능성, grasp point 타당성을 멀티턴 VLM 파이프라인에서 사전 확인. 물리적으로 불가능한 실행 계획을 사전 차단.

### 3.3 Agent 3 성과 — 보고서 작성 시 필요한 정량 지표

> ⚠️ 아래는 측정이 필요한 항목 목록이다. 실제 값은 실험 후 기입해야 한다.

| 지표 | 정의 | 측정 방법 | 현재 상태 |
|---|---|---|---|
| **Code Runnability** | 생성 코드의 오류 없이 실행 가능한 비율 | 테스트 태스크 N개 대비 실행 성공 수 / N | **측정 필요** |
| **코드 생성 성공률 (태스크별)** | 태스크 유형별 (pick-place, stack, press 등) 실행 가능 비율 | 태스크별 분류 후 측정 | **측정 필요** |
| **자가 수정 성공률** | 첫 실행 실패 후 재생성으로 성공한 비율 | 재생성 후 Judge TRUE 비율 | **측정 필요** |
| **에피소드 수집 수** | 총 수집된 에피소드 수 | 데이터셋 에피소드 카운트 | **측정 필요** |
| **시간당 에피소드 수** | 시간당 수집 처리량 | 총 에피소드 / 총 수집 시간 | **측정 필요** |
| **코드 생성 시간** | LLM 코드 생성 평균 소요 시간 | Turn 0~3 총 latency 측정 | **측정 필요** |
| **코드 캐시 히트율** | 캐시 활용 비율 | 캐시 사용 에피소드 / 총 에피소드 | **측정 필요** |

### 3.4 Agent 5 성과 — 보고서 작성 시 필요한 정량 지표

| 지표 | 정의 | 측정 방법 | 현재 상태 |
|---|---|---|---|
| **Judge 정확도** | VLM 판정과 사람 판정의 일치율 | 사람 라벨링 대비 Judge 출력 비교 | **측정 필요** |
| **False Positive율** | 실패를 성공으로 오판한 비율 | Judge TRUE & 사람 FALSE / 사람 FALSE | **측정 필요** |
| **False Negative율** | 성공을 실패로 오판한 비율 | Judge FALSE & 사람 TRUE / 사람 TRUE | **측정 필요** |
| **UNCERTAIN 비율** | 판단 불가 비율 | UNCERTAIN 수 / 총 판정 수 | **측정 필요** |
| **Judge 응답 시간** | VLM 판정 평균 소요 시간 | API 호출 latency 측정 | **측정 필요** |

### 3.5 기술적 어려움 대응 — 보고서 6장 참고

#### Agent 3 관련

| 어려움 | 대응 내용 |
|---|---|
| **LLM 코드 생성 실패** | 멀티턴 파이프라인으로 인식/생성 분리, system prompt에 skill 사용 패턴 예시 포함, 코드 스켈레톤 제공 |
| **물체 위치 오차** | Crop-then-point 2단계 탐지로 정확도 향상, depth 기반 3D 변환으로 2D→3D 정확도 개선 |
| **반복 수집 효율** | 코드 캐싱 전략, Forward-Reset 자동 루프, 랜덤 셔플로 다양성 확보 |
| **양팔 로봇 지원** | UnifiedMultiArmPipeline + MultiArmSkills로 독립 제어/동기화 제어 모두 지원 |

#### Agent 5 관련

| 어려움 | 대응 내용 |
|---|---|
| **Z3 SMT Solver 적용 어려움** | 로봇 제어 코드의 동적 특성상 정적 분석이 어려워, perception 기반 사전 검증으로 전환 |
| **VLM 판정 일관성** | Chain-of-Thought 5단계 프로토콜로 구조화, 초기/최종 이미지 비교 방식으로 일관성 확보 |
| **미세한 변화 감지** | 물체 좌표 + 실행 코드를 VLM에 추가 컨텍스트로 제공하여 시각 정보 보완 |

### 3.6 파일 경로 참조 (보고서 작성 시 코드 인용용)

| 구성 요소 | 핵심 파일 |
|---|---|
| 전체 오케스트레이터 | `execution_forward_and_reset.py` |
| LLM 라우터 | `code_gen_lerobot/llm.py` |
| Forward 코드 생성 | `code_gen_lerobot/forward_execution/code_gen.py` |
| Forward 프롬프트 (Turn 3) | `code_gen_lerobot/forward_execution/user_prompt.py` |
| System Prompt | `code_gen_lerobot/forward_execution/system_prompt.py` |
| Reset 코드 생성 | `code_gen_lerobot/reset_execution/code_gen.py` |
| Reset 프롬프트 | `code_gen_lerobot/reset_execution/prompt.py` |
| Skill API | `skills/skills_lerobot.py` |
| 양팔 Skill API | `skills/multi_arm_skills.py` |
| Judge (사후 검증) | `judge/forward_execution/judge.py` |
| Judge 프롬프트 | `judge/forward_execution/prompt.py` |
| 좌표 변환 | `object_detection/localization/coordinate_transform.py` |
| 레코딩 | `record_dataset/recorder.py` |
| Skill 레코딩 래퍼 | `record_dataset/skills_wrapper.py` |
| 실행 컨텍스트 | `code_gen_lerobot/execution_context.py` |
| 파이프라인 설정 | `pipeline_config/paid_api_config.yaml` |
| 레코딩 설정 | `pipeline_config/recording_config.yaml` |
| 양팔 파이프라인 | `unified_multi_arm.py` |
