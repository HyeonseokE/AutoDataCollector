# TECHSPEC: LeRobot Dataset Recording Pipeline

## Overview

이 문서는 `run_forward_and_reset.sh` 파이프라인의 **forward 과정**에서 발생하는 로봇 움직임을 **LeRobot Dataset v3.0 포맷**으로 레코딩하기 위한 기술 명세서입니다.

---

## 1. 구현할 것 (In Scope)

### 1.1 Core Components (`./record_dataset/`)

#### 1.1.1 `DatasetRecorder` 클래스
- **위치**: `record_dataset/recorder.py`
- **역할**: LeRobotDataset API를 래핑하여 레코딩 관리
- **기능**:
  - 데이터셋 생성 (`LeRobotDataset.create()`)
  - 프레임 추가 (`add_frame()`)
  - 에피소드 저장 (`save_episode()`)
  - 데이터셋 완료 (`finalize()`)

```python
class DatasetRecorder:
    def __init__(self, repo_id: str, fps: int, robot_type: str = "so101")
    def start_episode(self, task: str)
    def record_frame(self, observation: dict, action: np.ndarray, image: np.ndarray)
    def end_episode(self)
    def finalize(self)
```

#### 1.1.2 `RecordingCallback` 클래스
- **위치**: `record_dataset/callback.py`
- **역할**: `LeRobotSkills`의 제어 루프에 삽입될 콜백
- **기능**:
  - 매 제어 스텝에서 호출
  - 현재 상태(state), 액션(action), 이미지(image) 캡처
  - FPS 동기화 (50Hz 제어 → 30Hz 레코딩)

```python
class RecordingCallback:
    def __init__(self, recorder: DatasetRecorder, camera, target_fps: int = 30)
    def on_step(self, current_state: np.ndarray, action: np.ndarray)
    def should_record(self) -> bool  # FPS 동기화
```

#### 1.1.3 `SkillsRecordingWrapper` 클래스
- **위치**: `record_dataset/skills_wrapper.py`
- **역할**: `LeRobotSkills`를 래핑하여 레코딩 기능 추가
- **기능**:
  - 기존 Skills API 유지
  - 내부 제어 루프에 RecordingCallback 주입
  - 에피소드 자동 관리

```python
class SkillsRecordingWrapper:
    def __init__(self, skills: LeRobotSkills, recorder: DatasetRecorder, camera)
    def move_to_position(self, ...) -> bool  # 레코딩 포함
    def gripper_open(self, ...)
    def gripper_close(self, ...)
    # ... 기타 스킬 메서드들
```

### 1.2 데이터셋 스키마 (Features)

```python
DATASET_FEATURES = {
    # 로봇 상태 (6 joints: 5 arm + 1 gripper)
    "observation.state": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]
    },

    # 액션 (다음 목표 위치)
    "action": {
        "dtype": "float32",
        "shape": (6,),
        "names": ["shoulder_pan", "shoulder_lift", "elbow_flex",
                  "wrist_flex", "wrist_roll", "gripper"]
    },

    # 카메라 이미지
    "observation.images.front": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"]
    }
}
```

### 1.3 파이프라인 통합

#### 1.3.1 `execution_forward_and_reset.py` 수정
- `--record` 플래그 추가
- `--dataset-repo-id` 인자 추가
- Forward 실행 시 레코딩 활성화

#### 1.3.2 `run_forward_and_reset.sh` 수정
- `RECORD_DATASET` 옵션 추가
- `DATASET_REPO_ID` 옵션 추가

### 1.4 데이터 포맷 변환

| 현재 시스템 | LeRobot 표준 | 변환 방법 |
|------------|-------------|----------|
| normalized (-100 to +100) | normalized (-100 to +100) | 그대로 사용 |
| 50Hz 제어 루프 | 30Hz 레코딩 | 프레임 스킵 (매 1.67 스텝마다 레코딩) |
| RGB numpy array | RGB numpy array | 그대로 사용 |

### 1.5 저장 구조

```
~/.cache/huggingface/lerobot/{repo_id}/
├── meta/
│   ├── info.json          # 데이터셋 메타정보
│   ├── stats.json         # 정규화 통계
│   ├── tasks.parquet      # 태스크 목록
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── observation.images.front/
        └── chunk-000/
            └── file-000.mp4
```

---

## 2. 구현하지 않을 것 (Out of Scope)

### 2.1 lerobot 프레임워크 전체 편입
- **이유**: 현재 CaP(Code as Policies) 파이프라인의 강점 유지
- **대안**: `LeRobotDataset` API만 독립적으로 사용

### 2.2 Teleoperation 기반 레코딩
- **이유**: 현재 시스템은 LLM이 생성한 코드로 자율 실행
- **대안**: 자율 실행 중 데이터 캡처

### 2.3 lerobot의 Robot 인터페이스 구현
- **이유**: 기존 `FeetechController` + `LeRobotSkills` 유지
- **대안**: 데이터 포맷만 호환

### 2.4 Reset 과정 레코딩
- **이유**: Forward(태스크 수행) 데이터만 학습에 유의미
- **대안**: Forward 구간만 레코딩

### 2.5 실시간 Hub 업로드
- **이유**: 네트워크 지연으로 인한 제어 불안정
- **대안**: 세션 종료 후 일괄 업로드 (`push_to_hub`)

### 2.6 Detection/Judge 단계 데이터 포함
- **이유**: 로봇 움직임(state/action) 데이터에만 집중
- **대안**: Forward 실행 구간만 레코딩

### 2.7 기존 `skills_lerobot.py` 직접 수정
- **이유**: 기존 코드 안정성 유지
- **대안**: Wrapper 패턴으로 기능 추가

---

## 3. 이 구현을 통해 이루고자 하는 목적 (Goals)

### 3.1 주요 목적

#### 3.1.1 Policy Distillation을 위한 데이터셋 구축
- **설명**: LLM이 생성한 코드로 수행된 로봇 동작을 데이터셋으로 변환
- **활용**: ACT, Diffusion Policy 등의 모방학습 모델 학습
- **기대효과**: LLM 없이도 학습된 정책으로 태스크 수행 가능

#### 3.1.2 LeRobot 생태계 호환
- **설명**: HuggingFace LeRobot Dataset v3.0 포맷 준수
- **활용**: lerobot 학습 파이프라인 직접 사용 가능
- **기대효과**: 커뮤니티 모델/도구와의 호환성

#### 3.1.3 CaP 파이프라인 강점 유지
- **설명**: Detection → Code Gen → Execution → Judge 파이프라인 보존
- **활용**: LLM 기반 제로샷 태스크 수행 + 데이터 수집 동시 진행
- **기대효과**: 별도 텔레옵 없이 자동화된 데이터 수집

### 3.2 세부 목적

#### 3.2.1 최소 침습적 통합 (Minimal Invasive Integration)
- 기존 코드 수정 최소화
- Wrapper/Callback 패턴으로 기능 추가
- 레코딩 비활성화 시 기존 동작과 동일

#### 3.2.2 유연한 데이터 수집
- 단일 에피소드 또는 다중 에피소드 레코딩 지원
- 태스크별 데이터셋 분리 가능
- 세션 중 실패 에피소드 제외 가능

#### 3.2.3 데이터 품질 보장
- FPS 동기화로 일관된 시간 간격
- 정규화 통계 자동 계산
- 비디오 인코딩으로 저장 공간 효율화

---

## 4. 이루지 않고자 하는 목적 (Non-Goals)

### 4.1 실시간 학습 (Online Learning)
- **설명**: 레코딩 중 실시간으로 정책 업데이트하지 않음
- **이유**: 복잡도 증가, 안정성 저하
- **대안**: 오프라인 학습 후 정책 배포

### 4.2 모든 센서 데이터 수집
- **설명**: Depth 이미지, Force/Torque 센서 등 미수집
- **이유**: 현재 하드웨어에서 불필요
- **대안**: RGB 이미지 + Joint State만 수집

### 4.3 Failure Recovery 데이터 수집
- **설명**: 실패 후 복구 과정 데이터 미수집
- **이유**: Forward 성공 데이터에 집중
- **대안**: Judge=TRUE인 에피소드만 저장 옵션

### 4.4 Multi-robot 동시 레코딩
- **설명**: 여러 로봇 동시 제어/레코딩 미지원
- **이유**: 현재 단일 로봇 설정
- **대안**: 단일 로봇 레코딩 후 데이터셋 병합

### 4.5 완전 자동화된 데이터 증강
- **설명**: 레코딩 시점에 이미지 변환 미적용
- **이유**: 원본 데이터 보존 우선
- **대안**: 학습 시점에 ImageTransforms 적용

### 4.6 기존 결과 폴더 구조 변경
- **설명**: `./results/` 폴더의 기존 저장 방식 유지
- **이유**: 기존 분석 도구 호환성
- **대안**: LeRobot 데이터셋은 별도 경로에 저장

---

## 5. 구현 세부사항

### 5.1 의존성

```python
# ./record_dataset/requirements.txt
lerobot>=0.4.0  # 또는 main branch
torch>=2.0.0
numpy>=1.24.0
pillow>=10.0.0
pyarrow>=14.0.0
```

### 5.2 파일 구조

```
./record_dataset/
├── __init__.py
├── recorder.py          # DatasetRecorder 클래스
├── callback.py          # RecordingCallback 클래스
├── skills_wrapper.py    # SkillsRecordingWrapper 클래스
├── config.py            # 설정 상수 (FPS, features 등)
├── utils.py             # 유틸리티 함수
└── requirements.txt
```

### 5.3 API 사용 흐름

```python
# 1. 레코더 초기화
recorder = DatasetRecorder(
    repo_id="user/cap_pick_and_place",
    fps=30,
    robot_type="so101"
)

# 2. 카메라 초기화
camera = RealSenseD435(width=640, height=480, fps=30)
camera.start()

# 3. Skills 래퍼 생성
skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml")
skills.connect()
recording_skills = SkillsRecordingWrapper(skills, recorder, camera)

# 4. 에피소드 시작
recorder.start_episode(task="pick up the yellow dice and place it on the blue dish")

# 5. 태스크 실행 (내부에서 자동 레코딩)
recording_skills.move_to_initial_state()
recording_skills.rotate_90degree()
recording_skills.move_to_position(pick_pos)
recording_skills.gripper_close()
# ... 나머지 동작

# 6. 에피소드 종료
recorder.end_episode()

# 7. 데이터셋 완료 (필수!)
recorder.finalize()

# 8. (선택) Hub에 업로드
recorder.push_to_hub()
```

### 5.4 제어 루프 레코딩 상세

```python
# skills_wrapper.py 내부 구현 개념
def _execute_planned_trajectory_with_recording(self, ...):
    frame_counter = 0
    control_hz = 50
    record_hz = 30
    record_interval = control_hz / record_hz  # 1.67

    while not self._stop_requested:
        # 현재 상태 읽기
        actual_norm, actual_rad, current_ee = self._get_current_state()

        # 액션 계산
        action = trajectory.get_state_at_time(elapsed)
        action_normalized = self._radians_to_normalized(action)

        # 레코딩 (FPS 동기화)
        if frame_counter % record_interval < 1:
            image = self.camera.get_frames()[0]  # RGB
            self.recorder.record_frame(
                observation=actual_norm,
                action=action_normalized,
                image=image
            )

        # 로봇에 명령 전송
        self.robot.write_positions(...)

        frame_counter += 1
        time.sleep(0.02)  # 50Hz
```

---

## 6. 테스트 계획

### 6.1 단위 테스트
- [ ] `DatasetRecorder` 생성/종료
- [ ] 프레임 추가 및 에피소드 저장
- [ ] FPS 동기화 로직
- [ ] 데이터 포맷 변환

### 6.2 통합 테스트
- [ ] 단일 에피소드 레코딩 후 로드
- [ ] 다중 에피소드 연속 레코딩
- [ ] `run_forward_and_reset.sh` 레코딩 모드 실행
- [ ] LeRobot 학습 파이프라인에서 데이터셋 로드

### 6.3 검증 항목
- [ ] 생성된 데이터셋이 `LeRobotDataset(repo_id)`로 로드 가능
- [ ] `observation.state` shape = (6,)
- [ ] `action` shape = (6,)
- [ ] 비디오 프레임 FPS = 30
- [ ] `meta/stats.json` 정규화 통계 포함

---

## 7. 마이그레이션 가이드

### 7.1 기존 파이프라인 (레코딩 없이)
```bash
./run_forward_and_reset.sh
```

### 7.2 레코딩 활성화
```bash
# run_forward_and_reset.sh에 추가
RECORD_DATASET=true
DATASET_REPO_ID="user/cap_pick_and_place_v1"

./run_forward_and_reset.sh
```

### 7.3 데이터셋 사용
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 로컬에서 로드
dataset = LeRobotDataset("user/cap_pick_and_place_v1")

# 학습에 사용
for batch in DataLoader(dataset, batch_size=16):
    observations = batch["observation.state"]
    actions = batch["action"]
    images = batch["observation.images.front"]
```

---

## 8. 참고 자료

- [LeRobot Dataset v3.0 문서](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [LeRobot Dataset Tools](https://huggingface.co/docs/lerobot/using_dataset_tools)
- [Porting Datasets to v3.0](https://huggingface.co/docs/lerobot/porting_datasets_v3)
- [lerobot GitHub Repository](https://github.com/huggingface/lerobot)
