# LeRobot CaP Distillation

**Code-as-Policies (CaP) 기반 로봇 제어 프레임워크**

자연어 명령을 실행 가능한 로봇 코드로 변환하고, 실행 결과를 자동으로 평가하는 End-to-End 로봇 조작 시스템입니다.

---

## Overview

이 프로젝트는 두 가지 레벨의 로봇 제어를 제공합니다:

| Level | 설명 | 주요 모듈 |
|-------|------|----------|
| **High-Level** | CaP 기반 자연어 → 코드 생성 → 실행 → 평가 | `code_gen_lerobot/`, `skills/` |
| **Low-Level** | 하드웨어 직접 제어 (IK/FK, 모터, 궤적) | `src/lerobot_cap/` |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CaP-based Robot Control Pipeline                      │
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │   Natural    │    │     Code     │    │    Robot     │    │   Task    │  │
│  │   Language   │───▶│  Generation  │───▶│  Execution   │───▶│   Judge   │  │
│  │  Instruction │    │    (LLM)     │    │   (Skills)   │    │   (VLM)   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘  │
│         │                   │                   │                   │        │
│         ▼                   ▼                   ▼                   ▼        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │    Object    │    │   Primitive  │    │  Low-Level   │    │  Success  │  │
│  │  Detection   │    │    Skills    │    │   Control    │    │ TRUE/FALSE│  │
│  │  (VLM+RGBD)  │    │   Template   │    │  (IK/Motor)  │    │           │  │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

# Part 1: CaP-based Robot Control (High-Level)

자연어 명령을 받아 로봇이 자율적으로 태스크를 수행하는 파이프라인입니다.

## Pipeline Overview

```
"Pick up the red cup and place it in the blue box"
                    │
                    ▼
    ┌───────────────────────────────┐
    │  Step 1: Object Detection     │
    │  - RealSense D435 (RGB-D)     │
    │  - Grounding DINO (Zero-shot) │
    │  - Pixel → Robot Coordinates  │
    └───────────────┬───────────────┘
                    │ {"red cup": [0.15, 0.05, 0.02],
                    │  "blue box": [0.20, -0.05, 0.03]}
                    ▼
    ┌───────────────────────────────┐
    │  Step 2: Code Generation      │
    │  - LLM (GPT-4, Gemini, LLaMA) │
    │  - Primitive Skills Prompt    │
    │  - Executable Python Code     │
    └───────────────┬───────────────┘
                    │ skills.move_to_position(...)
                    │ skills.gripper_close()
                    ▼
    ┌───────────────────────────────┐
    │  Step 3: Robot Execution      │
    │  - LeRobotSkills primitives   │
    │  - IK/FK + Trajectory Plan    │
    │  - Motor Control (Feetech)    │
    └───────────────┬───────────────┘
                    │ Robot moves physically
                    ▼
    ┌───────────────────────────────┐
    │  Step 4: Task Judge           │
    │  - Before/After Images        │
    │  - GPT-4o Vision Analysis     │
    │  - TRUE / FALSE / UNCERTAIN   │
    └───────────────────────────────┘
```

## Quick Start (CaP Pipeline)

### 1. 전체 파이프라인 실행

```bash
python3 run_code_gen_with_judge.py \
  --instruction "Pick up the red cup and place it in the blue box" \
  --objects "red cup" "blue box" \
  --robot 3 \
  --llm openai
```

### 2. 단계별 실행

```python
from code_gen_lerobot.code_gen_with_skill import lerobot_code_gen
from code_gen_lerobot.judge import TaskJudge

# Step 1-2: Detection + Code Generation
code = lerobot_code_gen(
    instruction="Pick up the red cup",
    object_positions={"red cup": [0.15, 0.05, 0.02]},
    robot_id=3,
    llm_provider="openai"
)

# Step 3: Execute generated code
exec(code)

# Step 4: Judge task completion
judge = TaskJudge()
result = judge.judge(
    instruction="Pick up the red cup",
    initial_image=before_img,
    final_image=after_img
)
print(result['prediction'])  # TRUE / FALSE / UNCERTAIN
```

## Module Details

### 1. Object Detection (`object_detection/`)

Intel RealSense D435와 Grounding DINO를 사용한 zero-shot 객체 검출

```
object_detection/
├── camera/              # RealSense D435 인터페이스
│   └── realsense.py     # RGB-D 프레임 캡처
├── detection/           # Vision-Language 모델
│   └── grounding_detector.py  # Grounding DINO
├── localization/        # 좌표 변환
│   └── coordinate_transform.py
└── main.py              # ObjectDetectionSystem
```

**사용 예시:**
```python
from object_detection.main import ObjectLocalizationSystem

system = ObjectLocalizationSystem()
system.initialize()

# 자연어 쿼리로 물체 검출
position = system.find_object("red cup")
print(f"Red cup position: {position}")  # [0.15, 0.05, 0.02] (meters)

system.shutdown()
```

### 2. Code Generation (`code_gen_lerobot/`)

LLM을 사용하여 자연어 명령을 실행 가능한 Python 코드로 변환

```
code_gen_lerobot/
├── code_gen_with_skill.py   # 메인 코드 생성 함수
├── object_positions.py      # Detection 시스템 래퍼
├── llm.py                   # LLM 라우터 (OpenAI, Gemini, LLaMA)
├── prompt/
│   ├── code_gen.py          # 코드 생성 프롬프트
│   ├── spec_gen.py          # 스펙 생성 프롬프트
│   └── reset_*.py           # 리셋 프롬프트
└── judge/                   # Task 평가 모듈
    ├── judge.py             # TaskJudge (VLM 평가)
    ├── prompt.py            # Judge 프롬프트
    └── image_capture.py     # 이미지 캡처
```

**지원 LLM:**
| Provider | Model | 설명 |
|----------|-------|------|
| `openai` | GPT-4, GPT-3.5 | 가장 안정적 |
| `gemini` | Gemini Pro | Google API |
| `llama` | LLaMA 2/3 | 로컬 실행 가능 |

### 3. Primitive Skills (`skills/skills_lerobot.py`)

LLM이 생성한 코드가 호출하는 로봇 동작 primitives

| Skill | 설명 | 파라미터 |
|-------|------|----------|
| `connect()` | 로봇 연결 | - |
| `disconnect()` | 로봇 연결 해제 | - |
| `move_to_position(pos)` | 위치 이동 (현재 wrist_roll 유지) | `position`, `apply_gripper_offset` |
| `gripper_open()` | 그리퍼 열기 | - |
| `gripper_close()` | 그리퍼 닫기 | - |
| `rotate_90degree(dir)` | 그리퍼 90도 회전 | `direction`: 1(CW), -1(CCW) |
| `move_to_initial_state()` | 홈 위치로 이동 | - |
| `move_to_free_state()` | 안전 주차 위치 | - |

**Pick & Place 패턴:**
```python
from skills.skills_lerobot import LeRobotSkills

skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot3.yaml")
skills.connect()

# Setup
skills.move_to_initial_state()
skills.rotate_90degree(direction=1)  # 그리퍼 90도 회전

# PICK (apply_gripper_offset=True로 비대칭 그리퍼 충돌 방지)
skills.gripper_open()
skills.move_to_position([0.15, 0.05, 0.07], apply_gripper_offset=True)  # approach
skills.move_to_position([0.15, 0.05, 0.02], apply_gripper_offset=True)  # descend
skills.gripper_close()
skills.move_to_position([0.15, 0.05, 0.07])  # lift

# PLACE
skills.move_to_position([0.20, -0.05, 0.07])  # approach
skills.move_to_position([0.20, -0.05, 0.02])  # descend
skills.gripper_open()
skills.move_to_position([0.20, -0.05, 0.07])  # retract

# Cleanup
skills.move_to_initial_state()
skills.move_to_free_state()
skills.disconnect()
```

### 4. Task Judge (`code_gen_lerobot/judge/`)

GPT-4o Vision을 사용하여 태스크 성공 여부 자동 평가

```python
from code_gen_lerobot.judge import TaskJudge

judge = TaskJudge()
result = judge.judge(
    instruction="Pick up the red cup and place it in the blue box",
    initial_image=before_image,  # numpy array (BGR)
    final_image=after_image,
    object_positions={"red cup": [0.15, 0.05, 0.02]},
    executed_code=generated_code
)

print(result['prediction'])  # 'TRUE', 'FALSE', or 'UNCERTAIN'
print(result['reasoning'])   # 상세 분석 결과
```

---

# Part 2: Low-Level Robot Control

하드웨어를 직접 제어하는 저수준 API입니다.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│     (Skills, Teleoperation, Data Collection, Learning)      │
├─────────────────────────────────────────────────────────────┤
│                     Planning Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Kinematics  │  │ Trajectory  │  │ Frame Transformer   │  │
│  │ Engine      │  │ Planner     │  │ (World↔Base Link)   │  │
│  │ (Pinocchio) │  │ (Multi-IK)  │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Control Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Calibration │  │Compensation │  │ Safety Limits       │  │
│  │ Limits      │  │ (Gravity)   │  │ (Joint/Velocity)    │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
├─────────────────────────────────────────────────────────────┤
│                     Hardware Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Feetech     │  │ Motor       │  │ Camera              │  │
│  │ Controller  │  │ Calibration │  │ (RealSense)         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/lerobot_cap/
├── hardware/           # 하드웨어 인터페이스
│   ├── controller.py   # FeetechController (모터 통신)
│   └── calibration.py  # MotorCalibration (엔코더 변환)
├── kinematics/         # 기구학
│   ├── engine.py       # KinematicsEngine (Pinocchio FK/IK)
│   └── calibration_limits.py  # 관절 한계 변환
├── planning/           # 궤적 계획
│   └── trajectory.py   # TrajectoryPlanner (Multi-IK, 보간)
└── compensation.py     # AdaptiveCompensator (중력 보상)
```

## Core Components

### 1. FeetechController (`hardware/controller.py`)

Feetech STS3215 서보 모터 직접 제어

```python
from lerobot_cap.hardware import FeetechController

# 초기화
robot = FeetechController(
    port="/dev/ttyACM0",
    baudrate=1000000,
    motor_ids=[1, 2, 3, 4, 5, 6]  # 6 motors (5 arm + 1 gripper)
)
robot.connect()
robot.enable_torque()

# 현재 위치 읽기 (정규화: -100 ~ +100)
positions = robot.read_positions(normalize=True)
print(f"Current positions: {positions}")

# 위치 명령 (정규화 값)
target = [0.0, -30.0, 45.0, 20.0, 0.0, 50.0]
robot.write_positions(target, normalize=True)

# 정리
robot.disable_torque()
robot.disconnect()
```

### 2. KinematicsEngine (`kinematics/engine.py`)

Pinocchio 기반 Forward/Inverse Kinematics

```python
from lerobot_cap.kinematics import KinematicsEngine

# URDF 로드
kinematics = KinematicsEngine(
    urdf_path="assets/urdf/so101.urdf",
    end_effector_frame="gripper_frame_link",
    joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex",
                 "wrist_flex", "wrist_roll"]
)

# Forward Kinematics: 관절 각도 → End-Effector 위치
joint_angles = [0.0, -0.5, 1.0, 0.5, 0.0]  # radians
ee_position = kinematics.get_ee_position(joint_angles)
print(f"EE position: {ee_position}")  # [x, y, z] meters

# Inverse Kinematics: 목표 위치 → 관절 각도
target_pos = [0.2, 0.0, 0.15]
solution, converged = kinematics.inverse_kinematics(
    target_pos,
    q_init=joint_angles
)
print(f"IK solution: {solution}, converged: {converged}")
```

### 3. TrajectoryPlanner (`planning/trajectory.py`)

Multi-IK 기반 궤적 계획

```python
from lerobot_cap.planning import TrajectoryPlanner
from lerobot_cap.kinematics import KinematicsEngine, load_calibration_limits

kinematics = KinematicsEngine("assets/urdf/so101.urdf")
calibration_limits = load_calibration_limits("robot_configs/motor_calibration/so101/robot3_calibration.json")

planner = TrajectoryPlanner(
    kinematics,
    max_velocity=1.0,        # rad/s
    max_acceleration=2.0,    # rad/s²
    interpolation_points=50,
    calibration_limits=calibration_limits
)

# 목표 위치로 궤적 계획 (Multi-IK: 여러 초기값으로 최적 해 탐색)
current_joints = [0.0, -0.5, 1.0, 0.5, 0.0]
target_position = [0.2, 0.05, 0.1]

trajectory, ik_info = planner.plan_to_position_multi(
    target_position,
    current_joints,
    duration=3.0,
    num_random_samples=10,
    fixed_joints=[4]  # wrist_roll 고정
)

print(f"IK solutions: {ik_info['num_valid']}/{ik_info['num_solutions']} valid")
print(f"Trajectory duration: {trajectory.duration}s")
print(f"Waypoints: {len(trajectory.joint_positions)}")
```

### 4. CalibrationLimits (`kinematics/calibration_limits.py`)

정규화 ↔ 라디안 변환

```python
from lerobot_cap.kinematics import load_calibration_limits

limits = load_calibration_limits("robot_configs/motor_calibration/so101/robot3_calibration.json")

# 정규화 값 (-100 ~ +100) → 라디안
normalized = [0.0, -30.0, 45.0, 20.0, 0.0]
radians = limits.normalized_to_radians(normalized)

# 라디안 → 정규화 값
radians = [0.0, -0.5, 1.0, 0.5, 0.0]
normalized = limits.radians_to_normalized(radians)
```

## Configuration Files

```
robot_configs/
├── robot/
│   ├── so101_robot2.yaml    # Robot 2 설정
│   └── so101_robot3.yaml    # Robot 3 설정
├── frames/
│   └── robot3_matrix.json   # World-Robot 변환 행렬
├── initial_state.json       # 초기 상태 (홈 포지션)
└── free_state.json          # 안전 주차 상태

robot_configs/motor_calibration/so101/
├── robot2_calibration.json  # Robot 2 모터 캘리브레이션
├── robot3_calibration.json  # Robot 3 모터 캘리브레이션
└── robot3_compensation.json # 중력 보상 LUT
```

**Robot Config 예시 (`so101_robot3.yaml`):**
```yaml
port: /dev/ttyACM0
baudrate: 1000000

motors:
  motor_1: {id: 1, name: shoulder_pan}
  motor_2: {id: 2, name: shoulder_lift}
  motor_3: {id: 3, name: elbow_flex}
  motor_4: {id: 4, name: wrist_flex}
  motor_5: {id: 5, name: wrist_roll}
  motor_6: {id: 6, name: gripper}

kinematics:
  urdf_path: assets/urdf/so101_robot3.urdf
  end_effector_frame: gripper_frame_link

calibration_file: robot_configs/motor_calibration/so101/robot3_calibration.json
compensation_file: robot_configs/motor_calibration/so101/robot3_compensation.json
```

---

## Installation

```bash
# Clone repository
git clone https://github.com/your-repo/lerobot_CaP_distillation.git
cd lerobot_CaP_distillation

# Install dependencies
pip install -e ".[all]"

# Pinocchio (IK/FK)
conda install -c conda-forge pinocchio

# Object Detection dependencies
pip install torch torchvision groundingdino-py
pip install pyrealsense2  # RealSense camera

# LLM APIs
pip install openai google-generativeai
```

## Hardware Requirements

| Component | Model | 용도 |
|-----------|-------|------|
| Robot Arm | SO-101 (Feetech STS3215) | 5-DOF 매니퓰레이터 + 그리퍼 |
| Camera | Intel RealSense D435 | RGB-D Object Detection |
| GPU | NVIDIA (CUDA) | Grounding DINO, VLM |

## Supported Robots

- **SO-100** (Feetech STS3215)
- **SO-101** (Feetech STS3215) - 5-DOF + Gripper
- Custom robots via URDF

---

## Project Structure

```
lerobot_CaP_distillation/
├── src/lerobot_cap/        # Low-level control modules
│   ├── hardware/           # Motor controllers
│   ├── kinematics/         # FK/IK (Pinocchio)
│   ├── planning/           # Trajectory planning
│   └── compensation.py     # Gravity compensation
├── skills/                 # High-level primitive skills
│   ├── skills_lerobot.py   # LeRobotSkills class
│   └── About_skills.md     # Skills documentation
├── code_gen_lerobot/       # Code generation pipeline
│   ├── code_gen_with_skill.py
│   ├── prompt/             # LLM prompts
│   └── judge/              # Task evaluation
├── object_detection/       # Vision-language detection
│   ├── camera/
│   ├── detection/
├── robot_configs/                # Configuration files
│   └── motor_calibration/  # Motor calibration data
├── assets/urdf/            # Robot URDF files
├── scripts/                # Utility scripts
└── run_code_gen_with_judge.py  # Main pipeline script
```

---

## License

MIT License
