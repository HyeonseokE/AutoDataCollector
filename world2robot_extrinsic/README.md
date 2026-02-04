# World-to-Robot Extrinsic Calibration

World 좌표계(카메라/Detection)와 Robot Base 좌표계 간의 외부 캘리브레이션을 수행하는 도구입니다.

## 목적

카메라나 Object Detection 시스템에서 검출된 객체의 **World 좌표**를 로봇이 이해할 수 있는 **Robot Base 좌표**로 변환하기 위한 변환 행렬을 계산합니다.

```
World 좌표 (Detection) ──[변환 행렬]──> Robot Base 좌표 (로봇 제어)
```

## 프레임워크 내 역할

```
┌─────────────────────────────────────────────────────────────┐
│                    Pipeline Flow                             │
├─────────────────────────────────────────────────────────────┤
│  1. Object Detection (Grounding DINO)                        │
│     └─> 객체 위치 (Pixel 좌표)                               │
│                                                              │
│  2. Pixel → World 변환 (camera_extrinsic)                    │
│     └─> 객체 위치 (World 좌표)                               │
│                                                              │
│  3. World → Robot Base 변환 ★ [이 폴더]                      │
│     └─> 객체 위치 (Robot Base 좌표)                          │
│                                                              │
│  4. Code Generation (LLM)                                    │
│     └─> 로봇 제어 Python 코드                                │
│                                                              │
│  5. Robot Execution                                          │
│     └─> 실제 로봇 동작                                       │
└─────────────────────────────────────────────────────────────┘
```

변환 결과는 `robot_configs/world2robot_matrices/robot{N}_matrix.json`에 저장되어 파이프라인에서 사용됩니다.

## 캘리브레이션 과정

### Step 1: 매칭점 수집

World 좌표와 해당 위치에서의 Robot Base TCP 좌표를 쌍으로 수집합니다.

```bash
# 로봇 3번 기준, World 좌표 (0.15, 0.0, 0.0)에 대한 매칭점 수집
./find_matching_point.sh 3 0.15 0.0 0.0
```

**수집 과정:**
1. 스크립트 실행 시 로봇 토크가 비활성화됨
2. 작업자가 로봇 그리퍼를 **지정된 World 좌표 위치**로 수동 이동
3. Enter 키를 누르면 현재 TCP 위치(Robot Base 좌표)가 기록됨
4. 이 과정을 **최소 10회** 반복 (다양한 위치에서)

**권장 수집 패턴:**
```
World X: 0.05 ~ 0.25m (5cm 간격)
World Y: -0.09 ~ 0.05m (5cm 간격)
World Z: 0.0m (테이블 표면)
```

### Step 2: 변환 행렬 계산

수집된 매칭점을 사용하여 Kabsch-Umeyama 알고리즘으로 변환 행렬을 계산합니다.

```bash
# 로봇 3번의 변환 행렬 계산
./calculate_world2robot_transform_matrix.sh 3
```

**출력:**
- 회전 행렬 (R)
- 평행이동 벡터 (t)
- 4x4 동차 변환 행렬 (T)
- RMSE 오차 분석

### Step 3: 결과 확인

계산된 변환 행렬은 두 위치에 저장됩니다:

1. **파이프라인용**: `robot_configs/world2robot_matrices/robot{N}_matrix.json`
2. **상세 결과**: `extrinsics/robot{N}_calibration_result.json`

**품질 기준:**
| RMSE | 품질 |
|------|------|
| < 5mm | 우수 |
| < 10mm | 양호 |
| < 20mm | 보통 |
| >= 20mm | 불량 (재캘리브레이션 필요) |

## 폴더 구조

```
world2robot_extrinsic/
├── README.md                              # 이 문서
├── find_matching_point.py                 # 매칭점 수집 Python 스크립트
├── find_matching_point.sh                 # 매칭점 수집 Shell 래퍼
├── world_frame2robot_base_frame.py        # 변환 행렬 계산 Python 스크립트
├── calculate_world2robot_transform_matrix.sh  # 변환 행렬 계산 Shell 래퍼
├── matching_points/                       # 수집된 매칭점 데이터
│   ├── robot2_matching_points.json
│   └── robot3_matching_points.json
└── extrinsics/                            # 캘리브레이션 결과
    ├── robot2_calibration_result.json
    └── robot3_calibration_result.json
```

## 파일 설명

### Python 스크립트

| 파일 | 설명 |
|------|------|
| `find_matching_point.py` | 매칭점 수집 도구. 로봇 토크를 비활성화하고, 사용자가 그리퍼를 World 좌표 위치로 이동시키면 FK로 Robot Base TCP 좌표를 계산하여 저장 |
| `world_frame2robot_base_frame.py` | Kabsch-Umeyama 알고리즘을 사용하여 수집된 점들로부터 최적의 강체 변환(회전+평행이동)을 계산 |

### Shell 래퍼

| 파일 | 설명 |
|------|------|
| `find_matching_point.sh` | Python 스크립트 실행 래퍼. ttyACM 권한 설정 및 PYTHONPATH 구성 |
| `calculate_world2robot_transform_matrix.sh` | 변환 행렬 계산 스크립트 래퍼 |

### 데이터 폴더

| 폴더 | 설명 |
|------|------|
| `matching_points/` | 수집된 (World, Robot Base) 좌표 쌍. JSON 형식으로 누적 저장 |
| `extrinsics/` | 계산된 캘리브레이션 결과. 변환 행렬, 오차 통계 등 상세 정보 포함 |

## 알고리즘

**Kabsch-Umeyama Algorithm**

주어진 점 쌍 집합에서 최적의 강체 변환을 찾습니다:

```
minimize Σ || R·world_i + t - robot_base_i ||²
```

1. 양 점 집합의 중심(centroid) 계산
2. 중심 기준으로 점들 정규화
3. 교차 공분산 행렬 H 계산
4. SVD 분해: H = UΣV^T
5. 회전 행렬: R = V·U^T
6. 평행이동: t = centroid_target - R·centroid_source

## 재캘리브레이션

로봇 위치가 변경되거나 카메라가 재설치된 경우:

```bash
# 기존 데이터 삭제
rm matching_points/robot3_matching_points.json

# 새로 수집 (10회 이상)
./find_matching_point.sh 3 0.10 0.00 0.0
./find_matching_point.sh 3 0.15 0.00 0.0
./find_matching_point.sh 3 0.20 0.00 0.0
# ... 추가 점 수집

# 변환 행렬 재계산
./calculate_world2robot_transform_matrix.sh 3
```

## 사용 예시

```python
import json
import numpy as np

# 변환 행렬 로드
with open("robot_configs/world2robot_matrices/robot3_matrix.json") as f:
    config = json.load(f)

T = np.array(config["_raw_transform"]["transform_4x4"])

# World 좌표를 Robot Base 좌표로 변환
world_point = np.array([0.15, 0.0, 0.0, 1.0])  # homogeneous
robot_base_point = T @ world_point
print(f"Robot Base: {robot_base_point[:3]}")
```
