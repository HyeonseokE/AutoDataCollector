# Pix2Robot 캘리브레이션

## 핵심 개념
- **기존**: pixel → world → robot (3단계, depth 노이즈 문제)
- **변경**: pixel → robot 직접 변환 (호모그래피 + depth 높이 추정)

## 변환 방식
| 항목 | 설명 |
|------|------|
| **호모그래피 3x3** | 픽셀(u,v) → 로봇(x,y) 평면 매핑 (cv2.findHomography, RANSAC) |
| **table_depth** | 캘리브레이션 시 수집한 카메라→테이블 depth 평균 (meters) |
| **물체 높이(z)** | `table_depth - 물체_depth` → 테이블 위 물체 높이 (depth 미제공 시 table_z ≈ 0) |

## 파일 구조
```
pix2robot_calibrator/
├── __init__.py                # Pix2RobotCalibrator export
├── calibrator.py              # Pix2RobotCalibrator 클래스
└── run_calibration.py         # CLI 진입점

robot_configs/pix2robot_matrices/
├── robot{N}_pix2robot_data.npz   # homography, table_z, table_depth, points, depth_values, errors
└── robot{N}_pix2robot_data.json  # 사람이 읽을 수 있는 형식
```

## 실행 방법

### 캘리브레이션
```bash
# 새로 캘리브레이션
python pix2robot_calibrator/run_calibration.py --robot 3

# 기존 데이터에 이어서 추가
python pix2robot_calibrator/run_calibration.py --robot 3 --resume
```

**순서**: 카메라 프리뷰 `s`캡처 → 이미지 좌클릭(픽셀) → 로봇 수동 이동 → Enter(기록) → 반복(권장 8~12쌍) → `c`계산 → `s`저장

### 코드에서 사용
```python
from pix2robot_calibrator import Pix2RobotCalibrator

cal = Pix2RobotCalibrator(robot_id=3)
cal.load("robot_configs/pix2robot_matrices/robot3_pix2robot_data.npz")

# x,y만 (z ≈ 0)
pos = cal.pixel_to_robot(400, 300)

# x,y + 물체 높이 (depth 활용)
pos = cal.pixel_to_robot(400, 300, depth_m=0.65)
# → [0.2491, 0.0640, 0.0367]  (높이 3.7cm)
```

## 파이프라인 연동
- `_get_frame_for_robot(robot_id)`: pix2robot .npz 존재 시 `frame="base_link"` → 스킬에서 world→robot 변환 생략
- 소비자 파일(`code_gen_with_skill.py`, `run_detect.py`, `reset_execution/code_gen.py`): pix2robot 우선, 없으면 기존 3단계 fallback
