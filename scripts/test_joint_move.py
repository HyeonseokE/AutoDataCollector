#!/usr/bin/env python3
"""
각 조인트를 현재 위치에서 +-30도씩 움직이는 테스트 스크립트
/dev/ttyACM1 (robot2) 대상
"""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.lerobot_cap.hardware.feetech import FeetechController
from src.lerobot_cap.hardware.calibration import MotorCalibration

# === 설정 ===
PORT = "/dev/ttyACM1"
BAUDRATE = 1000000
CALIB_PATH = Path(__file__).resolve().parent.parent / "robot_configs/motor_calibration/so101/robot2_calibration.json"
DEGREES = 30
STEPS_PER_REVOLUTION = 4096  # STS3215: 4096 steps = 360 degrees
MOVE_WAIT = 1.5  # 각 이동 후 대기 시간(초)

# === 캘리브레이션 로드 ===
with open(CALIB_PATH, 'r') as f:
    calib_raw = json.load(f)

calibration = {}
joint_names = []
for name, data in calib_raw.items():
    motor_data = dict(data)
    if 'id' in motor_data and 'motor_id' not in motor_data:
        motor_data['motor_id'] = motor_data.pop('id')
    if 'model' not in motor_data:
        motor_data['model'] = 'sts3215'
    calib = MotorCalibration(**motor_data)
    calibration[calib.motor_id] = calib
    joint_names.append(name)

motor_ids = [calibration[mid].motor_id for mid in sorted(calibration.keys())]

# 30도를 각 관절의 normalized 단위로 변환
# range_min ~ range_max = 200 normalized units
# 30도 = 341 raw steps
steps_30deg = int(STEPS_PER_REVOLUTION * DEGREES / 360)

deg30_normalized = {}
for mid, calib in calibration.items():
    range_size = calib.range_max - calib.range_min
    norm_per_step = 200.0 / range_size
    deg30_norm = steps_30deg * norm_per_step
    deg30_normalized[mid] = deg30_norm

print(f"=== 조인트 +-{DEGREES}도 테스트 ===")
print(f"포트: {PORT}")
print(f"30도 = {steps_30deg} raw steps")
print()
for i, name in enumerate(joint_names):
    mid = sorted(calibration.keys())[i]
    print(f"  {name} (ID {mid}): +-{deg30_normalized[mid]:.1f} normalized units")
print()

# === 연결 ===
robot = FeetechController(
    port=PORT,
    baudrate=BAUDRATE,
    motor_ids=motor_ids,
    calibration=calibration,
)

if not robot.connect():
    print("연결 실패!")
    sys.exit(1)

try:
    # 현재 위치 읽기
    current_pos = robot.read_positions(normalize=True)
    print(f"\n현재 위치 (normalized):")
    for i, name in enumerate(joint_names):
        print(f"  {name}: {current_pos[i]:.1f}")

    # 토크 활성화
    robot.enable_torque()
    time.sleep(0.5)

    # 그리퍼(ID 6) 제외, 5개 관절만 테스트
    test_joints = list(range(5))  # index 0~4 (shoulder_pan ~ wrist_roll)
    original_pos = current_pos.copy()

    for joint_idx in test_joints:
        name = joint_names[joint_idx]
        mid = sorted(calibration.keys())[joint_idx]
        delta = deg30_normalized[mid]

        print(f"\n--- {name} (ID {mid}) 테스트 ---")

        # +30도
        target_plus = original_pos.copy()
        target_val = original_pos[joint_idx] + delta
        target_val = max(-100.0, min(100.0, target_val))
        target_plus[joint_idx] = target_val
        print(f"  +{DEGREES}도: {original_pos[joint_idx]:.1f} -> {target_val:.1f}")
        robot.write_positions(target_plus, normalize=True)
        time.sleep(MOVE_WAIT)

        # 원위치
        robot.write_positions(original_pos, normalize=True)
        time.sleep(MOVE_WAIT)

        # -30도
        target_minus = original_pos.copy()
        target_val = original_pos[joint_idx] - delta
        target_val = max(-100.0, min(100.0, target_val))
        target_minus[joint_idx] = target_val
        print(f"  -{DEGREES}도: {original_pos[joint_idx]:.1f} -> {target_val:.1f}")
        robot.write_positions(target_minus, normalize=True)
        time.sleep(MOVE_WAIT)

        # 원위치
        print(f"  원위치 복귀")
        robot.write_positions(original_pos, normalize=True)
        time.sleep(MOVE_WAIT)

    print("\n=== 테스트 완료 ===")

finally:
    robot.disable_torque()
    robot.disconnect()
