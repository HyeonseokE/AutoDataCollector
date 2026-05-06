#!/usr/bin/env python3
"""
Robot4를 initial_state → free_state로 이동시키는 스크립트.
Cosine smoothing 보간으로 부드럽게 이동합니다.
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
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--robot", type=int, default=4, help="Robot ID (e.g. 4 or 6)")
parser.add_argument("--port", type=str, default=None, help="Override serial port")
args = parser.parse_args()

ROBOT_ID = f"robot{args.robot}"
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "robot_configs"

# Load port from robot yaml config
import yaml
robot_yaml = CONFIG_DIR / f"robot/so101_{ROBOT_ID}.yaml"
with open(robot_yaml, 'r') as f:
    robot_config = yaml.safe_load(f)
PORT = args.port if args.port else robot_config["port"]
BAUDRATE = robot_config.get("baudrate", 1000000)

CALIB_PATH = CONFIG_DIR / f"motor_calibration/so101/{ROBOT_ID}_calibration.json"
INITIAL_STATE_PATH = CONFIG_DIR / f"initial_state/{ROBOT_ID}_initial_state.json"
FREE_STATE_PATH = CONFIG_DIR / f"free_state/{ROBOT_ID}_free_state.json"

DURATION = 3.0  # 이동 시간 (초)
FPS = 30

# === 상태 파일 로드 ===
with open(INITIAL_STATE_PATH, 'r') as f:
    initial_data = json.load(f)
with open(FREE_STATE_PATH, 'r') as f:
    free_data = json.load(f)

initial_state = np.array(initial_data["initial_state_normalized"])
initial_gripper = initial_data["gripper_normalized"]
free_state = np.array(free_data["initial_state_normalized"])
free_gripper = free_data["gripper_normalized"]

print(f"=== {ROBOT_ID}: initial_state → free_state ===")
print(f"Initial state: {initial_state}")
print(f"Initial gripper: {initial_gripper}")
print(f"Free state:    {free_state}")
print(f"Free gripper:  {free_gripper}")
print(f"Duration: {DURATION}s")
print()

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

# === 컨트롤러 연결 ===
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
    print(f"현재 위치 (normalized): {current_pos[:5]}")
    print(f"현재 gripper: {current_pos[5]:.1f}")
    print()

    # 토크 활성화
    robot.enable_torque()
    time.sleep(0.3)

    # --- Phase 1: 현재 위치 → initial_state ---
    print("[Phase 1] 현재 위치 → initial_state")
    start_arm = current_pos[:5].copy()
    start_gripper = current_pos[5]
    end_arm = initial_state.copy()
    end_gripper = initial_gripper

    start_time = time.time()
    loop_period = 1.0 / FPS

    while True:
        loop_start = time.perf_counter()
        elapsed = time.time() - start_time
        if elapsed >= DURATION:
            break

        alpha = min(elapsed / DURATION, 1.0)
        smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2

        arm_cmd = np.clip(start_arm + smooth_alpha * (end_arm - start_arm), -99.0, 99.0)
        gripper_cmd = start_gripper + smooth_alpha * (end_gripper - start_gripper)
        full_cmd = np.concatenate([arm_cmd, [gripper_cmd]])
        robot.write_positions(full_cmd, normalize=True)

        progress = alpha
        filled = int(30 * progress)
        print(f"\r  [{'=' * filled}{'-' * (30 - filled)}] {progress*100:5.1f}%", end="", flush=True)

        dt = time.perf_counter() - loop_start
        sleep_time = loop_period - dt
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Final position
    full_cmd = np.concatenate([np.clip(end_arm, -99.0, 99.0), [end_gripper]])
    robot.write_positions(full_cmd, normalize=True)
    print(f"\r  [{'=' * 30}] 100.0%")
    print("  initial_state 도착!")
    time.sleep(1.0)

    # --- Phase 2: initial_state → free_state ---
    print("\n[Phase 2] initial_state → free_state")
    start_arm = initial_state.copy()
    start_gripper = initial_gripper
    end_arm = free_state.copy()
    end_gripper = free_gripper

    start_time = time.time()

    while True:
        loop_start = time.perf_counter()
        elapsed = time.time() - start_time
        if elapsed >= DURATION:
            break

        alpha = min(elapsed / DURATION, 1.0)
        smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2

        arm_cmd = np.clip(start_arm + smooth_alpha * (end_arm - start_arm), -99.0, 99.0)
        gripper_cmd = start_gripper + smooth_alpha * (end_gripper - start_gripper)
        full_cmd = np.concatenate([arm_cmd, [gripper_cmd]])
        robot.write_positions(full_cmd, normalize=True)

        progress = alpha
        filled = int(30 * progress)
        print(f"\r  [{'=' * filled}{'-' * (30 - filled)}] {progress*100:5.1f}%", end="", flush=True)

        dt = time.perf_counter() - loop_start
        sleep_time = loop_period - dt
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Final position
    full_cmd = np.concatenate([np.clip(end_arm, -99.0, 99.0), [end_gripper]])
    robot.write_positions(full_cmd, normalize=True)
    print(f"\r  [{'=' * 30}] 100.0%")
    print("  free_state 도착!")

    time.sleep(0.5)
    final_pos = robot.read_positions(normalize=True)
    print(f"\n최종 위치: {final_pos[:5]}")
    print(f"최종 gripper: {final_pos[5]:.1f}")
    print("\n=== 이동 완료 ===")

finally:
    robot.disable_torque()
    robot.disconnect()
