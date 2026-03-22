#!/usr/bin/env python3
"""
그리퍼 단독 테스트 스크립트.

사용법:
    python scripts/test_gripper.py --config robot_configs/robot/so101_robot2.yaml
    python scripts/test_gripper.py --config robot_configs/robot/so101_robot3.yaml

동작:
    1. 로봇 연결 (그리퍼 모터만 토크 활성화, 팔은 자유 상태)
    2. 인터랙티브 메뉴:
       o - 그리퍼 열기
       c - 그리퍼 닫기
       h - 그리퍼 반만 열기 (50%)
       r - 현재 위치 읽기
       q - 종료
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lerobot_cap.hardware import FeetechController, MotorCalibration


def load_robot(config_path: str):
    """로봇 설정 로드 및 연결."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 캘리브레이션 로드
    calibration_file = config.get("calibration_file")
    calibration_by_id = {}
    if calibration_file and Path(calibration_file).exists():
        with open(calibration_file, 'r') as f:
            calib_data = json.load(f)
        for name, data in calib_data.items():
            motor_id = data.get('motor_id', data.get('id'))
            if motor_id is None:
                continue
            calibration_by_id[motor_id] = MotorCalibration(
                motor_id=motor_id,
                model=data.get('model', 'sts3215'),
                drive_mode=data.get('drive_mode', 0),
                homing_offset=data.get('homing_offset', 0),
                range_min=data.get('range_min', 0),
                range_max=data.get('range_max', 4095),
            )
        print(f"캘리브레이션 로드: {len(calibration_by_id)}개 모터 ({calibration_file})")

    # 모터 ID
    motor_ids = [config["motors"][f"motor_{i}"]["id"] for i in range(1, 7)]

    robot = FeetechController(
        port=config.get("port"),
        baudrate=config.get("baudrate", 1000000),
        motor_ids=motor_ids,
        calibration=calibration_by_id,
    )

    if not robot.connect():
        print("Error: 로봇 연결 실패")
        sys.exit(1)

    return robot, config


def move_gripper(robot: FeetechController, target_pos: float, duration: float = 1.0):
    """그리퍼만 목표 위치로 이동 (팔은 현재 위치 유지)."""
    current = robot.read_positions(normalize=True)
    start_gripper = current[5]

    num_steps = int(duration * 50)
    start_time = time.time()

    for i in range(num_steps):
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break

        alpha = min(elapsed / duration, 1.0)
        gripper_now = start_gripper + alpha * (target_pos - start_gripper)

        # 팔 5축은 현재 위치 유지, 그리퍼만 변경
        arm_pos = current[:5].copy()
        full_cmd = np.concatenate([arm_pos, [gripper_now]])
        robot.write_positions(full_cmd, normalize=True)

        next_time = start_time + (i + 1) * (duration / num_steps)
        sleep_time = next_time - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)

    # 최종 위치 전송
    current_arm = robot.read_positions(normalize=True)[:5]
    robot.write_positions(np.concatenate([current_arm, [target_pos]]), normalize=True)


def main():
    parser = argparse.ArgumentParser(description="그리퍼 단독 테스트")
    parser.add_argument("--config", type=str, default="robot_configs/robot/so101_robot2.yaml",
                        help="로봇 설정 파일")
    parser.add_argument("--open-pos", type=float, default=85.0,
                        help="그리퍼 열림 위치 (normalized, 기본: 85)")
    parser.add_argument("--close-pos", type=float, default=-95.0,
                        help="그리퍼 닫힘 위치 (normalized, 기본: -95)")
    parser.add_argument("--duration", type=float, default=1.0,
                        help="이동 시간 (초, 기본: 1.0)")
    args = parser.parse_args()

    robot, config = load_robot(args.config)

    GRIPPER_OPEN = args.open_pos
    GRIPPER_CLOSE = args.close_pos
    GRIPPER_HALF = (GRIPPER_OPEN + GRIPPER_CLOSE) / 2

    # 그리퍼(모터6)만 토크 활성화
    gripper_motor_id = config["motors"]["motor_6"]["id"]
    robot.enable_torque([gripper_motor_id])
    print(f"그리퍼 모터(ID={gripper_motor_id})만 토크 활성화")

    try:
        current = robot.read_positions(normalize=True)
        print(f"현재 그리퍼 위치: {current[5]:.1f} (normalized)")
        print(f"  열림={GRIPPER_OPEN}, 닫힘={GRIPPER_CLOSE}")
        print()
        print("명령어:")
        print("  o - 열기 (open)")
        print("  c - 닫기 (close)")
        print("  h - 반열기 (half, 50%)")
        print("  r - 현재 위치 읽기")
        print("  숫자 - 직접 위치 지정 (-100 ~ 100)")
        print("  q - 종료")
        print()

        while True:
            try:
                cmd = input("gripper> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if cmd == 'q':
                break
            elif cmd == 'o':
                print(f"그리퍼 열기 → {GRIPPER_OPEN:.0f}")
                move_gripper(robot, GRIPPER_OPEN, args.duration)
            elif cmd == 'c':
                print(f"그리퍼 닫기 → {GRIPPER_CLOSE:.0f}")
                move_gripper(robot, GRIPPER_CLOSE, args.duration)
            elif cmd == 'h':
                print(f"그리퍼 반열기 → {GRIPPER_HALF:.0f}")
                move_gripper(robot, GRIPPER_HALF, args.duration)
            elif cmd == 'r':
                pos = robot.read_positions(normalize=True)
                print(f"현재 그리퍼 위치: {pos[5]:.1f}")
            elif cmd == '':
                continue
            else:
                try:
                    target = float(cmd)
                    if -100 <= target <= 100:
                        print(f"그리퍼 이동 → {target:.1f}")
                        move_gripper(robot, target, args.duration)
                    else:
                        print("범위 초과: -100 ~ 100 사이 값 입력")
                except ValueError:
                    print("알 수 없는 명령어. o/c/h/r/q 또는 숫자 입력")

            # 이동 후 현재 위치 표시
            if cmd in ('o', 'c', 'h') or cmd.replace('-', '').replace('.', '').isdigit():
                time.sleep(0.1)
                pos = robot.read_positions(normalize=True)
                print(f"  → 현재 위치: {pos[5]:.1f}")

    finally:
        robot.disable_torque([gripper_motor_id])
        robot.disconnect()
        print("종료")


if __name__ == "__main__":
    main()
