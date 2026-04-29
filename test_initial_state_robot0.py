"""
Robot0 initial-state 테스트.

robot0 캘리브레이션과 초기상태를 로드해서 initial_state로 이동시킨다.
USB 포트: /dev/ttyACM0 (so101_robot0.yaml에 이미 설정됨)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import os
os.chdir(PROJECT_ROOT)

from skills.skills_lerobot import LeRobotSkills


def main():
    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot0.yaml",
        frame="base_link",
        movement_duration=3.0,
    )

    if not skills.connect():
        print("[FAIL] Robot connection failed")
        return 1

    try:
        print("\n[STEP] Moving to initial state...")
        ok = skills.move_to_initial_state()
        if not ok:
            print("[FAIL] move_to_initial_state returned False")
            return 1
        print("[OK] Reached initial state")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    finally:
        skills.disconnect()


if __name__ == "__main__":
    sys.exit(main())
