"""
debug_drawer_descent — drawer 핸들 z 도달 실패 원인 진단.

Approach (pitch IK 자유) 와 동일 z target 을 다음 3가지 변형으로 시도:
  1. maintain_pitch=False (현재 execute_pull Step 1 동작)
  2. maintain_pitch=True (approach pitch 유지)
  3. target_pitch=-45° (수직에 가까운 pitch)

각 케이스에서 도달 z error 비교 → motor saturation vs pitch 선택 문제 구분.

사용:
    python tools/debug_drawer_descent.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from skills.skills_lerobot import LeRobotSkills


def main():
    # 실패 episode 의 검출 좌표 (drawer handle)
    HANDLE_XY = (0.306, -0.004)
    HANDLE_Z = 0.166
    APPROACH_Z = 0.20

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot0.yaml",
        frame="world",
    )
    if not skills.connect():
        print("Robot connection failed")
        return

    try:
        results = []

        for label, kwargs in [
            ("FREE_IK   (current execute_pull Step 1)", {"maintain_pitch": False}),
            ("APPROACH_PITCH  (maintain_pitch=True)", {"maintain_pitch": True}),
            ("STEEP_-45deg     (target_pitch=-0.7854)", {"target_pitch": -0.7854}),
        ]:
            print(f"\n{'='*70}")
            print(f"  Test: {label}")
            print(f"{'='*70}")

            # 매 케이스 동일 출발: initial state → approach (free IK)
            skills.move_to_initial_state()
            skills.move_to_position(
                [HANDLE_XY[0], HANDLE_XY[1], APPROACH_Z],
                duration=3.0,
                gripper_action="open", gripper_start_fraction=0.3,
                skill_description="approach above handle",
            )

            # Descent 시도
            skills.move_to_position(
                [HANDLE_XY[0], HANDLE_XY[1], HANDLE_Z],
                duration=5.0,
                **kwargs,
                skill_description=f"descend to handle ({label})",
            )

            # 실제 도달 z 기록
            try:
                _, _, ee = skills._get_current_state()
                z_actual = float(ee[2])
                z_err_mm = (z_actual - HANDLE_Z) * 1000
                results.append((label, z_actual, z_err_mm))
                print(f"  → actual z = {z_actual*100:.2f}cm, err = {z_err_mm:+.1f}mm")
            except Exception as e:
                print(f"  → could not read state: {e}")
                results.append((label, None, None))

        print(f"\n{'='*70}")
        print(f"  SUMMARY (target z = {HANDLE_Z*100:.1f}cm)")
        print(f"{'='*70}")
        print(f"  {'Test':50s} | {'actual z':>10s} | {'err':>10s}")
        print(f"  {'-'*50:50s}-+-{'-'*10:>10s}-+-{'-'*10:>10s}")
        for label, z, err in results:
            if z is not None:
                print(f"  {label:50s} | {z*100:>8.2f}cm | {err:>+8.1f}mm")
            else:
                print(f"  {label:50s} | {'(failed)':>10s} | {'-':>10s}")

        skills.move_to_initial_state()
        skills.move_to_free_state()

    finally:
        skills.disconnect()


if __name__ == "__main__":
    main()
