#!/usr/bin/env python3
"""
Gravity Sag Compensation Test

Moves to the same positions that showed large z-errors in the log,
then reports actual errors with compensation enabled.
"""
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from skills.skills_lerobot import LeRobotSkills

# Test positions from the log (base_link frame, z=200mm approach height)
TEST_POSITIONS = [
    {"pos": [0.288, -0.133, 0.200], "label": "pos1 (reach=0.317)"},
    {"pos": [0.292,  0.149, 0.200], "label": "pos2 (reach=0.328)"},
    {"pos": [0.339, -0.000, 0.200], "label": "pos3 (reach=0.339)"},
    {"pos": [0.348, -0.126, 0.200], "label": "pos4 (reach=0.370)"},
    {"pos": [0.303,  0.139, 0.200], "label": "pos5 (reach=0.333)"},
    {"pos": [0.178, -0.129, 0.200], "label": "pos6 (reach=0.220)"},
]


def main():
    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot2.yaml",
        frame="base_link",
        verbose=True,
    )
    skills.connect()

    print("\n" + "=" * 60)
    print("Gravity Sag Compensation Test")
    print("=" * 60)

    if skills.gravity_sag is not None:
        info = skills.gravity_sag.get_info()
        print(f"Sag compensator: gain={info['gain']}, power={info['reach_power']}, "
              f"z_deadzone={info['z_deadzone']}m, max={info['max_offset']*1000:.0f}mm")
    else:
        print("WARNING: No gravity sag compensator loaded!")

    # Move to initial state first
    skills.move_to_initial_state()
    skills.gripper_open()

    results = []

    for i, test in enumerate(TEST_POSITIONS):
        pos = test["pos"]
        label = test["label"]
        print(f"\n{'─' * 50}")
        print(f"Test {i+1}/{len(TEST_POSITIONS)}: {label}")
        print(f"  Target: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        success = skills.move_to_position(position=pos)

        err = skills.last_error
        if err:
            z_err = err.get("z_mm", 0)
            total_err = err.get("total_mm", 0)
            results.append({
                "label": label,
                "target": pos,
                "z_err_mm": z_err,
                "total_err_mm": total_err,
                "success": success,
            })
            print(f"  Result: z_err={z_err:+.1f}mm, total={total_err:.1f}mm, success={success}")

    # Return to initial
    skills.move_to_initial_state()

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Label':>25s}  {'z_err':>8s}  {'total':>8s}  {'ok':>4s}")
    print("-" * 55)
    for r in results:
        print(f"{r['label']:>25s}  {r['z_err_mm']:+8.1f}  {r['total_err_mm']:8.1f}  {'✓' if r['success'] else '✗':>4s}")

    z_errors = [abs(r["z_err_mm"]) for r in results]
    if z_errors:
        print(f"\nz-error: mean={np.mean(z_errors):.1f}mm, max={np.max(z_errors):.1f}mm")

    skills.disconnect()


if __name__ == "__main__":
    main()
