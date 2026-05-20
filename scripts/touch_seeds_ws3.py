"""Touch each of 10 seed positions with EE on ws3/robot4.

Sequence: initial → seed1 → initial → seed2 → ... → seed10 → initial → free.
Target object per seed: "red block" (primary manipulable). If a seed lacks z,
use 0.0 (table floor) per user specification.

Run from repo root:
    python scripts/touch_seeds_ws3.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from skills.skills_lerobot import LeRobotSkills

SESSION_DIR = REPO / "results" / "session_20260519_225321"
ROBOT_CONFIG = "robot_configs/robot/so101_robot4.yaml"
TARGET_KEY = "red block"
# User-specified: touch the actual floor regardless of what the seed records.
FORCE_FLOOR_Z = 0.0
# Slower duration → gives Hold phase more time to overcome friction.
MOVEMENT_DURATION_S = 8.0

# --- Floor-contact compensation ---
# Empirically z always settles +6~10mm above commanded value (controller plateau
# + gravity sag residue). To actually touch table (actual z = 0), command z
# slightly BELOW table so the plateau lands on z=0. -0.008m chosen so worst-case
# overshoot (>10mm bias) lands at most 2mm below table (fingertip pinch can
# compress / arm compliance absorbs).
Z_FLOOR_OVERSHOOT = -0.008

# --- Radial overshoot ONLY for far seeds ---
# Logs show xy plateau is small (±3mm) for most reach values, but reach ≥ 0.30
# has a systematic -9mm radial undershoot (elbow_flex deadband). Apply +0.009m
# radial overshoot only when reach exceeds threshold — leaves near/medium seeds
# untouched.
FAR_REACH_THRESHOLD = 0.30
FAR_RADIAL_OVERSHOOT = 0.009

# --- Closed-loop refinement ---
# After each move, read actual EE and re-issue command nudged toward target.
# Damping < 1 avoids overshoot from controller non-linearity (z changes often
# push pitch selection in IK, which couples back into xy plateau). Clamps keep
# cmd within safe envelope.
REFINE_TOL_MM = 3.0
MAX_REFINE_ITERS = 3
REFINE_DAMPING = 0.5            # cmd += DAMPING * residual_error
Z_CMD_MIN = -0.020              # never push cmd z below -2cm (avoid IK pitch switch)
RADIAL_CMD_EXTRA_MAX = 0.020    # cumulative radial push cap from initial cmd


def load_seed_xyz(seed_dir: Path) -> tuple[float, float, float] | None:
    """Return (x, y, z) for the seed's target object. z is always forced to
    ``FORCE_FLOOR_Z`` per user spec ("바닥을 찍어줬으면 좋겠어")."""
    fp = seed_dir / "seed_positions.json"
    if not fp.exists():
        return None
    data = json.loads(fp.read_text())
    info = data.get("positions", {}).get(TARGET_KEY)
    if not info:
        return None
    pos = info.get("position") or info.get("points", {}).get("grasp center")
    if not pos:
        return None
    return (float(pos[0]), float(pos[1]), FORCE_FLOOR_Z)


def main() -> int:
    targets: list[tuple[int, tuple[float, float, float]]] = []
    for i in range(1, 11):
        sdir = SESSION_DIR / f"seed_{i:02d}_setup"
        xyz = load_seed_xyz(sdir)
        if xyz is None:
            print(f"  seed_{i:02d}: MISSING — skip")
            continue
        targets.append((i, xyz))

    print(f"Loaded {len(targets)} seed targets from {SESSION_DIR.name}:")
    for i, (x, y, z) in targets:
        print(f"  seed_{i:02d}:  xyz = ({x:+.4f}, {y:+.4f}, {z:+.4f})")
    if not targets:
        print("No targets — aborting.")
        return 1

    print(f"\nConnecting to robot4 ({ROBOT_CONFIG}) ...")
    # movement_duration=MOVEMENT_DURATION_S → 50% slower than default 3.0s.
    # Propagates to move_to_initial_state / move_to_position / move_to_free_state
    # via their default `duration = duration or self.movement_duration`.
    skills = LeRobotSkills(
        robot_config=ROBOT_CONFIG,
        frame="base_link",
        movement_duration=MOVEMENT_DURATION_S,
    )
    skills.connect()
    try:
        print(f"\n--- Home (initial state, {MOVEMENT_DURATION_S:.1f}s) ---")
        skills.move_to_initial_state(
            skill_description="Home before touring seeds",
            verification_question=None,
        )
        for i, xyz in targets:
            x_t, y_t, z_t = xyz
            # Initial command — heuristic overshoot to land near target.
            reach = math.hypot(x_t, y_t)
            radial_push = FAR_RADIAL_OVERSHOOT if reach >= FAR_REACH_THRESHOLD else 0.0
            scale = 1.0 + (radial_push / reach) if reach > 1e-3 else 1.0
            x_cmd, y_cmd = x_t * scale, y_t * scale
            z_cmd = z_t + Z_FLOOR_OVERSHOOT

            target = (x_t, y_t, z_t)
            print(
                f"\n--- seed_{i:02d} → tgt=({x_t:+.4f},{y_t:+.4f},{z_t:+.4f}) "
                f"reach={reach:.3f}m  closed-loop start ---"
            )

            # Closed-loop refinement: damped (avoids overshoot from controller
            # non-linearity) + z/radial clamps (avoid IK pitch switching).
            x_cmd_init, y_cmd_init = x_cmd, y_cmd
            success = False
            best_err = float("inf")
            best_actual: tuple[float, float, float] | None = None
            for iteration in range(MAX_REFINE_ITERS + 1):
                tag = "initial" if iteration == 0 else f"refine #{iteration}"
                print(
                    f"  [{tag}] cmd=({x_cmd:+.4f},{y_cmd:+.4f},{z_cmd:+.4f})"
                )
                ok = skills.move_to_position(
                    [x_cmd, y_cmd, z_cmd],
                    target_name=f"seed_{i:02d}",
                    skill_description=f"Touch seed_{i:02d} ({tag})",
                    is_transit=False,
                    xy_lead_descent=True,
                )
                if not ok:
                    print(f"  [WARN] seed_{i:02d} {tag} unreachable — stop refining")
                    break
                _, _, ee_actual = skills._get_current_state()
                dx = target[0] - float(ee_actual[0])
                dy = target[1] - float(ee_actual[1])
                dz = target[2] - float(ee_actual[2])
                err_mm = 1000.0 * math.sqrt(dx * dx + dy * dy + dz * dz)
                print(
                    f"  [{tag}] actual=({ee_actual[0]:+.4f},{ee_actual[1]:+.4f},{ee_actual[2]:+.4f})"
                    f"  Δ from target=({dx*1000:+.2f}, {dy*1000:+.2f}, {dz*1000:+.2f})mm"
                    f"  |Δ|={err_mm:.2f}mm"
                )
                if err_mm < best_err:
                    best_err = err_mm
                    best_actual = (float(ee_actual[0]), float(ee_actual[1]), float(ee_actual[2]))
                if err_mm <= REFINE_TOL_MM:
                    print(f"  [{tag}] WITHIN TOLERANCE ({REFINE_TOL_MM}mm) ✓")
                    success = True
                    break
                if iteration == MAX_REFINE_ITERS:
                    print(f"  [{tag}] retries exhausted ({MAX_REFINE_ITERS})")
                    break
                # Damped nudge — half of residual per iteration to avoid overshoot.
                x_cmd_new = x_cmd + REFINE_DAMPING * dx
                y_cmd_new = y_cmd + REFINE_DAMPING * dy
                z_cmd_new = z_cmd + REFINE_DAMPING * dz
                # Clamps: prevent IK pitch flip from extreme z; cap radial push.
                z_cmd_new = max(z_cmd_new, Z_CMD_MIN)
                cum_radial = math.hypot(x_cmd_new, y_cmd_new) - math.hypot(x_cmd_init, y_cmd_init)
                if abs(cum_radial) > RADIAL_CMD_EXTRA_MAX:
                    # Scale back toward initial cmd along the radial direction.
                    init_r = math.hypot(x_cmd_init, y_cmd_init)
                    if init_r > 1e-3:
                        capped_r = init_r + math.copysign(RADIAL_CMD_EXTRA_MAX, cum_radial)
                        s = capped_r / math.hypot(x_cmd_new, y_cmd_new)
                        x_cmd_new *= s
                        y_cmd_new *= s
                x_cmd, y_cmd, z_cmd = x_cmd_new, y_cmd_new, z_cmd_new
            if not success and best_actual is not None:
                print(f"  [seed_{i:02d}] best residual = {best_err:.2f}mm "
                      f"(actual=({best_actual[0]:+.4f},{best_actual[1]:+.4f},{best_actual[2]:+.4f}))")
            print(f"--- back to home ---")
            skills.move_to_initial_state(
                skill_description=f"Home after seed_{i:02d}",
                verification_question=None,
            )
        print("\n--- Move to free state ---")
        skills.move_to_free_state(skill_description="Park")
    finally:
        skills.disconnect()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
