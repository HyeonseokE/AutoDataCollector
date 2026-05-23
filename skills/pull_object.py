"""
pull_object — 끌기 스킬 (drawer / door / articulated)

핸들/손잡이를 그래스핑한 뒤 직선으로 끌어 articulated 객체를 여는 스킬.
push_object 와 대칭 구조. 내부적으로 move_linear 를 핵심 동작으로 사용.

전제 조건 (LLM 코드에서 외부 처리):
    - gripper 가 이미 열려 있어야 함 (gripper_action="open" 으로 approach)
    - EE 가 이미 start 위 approach_height 에 있어야 함 (move_to_position)

동작 시퀀스:
    1. move_to_position  — start 로 하강 (z=start.z, IK 가 grasp 친화 pitch 선택)
    2. gripper_close     — 핸들 그래스핑
    3. move_linear       — start → end 직선 끌기 (z 고정, maintain_pitch=True)
    4. gripper_open      — 핸들 release (open_after=True 시)

    Retreat 단계는 의도적으로 없음. caller 가 이어서 move_to_initial_state()
    또는 다음 skill 을 호출해 핸들 영역에서 빠져나가는 게 표준 패턴.

                    approach_height (외부)
    ─────●
         │ Step 1
         │  하강
         ●─grasp─●─────→──────────────────────●─open  z = start.z
        (Step 2  (Step 3: linear pull            (Step 4
         close)   maintain_pitch=True)             open — 끝)

Usage:
    from skills.skills_lerobot import LeRobotSkills
    from skills.pull_object import pull_object

    skills = LeRobotSkills(robot_config="robot_configs/robot/so101_robot2.yaml")
    skills.connect()

    handle = [0.27, -0.10, 0.10]

    # 외부: start 위 approach 위치로 이동 (gripper open)
    skills.move_to_position(
        [handle[0], handle[1], 0.20],
        gripper_action="open", gripper_start_fraction=0.3,
        skill_description="approach drawer handle and open gripper",
    )

    # Pull (-x 방향 17cm 고정; 거리/방향 인자 없음)
    pull_object(
        skills,
        start_position=handle,
        object_name="top drawer",
        skill_description="pull drawer open",
    )

    skills.disconnect()
"""

from typing import List, Optional, Union

import numpy as np

from skills.move_linear import move_linear


# ======================================================================
# 모듈 상수
# ======================================================================
APPROACH_HEIGHT = 0.20      # 복귀 높이 20cm


def pull_object(
    skills,
    start_position: Union[List[float], np.ndarray],
    distance: float,
    duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
    verification_question: Optional[str] = None,
) -> bool:
    """
    핸들을 그래스핑하여 **-x 방향으로 `distance` 만큼** 끌기 (drawer/door OPEN).

    EE 가 이미 start 위 approach_height 에 있고 gripper 가 열려 있다고 가정.
    하강 → grasp → 직선 끌기 (-x, distance) → release → approach_height 로 복귀.

    Args:
        skills: LeRobotSkills 인스턴스
        start_position: 핸들 그래스 위치 [x, y, z] (meters). z 가 끌기 동안 EE 높이.
        distance: 끌기 거리 (meters, e.g., 0.10 = 10cm). 명령에서 추출 (VLM/LLM).
                  방향은 -x 로 고정. 양수 입력 권장.
        duration: 끌기 구간 이동 시간 (초). None=거리 기반 자동.
        object_name / skill_description / verification_question: 기록용

    Returns:
        bool: True 면 끌기 동작 성공
    """
    start_pos = np.array(start_position, dtype=float)
    # Apply radial xy offset (same compensation as execute_pick_object): push
    # the grasp xy outward by skills.pick_xy_offset along the base→handle
    # direction so the Hold-phase radial undershoot lands on the handle, not
    # short of it. No-op when offset = 0. End/pull_distance unchanged.
    _sx, _sy = skills._apply_pick_xy_offset(start_pos[0], start_pos[1])
    start_pos = np.array([_sx, _sy, start_pos[2]], dtype=float)
    # end = start + (-distance, 0, 0)  — -x 방향 (corrected start 기준)
    end_pos = np.array([start_pos[0] - float(distance), start_pos[1], start_pos[2]], dtype=float)
    approach_height = APPROACH_HEIGHT
    open_after = True   # drawer/door open 케이스 — 항상 release

    # 끌기 방향/거리
    pull_vec = end_pos[:2] - start_pos[:2]
    pull_distance = float(np.linalg.norm(pull_vec))
    if pull_distance < 1e-6:
        skills._log("ERROR: pull start and end positions are the same")
        return False

    # 끌기 z 는 start.z 로 고정 (핸들이 그리퍼에서 빠지지 않게).
    # end.z 는 무시되지만 너무 차이나면 경고.
    pull_z = float(start_pos[2])
    if abs(end_pos[2] - pull_z) > 0.01:
        skills._log(
            f"NOTE: end.z ({end_pos[2]*100:.1f}cm) ignored; pull stays at "
            f"start.z ({pull_z*100:.1f}cm) to keep handle in gripper"
        )

    desc_prefix = f"pull {object_name}" if object_name else "pull object"

    skills._log(f"\n{'='*60}")
    skills._log(f"[pull_object] {desc_prefix}")
    skills._log(f"  Start (grasp):   [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    skills._log(f"  End:             [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {pull_z:.3f}]")
    skills._log(f"  Pull distance: {pull_distance*100:.1f}cm at z={pull_z*100:.1f}cm")
    skills._log(f"{'='*60}")

    # Step 1: start 보다 한참 아래로 하강 명령 (over-descent).
    # - 이유: SO-101 모터는 핸들 z 영역에서 commanded z 보다 ~1.5cm 위에서
    #   saturate 함. handle z 그대로 명령하면 그리퍼가 핸들 위 1.5cm 에서
    #   닫혀 공기를 잡음. execute_pick_object 가 항상 잘 grasp 한 이유는
    #   pick_offset (2.5cm) + z_offset (-0.5cm) = 3cm 추가 하강 명령이
    #   이 saturation 을 자연스럽게 보상하기 때문.
    # - 같은 보상을 pull 에도 적용 — skills.pick_offset (default 2.5cm) 만큼
    #   더 내려가게 명령. 모터 saturation 으로 실제 도달은 핸들 z 부근에서
    #   멈춤 → 그리퍼가 핸들에 정확히 닿음.
    # - duration 5s — 수렴 시간 충분히
    # - disable_sag=True — payload 없는 descent 에서 base_sag over-correction 회피
    # over-descent = pick_offset (≈2.5cm) + 1cm 추가 (drawer/door 핸들은
    # block grasp 보다 모터 saturation 더 가팔라 1cm 더 깊게 명령).
    DESCENT_OVERSHOOT = float(getattr(skills, "pick_offset", 0.025)) + 0.01
    descent_target_z = pull_z - DESCENT_OVERSHOOT
    DESCENT_DURATION = 5.0
    skills._log("\n[Step 1] Descend to grasp position (sag bypassed, over-descent)")
    skills._log(f"  handle z = {pull_z*100:.1f}cm, descent commanded z = "
                f"{descent_target_z*100:.1f}cm  (over-descent {DESCENT_OVERSHOOT*100:.1f}cm)")
    if not skills.move_to_position(
        position=[start_pos[0], start_pos[1], descent_target_z],
        duration=DESCENT_DURATION,
        maintain_pitch=False,
        disable_sag=True,
        target_name=object_name,
        skill_description=f"{desc_prefix}: descend to grasp",
        is_transit=False,
    ):
        skills._log("ERROR: Failed to descend to grasp position (IK / workspace)")
        return False

    # 실제 도달 z 확인. 핸들 z 기준 ±1.5cm 안이면 OK (그리퍼 jaw range).
    # +1.5cm 초과: 핸들 위 공기를 잡을 위험 → abort.
    GRASP_Z_TOLERANCE_ABOVE = 0.015  # 1.5cm 위
    actual_grasp_z = pull_z   # fallback
    try:
        _, _, actual_ee = skills._get_current_state()
        actual_grasp_z = float(actual_ee[2])
        z_err = actual_grasp_z - pull_z
        if z_err > GRASP_Z_TOLERANCE_ABOVE:
            skills._log(
                f"ERROR: Descent saturated above handle. "
                f"handle z={pull_z*100:.1f}cm, actual={actual_grasp_z*100:.1f}cm, "
                f"+{z_err*1000:.1f}mm above handle (>{GRASP_Z_TOLERANCE_ABOVE*1000:.0f}mm). "
                f"Aborting before gripper close (would grasp air)."
            )
            return False
        skills._log(f"  Descent z OK: handle={pull_z*100:.1f}cm, actual={actual_grasp_z*100:.1f}cm "
                    f"(diff {z_err*1000:+.1f}mm)")
    except Exception as e:
        skills._log(f"WARNING: Could not verify descent z ({e}); proceeding anyway")

    # Step 2: 핸들 그래스핑
    skills._log("\n[Step 2] Close gripper on handle")
    skills.gripper_close(
        skill_description=f"{desc_prefix}: grasp handle",
    )

    # Step 3: actual_grasp_z 유지하며 직선 끌기 (Step 1 의 실제 도달 z 그대로 — pull
    # 도중 motor 가 위로 이동하려 하면 핸들이 그리퍼에서 빠짐)
    if duration is None:
        duration = max(pull_distance / 0.05, 2.0)  # 5cm/s, 최소 2초

    skills._log("\n[Step 3] Linear pull (start → end, z held at grasp, pitch locked)")
    pull_success = move_linear(
        skills,
        start=[start_pos[0], start_pos[1], actual_grasp_z],
        end=[end_pos[0], end_pos[1], actual_grasp_z],
        duration=duration,
        maintain_pitch=True,
        target_name=object_name,
        skill_description=skill_description or f"{desc_prefix}: pulling",
    )

    if not pull_success:
        skills._log("WARNING: Pull move_linear did not fully converge")

    # Step 4: gripper release (open_after=True 시)
    if open_after:
        skills._log("\n[Step 4] Open gripper to release handle")
        skills.gripper_open(
            skill_description=f"{desc_prefix}: release handle",
        )

    # Retreat 단계 없음 — caller 가 다음에 move_to_initial_state 를 호출하는
    # 게 표준 패턴이라 pull_object 내부에서 중복 retreat 을 만들지 않는다.
    # 기존 approach_height retreat 은 Phase1 perturbation 이 핸들 z 영역으로
    # subgoal 을 끌어내려 그리퍼가 다시 걸리는 문제가 있었음.

    skills._log(f"\n[pull_object] Complete (pull_success={pull_success})")
    return pull_success


if __name__ == "__main__":
    from skills.skills_lerobot import LeRobotSkills

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot2.yaml",
        frame="world",
    )
    if not skills.connect():
        print("Robot connection failed")
        exit(1)

    try:
        skills.move_to_initial_state()

        start_pos = [0.27, -0.10, 0.10]                      # drawer handle
        end_pos = [start_pos[0] - 0.08, start_pos[1], start_pos[2]]   # 8cm -x

        # 외부: start 위 approach 위치로 이동 (gripper open)
        skills.move_to_position(
            [start_pos[0], start_pos[1], 0.20],
            gripper_action="open", gripper_start_fraction=0.3,
            skill_description="test: approach drawer handle",
        )

        # Pull (-x 방향, 명시 distance)
        success = pull_object(
            skills,
            start_position=start_pos,
            distance=0.10,   # 10cm
            object_name="test drawer",
            skill_description="test: pull drawer open",
        )
        print(f"Result: {'success' if success else 'failed'}")

        skills.move_to_free_state()
    finally:
        skills.disconnect()
