"""
place_lid — lid 안착 + 4방향 settle wiggle 스킬

일반 execute_place_object와 달리, lid를 컨테이너 위에 내려놓은 뒤 그리퍼를
닫은 채로 ±x / ±y 4 방향으로 1cm 씩 왕복하며 (out → back to descend)
lid 를 rim 에 안정적으로 안착시킨다. 모든 방향 왕복 끝난 뒤 그리퍼 release.
순 변위는 0 — release 지점은 descend xy 그대로.

도입 배경:
    pot lid 검출/캘리브레이션 편향 + rim 마찰로 release 직후 lid 가 미세히
    틀어진 채로 안착되는 경향이 있다. 한 방향 drag (이전 ±x 만) 대신 4 방향
    settle wiggle 로 rim 안쪽에 lid 가 자리 잡도록 유도.

전제 조건:
    - LLM 생성 코드가 이미 pick + place_approach (target 위 approach_height) 까지 수행
    - 그리퍼는 lid를 잡고 있는 상태 (closed)
    - execute_pick_object 단계에서 _pick_z, _saved_pitch가 저장되어 있음

동작 시퀀스:
    1. descend       — place_position 의 (x, y) 위에서 (surface_z + pick_z + EXTRA) 로 하강 (saved pitch 복원)
    2. wiggle ±x ±y — 각 방향 1cm 왕복 (out → return), 4 leg × 2 = 8 move_linear
    3. gripper_open  — release at descend xy

    descend (place 위치, gripper closed)
        ↓        ┌─────────────────────────────┐
        ●  ── +x 1cm ──→ ● ── −x 1cm ──→ descend
                  │
                  ├─ +x ── back ── +y ── back ── −x ── back ── −y ── back ─┐
                  ▼                                                          │
                gripper_open at descend  ←──────────────────────────────────┘

Usage (LLM-generated code):
    pot_pos = positions["pot"]["position"]   # pot z는 미리 +offset 적용됨
    skills.execute_place_lid(
        pot_pos,
        gripper_open_ratio=0.7,
        target_name="pot",
        skill_description="Place lid on pot and seat it",
        verification_question="Is the lid centered and seated on the pot?",
    )
"""

from typing import List, Optional, Union

import numpy as np

from skills.move_linear import move_linear


def place_lid(
    skills,
    place_position: Union[List[float], np.ndarray],
    pull_distance: float = 0.02,
    gripper_open_ratio: float = 0.7,
    target_name: Optional[str] = None,
    skill_description: Optional[str] = None,
    verification_question: Optional[str] = None,
) -> bool:
    """
    Lid를 target 위에 내려놓고 4방향 (±x ±y) 1cm 씩 wiggle 후 release.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 + execute_pick_object 선행)
        place_position: 안착할 표면(=container 상단)의 [x, y, z] (meters)
                        - z는 surface 높이로 사용 (execute_place_object의 is_table=False와 동일 의미)
                        - 예: positions["pot"]["position"] (pot z는 사전 offset 보정된 상태)
        pull_distance: (deprecated) 이전 -x drag 거리. 새 4-direction wiggle 에서는
                        무시됨 — wiggle 거리는 내부 상수 WIGGLE_DIST=0.01 로 고정.
                        API 호환 위해 시그니처 유지.
        gripper_open_ratio: release 시 그리퍼 열림 비율 (0.0–1.0). default=0.7
        target_name: 안착 대상 라벨 (subgoal 라벨용). 예: "pot"
        skill_description: 스킬 동작 설명 (recording 라벨)
        verification_question: 검증 질문 (recording 메타)

    Returns:
        True if 안착 + wiggle + release 모두 성공
    """
    place_position = np.array(place_position, dtype=float)
    target_surface_height = float(place_position[2])

    MIN_PLACE_Z = 0.005  # 최소 안착 높이 (테이블 마진)
    # 추가 하강: pick 시 z_offset 보정으로 pick_z 가 평소보다 높게 저장되는데,
    # place_lid 는 그 값을 surface 위에 더하므로 lid 가 rim 위 공중에서 release 됨.
    # surface 에 직접 안착시키려면 그만큼 빼서 보정. (signed, 음수 = 추가 하강)
    PLACE_LID_EXTRA_DESCENT_M = -0.04
    pick_z = getattr(skills, "_pick_z", skills.pick_offset)
    place_z = max(target_surface_height + pick_z + PLACE_LID_EXTRA_DESCENT_M, MIN_PLACE_Z)
    if target_surface_height + pick_z + PLACE_LID_EXTRA_DESCENT_M < MIN_PLACE_Z:
        skills._log(
            f"  [Place Lid Z-Fix] {(target_surface_height + pick_z + PLACE_LID_EXTRA_DESCENT_M)*100:.1f}cm "
            f"< min {MIN_PLACE_Z*100:.0f}cm, clamping to {MIN_PLACE_Z*100:.0f}cm"
        )

    descend_position = [float(place_position[0]), float(place_position[1]), place_z]
    # 4-direction settle wiggle: 내려간 뒤 ±x / ±y 각 1cm 씩 왕복하며 lid 를
    # rim 위에 안착시킨다. 매 단계 descend_position 으로 복귀해 +/-1cm 만
    # 벗어났다 돌아오므로 lid 의 net 변위는 0 — descend xy 그대로 release.
    WIGGLE_DIST = 0.01                          # 한 방향 이동 거리 (1cm)
    wiggle_directions = [
        (+WIGGLE_DIST, 0.0, "+x"),
        (-WIGGLE_DIST, 0.0, "-x"),
        (0.0, +WIGGLE_DIST, "+y"),
        (0.0, -WIGGLE_DIST, "-y"),
    ]

    saved_pitch = getattr(skills, "_saved_pitch", None)

    skills._log("\n[Execute Place Lid]")
    skills._log(f"  Target surface: z={target_surface_height*100:.1f}cm")
    skills._log(
        f"  Place point:    z={place_z*100:.1f}cm "
        f"(pick_z={pick_z*100:.1f}cm, min={MIN_PLACE_Z*100:.0f}cm)"
    )
    skills._log(
        f"  Descend to: [{descend_position[0]:.3f}, {descend_position[1]:.3f}, {descend_position[2]:.3f}]"
    )
    skills._log(
        f"  Settle wiggle: ±x / ±y by {WIGGLE_DIST*100:.0f}cm each, return to descend between every leg"
    )
    if saved_pitch is not None:
        skills._log(f"  Restoring pitch: {np.degrees(saved_pitch):.1f}°")

    # 1. Descend to place position (interaction subgoal — must not be perturbed).
    # xy_lead_descent: continuous Bezier that closes the xy offset left by the
    # perturbed approach transit before z reaches the rim (vertical contact),
    # so the lid lands on the pot centre instead of off-edge (mirrors
    # execute_place_object's descent).
    descend_label = f"place lid on {target_name}" if target_name else "place lid"
    if not skills.move_to_position(
        descend_position,
        target_pitch=saved_pitch,
        target_name=descend_label,
        skill_description=skill_description,
        verification_question=verification_question,
        is_transit=False,
        xy_lead_descent=True,
    ):
        skills._log("Error: Failed to reach lid place position")
        return False

    # 2. Settle wiggle — ±x, ±y 1cm 씩 왕복. 매 leg 가 (descend → offset → descend)
    # 두 move_linear 로 구성돼 net 변위 0 을 보장. pitch 는 유지.
    for dx, dy, dir_label in wiggle_directions:
        offset_position = [
            descend_position[0] + dx,
            descend_position[1] + dy,
            place_z,
        ]
        out_label = (
            f"wiggle lid {dir_label} +{WIGGLE_DIST*100:.0f}cm on {target_name}"
            if target_name else f"wiggle lid {dir_label} +{WIGGLE_DIST*100:.0f}cm"
        )
        out_ok = move_linear(
            skills,
            start=descend_position,
            end=offset_position,
            maintain_pitch=True,
            target_name=target_name,
            skill_description=out_label,
        )
        if not out_ok:
            skills._log(f"WARNING: lid wiggle {dir_label} outward did not fully converge")

        back_label = (
            f"return lid to descend ({dir_label}-back) on {target_name}"
            if target_name else f"return lid to descend ({dir_label}-back)"
        )
        back_ok = move_linear(
            skills,
            start=offset_position,
            end=descend_position,
            maintain_pitch=True,
            target_name=target_name,
            skill_description=back_label,
        )
        if not back_ok:
            skills._log(f"WARNING: lid wiggle {dir_label} return did not fully converge")

    # 3. Open gripper to release lid at the descend position (net 변위 0)
    release_desc = f"release lid on {target_name}" if target_name else "release lid"
    skills.gripper_open(ratio=gripper_open_ratio, skill_description=release_desc)

    # Clear saved state from execute_pick_object
    skills._pick_z = None
    skills._saved_pitch = None

    skills._log("[Execute Place Lid] Complete")
    return True


if __name__ == "__main__":
    # Manual test harness — assumes LLM-generated code performed approach + grasp.
    from skills.skills_lerobot import LeRobotSkills

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot6.yaml",
        frame="world",
    )
    if not skills.connect():
        print("Robot connection failed")
        exit(1)

    try:
        skills.move_to_initial_state()
        # Simulate prior pick: assume lid is already grasped at approach above pot.
        pot_top = [0.36, 0.0, 0.07]
        skills.move_to_position([pot_top[0], pot_top[1], 0.20])
        success = place_lid(
            skills,
            place_position=pot_top,
            pull_distance=0.02,
            target_name="pot",
            skill_description="test: place lid on pot with -x drag",
        )
        print(f"Result: {'success' if success else 'failed'}")
        skills.move_to_position([pot_top[0], pot_top[1], 0.20])
        skills.move_to_initial_state()
        skills.move_to_free_state()
    finally:
        skills.disconnect()
