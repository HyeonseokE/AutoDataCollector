"""
place_lid — lid 안착 + 로봇 방향 끌어당김 스킬

일반 execute_place_object와 달리, lid를 컨테이너 위에 내려놓은 뒤
그리퍼를 닫은 채로 -x 방향(로봇 base 쪽)으로 약간 끌어당겨 lid 위치를 보정한다.
끌어당긴 직후 그리퍼를 열어 lid를 release.

도입 배경:
    pot lid를 pot 위에 placement 할 때, 검출/캘리브레이션 편향으로 lid가
    로봇에서 먼 쪽(+x)으로 약간 어긋나게 안착되는 경향이 있음. 이를 보정하기
    위해 placement 직후 -x 방향으로 약 2cm 끌어와 정중앙에 안착시킨다.

전제 조건:
    - LLM 생성 코드가 이미 pick + place_approach (target 위 approach_height) 까지 수행
    - 그리퍼는 lid를 잡고 있는 상태 (closed)
    - execute_pick_object 단계에서 _pick_z, _saved_pitch가 저장되어 있음

동작 시퀀스:
    1. descend       — place_position의 (x, y) 위에서 (surface_z + pick_z) 까지 하강 (saved pitch 복원)
    2. drag (-x)     — Cartesian 직선으로 -x 방향 pull_distance 만큼 끌어옴 (그리퍼 닫힌 채, pitch 유지)
    3. gripper_open  — release

    descend (place 위치, gripper closed)
        ↓
        ●  (place point: surface + pick_z)
        ↓ -x drag
        ●  (release point: place_x - pull_distance)
        → gripper_open

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
    Lid를 target 위에 내려놓고 -x 방향으로 끌어당긴 뒤 release.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 + execute_pick_object 선행)
        place_position: 안착할 표면(=container 상단)의 [x, y, z] (meters)
                        - z는 surface 높이로 사용 (execute_place_object의 is_table=False와 동일 의미)
                        - 예: positions["pot"]["position"] (pot z는 사전 offset 보정된 상태)
        pull_distance: -x 방향 끌어당김 거리 (meters). default=0.02 (2cm)
        gripper_open_ratio: release 시 그리퍼 열림 비율 (0.0–1.0). default=0.7
        target_name: 안착 대상 라벨 (subgoal 라벨용). 예: "pot"
        skill_description: 스킬 동작 설명 (recording 라벨)
        verification_question: 검증 질문 (recording 메타)

    Returns:
        True if 안착 + drag + release 모두 성공
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
    push_distance = pull_distance               # +x 로 push 하는 거리 (= pull_distance)
    pull_back_distance = 0.005                  # -x 로 되돌아오는 거리 (고정 0.5cm)
    push_position = [descend_position[0] + push_distance, descend_position[1], place_z]
    release_position = [push_position[0] - pull_back_distance, descend_position[1], place_z]

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
        f"  Push +x by {push_distance*100:.1f}cm → "
        f"[{push_position[0]:.3f}, {push_position[1]:.3f}, {push_position[2]:.3f}]"
    )
    skills._log(
        f"  Pull -x by {pull_back_distance*100:.1f}cm → "
        f"[{release_position[0]:.3f}, {release_position[1]:.3f}, {release_position[2]:.3f}]"
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

    # 2. Push +x by push_distance (gripper still closed, lid pushes against rim)
    push_label = (
        f"push lid +{push_distance*100:.0f}cm on {target_name}"
        if target_name else f"push lid +{push_distance*100:.0f}cm"
    )
    push_success = move_linear(
        skills,
        start=descend_position,
        end=push_position,
        maintain_pitch=True,
        target_name=target_name,
        skill_description=push_label,
    )
    if not push_success:
        skills._log("WARNING: lid push did not fully converge")

    # 3. Pull back -x by push_distance/2 (settle lid into rim)
    pull_label = (
        f"pull lid -{pull_back_distance*100:.0f}cm on {target_name}"
        if target_name else f"pull lid -{pull_back_distance*100:.0f}cm"
    )
    pull_success = move_linear(
        skills,
        start=push_position,
        end=release_position,
        maintain_pitch=True,
        target_name=target_name,
        skill_description=pull_label,
    )
    if not pull_success:
        skills._log("WARNING: lid pull-back did not fully converge")

    # 4. Open gripper to release lid at the corrected position
    release_desc = f"release lid on {target_name}" if target_name else "release lid"
    skills.gripper_open(ratio=gripper_open_ratio, skill_description=release_desc)

    # 5. Retreat to approach height — release 위치(=pot 표면 근처)에서 같은 (x, y)
    #    로 수직 상승. release 직후 그리퍼는 연 상태이고 gripper_action 을 주지
    #    않으므로(None) 닫지 않고 open 그대로 이동한다. retreat 없이 바로 initial
    #    로 가면 낮은 z 에서 수평 이동하며 lid/pot 를 스칠 수 있어 이를 방지.
    #    is_transit=False — descend 와 동일하게 perturb 회피(정확히 18 cm).
    #    pitch 는 유지하지 않음(target_pitch=None) — 상승만 하면 되므로 IK 가 자유롭게.
    RETREAT_HEIGHT_M = 0.18
    retreat_position = [release_position[0], release_position[1], RETREAT_HEIGHT_M]
    retreat_desc = (
        f"retreat to approach height above {target_name}"
        if target_name else "retreat to approach height"
    )
    skills._log(
        f"  Retreat to: [{retreat_position[0]:.3f}, {retreat_position[1]:.3f}, "
        f"{retreat_position[2]:.3f}] (gripper open 유지)"
    )
    if not skills.move_to_position(
        retreat_position,
        skill_description=retreat_desc,
        is_transit=False,
    ):
        skills._log("WARNING: retreat to approach height did not fully converge")

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
