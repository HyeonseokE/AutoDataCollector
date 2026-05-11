"""
push_object — 밀기 스킬 (move_linear 의 semantic wrapper)

물체나 핸들을 직선으로 미는 스킬. 그리퍼 상태는 caller 가 결정 (스킬은 안 건드림).
내부 핵심은 move_linear.

3-단계 구성:
    1. Approach (descent) — pick 처럼 push_height 보다 더 깊게 명령 (모터 saturation
       보상). over-descent 후 실제 도달 z 를 push 동안 유지.
    2. Linear push — 정해진 방향/거리로 직선 이동 (gripper 변화 없음).
    3. Retreat — push 반대방향으로 살짝 빠진 뒤 approach_height 로 상승.

전제 조건 (caller 책임):
    - 사용 의도에 맞는 그리퍼 상태 (block 밀기: closed / 핸들 밀기: open)
    - EE 가 이미 start 위 approach_height 에 있어야 함

                    approach_height (외부)
    ─────●                                          ●
         │ Step 1                            Step 3 │
         │  하강                                상승 │
         ●────→──●─────────────────────────→──●──→──●  z = actual descent z
    pre-contact  start                       end  clear
                 (Step 2: linear push)

Usage:
    # 블럭 밀기 (gripper closed)
    skills.gripper_close(...)
    skills.move_to_position([start[0], start[1], 0.20], ...)
    skills.execute_push(start, end, push_height=0.01, ...)

    # 서랍 닫기 (gripper open, run_up 없음)
    skills.move_to_position([handle[0], handle[1], 0.20], gripper_action="open", ...)
    skills.execute_push(handle, close_end, push_height=handle[2], run_up_distance=0.0, ...)
    skills.move_to_initial_state()
"""

from typing import List, Optional, Union

import numpy as np

from skills.move_linear import move_linear


def push_object(
    skills,
    start_position: Union[List[float], np.ndarray],
    end_position: Union[List[float], np.ndarray],
    push_height: float = 0.01,
    run_up_distance: float = 0.04,
    approach_height: float = 0.20,
    duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
    compliance_torque: Optional[int] = None,
) -> bool:
    """
    물체를 접촉하여 직선으로 밀기.

    EE가 이미 start 위 approach_height에 있고 gripper가 닫혀 있다고 가정.
    pre-contact로 하강 → run-up + 밀기 직선 이동 → approach_height로 복귀.

    Args:
        skills: LeRobotSkills 인스턴스 (connect() 완료 상태)

        start_position: 물체 접촉 위치 [x, y, z] (단위: meters)
                        - interaction point (예: 물체 왼쪽 가장자리)
                        - z값은 물체 높이 참조용 (실제 밀기 높이는 push_height)

        end_position: 밀기 끝 위치 [x, y, z] (단위: meters)
                      - xy 평면에서 밀기 방향과 거리를 결정
                      - z값은 사용되지 않음 (push_height가 우선)

        push_height: 밀기 시 EE 높이 (단위: meters). default=0.01 (1cm)
                     - 물체 높이의 1/3~1/2 권장 (예: 3cm 물체 → push_height=0.01)

        run_up_distance: pre-contact offset 거리 (단위: meters). default=0.03 (3cm)
                         - start에서 push 반대 방향으로 이 거리만큼 뒤로 뺀 위치에서 하강
                         - run-up 구간을 통해 물체 접촉 전 안정적 이동 확보

        approach_height: 복귀 높이 (단위: meters). default=0.20 (20cm)
                         - Step 3에서 end 위 이 높이로 상승

        duration: 밀기 구간 이동 시간 (단위: 초). default=None
                  - None: 거리 기반 자동 계산 (5cm/s, 최소 2초)

        object_name: 밀 대상 물체 이름 (선택). default=None

        skill_description: 스킬 동작 설명 (선택). default=None

    Returns:
        bool: True면 밀기 동작 성공, False면 실패
    """
    start_pos = np.array(start_position, dtype=float)
    end_pos = np.array(end_position, dtype=float)

    # push_height 최소값 보장 (IK 안정성)
    MIN_PUSH_HEIGHT = 0.01  # 1cm
    if push_height < MIN_PUSH_HEIGHT:
        skills._log(f"WARNING: push_height {push_height*100:.1f}mm < {MIN_PUSH_HEIGHT*100:.0f}mm, "
                     f"clamping to {MIN_PUSH_HEIGHT*100:.0f}mm")
        push_height = MIN_PUSH_HEIGHT

    # push 방향 벡터 계산
    push_vec = end_pos[:2] - start_pos[:2]
    push_distance = np.linalg.norm(push_vec)
    if push_distance < 1e-6:
        skills._log("ERROR: start and end positions are the same")
        return False
    push_dir = push_vec / push_distance

    # pre-contact: start에서 push 반대방향으로 run_up_distance만큼 뒤로
    pre_contact = start_pos[:2] - push_dir * run_up_distance

    desc_prefix = f"push {object_name}" if object_name else "push object"

    # Over-descent + sag bypass — pick / pull 과 동일한 패턴.
    # 이유: 이 robot 의 모터는 commanded z 보다 ~1.5cm 위에서 saturate. 그래서
    # push_height 그대로 명령하면 그리퍼가 push_height 위 1.5cm 에서 멈춰
    # 물체 / 핸들에 닿지 않음. pick_offset 만큼 더 깊게 명령해 saturation 을
    # 보상하면 실제 도달 z 가 push_height 부근에 정착.
    DESCENT_OVERSHOOT = float(getattr(skills, "pick_offset", 0.025))
    MIN_PUSH_Z = -0.025   # 책상 아래 -2.5cm 까지 명령 허용 (모터 자체 floor 이 보호)
    descent_target_z = max(push_height - DESCENT_OVERSHOOT, MIN_PUSH_Z)

    skills._log(f"\n{'='*60}")
    skills._log(f"[push_object] {desc_prefix}")
    skills._log(f"  Start (contact): [{start_pos[0]:.3f}, {start_pos[1]:.3f}, {start_pos[2]:.3f}]")
    skills._log(f"  End:             [{end_pos[0]:.3f}, {end_pos[1]:.3f}, {end_pos[2]:.3f}]")
    skills._log(f"  Pre-contact:     [{pre_contact[0]:.3f}, {pre_contact[1]:.3f}]")
    skills._log(f"  Push height: {push_height*100:.1f}cm  (descent commanded {descent_target_z*100:.1f}cm "
                f"= push_height - {DESCENT_OVERSHOOT*100:.1f}cm over-descent)")
    skills._log(f"  Run-up: {run_up_distance*100:.1f}cm,  push distance: {push_distance*100:.1f}cm")
    skills._log(f"{'='*60}")

    # Step 1: pre-contact 위치로 하강 (over-descent + sag bypass).
    skills._log("\n[Step 1] Descend to pre-contact (over-descent, sag bypassed)")
    if not skills.move_to_position(
        position=[pre_contact[0], pre_contact[1], descent_target_z],
        maintain_pitch=False,
        disable_sag=True,
        target_name=object_name,
        skill_description=f"{desc_prefix}: descend to pre-contact",
    ):
        skills._log("ERROR: Failed to descend to pre-contact position")
        return False

    # 실제 도달 z 기록 — 이후 Step 2/3 모두 이 z 에서 진행 (motor saturation 으로
    # 명령 z 와 다를 수 있음. push_height 로 다시 올라가면 핸들 / 물체에서 이탈).
    actual_push_z = push_height
    try:
        _, _, actual_ee = skills._get_current_state()
        actual_push_z = float(actual_ee[2])
        skills._log(f"  Descent z: commanded={descent_target_z*100:.1f}cm, "
                    f"actual={actual_push_z*100:.1f}cm (Step 2/3 use actual)")
    except Exception as e:
        skills._log(f"WARNING: Could not read actual descent z ({e}); using push_height")

    # Step 2: pre-contact → end 직선 밀기 (run-up + push 한 번에). z 는 actual 유지.
    # compliance_torque 가 지정된 경우 Step 2 동안만 토크 제한 (drawer-stop 보호).
    # Step 1 (descent) 와 Step 3 (retreat) 는 full 토크 — 그래야 도달/복귀 가능.
    total_linear_dist = np.linalg.norm(end_pos[:2] - pre_contact)
    if duration is None:
        duration = max(total_linear_dist / 0.05, 2.0)  # 5cm/s, 최소 2초

    arm_motor_ids = [1, 2, 3, 4, 5]   # gripper(6) 제외
    DEFAULT_TORQUE_LIMIT = 1000

    if compliance_torque is not None and compliance_torque < DEFAULT_TORQUE_LIMIT:
        skills._log(f"\n[Compliance] Setting motor torque limit to {compliance_torque}/1000 "
                    f"for Step 2 push (drawer-stop protection)")
        try:
            skills.robot.set_torque_limit(compliance_torque, motor_ids=arm_motor_ids)
        except Exception as e:
            skills._log(f"  [Compliance] WARNING: set_torque_limit failed: {e}")

    skills._log("\n[Step 2] Linear push (pre-contact → end, pitch locked)")
    try:
        push_success = move_linear(
            skills,
            start=[pre_contact[0], pre_contact[1], actual_push_z],
            end=[end_pos[0], end_pos[1], actual_push_z],
            duration=duration,
            maintain_pitch=True,
            target_name=object_name,
            skill_description=skill_description or f"{desc_prefix}: pushing",
        )
    finally:
        if compliance_torque is not None and compliance_torque < DEFAULT_TORQUE_LIMIT:
            skills._log(f"[Compliance] Restoring motor torque limit to {DEFAULT_TORQUE_LIMIT}/1000 "
                        f"before Step 3 retreat")
            try:
                skills.robot.set_torque_limit(DEFAULT_TORQUE_LIMIT, motor_ids=arm_motor_ids)
            except Exception as e:
                skills._log(f"  [Compliance] WARNING: restore set_torque_limit failed: {e}")

    if not push_success:
        skills._log("WARNING: Push move_linear did not fully converge")

    # Step 3: push 반대 방향으로 run_up 만큼 되돌아가서 물체에서 이탈 (run_up_distance=0
    # 이면 사실상 stay-in-place, 다음 lift 만 수행).
    retract_pos = end_pos[:2] - push_dir * run_up_distance
    if run_up_distance > 1e-6:
        skills._log(f"\n[Step 3a] Retract opposite push direction (-{run_up_distance*100:.1f}cm)")
        skills.move_to_position(
            position=[retract_pos[0], retract_pos[1], actual_push_z],
            maintain_pitch=False,
            target_name=object_name,
            skill_description=f"{desc_prefix}: retract away from contact",
        )

    # Step 3b: approach_height 로 상승 (복귀)
    skills._log(f"\n[Step 3b] Retreat to approach height ({approach_height*100:.0f}cm)")
    skills.move_to_position(
        position=[retract_pos[0], retract_pos[1], approach_height],
        maintain_pitch=False,
        target_name=object_name,
        skill_description=f"{desc_prefix}: retreat after push",
    )

    skills._log(f"\n[push_object] Complete (push_success={push_success})")
    return push_success


# ======================================================================
# 모듈 상수 — drawer/door CLOSE wrapper
# ======================================================================
CLOSE_OVERSHOOT = 0.03   # 입력 distance 위에 추가로 +3cm 더 미는 여유 (서랍 끝까지 닫힘 보장)


def push_object_handle_close(
    skills,
    start_position,
    distance: float,
    duration: Optional[float] = None,
    object_name: Optional[str] = None,
    skill_description: Optional[str] = None,
    verification_question: Optional[str] = None,
) -> bool:
    """
    핸들을 +x 방향으로 **`distance + 3cm`** 만큼 push (drawer/door CLOSE 전용).
    **순응 제어 적용** — Step 2 (linear push) 동안만 torque 제한, Step 1 (descent)
    + Step 3 (retreat) 는 full 토크. 그래야 retreat 시 모터가 위로 못 올라오는
    문제 없음. 서랍 끝 stop 에 닿을 때만 토크 제한으로 손상 방지.

    push_object() 의 thin wrapper:
      actual_push_distance = distance + CLOSE_OVERSHOOT (+3cm 여유)
      end                  = start + (actual_push_distance, 0, 0)   = +x
      push_height          = start.z                                 (핸들 z)
      run_up_distance      = 0                                       (사전접근 없음)
      compliance_torque    = HANDLE_PUSH_TORQUE_LIMIT (500/1000) ← Step 2 만

    distance 는 caller 가 알고 있는 "열린 거리" — close 는 그것보다 2cm 더 밀어
    서랍이 완전히 닫히도록 보장. (Caller 가 ±2cm 이미 더해서 줄 필요 없음.)

    Pre: caller 가 gripper OPEN 상태로 approach_height 까지 이동시킨 상태.
         (open jaw 가 핸들 안쪽에서 밀어내는 동작)

    Args:
        skills: LeRobotSkills 인스턴스
        start_position: 현재 (open 상태) 핸들 위치 [x, y, z] meters
        distance: 닫을 거리 (meters, 일반적으로 forward 의 open 거리와 동일).
                  내부적으로 +3cm 더해서 push 함.
        duration: push 시간 (None=auto)
        object_name / skill_description / verification_question: 기록용
    """
    HANDLE_PUSH_TORQUE_LIMIT = 500   # 0-1000, Step 2 (linear push) 동안만 적용
                                     # 너무 낮으면 motor 가 못 푸시 / 못 holds; 500 이 적정선

    start_pos = np.array(start_position, dtype=float)
    actual_push_distance = float(distance) + CLOSE_OVERSHOOT
    end_pos = [float(start_pos[0]) + actual_push_distance, float(start_pos[1]), float(start_pos[2])]
    skills._log(f"\n[execute_push close] input distance={float(distance)*100:.1f}cm, "
                f"+overshoot {CLOSE_OVERSHOOT*100:.0f}cm = actual push {actual_push_distance*100:.1f}cm")

    # compliance_torque=HANDLE_PUSH_TORQUE_LIMIT 를 push_object 에 위임.
    # push_object 가 Step 2 (linear push) 직전에 set, 직후 restore — Step 1 (descent)
    # 와 Step 3 (retreat) 는 full 토크로 진행되어 도달/복귀 보장.
    return push_object(
        skills,
        start_position=start_pos,
        end_position=end_pos,
        push_height=float(start_pos[2]),
        run_up_distance=0.0,
        duration=duration,
        object_name=object_name,
        skill_description=skill_description,
        compliance_torque=HANDLE_PUSH_TORQUE_LIMIT,
    )


if __name__ == "__main__":
    from skills.skills_lerobot import LeRobotSkills

    skills = LeRobotSkills(
        robot_config="robot_configs/robot/so101_robot3.yaml",
        frame="world",
    )
    if not skills.connect():
        print("Robot connection failed")
        exit(1)

    try:
        skills.move_to_initial_state()

        start_pos = [0.20, -0.05, 0.03]
        end_pos = [0.20, 0.05, 0.03]

        # 외부: gripper 닫기
        skills.gripper_close(skill_description="test: close gripper for push")

        # 외부: start 위 approach 위치로 이동
        skills.move_to_position(
            [start_pos[0], start_pos[1], 0.20],
            skill_description="test: approach above start",
        )

        # Push (하강 + run-up + 밀기 + 복귀 포함)
        success = push_object(
            skills,
            start_position=start_pos,
            end_position=end_pos,
            push_height=0.01,
            object_name="test block",
            skill_description="test: push block along y-axis",
        )
        print(f"Result: {'success' if success else 'failed'}")

        skills.move_to_free_state()
    finally:
        skills.disconnect()
