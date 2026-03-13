"""
Multi-Robot Skill Functions

동기화된 멀티로봇 pick/place 스킬.
기존 단일 로봇 스킬을 래핑하여 sync_barrier를 내부에서 호출.
"""

from typing import List, Union, Optional
import numpy as np


def execute_multi_pick_object(
    skills,
    sync_barrier,
    object_position: Union[List[float], np.ndarray],
) -> bool:
    """
    멀티로봇용 pick 스킬 - gripper close 직전에 동기화

    Args:
        skills: LeRobotSkills 인스턴스
        sync_barrier: SyncBarrier 인스턴스
        object_position: 객체 위치 [x, y, z]

    Returns:
        성공 여부
    """
    object_position = np.array(object_position)
    object_height = object_position[2]

    # Calculate pick Z
    MIN_PICK_Z = 0.01
    pick_z = max(object_height - skills.pick_offset, MIN_PICK_Z)
    pick_position = [object_position[0], object_position[1], pick_z]

    skills._log(f"\n[Execute Multi Pick Object]")
    skills._log(f"  Object height: {object_height*100:.1f}cm")
    skills._log(f"  Pick point: {pick_z*100:.1f}cm")

    # Move to pick position
    move_success = skills.move_to_position(pick_position)
    if not move_success:
        print("Warning: Failed to reach pick position, but continuing for sync")

    # === SYNC: 두 로봇 모두 pick 위치에 도달한 후 gripper close ===
    # NOTE: Always call sync_barrier.wait() even on failure to prevent deadlock
    sync_barrier.wait("pick_ready")

    # Close gripper (동기화됨)
    skills.gripper_close()

    if not move_success:
        return False

    # Store state for place operation
    skills._pick_z = pick_z
    _, current_joints, _ = skills._get_current_state()
    skills._saved_pitch = skills.kinematics.get_gripper_pitch(current_joints)
    skills._log(f"  Saved pitch: {np.degrees(skills._saved_pitch):.1f}°")

    skills._log("[Execute Multi Pick Object] Complete")
    return True


def execute_multi_place_object(
    skills,
    sync_barrier,
    place_position: Union[List[float], np.ndarray],
    is_table: bool = True,
) -> bool:
    """
    멀티로봇용 place 스킬 - gripper open 직전에 동기화

    Args:
        skills: LeRobotSkills 인스턴스
        sync_barrier: SyncBarrier 인스턴스
        place_position: 목표 위치 [x, y, z]
        is_table: 테이블에 놓을지 여부

    Returns:
        성공 여부
    """
    place_position = np.array(place_position)
    target_surface_height = 0.0 if is_table else place_position[2]

    # Get saved pick_z
    pick_z = getattr(skills, '_pick_z', skills.pick_offset)

    # Calculate place Z
    place_z = target_surface_height + pick_z
    final_position = [place_position[0], place_position[1], place_z]

    # Get saved pitch
    saved_pitch = getattr(skills, '_saved_pitch', None)

    skills._log(f"\n[Execute Multi Place Object]")
    skills._log(f"  Target surface: z={target_surface_height*100:.1f}cm")
    skills._log(f"  Place point: z={place_z*100:.1f}cm")
    if saved_pitch is not None:
        skills._log(f"  Restoring pitch: {np.degrees(saved_pitch):.1f}°")

    # Move to place position
    move_success = skills.move_to_position(final_position,
                                           target_pitch=saved_pitch)
    if not move_success:
        print("Warning: Failed to reach place position, but continuing for sync")

    # === SYNC: 두 로봇 모두 place 위치에 도달한 후 gripper open ===
    # NOTE: Always call sync_barrier.wait() even on failure to prevent deadlock
    sync_barrier.wait("place_ready")

    # Open gripper (동기화됨)
    skills.gripper_open()

    if not move_success:
        return False

    # Clear saved state
    skills._pick_z = None
    skills._saved_pitch = None

    skills._log("[Execute Multi Place Object] Complete")
    return True


def execute_pause_for_sync(
    skills,
    sync_barrier,
    after_skill: str,
) -> bool:
    """
    스킬 완료 시점에서 동기화 대기

    다른 로봇이 같은 스킬을 완료할 때까지 대기합니다.

    Args:
        skills: LeRobotSkills 인스턴스
        sync_barrier: SyncBarrier 인스턴스
        after_skill: 완료된 스킬 이름 (실제 함수명 사용)
            - "execute_multi_pick_object"
            - "execute_multi_place_object"
            - "move_to_position"
            - "move_to_initial_state"
            - "gripper_open" / "gripper_close"
            - 등

    Returns:
        True (항상 성공)

    Example:
        # 두 로봇 모두 pick 완료 후 동기화
        execute_multi_pick_object(skills, sync_barrier, object_pos)
        execute_pause_for_sync(skills, sync_barrier, "execute_multi_pick_object")

        # 두 로봇 모두 approach 위치 도달 후 동기화
        skills.move_to_position(approach_pos)
        execute_pause_for_sync(skills, sync_barrier, "move_to_position")

        # 두 로봇 모두 gripper 열기 완료 후 동기화
        skills.gripper_open()
        execute_pause_for_sync(skills, sync_barrier, "gripper_open")
    """
    skills._log(f"\n[Pause for Sync: after {after_skill}]")
    skills._log(f"  Waiting for other robot...")

    sync_barrier.wait(after_skill)

    skills._log(f"[Sync Complete: {after_skill}]")
    return True
