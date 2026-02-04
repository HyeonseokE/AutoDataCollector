"""
Multi-Robot Skill Functions

동기화된 멀티로봇 pick/place 스킬.
기존 단일 로봇 스킬을 래핑하여 sync_barrier를 내부에서 호출.
"""

from typing import List, Union, Optional
import numpy as np


def compute_tcp_offset_for_pin(
    skills,
    pin_position_world: Union[List[float], np.ndarray],
    pick_position_world: Union[List[float], np.ndarray],
    base_tcp_offset: List[float] = None,
) -> List[float]:
    """
    PIN 위치에 맞는 tcp_offset 계산 (XY offset만 적용, Z는 무시)

    World 좌표계에서 PIN과 CENTER의 XY offset을 계산하여
    TCP offset에 직접 반영합니다.

    Args:
        skills: LeRobotSkills 인스턴스
        pin_position_world: PIN 위치 [x, y, z] (world frame)
        pick_position_world: PICK 위치 (CENTER) [x, y, z] (world frame)
        base_tcp_offset: 기본 TCP offset (None이면 skills.tcp_offset 사용)

    Returns:
        tcp_offset_override: PIN에 맞춘 tcp offset [x, y, z]

    Example:
        # Pick at CENTER
        execute_multi_pick_object(skills, sync_barrier, center_pos, gripper_offset=...)

        # Compute tcp_offset for PIN alignment
        assembly_tcp_offset = compute_tcp_offset_for_pin(
            skills,
            pin_position_world=gray_circle["position"],
            pick_position_world=pink_part["position"],
        )

        # Place with PIN-aligned TCP
        skills.move_to_position(
            target_pos,
            gripper_offset=...,
            tcp_offset_override=assembly_tcp_offset,
        )
    """
    pin_pos = np.array(pin_position_world)
    pick_pos = np.array(pick_position_world)

    # Calculate pin offset (PIN - CENTER), XY only
    pin_offset_x = pin_pos[0] - pick_pos[0]
    pin_offset_y = pin_pos[1] - pick_pos[1]
    # Z offset is ignored

    skills._log(f"\n[Compute TCP Offset for PIN]")
    skills._log(f"  PIN position: [{pin_pos[0]:.4f}, {pin_pos[1]:.4f}, {pin_pos[2]:.4f}]")
    skills._log(f"  PICK position: [{pick_pos[0]:.4f}, {pick_pos[1]:.4f}, {pick_pos[2]:.4f}]")
    skills._log(f"  XY offset: [{pin_offset_x:.4f}, {pin_offset_y:.4f}]")

    # Base TCP offset
    if base_tcp_offset is None:
        base_tcp_offset = getattr(skills, 'tcp_offset', [-0.02, 0.0, 0.0])

    # assembly_tcp_offset = base + pin_offset (XY only, Z unchanged)
    tcp_offset_override = [
        base_tcp_offset[0] + pin_offset_x,
        base_tcp_offset[1] + pin_offset_y,
        base_tcp_offset[2],  # Z unchanged
    ]

    skills._log(f"  Base TCP offset: {base_tcp_offset}")
    skills._log(f"  TCP offset override: {tcp_offset_override}")
    return tcp_offset_override


def execute_multi_pick_object(
    skills,
    sync_barrier,
    object_position: Union[List[float], np.ndarray],
    gripper_offset: float = 0.0,
) -> bool:
    """
    멀티로봇용 pick 스킬 - gripper close 직전에 동기화

    Args:
        skills: LeRobotSkills 인스턴스
        sync_barrier: SyncBarrier 인스턴스
        object_position: 객체 위치 [x, y, z]
        gripper_offset: 그리퍼 오프셋

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
    move_success = skills.move_to_position(pick_position, gripper_offset=gripper_offset)
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
    gripper_offset: float = 0.0,
    is_table: bool = True,
    gripper_open_ratio: float = 0.3,
) -> bool:
    """
    멀티로봇용 place 스킬 - gripper open 직전에 동기화

    Args:
        skills: LeRobotSkills 인스턴스
        sync_barrier: SyncBarrier 인스턴스
        place_position: 목표 위치 [x, y, z]
        gripper_offset: 그리퍼 오프셋
        is_table: 테이블에 놓을지 여부
        gripper_open_ratio: 그리퍼 열림 비율 (0.0-1.0)

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
                                           gripper_offset=gripper_offset,
                                           target_pitch=saved_pitch)
    if not move_success:
        print("Warning: Failed to reach place position, but continuing for sync")

    # === SYNC: 두 로봇 모두 place 위치에 도달한 후 gripper open ===
    # NOTE: Always call sync_barrier.wait() even on failure to prevent deadlock
    sync_barrier.wait("place_ready")

    # Open gripper (동기화됨)
    skills.gripper_open(ratio=gripper_open_ratio)

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
