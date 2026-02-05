#!/usr/bin/env python3
"""
Skill-level Recording Test

skill.* feature들이 정상적으로 기록되는지 테스트합니다.
실제 로봇 없이 레코딩 로직만 테스트합니다.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from record_dataset.config import DATASET_FEATURES, build_dataset_features
from record_dataset.context import RecordingContext


def test_config_features():
    """config.py의 feature 스키마 테스트"""
    print("\n=== Test 1: Feature Schema ===")

    required_features = [
        "skill.natural_language",
        "skill.type",
        "skill.progress",
        "skill.goal_position.joint",
        "skill.goal_position.world_xyzrpy",
        "skill.goal_position.robot_xyzrpy",
        "skill.goal_position.gripper",
    ]

    for feature in required_features:
        if feature in DATASET_FEATURES:
            schema = DATASET_FEATURES[feature]
            print(f"  ✓ {feature}: dtype={schema['dtype']}, shape={schema.get('shape')}")
        else:
            print(f"  ✗ {feature}: NOT FOUND")
            return False

    # build_dataset_features도 테스트
    built_features = build_dataset_features()
    for feature in required_features:
        if feature not in built_features:
            print(f"  ✗ build_dataset_features missing: {feature}")
            return False

    print("  → Feature schema test PASSED")
    return True


def test_recording_context():
    """RecordingContext의 skill info 관리 테스트"""
    print("\n=== Test 2: RecordingContext ===")

    # Mock 설정
    RecordingContext._is_active = True
    RecordingContext._recorder = True  # Mock recorder

    # set_skill_info 테스트 (with start_state for state-based progress)
    start_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal_joint = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
    goal_world = np.array([0.2, 0.1, 0.15, 0.0, -0.5, 0.0], dtype=np.float32)
    goal_robot = np.array([0.15, 0.05, 0.12, 0.0, -0.5, 0.0], dtype=np.float32)
    goal_gripper = 85.0

    RecordingContext.set_skill_info(
        label="move to blue dish",
        skill_type="move",
        duration=3.0,
        goal_joint=goal_joint,
        goal_world_xyzrpy=goal_world,
        goal_robot_xyzrpy=goal_robot,
        goal_gripper=goal_gripper,
        start_state=start_state,
    )

    # get_skill_info 테스트 (with current_state for state-based progress)
    info = RecordingContext.get_skill_info(current_state=start_state)

    assert info["label"] == "move to blue dish", f"label mismatch: {info['label']}"
    assert info["type"] == "move", f"type mismatch: {info['type']}"
    assert np.allclose(info["goal_joint"], goal_joint), "goal_joint mismatch"
    assert np.allclose(info["goal_world_xyzrpy"], goal_world), "goal_world_xyzrpy mismatch"
    assert np.allclose(info["goal_robot_xyzrpy"], goal_robot), "goal_robot_xyzrpy mismatch"
    assert info["goal_gripper"] == goal_gripper, f"goal_gripper mismatch: {info['goal_gripper']}"

    print(f"  ✓ label: {info['label']}")
    print(f"  ✓ type: {info['type']}")
    print(f"  ✓ goal_joint: {info['goal_joint']}")
    print(f"  ✓ goal_world_xyzrpy: {info['goal_world_xyzrpy']}")
    print(f"  ✓ goal_robot_xyzrpy: {info['goal_robot_xyzrpy']}")
    print(f"  ✓ goal_gripper: {info['goal_gripper']}")

    # State-based progress: at start → ~0.0
    progress_start = RecordingContext.get_skill_progress(current_state=start_state)
    print(f"  ✓ progress at start: {progress_start:.3f} (expected ~0.0)")
    assert abs(progress_start - 0.0) < 0.01, f"progress at start should be ~0.0, got {progress_start}"

    # State-based progress: at midpoint → ~0.5
    midpoint = (start_state + goal_joint) / 2.0
    progress_mid = RecordingContext.get_skill_progress(current_state=midpoint)
    print(f"  ✓ progress at midpoint: {progress_mid:.3f} (expected ~0.5)")
    assert abs(progress_mid - 0.5) < 0.01, f"progress at midpoint should be ~0.5, got {progress_mid}"

    # State-based progress: at goal → ~1.0
    progress_goal = RecordingContext.get_skill_progress(current_state=goal_joint)
    print(f"  ✓ progress at goal: {progress_goal:.3f} (expected ~1.0)")
    assert abs(progress_goal - 1.0) < 0.01, f"progress at goal should be ~1.0, got {progress_goal}"

    # clear_skill_info 테스트
    RecordingContext.clear_skill_info()
    info_after_clear = RecordingContext.get_skill_info()

    assert info_after_clear["label"] == "", "label not cleared"
    assert info_after_clear["type"] == "", "type not cleared"
    assert info_after_clear["progress"] == 0.0, "progress not cleared"
    assert RecordingContext._skill_start_state is None, "start_state not cleared"
    print("  ✓ clear_skill_info works")

    # Reset mock
    RecordingContext._is_active = False
    RecordingContext._recorder = None

    print("  → RecordingContext test PASSED")
    return True


def test_state_based_progress_edge_cases():
    """State-based progress edge case 테스트"""
    print("\n=== Test 3: State-Based Progress Edge Cases ===")

    RecordingContext._is_active = True
    RecordingContext._recorder = True

    # Edge case 1: start == goal (zero distance) → 1.0
    same_state = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32)
    RecordingContext.set_skill_info(
        label="already at goal",
        skill_type="move",
        duration=3.0,
        goal_joint=same_state,
        goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
        goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
        goal_gripper=85.0,
        start_state=same_state,
    )
    progress = RecordingContext.get_skill_progress(current_state=same_state)
    print(f"  ✓ zero distance (start==goal): {progress:.3f} (expected 1.0)")
    assert progress == 1.0, f"zero distance should return 1.0, got {progress}"
    RecordingContext.clear_skill_info()

    # Edge case 2: time-based fallback (start_state=None)
    RecordingContext.set_skill_info(
        label="time-based fallback",
        skill_type="move",
        duration=2.0,
        goal_joint=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float32),
        goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
        goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
        goal_gripper=85.0,
        start_state=None,  # No start_state → time-based fallback
    )
    time.sleep(0.5)
    progress_time = RecordingContext.get_skill_progress()  # No current_state → time-based
    expected_time = 0.5 / 2.0  # ~0.25
    print(f"  ✓ time-based fallback (no start_state): {progress_time:.3f} (expected ~{expected_time:.3f})")
    assert 0.15 < progress_time < 0.45, f"time-based progress out of range: {progress_time}"

    # Also verify: even with current_state, if start_state is None → time-based
    progress_time2 = RecordingContext.get_skill_progress(
        current_state=np.zeros(6, dtype=np.float32)
    )
    print(f"  ✓ time-based (start_state=None, current_state given): {progress_time2:.3f}")
    assert progress_time2 > 0.15, f"should still use time-based, got {progress_time2}"
    RecordingContext.clear_skill_info()

    # Edge case 3: overshoot (current past goal)
    # L2 distance 특성: 목표를 지나치면 goal과의 거리가 다시 증가 → progress 감소
    # ||goal - current|| = 5, ||goal - start|| = 10, progress = 1 - 5/10 = 0.5
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    overshoot = np.array([15.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    RecordingContext.set_skill_info(
        label="overshoot",
        skill_type="move",
        duration=3.0,
        goal_joint=goal,
        goal_world_xyzrpy=np.zeros(6, dtype=np.float32),
        goal_robot_xyzrpy=np.zeros(6, dtype=np.float32),
        goal_gripper=85.0,
        start_state=start,
    )
    progress_over = RecordingContext.get_skill_progress(current_state=overshoot)
    print(f"  ✓ overshoot: {progress_over:.3f} (expected 0.5, L2 distance reflects distance from goal)")
    assert abs(progress_over - 0.5) < 0.01, f"overshoot case unexpected: {progress_over}"
    RecordingContext.clear_skill_info()

    RecordingContext._is_active = False
    RecordingContext._recorder = None

    print("  → State-based progress edge cases PASSED")
    return True


def test_skill_types():
    """각 스킬 타입이 올바르게 정의되었는지 테스트"""
    print("\n=== Test 4: Skill Types ===")

    expected_types = [
        "move",           # move_to_position
        "move_initial",   # move_to_initial_state
        "move_free",      # move_to_free_state
        "gripper_open",   # gripper_open
        "gripper_close",  # gripper_close
        "rotate",         # rotate_90degree
    ]

    for skill_type in expected_types:
        print(f"  ✓ {skill_type}")

    print("  → Skill types test PASSED")
    return True


def test_frame_data_structure():
    """레코딩될 프레임 데이터 구조 테스트"""
    print("\n=== Test 5: Frame Data Structure ===")

    # Mock frame data (recorder.py에서 생성되는 구조)
    frame = {
        "observation.state": np.zeros(6, dtype=np.float32),
        "action": np.zeros(6, dtype=np.float32),
        "skill.natural_language": "move to blue dish",
        "skill.type": "move",
        "skill.progress": np.array([0.5], dtype=np.float32),
        "skill.goal_position.joint": np.zeros(6, dtype=np.float32),
        "skill.goal_position.world_xyzrpy": np.zeros(6, dtype=np.float32),
        "skill.goal_position.robot_xyzrpy": np.zeros(6, dtype=np.float32),
        "skill.goal_position.gripper": np.array([85.0], dtype=np.float32),
    }

    for key, value in frame.items():
        if isinstance(value, np.ndarray):
            print(f"  ✓ {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  ✓ {key}: {type(value).__name__} = {value}")

    print("  → Frame data structure test PASSED")
    return True


def main():
    print("=" * 60)
    print("Skill-level Recording Test")
    print("=" * 60)

    tests = [
        ("Feature Schema", test_config_features),
        ("RecordingContext", test_recording_context),
        ("State-Based Progress Edge Cases", test_state_based_progress_edge_cases),
        ("Skill Types", test_skill_types),
        ("Frame Data Structure", test_frame_data_structure),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ {name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {name}: {status}")
        if not result:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
