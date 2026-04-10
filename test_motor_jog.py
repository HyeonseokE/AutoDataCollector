"""
Robot0 모터 테스트: 각 모터를 +-15도씩 움직여보기
USB 허브 통신 안정성 확인용
"""
import time
import sys
import json
import numpy as np

sys.path.insert(0, "/home/lerobot/AutoDataCollector")
from src.lerobot_cap.hardware.feetech import FeetechController
from src.lerobot_cap.hardware.calibration import MotorCalibration

# --- Config ---
PORT = "/dev/ttyACM1"
BAUDRATE = 1000000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
CALIB_FILE = "robot_configs/motor_calibration/so101/robot2_calibration.json"
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TARGET_DEGREES = 15.0
STEPS_PER_REV = 4096  # STS3215: 4096 steps = 360 degrees
MOVE_WAIT = 3.0  # seconds to wait after each move
TORQUE_LIMIT = 800  # 0-1000, default was ~500

# --- Load calibration ---
with open(CALIB_FILE) as f:
    calib_raw = json.load(f)

calibration = {}
for joint_name in JOINT_NAMES:
    c = calib_raw[joint_name]
    calibration[c["id"]] = MotorCalibration(
        motor_id=c["id"],
        model="sts3215",
        drive_mode=c["drive_mode"],
        homing_offset=c["homing_offset"],
        range_min=c["range_min"],
        range_max=c["range_max"],
    )

# --- Calculate 15 degrees in normalized units for each motor ---
def degrees_to_normalized(motor_id, degrees):
    """Convert degrees to normalized units for a specific motor."""
    calib = calibration[motor_id]
    range_steps = calib.range_max - calib.range_min
    range_degrees = range_steps * 360.0 / STEPS_PER_REV
    normalized_per_degree = 200.0 / range_degrees
    return degrees * normalized_per_degree

print("=" * 60)
print("  Robot0 Motor Jog Test (+-15 degrees)")
print("=" * 60)

# Print normalized values for 15 degrees per motor
for i, (mid, name) in enumerate(zip(MOTOR_IDS, JOINT_NAMES)):
    calib = calibration[mid]
    range_steps = calib.range_max - calib.range_min
    range_deg = range_steps * 360.0 / STEPS_PER_REV
    norm_15 = degrees_to_normalized(mid, TARGET_DEGREES)
    print(f"  Motor {mid} ({name}): range={range_steps} steps ({range_deg:.1f}°), 15° = {norm_15:.2f} normalized")

print()

# --- Connect ---
controller = FeetechController(
    port=PORT,
    baudrate=BAUDRATE,
    motor_ids=MOTOR_IDS,
    calibration=calibration,
)

if not controller.connect():
    print("Failed to connect!")
    sys.exit(1)

controller.enable_torque()
controller.set_torque_limit(TORQUE_LIMIT)

try:
    # First move to a neutral middle position (all joints near 0)
    # Gripper stays open at ~0
    mid_pos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    print(f"\nMoving to neutral middle position (all ~0)...")

    # Read current position and move gradually
    cur = controller.read_positions(normalize=True)
    print(f"  Current: {np.array2string(cur, precision=1)}")

    # Move in 2 steps to avoid sudden jerks
    halfway = (cur + mid_pos) / 2.0
    controller.write_positions(halfway, normalize=True)
    time.sleep(2.0)
    controller.write_positions(mid_pos, normalize=True)
    time.sleep(3.0)

    # Read actual middle position as baseline
    initial_pos = controller.read_positions(normalize=True)
    print(f"  Arrived:  {np.array2string(initial_pos, precision=1)}")
    print()

    write_fail_count = 0

    for i, (motor_id, joint_name) in enumerate(zip(MOTOR_IDS, JOINT_NAMES)):
        delta = degrees_to_normalized(motor_id, TARGET_DEGREES)

        print(f"--- Motor {motor_id} ({joint_name}) ---")

        # +15 degrees
        target = initial_pos.copy()
        target[i] = initial_pos[i] + delta
        target[i] = np.clip(target[i], -95, 95)  # safety margin
        print(f"  Moving +15°: {initial_pos[i]:.2f} → {target[i]:.2f}")
        controller.write_positions(target, normalize=True)
        time.sleep(MOVE_WAIT)

        actual = controller.read_positions(normalize=True)
        error = abs(actual[i] - target[i])
        print(f"  Actual: {actual[i]:.2f} (error: {error:.2f})")

        # Back to initial
        controller.write_positions(initial_pos, normalize=True)
        time.sleep(MOVE_WAIT)

        # -15 degrees
        target = initial_pos.copy()
        target[i] = initial_pos[i] - delta
        target[i] = np.clip(target[i], -95, 95)
        print(f"  Moving -15°: {initial_pos[i]:.2f} → {target[i]:.2f}")
        controller.write_positions(target, normalize=True)
        time.sleep(MOVE_WAIT)

        actual = controller.read_positions(normalize=True)
        error = abs(actual[i] - target[i])
        print(f"  Actual: {actual[i]:.2f} (error: {error:.2f})")

        # Back to initial
        print(f"  Returning to initial...")
        controller.write_positions(initial_pos, normalize=True)
        time.sleep(MOVE_WAIT)
        print()

    print("=" * 60)
    print("  Test complete!")
    print("=" * 60)

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\nError: {e}")
finally:
    controller.disable_torque()
    controller.disconnect()
