"""Move each of the 6 SO-101 motors +/-15 degrees around the current position.

Uses raw motor ticks so no calibration is needed. 15 deg ~= 171 ticks (4096/360*15).

Usage:
    python wiggle_so101.py [PORT]
    (default PORT: /dev/ttyACM0)
"""

import sys
import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


DELTA_TICKS = round(15.0 / 360.0 * 4096)  # ~171
SETTLE_S = 1.0

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"

    bus = FeetechMotorsBus(
        port=port,
        motors={
            name: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100)
            for i, name in enumerate(MOTOR_NAMES)
        },
        calibration=None,
    )
    bus.connect()

    try:
        bus.disable_torque()
        for m in MOTOR_NAMES:
            bus.write("Operating_Mode", m, OperatingMode.POSITION.value)
        bus.enable_torque()

        start = bus.sync_read("Present_Position", normalize=False)
        print("start (raw ticks):", start)

        for motor in MOTOR_NAMES:
            base = start[motor]
            for offset in (+DELTA_TICKS, 0, -DELTA_TICKS, 0):
                target = int(base + offset)
                goal = {m: int(start[m]) for m in MOTOR_NAMES}
                goal[motor] = target
                print(f"{motor}: -> tick {target:5d} (offset {offset:+d})")
                bus.sync_write("Goal_Position", goal, normalize=False)
                time.sleep(SETTLE_S)
    finally:
        bus.disable_torque()
        bus.disconnect()


if __name__ == "__main__":
    main()
