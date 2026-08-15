#!/usr/bin/env python
"""Sequentially wiggle motors 1-6 on /dev/ttyACM2 by +/-15 degrees.

For each motor:
    home -> +15 deg -> home -> -15 deg -> home

Uses raw tick control (no calibration file needed). STS3215 has 4096 ticks
per full revolution -> 15 deg == 170 ticks.

Safety: run only when the arm is in a pose where every joint can swing +/-15 deg
without collision. Torque is disabled on exit.
"""

import time

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

PORT = "/dev/ttyACM2"
STEP_DEG = 30
TICKS_PER_DEG = 4096 / 360
STEP_TICKS = int(round(STEP_DEG * TICKS_PER_DEG))
SETTLE_S = 0.7


def main() -> None:
    bus = FeetechMotorsBus(
        port=PORT,
        motors={f"m{i}": Motor(i, "sts3215", MotorNormMode.DEGREES) for i in range(1, 7)},
        calibration=None,
    )
    bus.connect()
    try:
        bus.disable_torque()
        for name in bus.motors:
            bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", name, 16)
        bus.enable_torque()

        home = bus.sync_read("Present_Position", normalize=False)
        print(f"[home ticks] {home}")

        for i in range(1, 7):
            name = f"m{i}"
            base = home[name]
            print(f"\n=== motor id={i} ({name})  home={base} ===")
            for delta in (+STEP_TICKS, 0, -STEP_TICKS, 0):
                target = base + delta
                print(f"  goto {target}  (delta {delta:+d} ticks / {delta / TICKS_PER_DEG:+.1f} deg)")
                bus.write("Goal_Position", name, target, normalize=False)
                time.sleep(SETTLE_S)
    finally:
        bus.disconnect(disable_torque=True)
        print("\n[done] torque disabled, port closed.")


if __name__ == "__main__":
    main()
