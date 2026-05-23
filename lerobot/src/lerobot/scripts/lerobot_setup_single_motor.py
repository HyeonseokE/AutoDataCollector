# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Set up ONE motor on a SO-arm (replacement after burnout, etc).

Unlike `lerobot-setup-motors`, this script only configures a single motor
chosen interactively, instead of walking through every motor in the arm.

Example:

```shell
lerobot-setup-single-motor \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=robot2
```
"""

from dataclasses import dataclass

import draccus

from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so_leader,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_mini,
    so_leader,
)

COMPATIBLE_DEVICES = [
    "koch_follower",
    "koch_leader",
    "omx_follower",
    "omx_leader",
    "openarm_mini",
    "so100_follower",
    "so100_leader",
    "so101_follower",
    "so101_leader",
    "lekiwi",
]


@dataclass
class SetupSingleConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop


def _prompt_motor(motor_names: list[str]) -> str:
    print("\nAvailable motors:")
    for i, name in enumerate(motor_names, 1):
        print(f"  [{i}] {name}")
    while True:
        sel = input("\nWhich motor to set up? (number or name): ").strip()
        if sel.isdigit() and 1 <= int(sel) <= len(motor_names):
            return motor_names[int(sel) - 1]
        if sel in motor_names:
            return sel
        print(f"  invalid — enter 1..{len(motor_names)} or one of {motor_names}")


@draccus.wrap()
def setup_single_motor(cfg: SetupSingleConfig):
    if cfg.device.type not in COMPATIBLE_DEVICES:
        raise NotImplementedError(f"Device type '{cfg.device.type}' not supported.")

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    else:
        device = make_teleoperator_from_config(cfg.device)

    motor_names = list(device.bus.motors)
    motor = _prompt_motor(motor_names)

    input(f"\nConnect the controller board to ONLY the '{motor}' motor and press enter.")
    device.bus.setup_motor(motor)
    print(f"\n'{motor}' motor id set to {device.bus.motors[motor].id}")


def main():
    setup_single_motor()


if __name__ == "__main__":
    main()
