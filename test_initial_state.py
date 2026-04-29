"""
지정한 robot이 initial state로 이동하는지 테스트.

사용법:
    python test_initial_state.py                    # robot0, /dev/ttyACM0
    python test_initial_state.py --robot 1          # robot1, /dev/ttyACM0
    python test_initial_state.py --robot 1 --port /dev/ttyACM3
"""
import argparse
import atexit
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from skills.skills_lerobot import LeRobotSkills


def make_port_override_yaml(orig_cfg_path: Path, new_port: str) -> Path:
    """원본과 같은 디렉토리에 port만 덮어쓴 임시 yaml 생성.

    주의: 절대경로로 만들면 '/home/lerobot3/...' 의 'lerobot3' 가
    skills_lerobot.py의 regex `robot(\\d+)` 와 먼저 매칭되어 robot3로 인식됨.
    따라서 **상대 경로** 그대로 유지해야 한다.
    """
    with open(orig_cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["port"] = new_port
    # filename pattern: "tmpoverride__" + original → 끝에 robotN이 유지되도록
    tmp_path = orig_cfg_path.with_name("tmpoverride__" + orig_cfg_path.name)
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f)
    atexit.register(lambda: tmp_path.unlink(missing_ok=True))
    return tmp_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type=int, default=0, help="robot id (0/1/2/3)")
    parser.add_argument("--port", default="/dev/ttyACM0", help="override USB port")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    orig_cfg = Path(f"robot_configs/robot/so101_robot{args.robot}.yaml")
    assert orig_cfg.exists(), f"Config not found: {orig_cfg}"

    with open(orig_cfg) as f:
        orig_port = yaml.safe_load(f).get("port")

    cfg_path: Path = orig_cfg
    if orig_port != args.port:
        print(f"[Override] port: {orig_port} -> {args.port}")
        cfg_path = make_port_override_yaml(orig_cfg, args.port)

    skills = LeRobotSkills(
        robot_config=str(cfg_path),
        frame="base_link",
        movement_duration=args.duration,
    )

    if not skills.connect():
        print("[FAIL] Robot connection failed")
        return 1

    try:
        print(f"\n[STEP] robot{args.robot}: Moving to initial state...")
        ok = skills.move_to_initial_state()
        if not ok:
            print("[FAIL] move_to_initial_state returned False")
            return 1
        print("[OK] Reached initial state")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    finally:
        skills.disconnect()


if __name__ == "__main__":
    sys.exit(main())
