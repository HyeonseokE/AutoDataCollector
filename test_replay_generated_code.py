"""
Replay a saved generated_code.py with its seed positions.

기본: session_20260420_192558 / episode_01 / forward (red block -> blue dish)
positions: 같은 session의 seed_01_setup/seed_positions.json 에서 로드.

사용법:
    python test_replay_generated_code.py
    python test_replay_generated_code.py \
        --code results/.../forward/generated_code.py \
        --positions results/.../seed_01_setup/seed_positions.json \
        --robot-config robot_configs/robot/so101_robot0.yaml
"""
import argparse
import atexit
import json
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from skills.skills_lerobot import LeRobotSkills


DEFAULT_CODE = "results/session_20260420_192558/episode_01/forward/generated_code.py"
DEFAULT_POSITIONS = "results/session_20260420_192558/seed_01_setup/seed_positions.json"


def load_positions(path: Path) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    return data["positions"] if "positions" in data else data


def make_port_override_yaml(orig_cfg_path: Path, new_port: str) -> Path:
    """원본과 같은 디렉토리에 port만 덮어쓴 임시 yaml 생성.

    주의: 절대경로로 만들면 '/home/lerobot3/...' 의 'lerobot3' 가
    skills_lerobot.py의 regex `robot(\\d+)` 와 먼저 매칭되어 robot3로 인식됨.
    → 상대 경로를 유지해야 함.
    """
    with open(orig_cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["port"] = new_port
    tmp_path = orig_cfg_path.with_name("tmpoverride__" + orig_cfg_path.name)
    with open(tmp_path, "w") as f:
        yaml.safe_dump(cfg, f)
    atexit.register(lambda: tmp_path.unlink(missing_ok=True))
    return tmp_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default=DEFAULT_CODE)
    parser.add_argument("--positions", default=DEFAULT_POSITIONS)
    parser.add_argument("--robot", type=int, default=0, help="robot id (0/1/2/3)")
    parser.add_argument("--port", default="/dev/ttyACM0", help="override USB port")
    parser.add_argument("--frame", default="base_link")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    code_path = Path(args.code)
    pos_path = Path(args.positions)
    orig_cfg = Path(f"robot_configs/robot/so101_robot{args.robot}.yaml")

    assert code_path.exists(), f"Code not found: {code_path}"
    assert pos_path.exists(), f"Positions not found: {pos_path}"
    assert orig_cfg.exists(), f"Robot config not found: {orig_cfg}"

    with open(orig_cfg) as f:
        orig_port = yaml.safe_load(f).get("port")
    cfg_path = orig_cfg
    if orig_port != args.port:
        print(f"[Override] port: {orig_port} -> {args.port}")
        cfg_path = make_port_override_yaml(orig_cfg, args.port)

    positions = load_positions(pos_path)

    print(f"[Replay] code:      {code_path}")
    print(f"[Replay] positions: {pos_path}")
    print(f"[Replay] robot cfg: {cfg_path}")
    for name, info in positions.items():
        pts = info.get("points", {})
        print(f"  - {name}: {list(pts.keys())} -> { {k: [round(v, 4) for v in val] for k, val in pts.items()} }")

    skills = LeRobotSkills(
        robot_config=str(cfg_path),
        frame=args.frame,
        movement_duration=args.duration,
    )

    code_text = code_path.read_text()
    exec_globals = {
        "__name__": "__generated__",
        "skills": skills,
        "positions": positions,
    }

    try:
        exec(code_text, exec_globals)
        fn = exec_globals.get("execute_task") or exec_globals.get("execute_reset_task")
        if fn is None:
            print("[FAIL] No execute_task/execute_reset_task found in generated code")
            return 1
        fn()
        print("[OK] Replay finished")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"[FAIL] Execution error: {e}")
        raise
    finally:
        if getattr(skills, "is_connected", False):
            try:
                skills.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
