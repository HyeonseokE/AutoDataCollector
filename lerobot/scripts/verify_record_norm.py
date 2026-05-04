#!/usr/bin/env python3
"""
녹화된 데이터셋의 motor 정규화 모드 사후 검증.

5 본체 모터: -100~100 범위 ± 그리퍼: 0~100 범위인지 확인.
DEGREES 모드면 일반적으로 본체 모터가 |값| > 100 이거나 부호가 다르게 나옴.

사용법:
    python verify_record_norm.py outputs/datasets/<repo_id>
    python verify_record_norm.py outputs/datasets/<repo_id> --episode 0
"""

import argparse
import sys
from pathlib import Path

import numpy as np


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
]
GRIPPER_KEY = "gripper.pos"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as e:
        print(f"lerobot import 실패: {e}")
        sys.exit(1)

    print(f"Loading: {args.dataset_root}")
    ds = LeRobotDataset(repo_id="local", root=args.dataset_root)
    print(f"  Episodes: {ds.num_episodes}, frames: {ds.num_frames}")
    print()

    # 한 에피소드의 모든 프레임 모으기
    epi_idx = args.episode
    if epi_idx >= ds.num_episodes:
        print(f"  episode {epi_idx} 없음 (총 {ds.num_episodes})")
        sys.exit(1)

    # 에피소드 시작/끝 인덱스
    epi_data = ds.hf_dataset.filter(lambda r: r["episode_index"] == epi_idx)
    if len(epi_data) == 0:
        print(f"  episode {epi_idx} 비어있음")
        sys.exit(1)

    # 'observation.state' 또는 'action' 등에서 motor 값 가져오기
    # state/action은 각 motor별 별도 column으로 저장됨
    print(f"Episode {epi_idx} ({len(epi_data)} frames):")
    print()

    cols = epi_data.column_names
    motor_cols_obs = [f"observation.state" if "observation.state" in cols else None]

    # observation.state, action 둘 다 있는 경우 컬럼이 dict 형태일 수 있음
    sample = epi_data[0]
    print("  available columns:", [c for c in cols if "state" in c or "action" in c])
    print()

    # 일반적으로 'observation.state' 가 (6,) array — 정규화된 motor 값
    if "observation.state" in cols:
        states = np.stack([np.asarray(s) for s in epi_data["observation.state"]])
        print(f"  observation.state shape: {states.shape}")
        if states.shape[1] >= 6:
            print(f"\n  본체 5 motor (cols 0-4):")
            print(f"    min:  {states[:, :5].min(axis=0)}")
            print(f"    max:  {states[:, :5].max(axis=0)}")
            print(f"    범위 일관성: ", end="")
            body_min, body_max = states[:, :5].min(), states[:, :5].max()
            if -110 <= body_min and body_max <= 110:
                print(f"-100~100 모드 (실측 {body_min:.1f} ~ {body_max:.1f})  ✓ use_degrees=False")
            elif -200 <= body_min and body_max <= 200:
                print(f"DEGREES 모드 (실측 {body_min:.1f} ~ {body_max:.1f})  ✗ use_degrees=True")
            else:
                print(f"비정상 범위 ({body_min:.1f} ~ {body_max:.1f})")

            print(f"\n  gripper (col 5):")
            print(f"    min: {states[:, 5].min():.1f}")
            print(f"    max: {states[:, 5].max():.1f}")
            grip_min, grip_max = states[:, 5].min(), states[:, 5].max()
            if -10 <= grip_min and grip_max <= 110:
                print(f"    범위: 0~100 모드  ✓")
            else:
                print(f"    범위 비정상 ({grip_min:.1f} ~ {grip_max:.1f})")

    # action도 동일 검사
    if "action" in cols:
        actions = np.stack([np.asarray(a) for a in epi_data["action"]])
        print(f"\n  action shape: {actions.shape}")
        if actions.shape[1] >= 6:
            print(f"  body action range: [{actions[:, :5].min():.1f}, {actions[:, :5].max():.1f}]")
            print(f"  gripper action range: [{actions[:, 5].min():.1f}, {actions[:, 5].max():.1f}]")


if __name__ == "__main__":
    main()
