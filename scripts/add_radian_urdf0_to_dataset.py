#!/usr/bin/env python3
"""
기존 HuggingFace LeRobot 데이터셋에 URDF 0° 기준 radian feature 를 추가하는 독립 스크립트.

입력:
  - 소스 데이터셋 (예: skkuprism/teleop_pnp_100ep) — HF Hub or 로컬
  - 캘리브레이션 JSON (데이터 수집 당시의 로봇)

출력:
  - 새 로컬 데이터셋 폴더 (예: teleop_pnp_100ep_urdf0)
  - observation.state / action 은 그대로 두고
  - observation.state.radian_urdf0 (6D) 와 action.radian_urdf0 (6D) 컬럼 추가

사용법:
  python scripts/add_radian_urdf0_to_dataset.py \\
      --source-repo-id skkuprism/teleop_pnp_100ep \\
      --target-name teleop_pnp_100ep_urdf0 \\
      --calibration robot_configs/motor_calibration/so101/robot0_calibration.json

비디오 파일은 재인코딩하지 않고 하드링크로 재사용 — 빠르고 디스크 절약.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_cap.kinematics import (  # noqa: E402
    load_calibration_limits,
    load_gripper_radian_params,
)


RADIAN_STATE_KEY = "observation.state.radian_urdf0"
RADIAN_ACTION_KEY = "action.radian_urdf0"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# 공식 버그버전 (과거 데이터셋에 남아있을 수 있음) — 있으면 함께 제거
LEGACY_RADIAN_KEYS = [
    "observation.radian.state",
    "observation.radian.action",
    "observation.radian.state_urdf0",
    "observation.radian.action_urdf0",
]


def convert_to_rad_urdf0(
    norm_6d: np.ndarray,
    calib,
    g_half: float,
    g_offset: float,
) -> np.ndarray:
    """Normalized 6D → URDF 0° radian 6D.

    Arm 5D: calibration 객체의 정식 변환 함수 사용 (input -100~+100).
    Gripper 1D: RANGE_0_100 (input 0~100), g_offset도 0~100 스케일.
    """
    arm_rad = calib.normalized_to_radians(norm_6d[:5])
    if g_half > 0.0:
        gripper_rad = ((float(norm_6d[5]) - g_offset) / 50.0) * g_half
    else:
        gripper_rad = 0.0
    return np.concatenate([arm_rad, [gripper_rad]]).astype(np.float32)


def download_source_dataset(repo_id: str) -> Path:
    """HF Hub 에서 LeRobot 데이터셋 스냅샷 다운로드 → 로컬 경로 반환."""
    from huggingface_hub import snapshot_download

    print(f"[Download] {repo_id} ...")
    local_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
    print(f"[Download] → {local_dir}")
    return Path(local_dir)


def mirror_tree(
    src: Path,
    dst: Path,
    *,
    skip_names: set[str],
) -> None:
    """src 의 파일/폴더 구조를 dst 로 미러링.

    HF cache 는 symlink → blobs 구조라 반드시 resolve() 로 실제 파일을 타겟.
    - 일반 파일: 하드링크 시도 (blob 공유) → 실패 시 내용 복사
    - .gitattributes 같은 숨김파일은 건너뜀
    - skip_names 경로는 건너뜀 (별도 처리용)
    """
    for root, _dirs, files in os.walk(src, followlinks=True):
        rel = Path(root).relative_to(src)
        if any(part in skip_names for part in rel.parts):
            continue

        dst_dir = dst / rel
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fname in files:
            if fname in skip_names or fname.startswith("."):
                continue
            src_file = (Path(root) / fname).resolve()  # ← symlink → blob 해석
            if not src_file.exists():
                continue
            dst_file = dst_dir / fname
            if dst_file.exists():
                continue
            try:
                os.link(src_file, dst_file)
            except OSError:
                shutil.copy2(src_file, dst_file)


def update_info_json(
    info_path: Path,
    *,
    motor_names: List[str],
) -> dict:
    """features 에 새 radian_urdf0 키 추가, legacy 버그버전 제거."""
    with open(info_path, "r") as f:
        info = json.load(f)

    features = info.get("features", {})
    joint_names = [f"{m}.pos" for m in motor_names]

    # Legacy 제거
    for k in LEGACY_RADIAN_KEYS:
        features.pop(k, None)

    # 새 feature 추가 (이미 있으면 덮어씀)
    features[RADIAN_STATE_KEY] = {
        "dtype": "float32",
        "shape": [6],
        "names": joint_names,
    }
    features[RADIAN_ACTION_KEY] = {
        "dtype": "float32",
        "shape": [6],
        "names": joint_names,
    }

    info["features"] = features
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    return info


def process_parquet(
    path: Path,
    *,
    calib,
    g_half: float,
    g_offset: float,
) -> Dict[str, np.ndarray]:
    """Parquet 파일에 radian_urdf0 컬럼 추가 후 in-place 저장.

    Returns: 해당 파일에서 본 radian 값들 (stats 계산용)
    """
    table = pq.read_table(path)
    df = table.to_pandas()

    # Legacy 컬럼 제거
    for k in LEGACY_RADIAN_KEYS:
        if k in df.columns:
            df = df.drop(columns=[k])

    # 각 프레임 변환
    state_vals = np.stack(df[STATE_KEY].values).astype(np.float64)  # (N, 6)
    action_vals = np.stack(df[ACTION_KEY].values).astype(np.float64)

    rad_state = np.empty_like(state_vals, dtype=np.float32)
    rad_action = np.empty_like(action_vals, dtype=np.float32)
    for i in range(len(df)):
        rad_state[i] = convert_to_rad_urdf0(state_vals[i], calib, g_half, g_offset)
        rad_action[i] = convert_to_rad_urdf0(action_vals[i], calib, g_half, g_offset)

    df[RADIAN_STATE_KEY] = list(rad_state)
    df[RADIAN_ACTION_KEY] = list(rad_action)

    # 쓰기
    new_table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(new_table, path)

    return {
        RADIAN_STATE_KEY: rad_state,
        RADIAN_ACTION_KEY: rad_action,
    }


def compute_stats(values: np.ndarray) -> dict:
    """LeRobot 표준 stats 포맷 (min/max/mean/std)."""
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
    }


def update_stats_json(
    stats_path: Path,
    *,
    rad_state_all: np.ndarray,
    rad_action_all: np.ndarray,
) -> None:
    if stats_path.exists():
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = {}

    # Legacy 제거
    for k in LEGACY_RADIAN_KEYS:
        stats.pop(k, None)

    stats[RADIAN_STATE_KEY] = compute_stats(rad_state_all)
    stats[RADIAN_ACTION_KEY] = compute_stats(rad_action_all)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source-repo-id",
        default="skkuprism/teleop_pnp_100ep",
        help="소스 HF 데이터셋 repo_id (기본: skkuprism/teleop_pnp_100ep)",
    )
    parser.add_argument(
        "--target-name",
        default="teleop_pnp_100ep_urdf0",
        help="출력 데이터셋 폴더 이름 (기본: teleop_pnp_100ep_urdf0)",
    )
    parser.add_argument(
        "--target-root",
        default=str(Path.home() / ".cache/huggingface/lerobot/local"),
        help="출력 상위 디렉터리 (기본: ~/.cache/huggingface/lerobot/local)",
    )
    parser.add_argument(
        "--calibration",
        required=True,
        help="데이터 수집 당시 로봇의 calibration JSON 경로 "
        "(예: robot_configs/motor_calibration/so101/robot0_calibration.json)",
    )
    parser.add_argument(
        "--local-source",
        default=None,
        help="이미 로컬에 있는 데이터셋 경로 (지정 시 HF 다운로드 생략)",
    )
    args = parser.parse_args()

    # 1) 소스 위치 확보
    if args.local_source:
        src_root = Path(args.local_source).resolve()
        if not src_root.exists():
            raise FileNotFoundError(f"--local-source not found: {src_root}")
        print(f"[Source] local: {src_root}")
    else:
        src_root = download_source_dataset(args.source_repo_id)

    # 2) 캘리브레이션 로드
    calib_path = Path(args.calibration).resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    print(f"[Calibration] {calib_path}")
    calib = load_calibration_limits(str(calib_path))
    g_half, g_offset = load_gripper_radian_params(str(calib_path))
    print(f"  arm half_range_rad[0]: {calib.half_range_radians[0]:.4f}")
    print(f"  gripper half_range_rad: {g_half:.4f}, offset_norm: {g_offset:.2f}")

    # 3) 출력 경로 준비
    target_root = Path(args.target_root).resolve() / args.target_name
    if target_root.exists():
        raise FileExistsError(
            f"Target already exists: {target_root}\nRemove it or pick a different --target-name."
        )
    target_root.mkdir(parents=True, exist_ok=False)
    print(f"[Target] {target_root}")

    # 4) 파일 미러링 (parquet 와 stats.json 제외: 아래서 재생성)
    skip = {"stats.json"}  # stats.json 은 아래서 재생성
    print("[Mirror] copying non-data files ...")
    mirror_tree(src_root, target_root, skip_names=skip)

    # 5) meta/info.json 업데이트
    info_path = target_root / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"info.json missing: {info_path}")
    motor_names = [
        "shoulder_pan", "shoulder_lift", "elbow_flex",
        "wrist_flex", "wrist_roll", "gripper",
    ]
    info = update_info_json(info_path, motor_names=motor_names)
    print(f"[Meta] info.json updated (features: +{RADIAN_STATE_KEY}, +{RADIAN_ACTION_KEY})")

    # 6) 모든 parquet 파일 처리
    data_dir = target_root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise RuntimeError(f"No parquet files under {data_dir}")
    print(f"[Data] processing {len(parquet_files)} parquet file(s) ...")

    rad_state_buckets: List[np.ndarray] = []
    rad_action_buckets: List[np.ndarray] = []
    for pq_path in parquet_files:
        result = process_parquet(pq_path, calib=calib, g_half=g_half, g_offset=g_offset)
        rad_state_buckets.append(result[RADIAN_STATE_KEY])
        rad_action_buckets.append(result[RADIAN_ACTION_KEY])
        print(f"  done: {pq_path.relative_to(target_root)}  (+{len(result[RADIAN_STATE_KEY])} frames)")

    rad_state_all = np.concatenate(rad_state_buckets, axis=0)
    rad_action_all = np.concatenate(rad_action_buckets, axis=0)

    # 7) stats.json 갱신 (기존이 있으면 병합, 없으면 신규)
    src_stats = src_root / "meta" / "stats.json"
    tgt_stats = target_root / "meta" / "stats.json"
    if src_stats.exists():
        shutil.copy2(src_stats, tgt_stats)
    update_stats_json(
        tgt_stats,
        rad_state_all=rad_state_all,
        rad_action_all=rad_action_all,
    )
    print(f"[Stats] stats.json updated (+{RADIAN_STATE_KEY}, +{RADIAN_ACTION_KEY})")

    # 8) 요약 출력
    print("\n" + "=" * 60)
    print(f"DONE: {target_root}")
    print(f"  total frames: {len(rad_state_all)}")
    print(f"  {RADIAN_STATE_KEY}:")
    print(f"    min={rad_state_all.min(axis=0)}")
    print(f"    max={rad_state_all.max(axis=0)}")
    print(f"  {RADIAN_ACTION_KEY}:")
    print(f"    min={rad_action_all.min(axis=0)}")
    print(f"    max={rad_action_all.max(axis=0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
