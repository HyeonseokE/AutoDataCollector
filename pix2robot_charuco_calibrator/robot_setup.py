"""
로봇 하드웨어 셋업 — Phase 3, 4에서 공유.

기존 pix2robot_calibrator/calibrator.py 의 setup_robot 패턴을 그대로 차용하되
독립 모듈로 분리하여 재사용 가능하게 함.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot_cap.hardware.feetech import FeetechController
from lerobot_cap.hardware.calibration import MotorCalibration
from lerobot_cap.kinematics.engine import KinematicsEngine
from lerobot_cap.kinematics.calibration_limits import load_calibration_limits


JOINT_NAMES = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']


def load_robot(robot_id: int) -> Tuple[FeetechController, KinematicsEngine, "object"]:
    """
    로봇 컨트롤러 + 키네마틱스 + 캘리브레이션 리밋 로드 후 연결.

    Returns:
        (controller, kinematics, calibration_limits)
    """
    config_path = (
        Path(__file__).parent.parent
        / f"robot_configs/robot/so101_robot{robot_id}.yaml"
    )
    if not config_path.exists():
        raise FileNotFoundError(f"로봇 설정 파일 없음: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    yaml_dir = config_path.parent
    for key in ("calibration_file", "compensation_file"):
        val = config.get(key)
        if val and not Path(val).is_absolute():
            config[key] = str((yaml_dir / val).resolve())
    kin_cfg = config.get("kinematics", {})
    if kin_cfg.get("urdf_path") and not Path(kin_cfg["urdf_path"]).is_absolute():
        kin_cfg["urdf_path"] = str((yaml_dir / kin_cfg["urdf_path"]).resolve())

    # 모터 컨트롤러
    calibration_path = Path(config['calibration_file'])
    with open(calibration_path, 'r') as f:
        calib_raw = json.load(f)

    calibration = {}
    for name, data in calib_raw.items():
        motor_data = dict(data)
        if 'id' in motor_data and 'motor_id' not in motor_data:
            motor_data['motor_id'] = motor_data.pop('id')
        if 'model' not in motor_data:
            motor_data['model'] = 'sts3215'
        calib = MotorCalibration(**motor_data)
        calibration[calib.motor_id] = calib
    motor_ids = [m['id'] for m in config['motors'].values()]

    controller = FeetechController(
        port=config['port'],
        baudrate=config['baudrate'],
        motor_ids=motor_ids,
        calibration=calibration,
    )
    if not controller.connect():
        raise RuntimeError("모터 연결 실패")

    kinematics = KinematicsEngine(
        urdf_path=config['kinematics']['urdf_path'],
        end_effector_frame=config['kinematics']['end_effector_frame'],
    )

    calibration_limits = load_calibration_limits(
        config['calibration_file'],
        joint_names=JOINT_NAMES,
    )
    print(f"  로봇 {robot_id} 연결 완료")
    return controller, kinematics, calibration_limits


def get_tcp_position(
    controller: FeetechController,
    kinematics: KinematicsEngine,
    calibration_limits,
) -> np.ndarray:
    """현재 모터 → FK → TCP 위치 (m, base_link 기준)."""
    normalized = controller.read_positions(normalize=True)
    joint_radians = calibration_limits.normalized_to_radians(normalized[:5])
    return kinematics.get_ee_position(joint_radians)


def cleanup_robot(controller: Optional[FeetechController]):
    if controller is None:
        return
    try:
        controller.enable_torque()
    except Exception:
        pass
    controller.disconnect()
