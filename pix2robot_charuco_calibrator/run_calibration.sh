#!/usr/bin/env bash
# Charuco 4페이즈 캘리브레이션 실행 셸 진입점.
#
# 사용법:
#   ./run_calibration.sh --robot 0
#   ./run_calibration.sh --robot 0 --skip-intrinsics    # K, dist 재사용
#   ./run_calibration.sh --robot 0 --phases 3,4         # 특정 페이즈만
#   ./run_calibration.sh --robot 0 --board-config custom.yaml
#
# 모든 인자는 main.py로 그대로 전달됨.

set -e

# 스크립트 위치 → AutoDataCollector 루트로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Python 환경 (필요 시 conda 환경 활성화)
PYTHON="${PYTHON:-/home/lerobot3/miniconda3/envs/lerobot_cap/bin/python}"

if [ ! -x "$PYTHON" ]; then
    echo "Python 인터프리터를 찾을 수 없습니다: $PYTHON"
    echo "PYTHON 환경변수로 다른 인터프리터를 지정하세요."
    exit 1
fi

exec "$PYTHON" -m pix2robot_charuco_calibrator.main "$@"
