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

# Python 환경: PYTHON 환경변수 우선 → lerobot3 conda → 시스템 python3 fallback
if [ -n "$PYTHON" ] && [ -x "$PYTHON" ]; then
    :  # 사용자 지정 경로 사용
elif [ -x "/home/lerobot3/miniconda3/envs/lerobot_cap/bin/python" ]; then
    PYTHON="/home/lerobot3/miniconda3/envs/lerobot_cap/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
else
    echo "Python 인터프리터를 찾을 수 없습니다."
    echo "PYTHON 환경변수로 인터프리터 경로를 지정하세요."
    exit 1
fi

exec "$PYTHON" -m pix2robot_charuco_calibrator.main "$@"
