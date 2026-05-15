#!/usr/bin/env bash
# pix2robot_calibrator (호모그래피 + Charuco corner-snap) 실행 셸 진입점.
#
# 사용법:
#   ./pix2robot_calibrator/run_calibration.sh --robot 3
#   ./pix2robot_calibrator/run_calibration.sh --robot 3 --camera-serial 254622073196
#   ./pix2robot_calibrator/run_calibration.sh --robot 3 --resume
#   ./pix2robot_calibrator/run_calibration.sh --robot 3 \
#       --board-config pix2robot_calibrator/board_config.yaml
#
# 필수 인자:
#   --robot N              로봇 번호 (예: 0, 2, 3)
#
# 선택 인자 (run_calibration.py 로 그대로 전달):
#   --camera-serial SERIAL RealSense 시리얼 (기본: $DEFAULT_CAMERA_SERIAL)
#   --resume               기존 캘리브 데이터에 이어 수집
#   --board-config PATH    다른 보드 yaml 사용 (기본: pix2robot_calibrator/board_config.yaml)

set -e

# ── 기본값 (이 머신에 연결된 RealSense D435 시리얼) ─────
DEFAULT_CAMERA_SERIAL="254622073196"

usage() {
    cat <<EOF
Usage: $(basename "$0") --robot N [--camera-serial SERIAL] [--resume] [--board-config PATH]

Required:
  --robot N              robot id (e.g., 0, 2, 3)

Optional:
  --camera-serial SERIAL RealSense serial (default: $DEFAULT_CAMERA_SERIAL)
  --resume               resume from existing calibration data
  --board-config PATH    custom board yaml path
EOF
    exit 1
}

# ── 인자 파싱 ────────────────────────────────────────────
ROBOT_ID=""
CAMERA_SERIAL="$DEFAULT_CAMERA_SERIAL"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --robot)
            ROBOT_ID="$2"
            shift 2
            ;;
        --camera-serial)
            CAMERA_SERIAL="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            PASSTHROUGH+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$ROBOT_ID" ]]; then
    echo "ERROR: --robot 인자가 필요합니다" >&2
    usage
fi

# ── 프로젝트 루트로 이동 ─────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── 모터 포트 권한 부여 (sudo 필요) ──────────────────────
echo "Setting permissions for /dev/ttyACM* ports..."
for port in /dev/ttyACM*; do
    if [ -e "$port" ]; then
        sudo chmod 777 "$port"
        echo "  $port - OK"
    fi
done

# ── Python 실행 ──────────────────────────────────────────
PYTHON="${PYTHON:-python}"

echo
echo "=========================================="
echo "  Pix2Robot Calibration"
echo "  robot:         $ROBOT_ID"
echo "  camera serial: $CAMERA_SERIAL"
if [[ ${#PASSTHROUGH[@]} -gt 0 ]]; then
    echo "  extra args:    ${PASSTHROUGH[*]}"
fi
echo "=========================================="
echo

exec "$PYTHON" pix2robot_calibrator/run_calibration.py \
    --robot "$ROBOT_ID" \
    --camera-serial "$CAMERA_SERIAL" \
    "${PASSTHROUGH[@]}"
