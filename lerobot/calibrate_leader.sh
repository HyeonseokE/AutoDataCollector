#!/bin/bash
# SO-101 Leader 캘리브레이션 스크립트
#
# 사용법: 아래 설정값을 수정한 후 실행
#   ./calibrate_leader.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# 설정
# ============================================================
PORT="/dev/ttyACM3"
ROBOT_ID="robot0"

# ============================================================
# 실행
# ============================================================
PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH" python -m lerobot.scripts.lerobot_calibrate \
    --teleop.type=so101_leader \
    --teleop.port="$PORT" \
    --teleop.id="$ROBOT_ID"
