#!/bin/bash
# SO-101 Follower 캘리브레이션 스크립트
#
# 사용법: 아래 설정값을 수정한 후 실행
#   ./calibrate_follower.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================
# 설정
# ============================================================
PORT="/dev/ttyACM0"
ROBOT_ID="robot5"

# ============================================================
# 실행
# ============================================================
PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH" python -m lerobot.scripts.lerobot_calibrate \
    --robot.type=so101_follower \
    --robot.port="$PORT" \
    --robot.id="$ROBOT_ID"
