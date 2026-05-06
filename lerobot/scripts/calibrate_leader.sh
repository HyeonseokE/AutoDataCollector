#!/usr/bin/env bash
# SO-101 Robot2 Leader Calibration Script
# Port: /dev/ttyACM4 (robot2 leader)

set -euo pipefail

PORT="${PORT:-/dev/ttyACM2}"
ID="${ID:-so101_robot3_leader}"
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"

echo "========================================="
echo "  SO-101 Robot2 Leader Calibration"
echo "  Port:  $PORT"
echo "  ID:    $ID"
echo "  Type:  $TELEOP_TYPE"
echo "========================================="

if [ -e "$PORT" ]; then
    sudo chmod 777 "$PORT"
else
    echo "[ERROR] Serial port not found: $PORT" >&2
    exit 1
fi

lerobot-calibrate \
    --teleop.type="$TELEOP_TYPE" \
    --teleop.port="$PORT" \
    --teleop.id="$ID"
