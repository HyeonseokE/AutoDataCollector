#!/usr/bin/env bash
# SO-101 Robot1 Calibration Script
# Port: /dev/ttyACM3 (robot1)

set -euo pipefail

PORT="/dev/ttyACM0"
ID="so101_robot8"

echo "========================================="
echo "  SO-101 Robot1 Calibration"
echo "  Port: $PORT"
echo "  ID:   $ID"
echo "========================================="

sudo chmod 777 "$PORT"

lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port="$PORT" \
    --robot.id="$ID"
