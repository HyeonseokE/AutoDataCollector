#!/bin/bash

echo "============================================================"
echo "Move to Pose DRY RUN (IK + Trajectory Planning only)"
echo "============================================================"
echo ""
echo "This tests IK and trajectory planning WITHOUT moving the robot."
echo ""

# Default target position
X=${1:-0.15}
Y=${2:-0.0}
Z=${3:-0.20}

echo "Target position: x=$X, y=$Y, z=$Z"
echo ""

python scripts/move_to_pose.py \
    --config robot_configs/robot/so101_robot3.yaml \
    --x $X \
    --y $Y \
    --z $Z \
    --duration 3.0 \
    --dry-run
