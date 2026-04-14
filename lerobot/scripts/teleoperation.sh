#!/usr/bin/env bash
# =============================================================================
#  SO-101 Robot2 Teleoperation + Dataset Recording
#  - Follower: /dev/ttyACM1 (so101_follower, id=so101_robot2)
#  - Leader  : /dev/ttyACM4 (so101_leader,   id=so101_robot2_leader)
#  - Records via lerobot-record (official LeRobot CLI)
#
#  Usage:
#    ./teleoperation.sh
#    REPO_ID=my_user/my_dataset TASK="pick up block" NUM_EPISODES=20 ./teleoperation.sh
#    FOLLOWER_PORT=/dev/ttyACM1 LEADER_PORT=/dev/ttyACM4 ./teleoperation.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# -------- conda env --------
CONDA_ENV="${CONDA_ENV:-lerobot}"
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# -------- follower (robot2) --------
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM1}"
FOLLOWER_ID="${FOLLOWER_ID:-so101_robot2}"

# -------- leader (robot2 leader) --------
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM4}"
LEADER_ID="${LEADER_ID:-so101_robot2_leader}"

# -------- cameras --------
CAM_LEFT_WRIST_TYPE="${CAM_LEFT_WRIST_TYPE:-opencv}"
CAM_LEFT_WRIST_ID="${CAM_LEFT_WRIST_ID:-/dev/video18}"
CAM_LEFT_WRIST_WIDTH="${CAM_LEFT_WRIST_WIDTH:-640}"
CAM_LEFT_WRIST_HEIGHT="${CAM_LEFT_WRIST_HEIGHT:-480}"
CAM_LEFT_WRIST_FPS="${CAM_LEFT_WRIST_FPS:-30}"
CAM_LEFT_WRIST_FOURCC="${CAM_LEFT_WRIST_FOURCC:-MJPG}"

CAM_TOP_TYPE="${CAM_TOP_TYPE:-intelrealsense}"
CAM_TOP_ID="${CAM_TOP_ID:-335622072328}"
CAM_TOP_WIDTH="${CAM_TOP_WIDTH:-640}"
CAM_TOP_HEIGHT="${CAM_TOP_HEIGHT:-480}"
CAM_TOP_FPS="${CAM_TOP_FPS:-30}"

_cam_entry() {
    local type="$1" id="$2" w="$3" h="$4" fps="$5" fourcc="${6:-}"
    if [ "$type" = "opencv" ]; then
        if [ -n "$fourcc" ]; then
            echo "{type: opencv, index_or_path: $id, width: $w, height: $h, fps: $fps, fourcc: '$fourcc'}"
        else
            echo "{type: opencv, index_or_path: $id, width: $w, height: $h, fps: $fps}"
        fi
    else
        echo "{type: intelrealsense, serial_number_or_name: '$id', width: $w, height: $h, fps: $fps}"
    fi
}

if [ -z "${CAMERAS+x}" ]; then
    _lw=$(_cam_entry "$CAM_LEFT_WRIST_TYPE" "$CAM_LEFT_WRIST_ID" "$CAM_LEFT_WRIST_WIDTH" "$CAM_LEFT_WRIST_HEIGHT" "$CAM_LEFT_WRIST_FPS" "$CAM_LEFT_WRIST_FOURCC")
    _top=$(_cam_entry "$CAM_TOP_TYPE" "$CAM_TOP_ID" "$CAM_TOP_WIDTH" "$CAM_TOP_HEIGHT" "$CAM_TOP_FPS")
    CAMERAS="{ left_wrist: $_lw, top: $_top }"
    unset _lw _top
fi
unset -f _cam_entry

# -------- dataset --------
REPO_ID="${REPO_ID:-skkuprism/teleop_pnp_100ep}"
TASK="${TASK:-pick up the red block and place it on the blue dish.}"
FPS="${FPS:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-3}"
NUM_EPISODES="${NUM_EPISODES:-100}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/outputs/datasets/$REPO_ID}"

# -------- serial port permissions --------
for _port in "$FOLLOWER_PORT" "$LEADER_PORT"; do
    if [ -e "$_port" ]; then
        sudo chmod 777 "$_port"
    else
        echo "[ERROR] Serial port not found: $_port" >&2
        exit 1
    fi
done
unset _port

# -------- summary --------
cat <<EOF
========= SO-101 Teleoperation + Record =========
 env       : $CONDA_ENV ($(python --version 2>&1))
 follower  : $ROBOT_TYPE ($FOLLOWER_ID)  port=$FOLLOWER_PORT
 leader    : $TELEOP_TYPE ($LEADER_ID)   port=$LEADER_PORT
 cameras   : left_wrist($CAM_LEFT_WRIST_TYPE:$CAM_LEFT_WRIST_ID ${CAM_LEFT_WRIST_WIDTH}x${CAM_LEFT_WRIST_HEIGHT}@${CAM_LEFT_WRIST_FPS}fps)
             top($CAM_TOP_TYPE:$CAM_TOP_ID ${CAM_TOP_WIDTH}x${CAM_TOP_HEIGHT}@${CAM_TOP_FPS}fps)
 task      : $TASK
 dataset   : $REPO_ID  (root=$DATASET_ROOT)
 recording : ${NUM_EPISODES} episodes x ${EPISODE_TIME_S}s @ ${FPS}Hz  reset=${RESET_TIME_S}s
 push_hub  : $PUSH_TO_HUB
=================================================
EOF

# -------- run --------
RECORD_ARGS=(
    --robot.type="$ROBOT_TYPE"
    --robot.port="$FOLLOWER_PORT"
    --robot.id="$FOLLOWER_ID"
    --robot.cameras="$CAMERAS"
    --teleop.type="$TELEOP_TYPE"
    --teleop.port="$LEADER_PORT"
    --teleop.id="$LEADER_ID"
    --dataset.repo_id="$REPO_ID"
    --dataset.single_task="$TASK"
    --dataset.root="$DATASET_ROOT"
    --dataset.fps="$FPS"
    --dataset.episode_time_s="$EPISODE_TIME_S"
    --dataset.reset_time_s="$RESET_TIME_S"
    --dataset.num_episodes="$NUM_EPISODES"
    --dataset.push_to_hub="$PUSH_TO_HUB"
)

RECORD_ARGS+=("$@")

lerobot-record "${RECORD_ARGS[@]}"
