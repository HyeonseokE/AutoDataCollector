#!/usr/bin/env bash
# =============================================================================
#  SO-101 Dataset Recording (H.264 codec)
#
#  Wraps the official LeRobot CLI `lerobot-record` with H.264 encoding.
#
#  - 비디오 인코딩: H.264 (소프트웨어 'h264' 기본 / NVIDIA GPU 'h264_nvenc' 옵션)
#  - 텔레오퍼레이션 + 데이터셋 레코딩 동시 진행
#
#  Usage:
#    ./record_dataset.sh
#    REPO_ID=my_user/my_dataset TASK="pick up block" NUM_EPISODES=20 ./record_dataset.sh
#    VCODEC=h264_nvenc ./record_dataset.sh                     # NVIDIA GPU 가속
#    FOLLOWER_PORT=/dev/ttyACM2 LEADER_PORT=/dev/ttyACM4 ./record_dataset.sh
#    ./record_dataset.sh --dataset.fps=60                       # 추가 인자 직접 전달
#    RESUME=true ./record_dataset.sh                            # 기존 데이터셋에 이어서 녹화
#
#  Codec 옵션:
#    h264              — 소프트웨어 인코더 (기본, 호환성 가장 좋음)
#    h264_nvenc        — NVIDIA GPU (가속, 권장 시 GPU 있을 때)
#    h264_vaapi        — Linux Intel/AMD GPU
#    h264_qsv          — Intel Quick Sync
#    auto              — 사용 가능한 HW 인코더 자동 선택 (없으면 libsvtav1 fallback)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# -------- conda env --------
CONDA_ENV="${CONDA_ENV:-lerobot}"
# shellcheck disable=SC1091
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# -------- video codec (H.264) --------
# 기본은 소프트웨어 h264.
VCODEC="${VCODEC:-h264}" # always h264 - hscho

# -------- normalization mode --------
# false: 본체 5 모터 -100~100, gripper 0~100 (gripper는 항상 0~100 하드코딩)
# true:  본체 5 모터 degrees, gripper 0~100
USE_DEGREES="${USE_DEGREES:-false}" # always false - hscho

# -------- dataset --------
REPO_ID="${REPO_ID:-CoRL2026-CSI/teleop_close_lid}"
RESET_REPO_ID="${RESET_REPO_ID:-CoRL2026-CSI/teleop_open_lid}"
TASK="${TASK:-Close the pot lid.}"
RESET_TASK="${RESET_TASK:-Open the pot lid and place it on the floor.}"
FPS="${FPS:-10}"
EPISODE_TIME_S="${EPISODE_TIME_S:-100}"
RESET_TIME_S="${RESET_TIME_S:-100}"
RECORD_RESET="${RECORD_RESET:-true}"
RESET_AS_EPISODE="${RESET_AS_EPISODE:-true}"
NUM_EPISODES="${NUM_EPISODES:-100}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/outputs/datasets/$REPO_ID}"
RESET_DATASET_ROOT="${RESET_DATASET_ROOT:-$REPO_DIR/outputs/datasets/$RESET_REPO_ID}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

# -------- resume --------
RESUME="${RESUME:-false}"

# -------- follower (robot) --------
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM2}"
FOLLOWER_ID="${FOLLOWER_ID:-so101_robot0}"

# -------- leader (teleop) --------
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM3}"
LEADER_ID="${LEADER_ID:-robot0}"

# -------- cameras --------
# ws2 (recording_config_ws2.yaml) 와 동일 설정:
#   shared/topview: RealSense 254622079503  640x480@30fps
#   left_arm/wrist: OpenCV /dev/video6      640x480@30fps  (MJPG)
CAM_LEFT_WRIST_TYPE="${CAM_LEFT_WRIST_TYPE:-opencv}"
CAM_LEFT_WRIST_ID="${CAM_LEFT_WRIST_ID:-/dev/video6}"
CAM_LEFT_WRIST_WIDTH="${CAM_LEFT_WRIST_WIDTH:-640}"
CAM_LEFT_WRIST_HEIGHT="${CAM_LEFT_WRIST_HEIGHT:-480}"
CAM_LEFT_WRIST_FPS="${CAM_LEFT_WRIST_FPS:-30}"
CAM_LEFT_WRIST_FOURCC="${CAM_LEFT_WRIST_FOURCC:-MJPG}"

CAM_TOP_NAME="${CAM_TOP_NAME:-topview}"
CAM_TOP_TYPE="${CAM_TOP_TYPE:-intelrealsense}"
CAM_TOP_ID="${CAM_TOP_ID:-254622079503}"
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
    CAMERAS="{ left_wrist: $_lw, $CAM_TOP_NAME: $_top }"
    unset _lw _top
fi
unset -f _cam_entry

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
========= SO-101 Dataset Recording (H.264) =========
 env       : $CONDA_ENV ($(python --version 2>&1))
 codec     : $VCODEC  (lerobot default streaming/threads)
 norm mode : use_degrees=$USE_DEGREES  (false → 본체 5DoF -100~100, gripper 0~100)
 follower  : $ROBOT_TYPE ($FOLLOWER_ID)  port=$FOLLOWER_PORT
 leader    : $TELEOP_TYPE ($LEADER_ID)   port=$LEADER_PORT
 cameras   : left_wrist($CAM_LEFT_WRIST_TYPE:$CAM_LEFT_WRIST_ID ${CAM_LEFT_WRIST_WIDTH}x${CAM_LEFT_WRIST_HEIGHT}@${CAM_LEFT_WRIST_FPS}fps)
             $CAM_TOP_NAME($CAM_TOP_TYPE:$CAM_TOP_ID ${CAM_TOP_WIDTH}x${CAM_TOP_HEIGHT}@${CAM_TOP_FPS}fps)
 task      : $TASK
 reset task: $RESET_TASK
 close ds  : $REPO_ID  (root=$DATASET_ROOT)
 open ds   : $RESET_REPO_ID  (root=$RESET_DATASET_ROOT)
 recording : ${NUM_EPISODES} close/open cycles @ ${FPS}Hz
             close=${EPISODE_TIME_S}s  open=${RESET_TIME_S}s  record_reset=$RECORD_RESET  reset_as_episode=$RESET_AS_EPISODE
 resume    : $RESUME  (true → 기존 데이터셋에 이어서)
 push_hub  : $PUSH_TO_HUB
====================================================
EOF

# -------- run --------
RECORD_ARGS=(
    --robot.type="$ROBOT_TYPE"
    --robot.port="$FOLLOWER_PORT"
    --robot.id="$FOLLOWER_ID"
    --robot.use_degrees="$USE_DEGREES"
    --robot.cameras="$CAMERAS"
    --teleop.type="$TELEOP_TYPE"
    --teleop.port="$LEADER_PORT"
    --teleop.id="$LEADER_ID"
    --teleop.use_degrees="$USE_DEGREES"
    --dataset.repo_id="$REPO_ID"
    --dataset.single_task="$TASK"
    --dataset.root="$DATASET_ROOT"
    --dataset.reset_repo_id="$RESET_REPO_ID"
    --dataset.reset_root="$RESET_DATASET_ROOT"
    --dataset.fps="$FPS"
    --dataset.episode_time_s="$EPISODE_TIME_S"
    --dataset.reset_time_s="$RESET_TIME_S"
    --dataset.record_reset="$RECORD_RESET"
    --dataset.reset_as_episode="$RESET_AS_EPISODE"
    --dataset.reset_task="$RESET_TASK"
    --dataset.num_episodes="$NUM_EPISODES"
    --dataset.push_to_hub="$PUSH_TO_HUB"
    --dataset.vcodec="$VCODEC"
    --display_data="$DISPLAY_DATA"
    --resume="$RESUME"
)

# 추가 인자 (예: --dataset.fps=60) 그대로 전달
RECORD_ARGS+=("$@")

python -m lerobot.scripts.lerobot_record "${RECORD_ARGS[@]}"
