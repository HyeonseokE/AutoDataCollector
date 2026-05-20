#!/usr/bin/env bash
# =============================================================================
#  SO-101 Single-Arm Dataset Recording — WS4 (H.264 codec)
#
#  Wraps the official LeRobot CLI `lerobot-record` with H.264 encoding.
#  ws3 single-arm 스크립트 구조 + ws4 하드웨어 매핑.
#
#  ARM_SIDE 로 좌/우 팔 선택 (default: left = robot6):
#    ARM_SIDE=left  → follower=so101_robot6 (/dev/ttyACM1)
#                     leader=so101_robot2_leader (/dev/ttyACM3)
#                     wrist cam=/dev/video0     (USB 04:00.0-2)
#    ARM_SIDE=right → follower=so101_robot7 (/dev/ttyACM4)
#                     leader=so101_robot3_leader (/dev/ttyACM5)
#                     wrist cam=/dev/video2     (USB 16:00.4-2)
#
#  Top camera (RealSense 254622075836) 는 항상 활성화. WS4 전용 RS.
#
#  Usage:
#    ./record_dataset_ws4.sh
#    ARM_SIDE=right ./record_dataset_ws4.sh
#    REPO_ID=my_user/single_dataset TASK="pick up block" NUM_EPISODES=20 \
#      ./record_dataset_ws4.sh
#    VCODEC=h264_nvenc ./record_dataset_ws4.sh                  # NVIDIA GPU
#    FOLLOWER_PORT=/dev/ttyACMx LEADER_PORT=/dev/ttyACMy ./record_dataset_ws4.sh
#    RESUME=true ./record_dataset_ws4.sh                        # 기존 데이터셋 이어서
#
#  Codec 옵션:
#    h264 (기본) / h264_nvenc / h264_vaapi / h264_qsv / auto
#
#  ※ 캘리브레이션 주의: lerobot-record는 robot.type=so101_follower 일 때
#    `~/.cache/huggingface/lerobot/calibration/robots/so_follower/{FOLLOWER_ID}.json`
#    을 찾음. bi_arm용 calibration(so101_dual_robot6_robot7_left.json 등)을 그대로
#    재사용하려면 다음처럼 복사:
#      cp ~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_dual_robot6_robot7_left.json \
#         ~/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_robot6.json
#    (right 쪽은 _right → so101_robot7 으로)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# -------- conda env --------
CONDA_ENV="${CONDA_ENV:-lerobot}"
# shellcheck disable=SC1091
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# -------- video codec (H.264) --------
VCODEC="${VCODEC:-h264}"
STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"

# -------- normalization mode --------
# false: 본체 5 모터 -100~100, gripper 0~100 (gripper는 항상 0~100 하드코딩)
# true:  본체 5 모터 degrees, gripper 0~100
USE_DEGREES="${USE_DEGREES:-false}"

# -------- arm side selection --------
ARM_SIDE="${ARM_SIDE:-left}"
case "$ARM_SIDE" in
    left)
        _DEFAULT_FOLLOWER_PORT="/dev/ttyACM1"
        _DEFAULT_FOLLOWER_ID="so101_robot6"
        _DEFAULT_LEADER_PORT="/dev/ttyACM3"
        _DEFAULT_LEADER_ID="so101_robot2_leader"
        _DEFAULT_WRIST_CAM_ID="/dev/video0"
        ;;
    right)
        _DEFAULT_FOLLOWER_PORT="/dev/ttyACM4"
        _DEFAULT_FOLLOWER_ID="so101_robot7"
        _DEFAULT_LEADER_PORT="/dev/ttyACM5"
        _DEFAULT_LEADER_ID="so101_robot3_leader"
        _DEFAULT_WRIST_CAM_ID="/dev/video2"
        ;;
    *)
        echo "[ERROR] ARM_SIDE 는 left|right 중 하나여야 합니다 (현재: '$ARM_SIDE')" >&2
        exit 1
        ;;
esac

# -------- dataset --------
REPO_ID="${REPO_ID:-CoRL2026-CSI/sort_blocks}"
TASK="${TASK:-Sort each colored block onto the plate of the matching color.}"
FPS="${FPS:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-10}"
NUM_EPISODES="${NUM_EPISODES:-100}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/outputs/datasets/$REPO_ID}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

# -------- resume --------
RESUME="${RESUME:-false}"

# -------- follower (robot) --------
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
FOLLOWER_PORT="${FOLLOWER_PORT:-$_DEFAULT_FOLLOWER_PORT}"
FOLLOWER_ID="${FOLLOWER_ID:-$_DEFAULT_FOLLOWER_ID}"

# -------- leader (teleop) --------
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
LEADER_PORT="${LEADER_PORT:-$_DEFAULT_LEADER_PORT}"
LEADER_ID="${LEADER_ID:-$_DEFAULT_LEADER_ID}"

# follower / leader 포트 중복 검사
if [ "$FOLLOWER_PORT" = "$LEADER_PORT" ]; then
    echo "[ERROR] FOLLOWER_PORT 와 LEADER_PORT 가 동일합니다: $FOLLOWER_PORT" >&2
    exit 1
fi

# -------- cameras --------
# WS4: top RealSense 254622075836, wrist camera는 ARM_SIDE 에 따라 다름
CAM_WRIST_TYPE="${CAM_WRIST_TYPE:-opencv}"
CAM_WRIST_ID="${CAM_WRIST_ID:-$_DEFAULT_WRIST_CAM_ID}"
CAM_WRIST_WIDTH="${CAM_WRIST_WIDTH:-640}"
CAM_WRIST_HEIGHT="${CAM_WRIST_HEIGHT:-480}"
CAM_WRIST_FPS="${CAM_WRIST_FPS:-30}"
CAM_WRIST_FOURCC="${CAM_WRIST_FOURCC:-MJPG}"

CAM_TOP_TYPE="${CAM_TOP_TYPE:-intelrealsense}"
CAM_TOP_ID="${CAM_TOP_ID:-254622075836}"
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
    _wr=$(_cam_entry "$CAM_WRIST_TYPE" "$CAM_WRIST_ID" "$CAM_WRIST_WIDTH" "$CAM_WRIST_HEIGHT" "$CAM_WRIST_FPS" "$CAM_WRIST_FOURCC")
    _top=$(_cam_entry "$CAM_TOP_TYPE"   "$CAM_TOP_ID"   "$CAM_TOP_WIDTH"   "$CAM_TOP_HEIGHT"   "$CAM_TOP_FPS")
    CAMERAS="{ wrist: $_wr, top: $_top }"
    unset _wr _top
fi
unset -f _cam_entry

# -------- serial port permissions --------
for _port in "$FOLLOWER_PORT" "$LEADER_PORT"; do
    if [ -e "$_port" ]; then
        sudo chmod 777 "$_port"
    else
        echo "[ERROR] Serial port not found: $_port" >&2
        echo "        현재 연결된 시리얼 장치:" >&2
        ls /dev/ttyACM* 2>&1 | sed 's/^/          /' >&2 || true
        exit 1
    fi
done
unset _port

# -------- summary --------
cat <<EOF
======= SO-101 Single-Arm Dataset Recording — WS4 ($ARM_SIDE arm) =======
 env       : $CONDA_ENV ($(python --version 2>&1))
 codec     : $VCODEC  (streaming=$STREAMING_ENCODING, threads=$ENCODER_THREADS)
 norm mode : use_degrees=$USE_DEGREES  (false → 본체 5DoF -100~100, gripper 0~100)
 arm side  : $ARM_SIDE
 follower  : $ROBOT_TYPE ($FOLLOWER_ID)  port=$FOLLOWER_PORT
 leader    : $TELEOP_TYPE ($LEADER_ID)   port=$LEADER_PORT
 cameras   : wrist ($CAM_WRIST_TYPE:$CAM_WRIST_ID ${CAM_WRIST_WIDTH}x${CAM_WRIST_HEIGHT}@${CAM_WRIST_FPS}fps)
             top   ($CAM_TOP_TYPE:$CAM_TOP_ID ${CAM_TOP_WIDTH}x${CAM_TOP_HEIGHT}@${CAM_TOP_FPS}fps)
 task      : $TASK
 dataset   : $REPO_ID  (root=$DATASET_ROOT)
 recording : ${NUM_EPISODES} episodes x ${EPISODE_TIME_S}s @ ${FPS}Hz  reset=${RESET_TIME_S}s
 resume    : $RESUME  (true → 기존 데이터셋에 이어서)
 push_hub  : $PUSH_TO_HUB
==========================================================================
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
    --dataset.fps="$FPS"
    --dataset.episode_time_s="$EPISODE_TIME_S"
    --dataset.reset_time_s="$RESET_TIME_S"
    --dataset.num_episodes="$NUM_EPISODES"
    --dataset.push_to_hub="$PUSH_TO_HUB"
    --dataset.vcodec="$VCODEC"
    --dataset.streaming_encoding="$STREAMING_ENCODING"
    --dataset.encoder_threads="$ENCODER_THREADS"
    --display_data="$DISPLAY_DATA"
    --resume="$RESUME"
)

# 추가 인자 (예: --dataset.fps=60) 그대로 전달
RECORD_ARGS+=("$@")

lerobot-record "${RECORD_ARGS[@]}"
