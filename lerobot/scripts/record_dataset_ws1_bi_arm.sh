#!/usr/bin/env bash
# =============================================================================
#  SO-101 Dual-Arm Dataset Recording (H.264 codec, bi_so_follower)
#
#  Wraps the official LeRobot CLI `lerobot-record` with bi_so_follower /
#  bi_so_leader for bimanual teleoperation + dataset recording.
#
#  - 비디오 인코딩: H.264 (소프트웨어 'h264' 기본 / NVIDIA GPU 'h264_nvenc' 옵션)
#  - 좌/우 follower + 좌/우 leader 텔레오퍼레이션 동시 진행
#  - 카메라: left wrist / right wrist / top (한 쪽 arm에만 부착)
#
#  Usage:
#    ./record_dataset_ws1_bi_arm.sh
#    REPO_ID=my_user/dual_dataset NUM_EPISODES=20 ./record_dataset_ws1_bi_arm.sh
#    VCODEC=h264_nvenc ./record_dataset_ws1_bi_arm.sh        # NVIDIA GPU 가속
#    LEFT_FOLLOWER_PORT=/dev/ttyACM0 RIGHT_FOLLOWER_PORT=/dev/ttyACM3 \
#    LEFT_LEADER_PORT=/dev/ttyACM4   RIGHT_LEADER_PORT=/dev/ttyACM5 \
#      ./record_dataset_ws1_bi_arm.sh
#    TOP_ARM_SIDE=right ./record_dataset_ws1_bi_arm.sh         # top을 right_arm에
#    TOP_ARM_SIDE=none  ./record_dataset_ws1_bi_arm.sh         # top 비활성
#    RESUME=true ./record_dataset_ws1_bi_arm.sh                # 기존 데이터셋 이어서
#
#  데이터셋 키 (bi_so_follower 자동 prefix):
#    observation.images.left_wrist   ← left arm wrist
#    observation.images.left_top     ← top (TOP_ARM_SIDE=left default)
#    observation.images.right_wrist  ← right arm wrist
#
#  ws1 single-arm 데이터셋과 키 통일 필요 시:
#    --dataset.rename_map='{"observation.images.left_top": "observation.images.top"}'
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

# -------- video codec (H.264) --------
VCODEC="${VCODEC:-h264}"

# -------- normalization mode --------
# false: 본체 5 모터 -100~100, gripper 0~100 (gripper는 항상 0~100 하드코딩)
# true:  본체 5 모터 degrees, gripper 0~100
USE_DEGREES="${USE_DEGREES:-false}"

# -------- dataset --------
REPO_ID="${REPO_ID:-CoRL2026-CSI/fold_towel}"
TASK="${TASK:-Fold the towel in half.}"
FPS="${FPS:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-40}"
RESET_TIME_S="${RESET_TIME_S:-7}"
NUM_EPISODES="${NUM_EPISODES:-100}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/outputs/datasets/$REPO_ID}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

# -------- resume --------
RESUME="${RESUME:-true}"

# -------- bi_so_follower (좌/우 follower) --------
ROBOT_TYPE="${ROBOT_TYPE:-bi_so_follower}"
ROBOT_ID="${ROBOT_ID:-so101_dual_robot2_robot3}"

LEFT_FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/ttyACM0}"
LEFT_FOLLOWER_ID="${LEFT_FOLLOWER_ID:-so101_robot2}"

RIGHT_FOLLOWER_PORT="${RIGHT_FOLLOWER_PORT:-/dev/ttyACM4}"
RIGHT_FOLLOWER_ID="${RIGHT_FOLLOWER_ID:-so101_robot3}"

# -------- bi_so_leader (좌/우 leader) --------
TELEOP_TYPE="${TELEOP_TYPE:-bi_so_leader}"
TELEOP_ID="${TELEOP_ID:-so101_dual_leader}"

# leader 포트는 default 없음 — 사용자가 명시 (follower 포트와 충돌 방지)
LEFT_LEADER_PORT="${LEFT_LEADER_PORT:-/dev/ttyACM3}"
LEFT_LEADER_ID="${LEFT_LEADER_ID:-so101_robot2_leader}"

RIGHT_LEADER_PORT="${RIGHT_LEADER_PORT:-/dev/ttyACM2}"
RIGHT_LEADER_ID="${RIGHT_LEADER_ID:-so101_robot3_leader}"

if [ -z "$LEFT_LEADER_PORT" ] || [ -z "$RIGHT_LEADER_PORT" ]; then
    echo "[ERROR] LEFT_LEADER_PORT / RIGHT_LEADER_PORT 환경변수 필수 — leader 시리얼 포트를 지정하세요." >&2
    echo "        예: LEFT_LEADER_PORT=/dev/ttyACM2 RIGHT_LEADER_PORT=/dev/ttyACM5 $0" >&2
    echo "        현재 연결된 시리얼 장치:" >&2
    ls /dev/ttyACM* 2>&1 | sed 's/^/          /' >&2 || true
    echo "        (현재 follower 사용 중: LEFT=$LEFT_FOLLOWER_PORT, RIGHT=$RIGHT_FOLLOWER_PORT)" >&2
    exit 1
fi

# follower / leader 포트 중복 검사
declare -A _seen_ports=()
for _port_name in "LEFT_FOLLOWER_PORT" "RIGHT_FOLLOWER_PORT" "LEFT_LEADER_PORT" "RIGHT_LEADER_PORT"; do
    _port_val="${!_port_name}"
    if [ -n "${_seen_ports[$_port_val]:-}" ]; then
        echo "[ERROR] 시리얼 포트 중복: $_port_name=$_port_val 가 ${_seen_ports[$_port_val]}와 동일." >&2
        exit 1
    fi
    _seen_ports[$_port_val]="$_port_name"
done
unset _seen_ports _port_name _port_val

# -------- cameras --------
# ws1 (recording_config_ws1.yaml) 와 동일 설정:
#   left wrist:  OpenCV /dev/video6   640x480@30fps  (MJPG)
#   right wrist: OpenCV /dev/video16  640x480@30fps  (MJPG)
#   top (shared, RealSense):  335622072328  640x480@30fps
#
# bi_so_follower 는 카메라 키에 자동으로 left_/right_ prefix 를 붙입니다.
# top 은 한 쪽 arm 에만 추가해야 RealSense 충돌(device busy) 회피 가능.
# TOP_ARM_SIDE: left | right | none
LEFT_WRIST_CAM_TYPE="${LEFT_WRIST_CAM_TYPE:-opencv}"
LEFT_WRIST_CAM_ID="${LEFT_WRIST_CAM_ID:-/dev/video6}"
LEFT_WRIST_CAM_WIDTH="${LEFT_WRIST_CAM_WIDTH:-640}"
LEFT_WRIST_CAM_HEIGHT="${LEFT_WRIST_CAM_HEIGHT:-480}"
LEFT_WRIST_CAM_FPS="${LEFT_WRIST_CAM_FPS:-30}"
LEFT_WRIST_CAM_FOURCC="${LEFT_WRIST_CAM_FOURCC:-MJPG}"

RIGHT_WRIST_CAM_TYPE="${RIGHT_WRIST_CAM_TYPE:-opencv}"
RIGHT_WRIST_CAM_ID="${RIGHT_WRIST_CAM_ID:-/dev/video16}"
RIGHT_WRIST_CAM_WIDTH="${RIGHT_WRIST_CAM_WIDTH:-640}"
RIGHT_WRIST_CAM_HEIGHT="${RIGHT_WRIST_CAM_HEIGHT:-480}"
RIGHT_WRIST_CAM_FPS="${RIGHT_WRIST_CAM_FPS:-30}"
RIGHT_WRIST_CAM_FOURCC="${RIGHT_WRIST_CAM_FOURCC:-MJPG}"

TOP_CAM_TYPE="${TOP_CAM_TYPE:-intelrealsense}"
TOP_CAM_ID="${TOP_CAM_ID:-335622072328}"
TOP_CAM_WIDTH="${TOP_CAM_WIDTH:-640}"
TOP_CAM_HEIGHT="${TOP_CAM_HEIGHT:-480}"
TOP_CAM_FPS="${TOP_CAM_FPS:-30}"

TOP_ARM_SIDE="${TOP_ARM_SIDE:-left}"
case "$TOP_ARM_SIDE" in
    left|right|none) ;;
    *)
        echo "[ERROR] TOP_ARM_SIDE 는 left|right|none 중 하나여야 합니다 (현재: '$TOP_ARM_SIDE')" >&2
        exit 1
        ;;
esac

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

if [ -z "${LEFT_CAMERAS+x}" ] || [ -z "${RIGHT_CAMERAS+x}" ]; then
    _lw=$(_cam_entry "$LEFT_WRIST_CAM_TYPE"  "$LEFT_WRIST_CAM_ID"  "$LEFT_WRIST_CAM_WIDTH"  "$LEFT_WRIST_CAM_HEIGHT"  "$LEFT_WRIST_CAM_FPS"  "$LEFT_WRIST_CAM_FOURCC")
    _rw=$(_cam_entry "$RIGHT_WRIST_CAM_TYPE" "$RIGHT_WRIST_CAM_ID" "$RIGHT_WRIST_CAM_WIDTH" "$RIGHT_WRIST_CAM_HEIGHT" "$RIGHT_WRIST_CAM_FPS" "$RIGHT_WRIST_CAM_FOURCC")
    _top=$(_cam_entry "$TOP_CAM_TYPE"         "$TOP_CAM_ID"         "$TOP_CAM_WIDTH"         "$TOP_CAM_HEIGHT"         "$TOP_CAM_FPS")

    case "$TOP_ARM_SIDE" in
        left)
            LEFT_CAMERAS="{ wrist: $_lw, top: $_top }"
            RIGHT_CAMERAS="{ wrist: $_rw }"
            ;;
        right)
            LEFT_CAMERAS="{ wrist: $_lw }"
            RIGHT_CAMERAS="{ wrist: $_rw, top: $_top }"
            ;;
        none)
            LEFT_CAMERAS="{ wrist: $_lw }"
            RIGHT_CAMERAS="{ wrist: $_rw }"
            ;;
    esac
    unset _lw _rw _top
fi
unset -f _cam_entry

# -------- serial port permissions --------
for _port in "$LEFT_FOLLOWER_PORT" "$RIGHT_FOLLOWER_PORT" "$LEFT_LEADER_PORT" "$RIGHT_LEADER_PORT"; do
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
======= SO-101 Dual-Arm Dataset Recording (H.264, bi_so_follower) =======
 env       : $CONDA_ENV ($(python --version 2>&1))
 codec     : $VCODEC  (lerobot default streaming/threads)
 norm mode : use_degrees=$USE_DEGREES  (false → 본체 5DoF -100~100, gripper 0~100)
 follower  : $ROBOT_TYPE ($ROBOT_ID)
   left    : $LEFT_FOLLOWER_ID   port=$LEFT_FOLLOWER_PORT
   right   : $RIGHT_FOLLOWER_ID  port=$RIGHT_FOLLOWER_PORT
 leader    : $TELEOP_TYPE ($TELEOP_ID)
   left    : $LEFT_LEADER_ID    port=$LEFT_LEADER_PORT
   right   : $RIGHT_LEADER_ID   port=$RIGHT_LEADER_PORT
 cameras   : left_wrist  ($LEFT_WRIST_CAM_TYPE:$LEFT_WRIST_CAM_ID   ${LEFT_WRIST_CAM_WIDTH}x${LEFT_WRIST_CAM_HEIGHT}@${LEFT_WRIST_CAM_FPS}fps)
             right_wrist ($RIGHT_WRIST_CAM_TYPE:$RIGHT_WRIST_CAM_ID ${RIGHT_WRIST_CAM_WIDTH}x${RIGHT_WRIST_CAM_HEIGHT}@${RIGHT_WRIST_CAM_FPS}fps)
             top         ($TOP_CAM_TYPE:$TOP_CAM_ID ${TOP_CAM_WIDTH}x${TOP_CAM_HEIGHT}@${TOP_CAM_FPS}fps)
 top side  : TOP_ARM_SIDE=$TOP_ARM_SIDE  → 데이터셋 키: ${TOP_ARM_SIDE}_top (none이면 미저장)
 task      : $TASK
 dataset   : $REPO_ID  (root=$DATASET_ROOT)
 recording : ${NUM_EPISODES} episodes x ${EPISODE_TIME_S}s @ ${FPS}Hz  reset=${RESET_TIME_S}s
 resume    : $RESUME  (true → 기존 데이터셋에 이어서)
 push_hub  : $PUSH_TO_HUB
=========================================================================
EOF

# -------- run --------
RECORD_ARGS=(
    --robot.type="$ROBOT_TYPE"
    --robot.id="$ROBOT_ID"
    --robot.left_arm_config.port="$LEFT_FOLLOWER_PORT"
    --robot.left_arm_config.use_degrees="$USE_DEGREES"
    --robot.left_arm_config.cameras="$LEFT_CAMERAS"
    --robot.right_arm_config.port="$RIGHT_FOLLOWER_PORT"
    --robot.right_arm_config.use_degrees="$USE_DEGREES"
    --robot.right_arm_config.cameras="$RIGHT_CAMERAS"
    --teleop.type="$TELEOP_TYPE"
    --teleop.id="$TELEOP_ID"
    --teleop.left_arm_config.port="$LEFT_LEADER_PORT"
    --teleop.left_arm_config.use_degrees="$USE_DEGREES"
    --teleop.right_arm_config.port="$RIGHT_LEADER_PORT"
    --teleop.right_arm_config.use_degrees="$USE_DEGREES"
    --dataset.repo_id="$REPO_ID"
    --dataset.single_task="$TASK"
    --dataset.root="$DATASET_ROOT"
    --dataset.fps="$FPS"
    --dataset.episode_time_s="$EPISODE_TIME_S"
    --dataset.reset_time_s="$RESET_TIME_S"
    --dataset.num_episodes="$NUM_EPISODES"
    --dataset.push_to_hub="$PUSH_TO_HUB"
    --dataset.vcodec="$VCODEC"
    --display_data="$DISPLAY_DATA"
    --resume="$RESUME"
)

# 추가 인자 (예: --dataset.fps=60, --dataset.rename_map='{...}') 그대로 전달
RECORD_ARGS+=("$@")

lerobot-record "${RECORD_ARGS[@]}"
