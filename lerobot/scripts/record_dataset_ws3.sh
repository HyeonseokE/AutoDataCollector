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
#    FOLLOWER_PORT=/dev/ttyACM1 LEADER_PORT=/dev/ttyACM4 ./record_dataset.sh
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
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# AutoDataCollector root (parent of the vendored lerobot dir) — needed so the
# EE-trace add-on can import lerobot_cap.kinematics (ADC_ROOT/src) and the
# record_dataset package (ADC_ROOT).
ADC_ROOT="$(cd "$REPO_DIR/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:$ADC_ROOT:$ADC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# -------- video codec (H.264) --------
# 기본은 소프트웨어 h264.
VCODEC="${VCODEC:-h264}" # always h264 - hscho
STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"

# -------- normalization mode --------
# false: 본체 5 모터 -100~100, gripper 0~100 (gripper는 항상 0~100 하드코딩)
# true:  본체 5 모터 degrees, gripper 0~100
USE_DEGREES="${USE_DEGREES:-${ROBOT_USE_DEGREES:-false}}" # always false - hscho

# -------- dataset --------
REPO_ID="${REPO_ID:-CoRL2026-CSI/teleop_open_drawer_temp}"
TASK="${TASK:-Open the top drawer.}"
FPS="${FPS:-30}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
RESET_TIME_S="${RESET_TIME_S:-5}"
NUM_EPISODES="${NUM_EPISODES:-100}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DATASET_ROOT="${DATASET_ROOT:-$REPO_DIR/outputs/datasets/$REPO_ID}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"

# -------- resume --------
RESUME="${RESUME:-false}"
if [ "$RESUME" = "true" ] && [ ! -f "$DATASET_ROOT/meta/info.json" ]; then
    echo "[WARN] RESUME=true but no local dataset metadata found: $DATASET_ROOT/meta/info.json" >&2
    echo "[WARN] Starting a new local dataset instead. Set RESUME=true after the first successful recording." >&2
    RESUME=false
fi

# -------- follower (robot) --------
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
FOLLOWER_PORT="${FOLLOWER_PORT:-${ROBOT_PORT:-/dev/ttyACM0}}"
FOLLOWER_ID="${FOLLOWER_ID:-${ROBOT_ID:-so101_robot4}}"
FOLLOWER_CALIBRATION_DIR="${FOLLOWER_CALIBRATION_DIR:-${ROBOT_CALIBRATION_DIR:-$ADC_ROOT/robot_configs/motor_calibration/so101}}"

# -------- leader (teleop) --------
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"
LEADER_ID="${LEADER_ID:-so101_robot4_leader}"
LEADER_CALIBRATION_DIR="${LEADER_CALIBRATION_DIR:-$HOME/.cache/huggingface/lerobot/calibration/teleoperators/so_leader}"

# -------- live 3D EE-trajectory trace (rerun) --------
# 텔레오퍼레이션 데모의 EE (x,y,z) 경로를 3D 에 실시간 누적, 에피소드마다 다른 색.
# lerobot 의 display_data(rerun) 세션에 piggyback → DISPLAY_DATA=true 필요.
# robot id 는 FOLLOWER_ID(so101_robot<N>) 에서 자동 도출. URDF/calib 는
# AutoDataCollector/assets, robot_configs 에서 자동 해석 (override 가능).
# reset 구간은 제외(데모 에피소드만), 종료 시 EE_TRACE_NPZ 로 덤프 →
# python scripts/render_ee_trace_video.py 로 mp4 렌더.
EE_TRACE="${EE_TRACE:-true}"
if [ "$EE_TRACE" = "true" ]; then
    export EE_TRACE_ENABLED=true
    # so101_robot4 → 4
    export EE_TRACE_ROBOT_ID="${EE_TRACE_ROBOT_ID:-$(echo "$FOLLOWER_ID" | grep -oE '[0-9]+$' || true)}"
    export EE_TRACE_NPZ="${EE_TRACE_NPZ:-$DATASET_ROOT/ee_trace.npz}"
    # 선택 override:
    # export EE_TRACE_URDF="$ADC_ROOT/assets/urdf/so101_robot${EE_TRACE_ROBOT_ID}.urdf"
    # export EE_TRACE_CALIB="$ADC_ROOT/robot_configs/motor_calibration/so101/robot${EE_TRACE_ROBOT_ID}_calibration.json"
fi

# -------- cameras --------
# WS3 inference와 동일한 카메라 식별자.
# /dev/videoN은 USB 재연결/재부팅마다 바뀔 수 있으므로 wrist는 by-id 경로를 기본값으로 사용.
CAM_LEFT_WRIST_TYPE="${CAM_LEFT_WRIST_TYPE:-opencv}"
CAM_LEFT_WRIST_ID="${CAM_LEFT_WRIST_ID:-/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0}"
CAM_LEFT_WRIST_WIDTH="${CAM_LEFT_WRIST_WIDTH:-640}"
CAM_LEFT_WRIST_HEIGHT="${CAM_LEFT_WRIST_HEIGHT:-480}"
CAM_LEFT_WRIST_FPS="${CAM_LEFT_WRIST_FPS:-30}"
CAM_LEFT_WRIST_FOURCC="${CAM_LEFT_WRIST_FOURCC:-MJPG}"

CAM_TOP_TYPE="${CAM_TOP_TYPE:-intelrealsense}"
CAM_TOP_ID="${CAM_TOP_ID:-254622073196}"
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
 codec     : $VCODEC  (streaming=$STREAMING_ENCODING, threads=$ENCODER_THREADS)
 norm mode : use_degrees=$USE_DEGREES  (false → 본체 5DoF -100~100, gripper 0~100)
 follower  : $ROBOT_TYPE ($FOLLOWER_ID)  port=$FOLLOWER_PORT
 calib     : $FOLLOWER_CALIBRATION_DIR
 leader    : $TELEOP_TYPE ($LEADER_ID)   port=$LEADER_PORT
 leader cal: $LEADER_CALIBRATION_DIR
 cameras   : left_wrist($CAM_LEFT_WRIST_TYPE:$CAM_LEFT_WRIST_ID ${CAM_LEFT_WRIST_WIDTH}x${CAM_LEFT_WRIST_HEIGHT}@${CAM_LEFT_WRIST_FPS}fps)
             top($CAM_TOP_TYPE:$CAM_TOP_ID ${CAM_TOP_WIDTH}x${CAM_TOP_HEIGHT}@${CAM_TOP_FPS}fps)
 task      : $TASK
 dataset   : $REPO_ID  (root=$DATASET_ROOT)
 recording : ${NUM_EPISODES} episodes x ${EPISODE_TIME_S}s @ ${FPS}Hz  reset=${RESET_TIME_S}s
 resume    : $RESUME  (true → 기존 데이터셋에 이어서)
 push_hub  : $PUSH_TO_HUB
====================================================
EOF

# -------- run --------
RECORD_ARGS=(
    --robot.type="$ROBOT_TYPE"
    --robot.port="$FOLLOWER_PORT"
    --robot.id="$FOLLOWER_ID"
    --robot.calibration_dir="$FOLLOWER_CALIBRATION_DIR"
    --robot.use_degrees="$USE_DEGREES"
    --robot.cameras="$CAMERAS"
    --teleop.type="$TELEOP_TYPE"
    --teleop.port="$LEADER_PORT"
    --teleop.id="$LEADER_ID"
    --teleop.calibration_dir="$LEADER_CALIBRATION_DIR"
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
