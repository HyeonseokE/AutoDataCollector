#!/usr/bin/env bash
# =============================================================================
#  VLA Inference Script for LeRobot
#  Runs Vision-Language-Action models on real robots using RTC pipeline.
#  Supported policies: SmolVLA, Pi0, Pi0.5, GR00T, etc.
#
#   ./inference.sh                                          # 기본값으로 inference
#   POLICY_PATH=outputs/train/my_model/pretrained_model ./inference.sh
#   TASK="Pick up the red block" DURATION=60 ./inference.sh
#   ./inference.sh --rtc.max_guidance_weight=1.5            # draccus 인자 직접
#
# 카메라 이름은 학습 시 데이터셋의 카메라 이름과 동일해야 합니다.
# 학습 시 rename_map 이 preprocessor 에 저장되므로, inference 시
# 로봇 카메라를 데이터셋 이름(left_wrist, top)으로 설정하면
# preprocessor 가 자동으로 policy 이름(camera1, camera2)으로 변환합니다.
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
export PYTHONNOUSERSITE=1   # ~/.local (user-site) 차단
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -------- PYTHONPATH: AutoDataCollector lerobot 사용 --------
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# -------- policy 설정 --------
POLICY_PATH="${POLICY_PATH:-skkuprism/test_model_teleop}"   # HF model ID 또는 로컬 체크포인트 경로
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"                # cuda / cpu / mps

# -------- robot 설정 --------
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"            # so100_follower / so101_follower / koch_follower / ...
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"              # 시리얼 포트
ROBOT_ID="${ROBOT_ID:-so101_robot0}"                  # 캘리브레이션 파일용 ID
# 액션 단위: true=degrees(arm) + gripper 0~100 / false=-100~+100(arm) + gripper 0~100
# 학습 데이터셋과 반드시 일치해야 함 (불일치 시 로봇 엉뚱한 동작)
ROBOT_USE_DEGREES="${ROBOT_USE_DEGREES:-false}"

# -------- 카메라 설정 --------
# 키 이름 = 학습 데이터셋의 카메라 이름 (rename_map 의 source 쪽)
# train_smolvla.sh 의 CAMERA_RENAME_PAIRS=("left_wrist:camera1" "top:camera2") 와 대응.
#
# 각 카메라별로 타입과 식별자를 독립 지정:
#   CAM_xxx_TYPE = opencv | intelrealsense
#   CAM_xxx_ID   = opencv일 때 경로/인덱스 ("/dev/video18", 0)
#                  realsense일 때 시리얼 번호 ("335622072328")
CAM_LEFT_WRIST_TYPE="${CAM_LEFT_WRIST_TYPE:-opencv}"
CAM_LEFT_WRIST_ID="${CAM_LEFT_WRIST_ID:-/dev/video6}"
CAM_LEFT_WRIST_WIDTH="${CAM_LEFT_WRIST_WIDTH:-640}"
CAM_LEFT_WRIST_HEIGHT="${CAM_LEFT_WRIST_HEIGHT:-480}"
CAM_LEFT_WRIST_FPS="${CAM_LEFT_WRIST_FPS:-30}"

CAM_TOP_TYPE="${CAM_TOP_TYPE:-intelrealsense}"
CAM_TOP_ID="${CAM_TOP_ID:-254622075836}"
CAM_TOP_WIDTH="${CAM_TOP_WIDTH:-640}"
CAM_TOP_HEIGHT="${CAM_TOP_HEIGHT:-480}"
CAM_TOP_FPS="${CAM_TOP_FPS:-30}"

# 타입에 따라 카메라 dict 항목 조립
_cam_entry() {
    local type="$1" id="$2" w="$3" h="$4" fps="$5"
    if [ "$type" = "opencv" ]; then
        echo "{type: opencv, index_or_path: $id, width: $w, height: $h, fps: $fps}"
    else
        echo "{type: intelrealsense, serial_number_or_name: '$id', width: $w, height: $h, fps: $fps}"
    fi
}

if [ -z "${CAMERAS+x}" ]; then
    _lw=$(_cam_entry "$CAM_LEFT_WRIST_TYPE" "$CAM_LEFT_WRIST_ID" "$CAM_LEFT_WRIST_WIDTH" "$CAM_LEFT_WRIST_HEIGHT" "$CAM_LEFT_WRIST_FPS")
    _top=$(_cam_entry "$CAM_TOP_TYPE" "$CAM_TOP_ID" "$CAM_TOP_WIDTH" "$CAM_TOP_HEIGHT" "$CAM_TOP_FPS")
    CAMERAS="{ left_wrist: $_lw, top: $_top }"
    unset _lw _top
fi
unset -f _cam_entry

# -------- task / 실행 설정 --------
TASK="${TASK:-pick up the red block and place it on the blue dish.}"                     # language instruction
DURATION="${DURATION:-120}"                           # 실행 시간 (초)
FPS="${FPS:-30}"                                      # action 실행 주파수 (Hz)

# -------- RTC (Real-Time Chunking) 설정 --------
RTC_ENABLED="${RTC_ENABLED:-false}"                    # true / false
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-20}"  # chunk 당 실행 스텝 수

# -------- torch compile (선택) --------
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-false}"       # true / false

# -------- action chunk 저장 (시각화용) --------
SAVE_CHUNKS="${SAVE_CHUNKS:-true}"                   # true / false
SAVE_CHUNKS_DIR="${SAVE_CHUNKS_DIR:-$REPO_DIR/outputs/action_chunks}"
SAVE_CHUNKS_MAX="${SAVE_CHUNKS_MAX:-15}"              # 저장할 chunk 수

# -------- 요약 출력 --------
cat <<EOF
========= VLA Inference =========
 env     : $CONDA_ENV ($(python --version 2>&1))
 lerobot : $REPO_DIR/src (PYTHONPATH)
 policy  : $POLICY_PATH
 device  : $POLICY_DEVICE
 robot   : $ROBOT_TYPE ($ROBOT_ID)  port=$ROBOT_PORT  use_degrees=$ROBOT_USE_DEGREES
 cameras : left_wrist($CAM_LEFT_WRIST_TYPE:$CAM_LEFT_WRIST_ID ${CAM_LEFT_WRIST_WIDTH}x${CAM_LEFT_WRIST_HEIGHT}@${CAM_LEFT_WRIST_FPS}fps)
           top($CAM_TOP_TYPE:$CAM_TOP_ID ${CAM_TOP_WIDTH}x${CAM_TOP_HEIGHT}@${CAM_TOP_FPS}fps)
 task    : $TASK
 rtc     : enabled=$RTC_ENABLED  horizon=$RTC_EXECUTION_HORIZON
 runtime : ${DURATION}s @ ${FPS}Hz  compile=$USE_TORCH_COMPILE
=================================
EOF

# -------- 시리얼 포트 권한 --------
if [ -e "$ROBOT_PORT" ]; then
    sudo chmod 777 "$ROBOT_PORT"
fi

# -------- 실행 --------
INFER_ARGS=(
    --policy.path="$POLICY_PATH"
    --policy.device="$POLICY_DEVICE"
    --rtc.enabled="$RTC_ENABLED"
    --rtc.execution_horizon="$RTC_EXECUTION_HORIZON"
    --robot.type="$ROBOT_TYPE"
    --robot.port="$ROBOT_PORT"
    --robot.id="$ROBOT_ID"
    --robot.use_degrees="$ROBOT_USE_DEGREES"
    --robot.cameras="$CAMERAS"
    --task="$TASK"
    --duration="$DURATION"
    --fps="$FPS"
    --use_torch_compile="$USE_TORCH_COMPILE"
    --save_chunks="$SAVE_CHUNKS"
    --save_chunks_dir="$SAVE_CHUNKS_DIR"
    --save_chunks_max="$SAVE_CHUNKS_MAX"
)

# 추가 draccus 인자 전달
INFER_ARGS+=("$@")

python "$REPO_DIR/examples/rtc/eval_with_real_robot.py" "${INFER_ARGS[@]}"
