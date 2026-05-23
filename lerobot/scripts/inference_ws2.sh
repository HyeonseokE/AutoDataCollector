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
CALL_DIR="$(pwd)"
cd "$REPO_DIR"

# -------- conda env --------
CONDA_ENV="${CONDA_ENV:-lerobot}"
# shellcheck disable=SC1091
set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
export PYTHONNOUSERSITE=1   # ~/.local (user-site) 차단
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -------- PYTHONPATH: AutoDataCollector lerobot 사용 --------
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# -------- policy 설정 --------
POLICY_PATH="${POLICY_PATH:-CoRL2026-CSI/smol_teleop_sort_block}"   # HF model ID 또는 로컬 체크포인트 경로
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"                # cuda / cpu / mps

# 일부 실험용 GR00T 체크포인트는 Hub에 policy_preprocessor 파일이 빠져 있어,
# 같은 카메라/로봇 구성의 기준 체크포인트 preprocessor를 붙인 로컬 overlay를 사용한다.
if [ "$POLICY_PATH" = "CoRL2026-CSI/Gr00t_CaP_Real_OpenTopDrawer_100ep_projector_actionexpert" ]; then
    _overlay="$REPO_DIR/outputs/local_checkpoints/Gr00t_CaP_Real_OpenTopDrawer_100ep_projector_actionexpert_with_preprocessor"
    if [ -d "$_overlay" ]; then
        POLICY_PATH="$_overlay"
    fi
    unset _overlay
fi
if [ "$POLICY_PATH" = "CoRL2026-CSI/Gr00t_CaP_Real_OpenTopDrawer_50ep_projector_actionexpert" ]; then
    _overlay="$REPO_DIR/outputs/local_checkpoints/Gr00t_CaP_Real_OpenTopDrawer_50ep_projector_actionexpert_with_preprocessor"
    if [ -d "$_overlay" ]; then
        POLICY_PATH="$_overlay"
    fi
    unset _overlay
fi
if [ "$POLICY_PATH" = "CoRL2026-CSI/Gr00t_CaP_Real_OpenTopDrawer_150ep_projector_actionexpert" ]; then
    _overlay="$REPO_DIR/outputs/local_checkpoints/Gr00t_CaP_Real_OpenTopDrawer_150ep_projector_actionexpert_with_preprocessor"
    if [ -d "$_overlay" ]; then
        POLICY_PATH="$_overlay"
    fi
    unset _overlay
fi

# -------- robot 설정 --------
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"            # so100_follower / so101_follower / koch_follower / ...
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"              # robot0 시리얼 포트
ROBOT_ID="${ROBOT_ID:-so101_robot0}"                  # 캘리브레이션 파일용 ID
# AutoDataCollector 자체 캘리브레이션 폴더 (lerobot 은 ${ROBOT_ID}.json 형식의 파일을 찾음;
# robotN_calibration.json 원본은 so101_robotN.json 심볼릭 링크로 매핑되어 있음)
ROBOT_CALIBRATION_DIR="${ROBOT_CALIBRATION_DIR:-$(cd "$REPO_DIR/.." && pwd)/robot_configs/motor_calibration/so101}"
# 액션 단위: true=degrees(arm) + gripper 0~100 / false=-100~+100(arm) + gripper 0~100
# 학습 데이터셋과 반드시 일치해야 함 (불일치 시 로봇 엉뚱한 동작)
ROBOT_USE_DEGREES="${ROBOT_USE_DEGREES:-false}"
# 현재 위치 대비 한 번에 보낼 수 있는 목표 변화량 제한. 비워두면 제한 없음.
ROBOT_MAX_RELATIVE_TARGET="${ROBOT_MAX_RELATIVE_TARGET:-}"

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
CAM_TOP_ID="${CAM_TOP_ID:-254622079503}"
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
DURATION="${DURATION:-999999}"                        # 실행 시간 (초)
FPS="${FPS:-10}"                                      # action 실행 주파수 (Hz)
STOP_AND_INFER="${STOP_AND_INFER:-false}"             # true면 chunk 실행 후 멈춰서 다음 chunk 추론

# -------- RTC (Real-Time Chunking) 설정 --------
RTC_ENABLED="${RTC_ENABLED:-false}"                    # true / false
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-20}"  # chunk 당 실행 스텝 수
CHUNK_TRANSITION_STEPS="${CHUNK_TRANSITION_STEPS:-4}" # non-RTC chunk 시작부 blending
ACTION_MAX_DELTA="${ACTION_MAX_DELTA:-}"              # action queue 단계 per-step delta 제한

# -------- torch compile (선택) --------
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-false}"       # true / false

# -------- action chunk 저장 (시각화용) --------
SAVE_CHUNKS="${SAVE_CHUNKS:-true}"                   # true / false
SAVE_CHUNKS_DIR="${SAVE_CHUNKS_DIR:-$REPO_DIR/outputs/action_chunks}"
SAVE_CHUNKS_MAX="${SAVE_CHUNKS_MAX:-15}"              # 저장할 chunk 수

# -------- Enter-pause 중 작은 로컬 카메라 창 --------
PAUSE_STREAM_ENABLED="${PAUSE_STREAM_ENABLED:-true}"  # true / false
PAUSE_STREAM_CAMERA="${PAUSE_STREAM_CAMERA:-top}"
PAUSE_STREAM_FPS="${PAUSE_STREAM_FPS:-15}"

# -------- 요약 출력 --------
cat <<EOF
========= VLA Inference =========
 env     : $CONDA_ENV ($(python --version 2>&1))
 lerobot : $REPO_DIR/src (PYTHONPATH)
 policy  : $POLICY_PATH
 device  : $POLICY_DEVICE
 robot   : $ROBOT_TYPE ($ROBOT_ID)  port=$ROBOT_PORT  use_degrees=$ROBOT_USE_DEGREES
 max_rel : ${ROBOT_MAX_RELATIVE_TARGET:-none}
 calib   : $ROBOT_CALIBRATION_DIR
 cameras : left_wrist($CAM_LEFT_WRIST_TYPE:$CAM_LEFT_WRIST_ID ${CAM_LEFT_WRIST_WIDTH}x${CAM_LEFT_WRIST_HEIGHT}@${CAM_LEFT_WRIST_FPS}fps)
           top($CAM_TOP_TYPE:$CAM_TOP_ID ${CAM_TOP_WIDTH}x${CAM_TOP_HEIGHT}@${CAM_TOP_FPS}fps)
 task    : $TASK
 mode    : stop_and_infer=$STOP_AND_INFER
 rtc     : enabled=$RTC_ENABLED  horizon=$RTC_EXECUTION_HORIZON
 smooth  : transition_steps=$CHUNK_TRANSITION_STEPS  action_max_delta=${ACTION_MAX_DELTA:-none}
 runtime : ${DURATION}s @ ${FPS}Hz  compile=$USE_TORCH_COMPILE
 pause   : window=$PAUSE_STREAM_ENABLED camera=$PAUSE_STREAM_CAMERA
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
    --chunk_transition_steps="$CHUNK_TRANSITION_STEPS"
    --robot.type="$ROBOT_TYPE"
    --robot.port="$ROBOT_PORT"
    --robot.id="$ROBOT_ID"
    --robot.calibration_dir="$ROBOT_CALIBRATION_DIR"
    --robot.use_degrees="$ROBOT_USE_DEGREES"
    --robot.cameras="$CAMERAS"
    --task="$TASK"
    --duration="$DURATION"
    --fps="$FPS"
    --stop_and_infer="$STOP_AND_INFER"
    --use_torch_compile="$USE_TORCH_COMPILE"
    --save_chunks="$SAVE_CHUNKS"
    --save_chunks_dir="$SAVE_CHUNKS_DIR"
    --save_chunks_max="$SAVE_CHUNKS_MAX"
    --pause_stream_enabled="$PAUSE_STREAM_ENABLED"
    --pause_stream_camera="$PAUSE_STREAM_CAMERA"
    --pause_stream_fps="$PAUSE_STREAM_FPS"
)

if [ -n "$ROBOT_MAX_RELATIVE_TARGET" ]; then
    INFER_ARGS+=(--robot.max_relative_target="$ROBOT_MAX_RELATIVE_TARGET")
fi
if [ -n "$ACTION_MAX_DELTA" ]; then
    INFER_ARGS+=(--action_max_delta="$ACTION_MAX_DELTA")
fi

# 추가 draccus 인자 전달
INFER_ARGS+=("$@")

python "$REPO_DIR/examples/rtc/eval_with_real_robot.py" "${INFER_ARGS[@]}"
