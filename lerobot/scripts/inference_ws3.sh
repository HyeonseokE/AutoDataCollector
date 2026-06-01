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
# miniconda3 / anaconda3 자동 탐지 (CONDA_BASE 로 명시적 override 가능)
if [ -z "${CONDA_BASE:-}" ]; then
    for _cand in "$HOME/miniconda3" "$HOME/anaconda3" "/opt/miniconda3" "/opt/anaconda3"; do
        if [ -f "$_cand/etc/profile.d/conda.sh" ]; then
            CONDA_BASE="$_cand"
            break
        fi
    done
fi
if [ -z "${CONDA_BASE:-}" ] || [ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "[ERROR] conda.sh 를 찾지 못했습니다. CONDA_BASE 환경변수로 지정하세요." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1   # ~/.local (user-site) 차단
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -------- PYTHONPATH: AutoDataCollector lerobot 사용 --------
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# -------- policy 설정 --------
POLICY_PATH="${POLICY_PATH:-CoRL2026-CSI/smolVLA_DistributeChoco_Ours_100epi_table1_new_50ep}"   # HF model ID 또는 로컬 체크포인트 경로
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"                # cuda / cpu / mps

# -------- robot 설정 --------
# WS3: run_forward_and_reset_ws3.sh ROBOT_IDS=(4) → robot4
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"            # so100_follower / so101_follower / koch_follower / ...
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"              # 시리얼 포트 (record_dataset_ws3.sh 와 동일)
ROBOT_ID="${ROBOT_ID:-so101_robot4}"                  # 캘리브레이션 파일용 ID (so101_robot4.json 심볼릭 링크 필요)
# AutoDataCollector 자체 캘리브레이션 폴더 (lerobot 은 ${ROBOT_ID}.json 형식의 파일을 찾음;
# robotN_calibration.json 원본은 so101_robotN.json 심볼릭 링크로 매핑되어 있음)
ROBOT_CALIBRATION_DIR="${ROBOT_CALIBRATION_DIR:-$(cd "$REPO_DIR/.." && pwd)/robot_configs/motor_calibration/so101}"
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
# WS3: recording_config_ws3.yaml + record_dataset_ws3.sh 와 동일한 카메라 식별자
#   - left_wrist : opencv Innomaker U20CAM (wrist) — MJPG
#   - top        : realsense 254622073196 (WS3 전용)
# 주의: /dev/videoN 인덱스는 USB 재연결/재부팅마다 바뀜 (video0→video1 등).
#       인덱스 대신 by-id 안정 경로를 기본값으로 사용 → 항상 같은 물리 카메라.
#       다른 wrist 캠이면 `ls /dev/v4l/by-id/` 로 경로 확인 후 override.
CAM_LEFT_WRIST_TYPE="${CAM_LEFT_WRIST_TYPE:-opencv}"
CAM_LEFT_WRIST_ID="${CAM_LEFT_WRIST_ID:-/dev/v4l/by-id/usb-Innomaker_Innomaker-U20CAM-1080p-S1_SN0001-video-index0}"
CAM_LEFT_WRIST_WIDTH="${CAM_LEFT_WRIST_WIDTH:-640}"
CAM_LEFT_WRIST_HEIGHT="${CAM_LEFT_WRIST_HEIGHT:-480}"
CAM_LEFT_WRIST_FPS="${CAM_LEFT_WRIST_FPS:-30}"
CAM_LEFT_WRIST_FOURCC="${CAM_LEFT_WRIST_FOURCC:-MJPG}"  # 학습 시 MJPG 로 캡처됨 — 동일 코덱 유지

CAM_TOP_TYPE="${CAM_TOP_TYPE:-intelrealsense}"
CAM_TOP_ID="${CAM_TOP_ID:-254622073196}"
CAM_TOP_WIDTH="${CAM_TOP_WIDTH:-640}"
CAM_TOP_HEIGHT="${CAM_TOP_HEIGHT:-480}"
CAM_TOP_FPS="${CAM_TOP_FPS:-30}"

# 타입에 따라 카메라 dict 항목 조립
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

# -------- task / 실행 설정 --------
TASK="${TASK:-distribute chocolate pies to each plate.}"                     # language instruction (run_forward_and_reset_ws3.sh INSTRUCTION 과 동일)
DURATION="${DURATION:-9999}"                           # 실행 시간 (초)
FPS="${FPS:-10}"                                      # action 실행 주파수 (Hz)

# -------- policy chunk 설정 (선택, 비우면 학습된 모델 config 그대로 사용) --------
# chunk_size      : policy 가 한 번에 예측하는 action chunk 길이 (학습 값과 일치시켜야 안전 — smolvla default=50)
# n_action_steps  : 그 중 실제로 실행할 step 수. chunk_size 보다 작게 줄이면 더 자주 replan
#                   (예: chunk_size=50, n_action_steps=20 → 50개 예측 → 앞 20개 실행 → 다음 chunk 요청)
POLICY_CHUNK_SIZE="${POLICY_CHUNK_SIZE:-50}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-20}"

# -------- RTC (Real-Time Chunking) 설정 --------
RTC_ENABLED="${RTC_ENABLED:-false}"                    # true / false
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-24}"  # chunk 당 실행 스텝 수 (RTC_ENABLED=true 일 때만 의미)

# -------- torch compile (선택) --------
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-false}"       # true / false

# -------- action chunk 저장 (시각화용) --------
SAVE_CHUNKS="${SAVE_CHUNKS:-true}"                   # true / false
SAVE_CHUNKS_DIR="${SAVE_CHUNKS_DIR:-$REPO_DIR/outputs/action_chunks}"
SAVE_CHUNKS_MAX="${SAVE_CHUNKS_MAX:-15}"              # 저장할 chunk 수

# -------- Enter-park 시 top-view 라이브 프리뷰 --------
# 1st Enter: free_state 로 복귀 + 아래 카메라를 local UI 로 라이브스트리밍 시작.
# 2nd Enter: 프리뷰 종료 + action buffer 비우고 inference 재개.
PAUSE_STREAM_ENABLED="${PAUSE_STREAM_ENABLED:-true}"  # true / false
PAUSE_STREAM_CAMERA="${PAUSE_STREAM_CAMERA:-top}"     # 데이터셋 카메라 키 (top = RealSense top-view)
PAUSE_STREAM_FPS="${PAUSE_STREAM_FPS:-15}"            # 프리뷰 갱신 주파수 (Hz)
PAUSE_STREAM_SCALE="${PAUSE_STREAM_SCALE:-2.0}"       # 프리뷰 창 확대 배율 (640x480 → 2.0배 = 1280x960)

# -------- 요약 출력 --------
cat <<EOF
========= VLA Inference =========
 env     : $CONDA_ENV ($(python --version 2>&1))
 lerobot : $REPO_DIR/src (PYTHONPATH)
 policy  : $POLICY_PATH
 device  : $POLICY_DEVICE
 robot   : $ROBOT_TYPE ($ROBOT_ID)  port=$ROBOT_PORT  use_degrees=$ROBOT_USE_DEGREES
 calib   : $ROBOT_CALIBRATION_DIR
 cameras : left_wrist($CAM_LEFT_WRIST_TYPE:$CAM_LEFT_WRIST_ID ${CAM_LEFT_WRIST_WIDTH}x${CAM_LEFT_WRIST_HEIGHT}@${CAM_LEFT_WRIST_FPS}fps fourcc=$CAM_LEFT_WRIST_FOURCC)
           top($CAM_TOP_TYPE:$CAM_TOP_ID ${CAM_TOP_WIDTH}x${CAM_TOP_HEIGHT}@${CAM_TOP_FPS}fps)
 task    : $TASK
 chunk   : chunk_size=${POLICY_CHUNK_SIZE:-<model default>}  n_action_steps=${POLICY_N_ACTION_STEPS:-<model default>}
 rtc     : enabled=$RTC_ENABLED  horizon=$RTC_EXECUTION_HORIZON
 pause   : stream=$PAUSE_STREAM_ENABLED  camera=$PAUSE_STREAM_CAMERA @ ${PAUSE_STREAM_FPS}fps  scale=${PAUSE_STREAM_SCALE}x (Enter=park+preview / Enter=quit+resume)
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
    --robot.calibration_dir="$ROBOT_CALIBRATION_DIR"
    --robot.use_degrees="$ROBOT_USE_DEGREES"
    --robot.cameras="$CAMERAS"
    --task="$TASK"
    --duration="$DURATION"
    --fps="$FPS"
    --use_torch_compile="$USE_TORCH_COMPILE"
    --save_chunks="$SAVE_CHUNKS"
    --save_chunks_dir="$SAVE_CHUNKS_DIR"
    --save_chunks_max="$SAVE_CHUNKS_MAX"
    --pause_stream_enabled="$PAUSE_STREAM_ENABLED"
    --pause_stream_camera="$PAUSE_STREAM_CAMERA"
    --pause_stream_fps="$PAUSE_STREAM_FPS"
    --pause_stream_scale="$PAUSE_STREAM_SCALE"
)

# 선택 인자: policy chunk 설정 (비어있으면 모델 config 값 유지)
if [ -n "$POLICY_CHUNK_SIZE" ]; then
    INFER_ARGS+=(--policy.chunk_size="$POLICY_CHUNK_SIZE")
fi
if [ -n "$POLICY_N_ACTION_STEPS" ]; then
    INFER_ARGS+=(--policy.n_action_steps="$POLICY_N_ACTION_STEPS")
fi

# 추가 draccus 인자 전달
INFER_ARGS+=("$@")

python "$REPO_DIR/examples/rtc/eval_with_real_robot.py" "${INFER_ARGS[@]}"
