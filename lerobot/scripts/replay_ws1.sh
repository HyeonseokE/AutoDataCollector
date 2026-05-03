#!/usr/bin/env bash
# =============================================================================
#  LeRobot Dataset Replay (Workspace 1)
#
#  로봇(follower)에 특정 dataset episode의 action 시퀀스를 재생합니다.
#  Replay 는 카메라를 사용하지 않습니다(action-only). 카메라 환경변수는
#  inference_ws1.sh 와의 일관성을 위해 그대로 받지만, 실제 실행에는 전달
#  되지 않고 로그에만 표시됩니다.
#
#  사용 예:
#    # 기본값 (DATASET_REPO_ID / EPISODE) 으로 재생
#    bash scripts/replay_ws1.sh
#
#    # 다른 에피소드 재생
#    EPISODE=3 bash scripts/replay_ws1.sh
#
#    # 다른 데이터셋 재생
#    DATASET_REPO_ID=skkuprism/SO101-single-pick_redblock_place_bluedish EPISODE=0 \
#        bash scripts/replay_ws1.sh
#
#    # 추가 draccus 인자 직접 전달
#    bash scripts/replay_ws1.sh --play_sounds=false
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

# -------- robot 설정 (inference_ws1.sh 와 동일 규약) --------
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-so101_robot2}"
# AutoDataCollector 자체 캘리브레이션 폴더 (lerobot 은 ${ROBOT_ID}.json 형식의 파일을 찾음;
# robotN_calibration.json 원본은 so101_robotN.json 심볼릭 링크로 매핑되어 있음)
ROBOT_CALIBRATION_DIR="${ROBOT_CALIBRATION_DIR:-$(cd "$REPO_DIR/.." && pwd)/robot_configs/motor_calibration/so101}"

# -------- 카메라 설정 (replay 에서는 미사용, 로그 출력용) --------
CAM_LEFT_WRIST_TYPE="${CAM_LEFT_WRIST_TYPE:-opencv}"
CAM_LEFT_WRIST_ID="${CAM_LEFT_WRIST_ID:-/dev/video18}"
CAM_LEFT_WRIST_WIDTH="${CAM_LEFT_WRIST_WIDTH:-640}"
CAM_LEFT_WRIST_HEIGHT="${CAM_LEFT_WRIST_HEIGHT:-480}"
CAM_LEFT_WRIST_FPS="${CAM_LEFT_WRIST_FPS:-30}"

CAM_TOP_TYPE="${CAM_TOP_TYPE:-intelrealsense}"
CAM_TOP_ID="${CAM_TOP_ID:-335622072328}"
CAM_TOP_WIDTH="${CAM_TOP_WIDTH:-640}"
CAM_TOP_HEIGHT="${CAM_TOP_HEIGHT:-480}"
CAM_TOP_FPS="${CAM_TOP_FPS:-30}"

# -------- dataset / 재생 설정 --------
DATASET_REPO_ID="${DATASET_REPO_ID:-skkuprism/test_pick_red_place_blue_50epi}"
EPISODE="${EPISODE:-1}"
DATASET_ROOT="${DATASET_ROOT:-}"     # 비워두면 HF cache 사용
FPS="${FPS:-30}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"

# -------- 요약 출력 --------
cat <<EOF
========= LeRobot Replay (WS1) =========
 env     : $CONDA_ENV ($(python --version 2>&1))
 lerobot : $REPO_DIR/src (PYTHONPATH)
 robot   : $ROBOT_TYPE ($ROBOT_ID)  port=$ROBOT_PORT
 calib   : $ROBOT_CALIBRATION_DIR
 cameras : (not used in replay — for reference only)
           left_wrist($CAM_LEFT_WRIST_TYPE:$CAM_LEFT_WRIST_ID ${CAM_LEFT_WRIST_WIDTH}x${CAM_LEFT_WRIST_HEIGHT}@${CAM_LEFT_WRIST_FPS}fps)
           top($CAM_TOP_TYPE:$CAM_TOP_ID ${CAM_TOP_WIDTH}x${CAM_TOP_HEIGHT}@${CAM_TOP_FPS}fps)
 dataset : $DATASET_REPO_ID  (episode=$EPISODE, fps=$FPS${DATASET_ROOT:+, root=$DATASET_ROOT})
 sounds  : $PLAY_SOUNDS
========================================
EOF

# -------- 시리얼 포트 권한 --------
if [ -e "$ROBOT_PORT" ]; then
    sudo chmod 777 "$ROBOT_PORT"
fi

# -------- 실행 --------
REPLAY_ARGS=(
    --robot.type="$ROBOT_TYPE"
    --robot.port="$ROBOT_PORT"
    --robot.id="$ROBOT_ID"
    --robot.calibration_dir="$ROBOT_CALIBRATION_DIR"
    --dataset.repo_id="$DATASET_REPO_ID"
    --dataset.episode="$EPISODE"
    --dataset.fps="$FPS"
    --play_sounds="$PLAY_SOUNDS"
)
if [ -n "$DATASET_ROOT" ]; then
    REPLAY_ARGS+=(--dataset.root="$DATASET_ROOT")
fi

# 추가 draccus 인자 전달
REPLAY_ARGS+=("$@")

python -m lerobot.scripts.lerobot_replay "${REPLAY_ARGS[@]}"
