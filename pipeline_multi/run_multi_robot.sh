#!/bin/bash
# Multi-Robot Pipeline Runner
# 여러 로봇을 동시에 제어하면서 데이터셋 레코딩을 수행하는 파이프라인
#
# Config 파일들 (pipeline_multi/pipeline_config_multi/):
#   - detection_config.yaml      : 객체 검출 설정
#   - paid_api_config.yaml       : 유료 API 설정 (USE_SERVER=false)
#   - free_api_config.yaml       : vLLM 서버 설정 (USE_SERVER=true)
#   - recording_config.yaml      : 레코딩 설정 (RECORD_DATASET=true)
#   - multi_robot_config.yaml    : 멀티로봇 전용 설정

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# ============================================================
# 핵심 설정 (Essential Configuration)
# 아래 값들은 config yaml 로드 후 override 됩니다.
# 직접 수정하거나 config yaml을 수정하세요.
# ============================================================

# [필수] 로봇 ID 목록 (공백으로 구분)
ROBOT_IDS=(2 3)

# [필수] 태스크 명령어
INSTRUCTION="assemble the red part and the pink part."

# [필수] 결과 저장 경로
SAVE_DIR="./pipeline_multi/results_multi"

# [필수] 에피소드 반복 횟수
NUM_EPISODES=1

# 서버 추론 사용 여부 (true: vLLM 서버, false: 유료 API)
USE_SERVER=true

# ============================================================
# Reset execution 설정
# ============================================================

# Reset 실행 여부 (false면 Forward + Judge만 실행)
EXECUTE_RESET=false

# Reset 모드 ("original": 원래 위치로 복귀, "random": 랜덤 위치로 배치)
RESET_MODE="original"

# ============================================================
# Judge execution 설정
# ============================================================

# Judge 실행 여부 (true면 Judge 단계 건너뛰기)
SKIP_JUDGE=true

# ============================================================
# Dataset Recording 설정
# ============================================================

# LeRobot 데이터셋 레코딩 활성화
RECORD_DATASET=true

# ============================================================
# Detection 설정
# ============================================================

# Detection timeout (초)
DETECTION_TIMEOUT=15.0

# Detection 시각화 여부
VISUALIZE_DETECTION=false

# ============================================================
# Config 파일 로드 함수
# pipeline_multi/pipeline_config_multi/ 디렉토리에서 로드
# ============================================================

CONFIG_DIR="$SCRIPT_DIR/pipeline_config_multi"
PARSE_YAML="$CONFIG_DIR/parse_yaml.py"

load_multi_robot_config() {
    local config_file="$CONFIG_DIR/multi_robot_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$PARSE_YAML" "$config_file")"
        echo "[Config] Loaded: multi_robot_config.yaml"

        # multi_robot_config에서 로봇 ID 추출
        if [ -n "$ROBOT_IDS" ]; then
            echo "  Robot IDs: ${ROBOT_IDS[*]}"
        fi

        # execution 설정 적용
        if [ -n "$EXECUTION_EXECUTE_RESET" ]; then
            EXECUTE_RESET="$EXECUTION_EXECUTE_RESET"
        fi
        if [ -n "$EXECUTION_RESET_MODE" ]; then
            RESET_MODE="$EXECUTION_RESET_MODE"
        fi
        if [ -n "$EXECUTION_SKIP_JUDGE" ]; then
            SKIP_JUDGE="$EXECUTION_SKIP_JUDGE"
        fi

        # recording 설정 적용
        if [ -n "$RECORDING_ENABLED" ]; then
            RECORD_DATASET="$RECORDING_ENABLED"
        fi
        if [ -n "$RECORDING_FPS" ]; then
            RECORDING_FPS="$RECORDING_FPS"
        fi

        # results 설정 적용
        if [ -n "$RESULTS_SAVE_DIR" ]; then
            SAVE_DIR="$RESULTS_SAVE_DIR"
        fi

        # task 설정 적용
        if [ -n "$TASK_INSTRUCTION" ]; then
            INSTRUCTION="$TASK_INSTRUCTION"
        fi
        if [ -n "$TASK_NUM_EPISODES" ]; then
            NUM_EPISODES="$TASK_NUM_EPISODES"
        fi
    else
        echo "[Config] Warning: multi_robot_config.yaml not found, using defaults"
    fi
}

load_detection_config() {
    local config_file="$CONFIG_DIR/detection_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$PARSE_YAML" "$config_file")"
        echo "[Config] Loaded: detection_config.yaml"
    else
        echo "[Config] Warning: detection_config.yaml not found, using defaults"
        OBJECTS=("green towel")
        DETECTION_TIMEOUT=15.0
        VISUALIZE_DETECTION=false
    fi
}

load_paid_api_config() {
    local config_file="$CONFIG_DIR/paid_api_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$PARSE_YAML" "$config_file")"
        echo "[Config] Loaded: paid_api_config.yaml"
    else
        echo "[Config] Warning: paid_api_config.yaml not found, using defaults"
        CODEGEN_LLM_MODEL="gpt-4o-mini"
        JUDGE_VLM_MODEL="gpt-4o"
        JUDGE_TIMEOUT=5.0
    fi
}

load_free_api_config() {
    local config_file="$CONFIG_DIR/free_api_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$PARSE_YAML" "$config_file")"
        echo "[Config] Loaded: free_api_config.yaml"
    else
        echo "[Config] Warning: free_api_config.yaml not found, using defaults"
        CODEGEN_SERVER_HOST="localhost"
        CODEGEN_SERVER_PORT=8001
        CODEGEN_SSH_PORT=22
        CODEGEN_SSH_USER="user"
        CODEGEN_MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
        JUDGE_SERVER_HOST="localhost"
        JUDGE_SERVER_PORT=8002
        JUDGE_SSH_PORT=22
        JUDGE_SSH_USER="user"
        JUDGE_MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
        JUDGE_TIMEOUT=3.0
    fi
}

load_recording_config() {
    local config_file="$CONFIG_DIR/recording_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$PARSE_YAML" "$config_file")"
        echo "[Config] Loaded: recording_config.yaml"
    else
        echo "[Config] Warning: recording_config.yaml not found, using defaults"
        RECORDING_FPS=30
    fi
}

# ============================================================
# Config 로드
# ============================================================

echo "========================================"
echo "Loading Configuration Files..."
echo "========================================"
echo "Config Directory: $CONFIG_DIR"
echo ""

# Multi-robot config 로드 (먼저 로드하여 기본값 설정)
load_multi_robot_config

# Detection config 로드
load_detection_config

# API config 로드 (USE_SERVER에 따라 선택)
if [ "$USE_SERVER" = true ]; then
    load_free_api_config
    LLM_MODEL="$CODEGEN_MODEL_NAME"
    JUDGE_MODEL="$JUDGE_MODEL_NAME"
else
    load_paid_api_config
    LLM_MODEL="$CODEGEN_LLM_MODEL"
    JUDGE_MODEL="$JUDGE_VLM_MODEL"
fi

# Recording config 로드 (RECORD_DATASET=true일 때만)
if [ "$RECORD_DATASET" = true ]; then
    load_recording_config
fi

echo ""

# ============================================================
# SSH 터널 설정 (USE_SERVER=true 시)
# ============================================================

TUNNEL_PIDS=()

cleanup_tunnels() {
    for pid in "${TUNNEL_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "[SSH Tunnel] Closing tunnel (PID: $pid)"
            kill "$pid" 2>/dev/null
        fi
    done
}
trap cleanup_tunnels EXIT

if [ "$USE_SERVER" = true ]; then
    echo "[Server] Setting up vLLM server connections..."

    # CodeGen 서버 연결 설정
    if [ "$CODEGEN_SSH_PORT" = "22" ]; then
        CODEGEN_SERVER_URL="http://${CODEGEN_SERVER_HOST}:${CODEGEN_SERVER_PORT}/v1"
        echo "[CodeGen] Direct connection: ${CODEGEN_SERVER_URL}"
    else
        LOCAL_CODEGEN_PORT=$CODEGEN_SERVER_PORT
        if ! pgrep -f "ssh -L ${LOCAL_CODEGEN_PORT}:localhost:${CODEGEN_SERVER_PORT}.*${CODEGEN_SERVER_HOST}" > /dev/null; then
            echo "[CodeGen] Creating SSH tunnel (localhost:${LOCAL_CODEGEN_PORT} → ${CODEGEN_SERVER_HOST}:${CODEGEN_SERVER_PORT})..."
            ssh -L ${LOCAL_CODEGEN_PORT}:localhost:${CODEGEN_SERVER_PORT} \
                -p ${CODEGEN_SSH_PORT} \
                ${CODEGEN_SSH_USER}@${CODEGEN_SERVER_HOST} \
                -N -f -o StrictHostKeyChecking=no -o ConnectTimeout=10
            TUNNEL_PIDS+=($!)
            sleep 2
        else
            echo "[CodeGen] SSH tunnel already exists"
        fi
        CODEGEN_SERVER_URL="http://localhost:${LOCAL_CODEGEN_PORT}/v1"
    fi

    # Judge 서버 연결 설정
    if [ "$JUDGE_SSH_PORT" = "22" ]; then
        JUDGE_SERVER_URL="http://${JUDGE_SERVER_HOST}:${JUDGE_SERVER_PORT}/v1"
        echo "[Judge] Direct connection: ${JUDGE_SERVER_URL}"
    else
        LOCAL_JUDGE_PORT=$JUDGE_SERVER_PORT
        if ! pgrep -f "ssh -L ${LOCAL_JUDGE_PORT}:localhost:${JUDGE_SERVER_PORT}.*${JUDGE_SERVER_HOST}" > /dev/null; then
            echo "[Judge] Creating SSH tunnel (localhost:${LOCAL_JUDGE_PORT} → ${JUDGE_SERVER_HOST}:${JUDGE_SERVER_PORT})..."
            ssh -L ${LOCAL_JUDGE_PORT}:localhost:${JUDGE_SERVER_PORT} \
                -p ${JUDGE_SSH_PORT} \
                ${JUDGE_SSH_USER}@${JUDGE_SERVER_HOST} \
                -N -f -o StrictHostKeyChecking=no -o ConnectTimeout=10
            TUNNEL_PIDS+=($!)
            sleep 2
        else
            echo "[Judge] SSH tunnel already exists"
        fi
        JUDGE_SERVER_URL="http://localhost:${LOCAL_JUDGE_PORT}/v1"
    fi

    echo ""
    echo "[Server] Checking connections..."

    if curl -s --connect-timeout 5 "${CODEGEN_SERVER_URL}/models" > /dev/null 2>&1; then
        echo "[Server] CodeGen LLM server OK (${CODEGEN_SERVER_URL})"
    else
        echo "[Server] WARNING: CodeGen LLM server not responding (${CODEGEN_SERVER_URL})"
    fi

    if curl -s --connect-timeout 5 "${JUDGE_SERVER_URL}/models" > /dev/null 2>&1; then
        echo "[Server] Judge VLM server OK (${JUDGE_SERVER_URL})"
    else
        echo "[Server] WARNING: Judge VLM server not responding (${JUDGE_SERVER_URL})"
    fi
    echo ""
fi

# ============================================================
# 설정 출력
# ============================================================

echo "========================================"
echo "Multi-Robot Pipeline"
echo "========================================"
echo "Instruction: $INSTRUCTION"
echo "Objects: ${OBJECTS[*]}"
echo "Robot IDs: ${ROBOT_IDS[*]}"
echo "Num Episodes: $NUM_EPISODES"
echo "Save Dir: $SAVE_DIR"
echo ""
echo "--- Feature Toggles ---"
echo "Execute Reset: $EXECUTE_RESET"
echo "Reset Mode: $RESET_MODE"
echo "Skip Judge: $SKIP_JUDGE"
echo "Use Server: $USE_SERVER"
echo "Record Dataset: $RECORD_DATASET"
echo ""
echo "--- Model Settings ---"
echo "LLM Model: $LLM_MODEL"
echo "Judge Model: $JUDGE_MODEL"
if [ "$USE_SERVER" = true ]; then
    echo "  CodeGen URL: $CODEGEN_SERVER_URL"
    echo "  Judge URL: $JUDGE_SERVER_URL"
fi
if [ "$RECORD_DATASET" = true ]; then
    echo ""
    echo "--- Recording Settings ---"
    echo "  Recording FPS: $RECORDING_FPS"
fi
echo "========================================"
echo ""

# ============================================================
# 환경 변수 설정 (vLLM 서버 사용 시)
# ============================================================

if [ "$USE_SERVER" = true ]; then
    export USE_LLM_SERVER=1
    export VLLM_SERVER_URL="$CODEGEN_SERVER_URL"
    export VLLM_MODEL_NAME="$CODEGEN_MODEL_NAME"
    export JUDGE_SERVER_URL="$JUDGE_SERVER_URL"
    export JUDGE_MODEL_NAME="$JUDGE_MODEL_NAME"
fi

# ============================================================
# 실행 인자 구성
# ============================================================

EXTRA_ARGS=""

# Robot IDs
ROBOT_ARGS=""
for robot_id in "${ROBOT_IDS[@]}"; do
    ROBOT_ARGS="$ROBOT_ARGS $robot_id"
done

# Detection 설정
if [ "$VISUALIZE_DETECTION" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --visualize-detection"
fi

# Reset 설정
if [ "$EXECUTE_RESET" = false ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip-reset"
fi

# Judge 설정
if [ "$SKIP_JUDGE" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip-judge"
fi

# Recording 설정
if [ "$RECORD_DATASET" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --record"
    EXTRA_ARGS="$EXTRA_ARGS --recording-fps $RECORDING_FPS"
fi

# ============================================================
# 파이프라인 실행
# ============================================================

python -m pipeline_multi.multi_robot_pipeline \
    --instruction "$INSTRUCTION" \
    --objects "${OBJECTS[@]}" \
    --robot-ids $ROBOT_ARGS \
    --llm-model "$LLM_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --reset-mode "$RESET_MODE" \
    --detection-timeout "$DETECTION_TIMEOUT" \
    --save-dir "$SAVE_DIR" \
    --num-episodes "$NUM_EPISODES" \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Multi-Robot Pipeline completed successfully!"
else
    echo "Multi-Robot Pipeline completed with errors (exit code: $EXIT_CODE)"
fi
echo "========================================"

exit $EXIT_CODE
