#!/bin/bash
# Recording Test Script - 2 Episodes
# Forward execution with LeRobot dataset recording

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# Recording Test Configuration
# ============================================================

ROBOT_ID=3
INSTRUCTION="pick up the yellow dice and place it on the blue dish"
OBJECTS=("yellow dice" "blue dish")

# LLM/VLM 설정
LLM_MODEL="gpt-4.1-mini"
JUDGE_MODEL="gpt-5-mini"

# 타임아웃 설정
DETECTION_TIMEOUT=5.0
JUDGE_TIMEOUT=2.0

# 검출 시각화
VISUALIZE_DETECTION=true

# 결과 저장 경로
SAVE_DIR="./results_recording_test"

# Reset 건너뛰기 (Forward만 테스트)
EXECUTE_RESET=false

# 서버 설정
USE_SERVER=true
CODEGEN_SERVER_HOST="115.145.175.11"
CODEGEN_SERVER_PORT=8001
CODEGEN_SSH_PORT=22
CODEGEN_SSH_USER="hscho"
CODEGEN_MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"

JUDGE_SERVER_HOST="115.145.173.248"
JUDGE_SERVER_PORT=8002
JUDGE_SSH_PORT=10001
JUDGE_SSH_USER="guest"
JUDGE_MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"

# ============================================================
# LeRobot Dataset Recording - ENABLED
# ============================================================
NUM_EPISODES=2
RECORD_DATASET=true
DATASET_REPO_ID="local/cap_recording_test"
RECORDING_FPS=30

# ============================================================
# Execution
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

    if [ "$CODEGEN_SSH_PORT" = "22" ]; then
        CODEGEN_SERVER_URL="http://${CODEGEN_SERVER_HOST}:${CODEGEN_SERVER_PORT}/v1"
        echo "[CodeGen] Direct connection: ${CODEGEN_SERVER_URL}"
    else
        LOCAL_CODEGEN_PORT=$CODEGEN_SERVER_PORT
        if ! pgrep -f "ssh -L ${LOCAL_CODEGEN_PORT}:localhost:${CODEGEN_SERVER_PORT}.*${CODEGEN_SERVER_HOST}" > /dev/null; then
            ssh -L ${LOCAL_CODEGEN_PORT}:localhost:${CODEGEN_SERVER_PORT} \
                -p ${CODEGEN_SSH_PORT} \
                ${CODEGEN_SSH_USER}@${CODEGEN_SERVER_HOST} \
                -N -f -o StrictHostKeyChecking=no -o ConnectTimeout=10
            TUNNEL_PIDS+=($!)
            sleep 2
        fi
        CODEGEN_SERVER_URL="http://localhost:${LOCAL_CODEGEN_PORT}/v1"
    fi

    if [ "$JUDGE_SSH_PORT" = "22" ]; then
        JUDGE_SERVER_URL="http://${JUDGE_SERVER_HOST}:${JUDGE_SERVER_PORT}/v1"
        echo "[Judge] Direct connection: ${JUDGE_SERVER_URL}"
    else
        LOCAL_JUDGE_PORT=$JUDGE_SERVER_PORT
        if ! pgrep -f "ssh -L ${LOCAL_JUDGE_PORT}:localhost:${JUDGE_SERVER_PORT}.*${JUDGE_SERVER_HOST}" > /dev/null; then
            ssh -L ${LOCAL_JUDGE_PORT}:localhost:${JUDGE_SERVER_PORT} \
                -p ${JUDGE_SSH_PORT} \
                ${JUDGE_SSH_USER}@${JUDGE_SERVER_HOST} \
                -N -f -o StrictHostKeyChecking=no -o ConnectTimeout=10
            TUNNEL_PIDS+=($!)
            sleep 2
        fi
        JUDGE_SERVER_URL="http://localhost:${LOCAL_JUDGE_PORT}/v1"
    fi
fi

echo "========================================"
echo "Recording Test Pipeline"
echo "========================================"
echo "Instruction: $INSTRUCTION"
echo "Objects: ${OBJECTS[*]}"
echo "Robot: $ROBOT_ID"
echo "Num Episodes: $NUM_EPISODES"
echo "Record Dataset: $RECORD_DATASET"
echo "Dataset Repo ID: $DATASET_REPO_ID"
echo "Recording FPS: $RECORDING_FPS"
echo "========================================"
echo ""

EXTRA_ARGS=""

if [ "$VISUALIZE_DETECTION" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --visualize-detection"
fi

if [ "$EXECUTE_RESET" = false ]; then
    EXTRA_ARGS="$EXTRA_ARGS --skip-reset"
fi

if [ "$USE_SERVER" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --use-server"
    EXTRA_ARGS="$EXTRA_ARGS --codegen-server-url $CODEGEN_SERVER_URL"
    EXTRA_ARGS="$EXTRA_ARGS --codegen-model $CODEGEN_MODEL_NAME"
    EXTRA_ARGS="$EXTRA_ARGS --judge-server-url $JUDGE_SERVER_URL"
    EXTRA_ARGS="$EXTRA_ARGS --judge-server-model $JUDGE_MODEL_NAME"
fi

if [ "$RECORD_DATASET" = true ]; then
    EXTRA_ARGS="$EXTRA_ARGS --record"
    if [ -n "$DATASET_REPO_ID" ]; then
        EXTRA_ARGS="$EXTRA_ARGS --dataset-repo-id $DATASET_REPO_ID"
    fi
    EXTRA_ARGS="$EXTRA_ARGS --recording-fps $RECORDING_FPS"
fi

python execution_forward_and_reset.py \
    --instruction "$INSTRUCTION" \
    --objects "${OBJECTS[@]}" \
    --robot "$ROBOT_ID" \
    --llm "$LLM_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --timeout "$DETECTION_TIMEOUT" \
    --judge-timeout "$JUDGE_TIMEOUT" \
    --save "$SAVE_DIR" \
    --num-episodes "$NUM_EPISODES" \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Recording test completed successfully!"
    echo "Dataset location: ~/.cache/huggingface/lerobot/$DATASET_REPO_ID"
else
    echo "Recording test completed with errors (exit code: $EXIT_CODE)"
fi
echo "========================================"

exit $EXIT_CODE
