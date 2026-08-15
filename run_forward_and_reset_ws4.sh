#!/bin/bash
# Forward + Reset Integrated Pipeline Runner
# Forward Execution → Judge (Evaluation) → Reset Execution 통합 파이프라인
#
# Config 파일들:
#   - pipeline_config/paid_api_config.yaml     : 유료 API 설정 (USE_SERVER=false)
#   - pipeline_config/free_api_config.yaml     : vLLM 서버 설정 (USE_SERVER=true)
#   - pipeline_config/recording_config.yaml    : 레코딩 설정 (RECORD_DATASET=true)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================
# 핵심 설정 (Essential Configuration) / 워크스페이스 명시 / 에피소드 갯수 명시
# ============================================================

# # Grasping:                                                                  
# (1, 완료) pick up the red block and place it on the blue plate
# (2, 완료) distribute chocolate pies to each plate                           
# (3) -

# # Arrangement:                                                               
# (1, 완료) place the yellow block between chocolate pies             
# (2, 완료) arrange yellow, red, and purple blocks in a line from left to right
# (3, 완료) stack the blocks in the order of red and yellow
# (3, 완료) stack the blocks in the order of red and yellow, purple

# # Non-grasping:
# (1, 성공) turn on the microphone by pressing the power button
# (2, 성공) Push the bowl of cereal 5cm from left to right
# (3, 성공) Open the trash can lid

# # Deformable:
# (1, 완료) fold the towel
# (2) sweep the floor with a towel
# (3) bend the microphone gooseneck leftward

# # Articulated:
# (1) open the drawers
# (2) close the drawers
# (3) beat the red block with a hammer

# # Insertion/Assembly:
# (1) assemble the battery pack
# (2) peg-in-hole
# (3) clean the desk

# # Rotation:
# (1) tighten the bolt
# (2) open the bottle
# (3) mix the tea

# # Contact-rich:
# (1) wipe the dish with a sponge
# (2) sweep the floor with a brush
# (3) shake the bottle

# INSTRUCTION="make sandwich using the ingredients on the table"
# INSTRUCTION="pick up the red block and place it on the blue dish"
# INSTRUCTION="fold the green towel"
# INSTRUCTION="pick up the brown peg and insert it into the hole of the gray structure"
# INSTRUCTION = "Pick up the banana and place it in the bowl. 
# You may need to handover the banana from one arm to the other if the initial arm picking the banana cannot reach the bowl. 
# After picking the banana with one arm, you can handover the banana by first placing it carefully on the table surface and then using the other arm to pick it up. 
# The placing position must be on the table, as far as possible from other objects but absolutely within the reachable table area of the other arm. 
# Make sure to move the picking arm out of the way before the receiving arm moves towards grasping the object."

# INSTRUCTION="Assemble the green hinge and red hinge.
# You need to carefully assemble the green hinge's male part to red hinge's hole part.
# since the green hinge's male part is upward, you need to rotate it downward first before assembling."

# [필수] 로봇 번호 배열 — 순서가 arm 그룹을 결정 (최대 4대):
#   ROBOT_IDS[0] → left_arm
#   ROBOT_IDS[1] → right_arm
#   ROBOT_IDS[2] → top_arm
#   ROBOT_IDS[3] → bottom_arm
# shared 카메라는 항상 포함. 제공된 ID 수만큼만 arm 그룹 활성화.
# 예: (0)       → shared + left_arm
#     (2 3)     → shared + left_arm(robot2) + right_arm(robot3)
#     (1 2 3 4) → shared + left_arm + right_arm + top_arm + bottom_arm
ROBOT_IDS=(6 7)

## Task_instruction 
# INSTRUCTION="stack red block at center, then place yellow block on top of red block"

### [single arm task]
## pick and place
# INSTRUCTION="close the pot’s lid."
# RESET_INSTRUCTION=""

INSTRUCTION="Fold the towel in half from top to bottom."
RESET_INSTRUCTION="Unfold the towel from bottom to top to recover its original flat state."


## stack red and yellow
# INSTRUCTION="stack the blocks in the order of red and yellow"
# RESET_INSTRUCTION=""

## stack RYP blocks
# INSTRUCTION="stack the blocks in the order of red and yellow, purple."
# RESET_INSTRUCTION=""

## distribute chocolate pies to each plate
# INSTRUCTION="distribute chocolate pies to each plate."
# RESET_INSTRUCTION=""

### [dual arm task]
## towel folding
# INSTRUCTION="fold the towel in half from top to bottom."
# RESET_INSTRUCTION="unfold the towel from bottom to top to recover its original flat state"

## move
# INSTRUCTION="move the yellow block from top-left area to bottom-right edge"
# RESET_INSTRUCTION="move the yellow block from bottom-right edge to top-left area"

## hand over the sponge
# INSTRUCTION="move the yellow block from top-left edge to bottom-right edge"
# RESET_INSTRUCTION="move the yellow block from bottom-right edge to top-left edge"

## Reset_instruction(Empty is default: "move objects to certain position")

# [필수] 에피소드 반복 횟수
NUM_EPISODES=100
NUM_RANDOM_SEEDS=1 # 배치 수 (1=초기 위치 유지, N>1=N종류 랜덤 배치, 에피소드를 N등분)

# [선택] 로봇별 reset 공간 제약 (all, all_wo_center, top-left, top-right, bottom-left, bottom-right)
# 로봇 순서대로 지정. 예: 단일 (top-left), 듀얼 (top-left top-right)
# all:           워크스페이스 전역
# all_wo_center: all 에서 이미지 중앙 세로 타원 (160 x 320 px) 영역만 제외
#                (위↔아래 1열 정렬 같은 task 에서 reset 위치가 중앙 라인에 떨어지지 않게 함)
# top-left 등:   테이블 4분면 중 해당 영역 ∩ 로봇 도달 범위
RESETSPACE_PER_ROBOT=(all) 

# [필수] 결과 저장 경로
SAVE_DIR="./results"

# [선택] Turn Test (Waypoint Trajectory) 스킵 여부
SKIP_TURN_TEST=true

# 서버 추론 사용 여부 (true: vLLM 서버, false: 유료 API)
USE_SERVER=false

# Reset execution 설정
EXECUTE_RESET=true # Reset 실행 여부

# Dataset Recording 설정
RECORD_DATASET=true

# ============================================================
# Method3 phase 토글 (final_method3_spec §2)
#   phase1 — Phase1 buffer-aware subgoal seeding (기본).
#   phase2 — Phase2 MI-based selection. P_phase1 vector DB 는 캐시 hit 면 그대로
#            로드, 없으면 pipeline_config/phase2_config.yaml 의
#            ``phase1_trained_vla_path`` + ``phase1_dataset_path`` 로 §6
#            re-embedding 자동 구축. HF repo_id ("user/name") 도 그대로 인식 —
#            로컬 캐시 miss 면 lerobot 가 다운로드.
# ============================================================
PHASE="phase1"

# Resume 설정 (이전 세션 이어받기)
# 비어있으면 새 세션, 경로 지정 시 이전 세션 이어받기
RESUME_SESSION="./results/session_20260512_181737"
# RESUME_SESSION="./results/session_20260319_174942"

# ============================================================
# Live 3D EE-trajectory trace (rerun) — forward 구간 EE (x,y,z) 를 3D 에
# 실시간 누적, 에피소드마다 다른 색. 제어 경로 비침투 (read-only polling).
# EE_TRACE_ENABLED=true 로 활성 (EE 폴링 + npz 누적 + 종료 시 mp4 렌더).
# 세션 종료 시 EE_TRACE_NPZ (미지정 시 <session_dir>/ee_trace.npz) 로 덤프 →
# scripts/render_ee_trace_video.py 로 mp4/gif 렌더 가능.
# URDF/calib 는 ROBOT_IDS[0] 로 자동 해석 (so101_robot<id>.urdf / robot<id>_calibration.json).
#
# TRAJ_VISUALIZER — 누적 경로 rerun *뷰어 창* 토글. 이 값이 true 일 때만 창이 뜬다.
#   true  → rerun 뷰어 spawn (실시간 3D 누적 경로 확인).
#   false → 창 안 뜸. rerun 로깅 자체를 건너뛰므로 (EE_TRACE_RRD 미지정 시)
#           sink 없는 in-memory 버퍼가 쌓이지 않는다. npz 누적 / mp4 렌더는 그대로.
# ============================================================
TRAJ_VISUALIZER=false

# 따로 찍는 real-time 30fps front-view 와 싱크되도록 30Hz 로 로깅 → real-time 1x mp4.
export EE_TRACE_ENABLED="${EE_TRACE_ENABLED:-true}"
export EE_TRACE_FPS="${EE_TRACE_FPS:-30}"            # 로깅 폴링레이트 (front-view fps 와 일치 권장)
export TRAJ_VISUALIZER="$TRAJ_VISUALIZER"
export EE_TRACE_SPAWN="$TRAJ_VISUALIZER"             # legacy 별칭 — TRAJ_VISUALIZER 를 따라감
export EE_TRACE_MAX_SPEED="${EE_TRACE_MAX_SPEED:-2.0}"  # 글리치 컷: EE 속도 임계(m/s) 초과 점 제거 (0=끔)
export EE_TRACE_MAX_JUMP="${EE_TRACE_MAX_JUMP:-0.08}"   # 글리치 컷: last-good 대비 점프(m) 초과 + stuck/frozen read 제거 (0=끔)
# export EE_TRACE_RRD="${EE_TRACE_RRD:-$SAVE_DIR/ee_trace.rrd}"   # 인터랙티브 replay 저장 시 주석 해제
# export EE_TRACE_NPZ="${EE_TRACE_NPZ:-$SAVE_DIR/ee_trace.npz}"   # 명시 경로 지정 시 주석 해제
# 파이프라인 종료(정상/Ctrl+C/에러) 시 ee_trace.npz → real-time mp4 자동 렌더 (EXIT trap).
EE_TRACE_RENDER="${EE_TRACE_RENDER:-true}"          # false → 자동 렌더 끔 (npz 만 남김)
EE_TRACE_RENDER_FPS="${EE_TRACE_RENDER_FPS:-30}"    # mp4 fps (front-view 와 동일하게)
EE_TRACE_ROTATE_DEG="${EE_TRACE_ROTATE_DEG:-0}"     # 0=고정 시점(싱크/합성용 권장), >0=공전
# 고정 시점: 카메라 eye(robot origin 기준 x y z) 에서 EE 초기 위치를 바라봄. 굵은 현재-EE 점.
EE_TRACE_EYE="${EE_TRACE_EYE:-0 -0.05 0.15}"        # 카메라 위치 (3값, 공백 구분)
EE_TRACE_HEAD_SIZE="${EE_TRACE_HEAD_SIZE:-110}"     # 현재 EE 강조 점 크기

# Multi-turn (crop-then-point) 코드 생성은 이제 기본 동작 — 토글 제거됨.
# 굳이 옛 single-turn (Grounding DINO 검출) 경로를 쓰려면 --no-multi-turn 을 붙일 것.
# CAD 참조 이미지 / side-view 이미지 옵션도 미사용이라 제거됨.

# ============================================================
# Config 파일 로드 함수
# ============================================================

CONFIG_DIR="$SCRIPT_DIR/pipeline_config"

load_paid_api_config() {
    local config_file="$CONFIG_DIR/paid_api_config.yaml"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$CONFIG_DIR/parse_yaml.py" "$config_file")"
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
        eval "$(python3 "$CONFIG_DIR/parse_yaml.py" "$config_file")"
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

RECORDING_CONFIG_FILE="$CONFIG_DIR/recording_config_ws4.yaml"

load_recording_config() {
    local config_file="$RECORDING_CONFIG_FILE"
    if [ -f "$config_file" ]; then
        eval "$(python3 "$CONFIG_DIR/parse_yaml.py" "$config_file")"
        echo "[Config] Loaded: $(basename $config_file)"
    else
        echo "[Config] Warning: $(basename $config_file) not found, using defaults"
        DATASET_REPO_ID=""
        RECORDING_FPS=30
    fi
}

# ============================================================
# Config 로드
# ============================================================

echo "========================================"
echo "Loading Configuration Files..."
echo "========================================"

# API config 로드 (USE_SERVER에 따라 선택)
if [ "$USE_SERVER" = true ]; then
    load_free_api_config
    LLM_MODEL="$CODEGEN_MODEL_NAME"
    JUDGE_MODEL="$JUDGE_MODEL_NAME"
else
    load_paid_api_config
    LLM_MODEL="$CODEGEN_LLM_MODEL"
    JUDGE_MODEL="$JUDGE_VLM_MODEL"
    CODEGEN_SESSION2_MODEL="${CODEGEN_SESSION2_MODEL:-}"
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
# 파이프라인 종료 시(정상 완료 / Ctrl+C / 에러 — 모두 EXIT trap 에서) 이번 run 의
# ee_trace.npz 를 찾아 real-time(1x) mp4 로 자동 렌더한다. npz 는 파이썬이
# atexit/증분으로 이미 저장하므로 여기서는 "그 파일 → 영상" 만 담당.
# real-time 렌더는 세션이 길면 수 분~수십 분 걸리므로 백그라운드(detached)로 돌려
# 터미널을 막지 않는다. mp4 는 렌더 완료 시 npz 옆에 나타난다.
# 한 개 mp4 를 백그라운드 렌더 (true real-time). $1=output mp4,
# $2=backdrop npz(회색 배경, "" 면 없음), 나머지($3...)=메인 입력 npz(컬러 누적).
# EE_TRACE_EYE 는 3토큰이라 따옴표 없이 전개.
_ee_render_one() {
    local mp4="$1"; shift
    local backdrop="$1"; shift
    local log="${mp4%.mp4}.render.log"
    local bd=()
    [ -n "$backdrop" ] && [ -f "$backdrop" ] && bd=(--backdrop "$backdrop")
    echo "[EETrace] 렌더 시작(백그라운드) → $mp4 (npz: $*, backdrop: ${backdrop:-none})"
    nohup python scripts/render_ee_trace_video.py \
        --npz "$@" --output "$mp4" "${bd[@]}" \
        --realtime --fps "${EE_TRACE_RENDER_FPS:-30}" \
        --rotate-deg "${EE_TRACE_ROTATE_DEG:-0}" \
        --eye ${EE_TRACE_EYE:-0 -0.05 0.15} \
        --head-size "${EE_TRACE_HEAD_SIZE:-110}" \
        --max-speed "${EE_TRACE_MAX_SPEED:-2.0}" \
        --max-jump "${EE_TRACE_MAX_JUMP:-0.08}" \
        > "$log" 2>&1 &
    disown 2>/dev/null || true
}

_render_ee_trace_on_exit() {
    [ "${EE_TRACE_ENABLED:-}" = "true" ] || return 0
    [ "${EE_TRACE_RENDER:-true}" = "true" ] || return 0

    # ── Phase2 (resume) → 두 모드 mp4 생성 (둘 다 true real-time) ──
    #   [1] phase2 단독:  <session>/phase2/ee_trace.mp4  (phase2 컬러 누적, front-view 싱크용)
    #   [2] phase1 배경 + phase2: <session>/ee_trace_phase1plus2.mp4
    #        phase1(1~10)을 회색 배경으로 깔고, 그 위에 phase2(11~20)를 컬러로 실시간 누적.
    #        타임라인은 phase2 만 → 길이 ≈ phase2 실제 수집시간 (gap 압축 없음).
    if [ "${PHASE:-}" = "phase2" ] && [ -n "${RESUME_SESSION:-}" ]; then
        local p1="$RESUME_SESSION/ee_trace.npz"
        local p2="$RESUME_SESSION/phase2/ee_trace.npz"
        if [ ! -f "$p2" ]; then
            echo "[EETrace] phase2 npz 없음 ($p2) — 렌더 건너뜀"
            return 0
        fi
        _ee_render_one "$RESUME_SESSION/phase2/ee_trace.mp4" "" "$p2"                # 모드 [1]
        if [ -f "$p1" ]; then
            _ee_render_one "$RESUME_SESSION/ee_trace_phase1plus2.mp4" "$p1" "$p2"    # 모드 [2] (p1=배경)
        else
            echo "[EETrace] phase1 npz 없음 ($p1) — 누적(모드2) 건너뜀, 단독(모드1)만 생성"
        fi
        echo "[EETrace] (렌더는 종료 후에도 백그라운드로 계속됩니다)"
        return 0
    fi

    # ── phase1 / 일반 단일 세션 ──
    local npz="${EE_TRACE_NPZ:-}"
    if [ -z "$npz" ]; then
        npz="$(ls -t "$SAVE_DIR"/session_*/ee_trace.npz 2>/dev/null | head -1)"
        if [ -z "$npz" ] && [ -n "${RESUME_SESSION:-}" ] && [ -f "$RESUME_SESSION/ee_trace.npz" ]; then
            npz="$RESUME_SESSION/ee_trace.npz"
        fi
    fi
    if [ -z "$npz" ] || [ ! -f "$npz" ]; then
        echo "[EETrace] npz 없음 — mp4 렌더 건너뜀 (trace 가 비활성였거나 점이 없음)"
        return 0
    fi
    _ee_render_one "${npz%.npz}.mp4" "" "$npz"
    echo "[EETrace] (렌더는 종료 후에도 백그라운드로 계속됩니다)"
}

# 단일 EXIT 핸들러로 렌더 + 터널정리 (EXIT trap 은 하나만 유효 → 합쳐서 등록).
_on_exit() { _render_ee_trace_on_exit; cleanup_tunnels; }
trap _on_exit EXIT

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
echo "Forward + Reset Pipeline"
echo "========================================"
echo "Instruction: $INSTRUCTION"
echo "Robot IDs: ${ROBOT_IDS[*]}"
echo "Num Episodes: $NUM_EPISODES"
echo "Save Dir: $SAVE_DIR"
echo ""
echo "--- Feature Toggles ---"
echo "Execute Reset: $EXECUTE_RESET"
echo "Random Seeds: $NUM_RANDOM_SEEDS"
echo "Use Server: $USE_SERVER"
echo "Record Dataset: $RECORD_DATASET"
echo ""
echo "--- Model Settings ---"
echo "LLM Model (Session 1): $LLM_MODEL"
if [ -n "$CODEGEN_SESSION2_MODEL" ]; then
    echo "CodeGen Model (Session 2): $CODEGEN_SESSION2_MODEL"
fi
echo "Judge Model: $JUDGE_MODEL"
echo "Judge Timeout: ${JUDGE_TIMEOUT}s"
if [ "$USE_SERVER" = true ]; then
    echo "  CodeGen URL: $CODEGEN_SERVER_URL"
    echo "  Judge URL: $JUDGE_SERVER_URL"
fi
if [ "$RECORD_DATASET" = true ]; then
    echo ""
    echo "--- Recording Settings ---"
    if [ -n "$DATASET_REPO_ID" ]; then
        echo "  Dataset Repo ID: $DATASET_REPO_ID"
    else
        echo "  Dataset Repo ID: (auto-generated)"
    fi
    echo "  Recording FPS: $RECORDING_FPS"
fi
echo "========================================"
echo ""

# ============================================================
# 실행 인자 구성
# ============================================================

EXTRA_ARGS=""

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

if [ -n "$RESUME_SESSION" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resume $RESUME_SESSION"
fi

if [ ${#RESETSPACE_PER_ROBOT[@]} -gt 0 ]; then
    EXTRA_ARGS="$EXTRA_ARGS --resetspace-per-robot ${RESETSPACE_PER_ROBOT[@]}"
fi

# Method3 phase 토글 — phase2 일 때 VLA/dataset 경로는 phase2_config.yaml 에서 자동 로드.
EXTRA_ARGS="$EXTRA_ARGS --phase $PHASE"

# ============================================================
# 파이프라인 실행
# ============================================================

CODEGEN_S2_ARG=""
if [ -n "$CODEGEN_SESSION2_MODEL" ]; then
    CODEGEN_S2_ARG="--codegen-session2-model $CODEGEN_SESSION2_MODEL"
fi

python execution_forward_and_reset.py \
    --instruction "$INSTRUCTION" \
    --robot ${ROBOT_IDS[@]} \
    --llm "$LLM_MODEL" \
    --judge-model "$JUDGE_MODEL" \
    --judge-timeout "$JUDGE_TIMEOUT" \
    --num-random-seeds "$NUM_RANDOM_SEEDS" \
    ${RESET_INSTRUCTION:+--reset-instruction "$RESET_INSTRUCTION"} \
    $( [ "$SKIP_TURN_TEST" = "true" ] && echo "--skip-turn-test" ) \
    --save "$SAVE_DIR" \
    --num-episodes "$NUM_EPISODES" \
    --recording-config "$RECORDING_CONFIG_FILE" \
    $CODEGEN_S2_ARG \
    $EXTRA_ARGS

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Pipeline completed successfully!"
else
    echo "Pipeline completed with errors (exit code: $EXIT_CODE)"
fi
echo "========================================"

exit $EXIT_CODE
