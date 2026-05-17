#!/bin/bash
# preselective_rpc 서버에 에피소드 하나를 ingest 하는 런처.
#
# 긴 경로 붙여넣기 사고를 피하려고, 데이터셋 경로 등은 아래 변수에서 편집한다
# (에디터로 한 번만 수정). 그 뒤 인자 없이 실행:
#   bash preselective_rpc/ingest_episode.sh
# 모든 값은 호출 시 환경변수로도 덮어쓸 수 있다:
#   DATASET_ROOT=/path EPISODE_INDEX=1 bash preselective_rpc/ingest_episode.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_ROOT"

# ============================================================
# 핵심 설정 (Essential Configuration)
# ============================================================

# [필수] 사용할 conda 환경
ENV_NAME="${ENV_NAME:-gpu_server}"

# [필수] 서버 gRPC 주소
ADDRESS="${ADDRESS:-127.0.0.1:50061}"

# [필수] LeRobot 데이터셋 경로 — 반드시 채울 것 (에디터로 수정)
DATASET_ROOT="${DATASET_ROOT:-skkuprism/teleop_pnp_100ep}"

# [선택] ingest 파라미터
CHUNK_SIZE="${CHUNK_SIZE:-50}"
SKILL_ID="${SKILL_ID:-move_to}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
MAX_FRAMES="${MAX_FRAMES:-12}"     # 빠른 점검용 프레임 상한
TIMEOUT="${TIMEOUT:-600}"

# ============================================================

PY="$HOME/miniconda3/envs/$ENV_NAME/bin/python"
if [ ! -x "$PY" ]; then
    echo "XX python not found: $PY" >&2
    echo "   ENV_NAME=<conda_env> 로 환경 이름을 지정하세요." >&2
    exit 1
fi
if [ -z "$DATASET_ROOT" ]; then
    echo "XX DATASET_ROOT 가 비어 있습니다." >&2
    echo "   ingest_episode.sh 상단의 DATASET_ROOT 를 편집하거나," >&2
    echo "   DATASET_ROOT=/path bash preselective_rpc/ingest_episode.sh 로 실행하세요." >&2
    exit 1
fi

# DATASET_ROOT 가 로컬 디렉터리가 아니고 'org/name' 형태이면 HF Hub 데이터셋으로
# 보고 lerobot 캐시로 내려받은 뒤, 그 로컬 경로를 사용한다.
# (open_chunked_dataset 은 repo_id 가 아니라 로컬 경로를 요구함)
if [ ! -d "$DATASET_ROOT" ]; then
    if [[ "$DATASET_ROOT" == */* && "$DATASET_ROOT" != /* && "$DATASET_ROOT" != ./* ]]; then
        LOCAL_DIR="$HOME/.cache/huggingface/lerobot/$DATASET_ROOT"
        if [ -d "$LOCAL_DIR/meta" ]; then
            echo "[fetch] 이미 받음: $LOCAL_DIR"
        else
            echo "[fetch] HF Hub dataset '$DATASET_ROOT' 내려받는 중 → $LOCAL_DIR"
            "$HOME/miniconda3/envs/$ENV_NAME/bin/hf" download \
                "$DATASET_ROOT" --repo-type dataset --local-dir "$LOCAL_DIR" || exit 1
        fi
        DATASET_ROOT="$LOCAL_DIR"
    else
        echo "XX DATASET_ROOT 디렉터리가 없습니다: $DATASET_ROOT" >&2
        exit 1
    fi
fi

export PYTHONNOUSERSITE=1
exec "$PY" -m preselective_rpc.tools.ingest_episode \
    --address "$ADDRESS" \
    --dataset-root "$DATASET_ROOT" \
    --chunk-size "$CHUNK_SIZE" \
    --skill-id "$SKILL_ID" \
    --episode-index "$EPISODE_INDEX" \
    --max-frames "$MAX_FRAMES" \
    --timeout "$TIMEOUT"
