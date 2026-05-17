#!/bin/bash
# preselective_rpc 서버 readiness 점검 런처.
#
# gpu_server 환경의 python 으로 check_ready 를 실행한다. conda activate 불필요.
#   bash preselective_rpc/check_ready.sh
#   bash preselective_rpc/check_ready.sh --address 127.0.0.1:50061 --timeout 10

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_ROOT"

# 사용할 conda 환경 (호출 시 ENV_NAME=... 으로 덮어쓰기 가능)
ENV_NAME="${ENV_NAME:-gpu_server}"
PY="$HOME/miniconda3/envs/$ENV_NAME/bin/python"

if [ ! -x "$PY" ]; then
    echo "XX python not found: $PY" >&2
    echo "   ENV_NAME=<conda_env> 로 환경 이름을 지정하세요." >&2
    exit 1
fi

export PYTHONNOUSERSITE=1
exec "$PY" -m preselective_rpc.tools.check_ready "$@"
