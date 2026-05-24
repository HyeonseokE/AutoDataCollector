#!/usr/bin/env bash
# switch_branch_and_relaunch.sh — server 측 git branch 전환 + 재시작 1 명령
#
# 사용:
#   bash grpc_server/switch_branch_and_relaunch.sh [branch]
#   기본 branch = master
#
# 동작 단계:
#   [1] server (REMOTE_HOST) 측에서 fetch + checkout + pull
#   [2] 현재 떠있는 server tmux 정리 (launch_remote_server.sh stop)
#   [3] 새 branch code + yaml 로 fresh boot (launch_remote_server.sh)
#   [4] Ready RPC + Phase2MISelector ready log 확인
#
# 환경 변수 (override 가능):
#   REMOTE_HOST  : server 호스트 (default = launch_remote_server.sh 의 yaml 값 사용)
#   REMOTE_DIR   : server 측 repo 경로 (default = /home/csiwoo/CORL2026/AutoDataCollector)

set -euo pipefail

BRANCH="${1:-master}"
REMOTE_HOST="${REMOTE_HOST:-csiwoo@61.107.202.230}"
REMOTE_DIR="${REMOTE_DIR:-/home/csiwoo/CORL2026/AutoDataCollector}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# conda env (lerobot_cap 등) 의 LD_LIBRARY_PATH/LD_PRELOAD 가 system ssh 의
# openssl 과 충돌해 "OpenSSL version mismatch. Built against 30000020, you have
# 30600020" error 발생. sync_artifacts.sh 와 동일 패턴 — ssh 호출 시 env 청소.
do_ssh() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD ssh "$@"
}

C_BOLD="\033[1m"; C_RST="\033[0m"
C_GREEN="\033[32m"; C_YELLOW="\033[33m"; C_RED="\033[31m"
info() { printf "${C_BOLD}%s${C_RST}\n" "$*"; }
ok()   { printf "${C_GREEN}  ✓ %s${C_RST}\n" "$*"; }
warn() { printf "${C_YELLOW}  ! %s${C_RST}\n" "$*"; }
err()  { printf "${C_RED}  ✗ %s${C_RST}\n" "$*"; }

echo
info "=========================================="
info " Switch branch + relaunch H100 server"
info "=========================================="
echo "  branch     : $BRANCH"
echo "  remote     : $REMOTE_HOST"
echo "  remote dir : $REMOTE_DIR"
echo

# -----------------------------------------------------------------------------
# [1] Server: fetch + checkout + pull
# -----------------------------------------------------------------------------
info "[1/3] Server: git fetch + checkout '$BRANCH' + pull"
if do_ssh -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new \
        "$REMOTE_HOST" "
    set -e
    cd '$REMOTE_DIR'
    git fetch origin
    # branch 가 local 에 없으면 -t 로 tracking 생성, 있으면 그냥 checkout
    if git rev-parse --verify --quiet '$BRANCH' >/dev/null; then
        git checkout '$BRANCH'
    else
        git checkout -t 'origin/$BRANCH'
    fi
    git pull origin '$BRANCH'
    echo '  server HEAD ='
    git log --oneline -1
"; then
    ok "server checked out + pulled '$BRANCH'"
else
    err "server checkout failed"
    exit 1
fi
echo

# -----------------------------------------------------------------------------
# [2] Stop old server (tmux + tunnel cleanup)
# -----------------------------------------------------------------------------
info "[2/3] Stop existing server (tmux + tunnel cleanup)"
# stop 실패해도 (= 떠있는 server 없음) 계속 진행 — fresh boot 만 보장하면 충분
if bash "$REPO_ROOT/grpc_server/launch_remote_server.sh" stop; then
    ok "old server stopped"
else
    warn "stop returned non-zero — likely nothing to stop, continuing"
fi
echo

# -----------------------------------------------------------------------------
# [3] Fresh boot with new branch code + yaml
# -----------------------------------------------------------------------------
info "[3/3] Launch fresh server (new branch code + yaml)"
bash "$REPO_ROOT/grpc_server/launch_remote_server.sh"
echo

# -----------------------------------------------------------------------------
# [4] Quick verification
# -----------------------------------------------------------------------------
info "[verify] Server startup log (selection_mode + listening)"
do_ssh -o ConnectTimeout=10 "$REMOTE_HOST" "
    tmux capture-pane -p -t phase2_server_pnp -S -500 2>&1 \
        | grep -iE 'Phase2MISelector ready|listening on|selection_mode|server HEAD' \
        | tail -5
" 2>&1 || warn "verify failed (server may still be booting — check tmux manually)"

echo
info "=========================================="
info " DONE — server now on branch '$BRANCH'"
info "=========================================="
echo "  next: bash run_forward_and_reset_ws3.sh   (client side)"
