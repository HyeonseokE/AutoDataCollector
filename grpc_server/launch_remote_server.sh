#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_remote_server.sh — *yaml-driven* 원격 H100 launcher
#
# 로컬 터미널에서 원격 H100 의 grpc_server 를 띄우고, 동시에
# SSH port forwarding tunnel 을 열어 *로컬 client* 가 yaml 의
# ``transport.address: 127.0.0.1:<port>`` 그대로 호출할 수 있게 한다.
#
# 모든 연결 정보는 ``pipeline_config/phase2_server_infer_settings.yaml`` 의
# ``remote:`` 섹션에서 자동으로 읽어온다. ~/.ssh/config 를 따로 건드릴 필요 없음.
#
# 우선순위:
#   shell env > yaml 값 > built-in default
#
# 사용:
#   bash grpc_server/launch_remote_server.sh                            # yaml 만 채우면 끝
#   REMOTE_HOST=other_host bash grpc_server/launch_remote_server.sh     # 임시 override
#   bash grpc_server/launch_remote_server.sh stop                       # tunnel + tmux 정리
# ---------------------------------------------------------------------------

set -euo pipefail

# ============================================================
# yaml 위치 — env 로 override 가능
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PHASE2_SERVER_YAML="${PHASE2_SERVER_YAML:-$PROJ_ROOT/pipeline_config/phase2_server_infer_settings.yaml}"

# ============================================================
# Helpers
# ============================================================
bold()  { printf "\033[1;36m== %s ==\033[0m\n" "$*"; }
info()  { printf "\033[36m  %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m!! %s\033[0m\n" "$*"; }
err()   { printf "\033[1;31mXX %s\033[0m\n" "$*" >&2; }

# system ssh 가 conda env 의 libssl/libcrypto 를 LD_LIBRARY_PATH 로 잘못 잡아
# "OpenSSL version mismatch. Built against XXXXXXXX, you have YYYYYYYY" 로
# 죽는 흔한 케이스 우회. ssh 호출에서만 LD_LIBRARY_PATH / LD_PRELOAD 를 비우고
# system binary 를 절대경로로 호출 — python (yaml 파서, check_ready 등) 은
# 그대로 conda lib 사용.
_SYS_SSH="$(command -v ssh || echo /usr/bin/ssh)"
ssh() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD "$_SYS_SSH" "$@"
}

# Python helper: yaml 의 remote + transport 섹션 → KEY='VAL' 라인 출력.
# (jq/yq 미설치 환경 대응 — 항상 있는 python yaml 만 사용)
read_yaml_settings() {
  python - "$PHASE2_SERVER_YAML" <<'PY'
import os, sys, re
try:
    import yaml
except ImportError:
    sys.stderr.write("PyYAML missing in current env — yaml load skipped\n")
    sys.exit(0)
path = sys.argv[1]
try:
    cfg = yaml.safe_load(open(path)) or {}
except FileNotFoundError:
    sys.stderr.write(f"yaml not found: {path}\n")
    sys.exit(0)
except Exception as e:
    sys.stderr.write(f"yaml parse failed: {e}\n")
    sys.exit(0)

r = cfg.get("remote") or {}
t = cfg.get("transport") or {}

def emit(k, v):
    if v is None: v = ""
    s = str(v).replace("'", "'\"'\"'")
    print(f"YAML_{k}='{s}'")

ssh_alias = (r.get("ssh_alias") or "").strip()
user      = (r.get("user") or "").strip()
hostname  = (r.get("hostname") or "").strip()

# alias 가 ~/.ssh/config 에 *실제로 등록되어 있을 때만* alias 우선.
# 사용자가 yaml 에 alias + user/hostname 을 둘 다 채운 경우, 등록 안 된 alias
# 는 무시하고 user@hostname 으로 직접 연결 — DNS 실패로 끊기는 흔한 함정 방지.
def _alias_in_ssh_config(a):
    if not a:
        return False
    cfg = os.path.expanduser("~/.ssh/config")
    if not os.path.exists(cfg):
        return False
    try:
        with open(cfg, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                low = s.lower()
                if low.startswith("host ") or low.startswith("host\t"):
                    hosts = s.split()[1:]  # 원문 case 유지
                    if a in hosts:
                        return True
    except Exception:
        return False
    return False

alias_registered = _alias_in_ssh_config(ssh_alias)
if ssh_alias and alias_registered:
    target = ssh_alias
    using_alias = "1"
elif user and hostname:
    target = f"{user}@{hostname}"
    using_alias = "0"
elif ssh_alias:
    # alias 가 yaml 에는 있지만 ~/.ssh/config 미등록 + user/hostname 도 비어있음
    # → 마지막 시도 (DNS 직접 resolve 가능할 수도 있음)
    target = ssh_alias
    using_alias = "1"
else:
    target = ""
    using_alias = "0"

emit("REMOTE_HOST",      target)
emit("SSH_ALIAS",        ssh_alias)
emit("SSH_USING_ALIAS",  using_alias)
emit("SSH_ALIAS_REGISTERED", "1" if alias_registered else "0")
emit("SSH_USER",         user)
emit("SSH_HOSTNAME",     hostname)
emit("SSH_PORT",         r.get("port", 22))
emit("SSH_IDENTITY",     os.path.expanduser(r.get("identity_file") or ""))
emit("REMOTE_PROJECT",   r.get("project_path") or "~/AutoDataCollector")
emit("SERVER_ENV_NAME",  r.get("conda_env") or "lerobot")
emit("SERVER_GPU_ID",    r.get("gpu_id", 0))
emit("TMUX_SESSION",     r.get("tmux_session") or "phase2_server")
emit("REMOTE_GIT_PULL",  1 if r.get("git_pull_before_start", False) else 0)
emit("SETTLE_S",         r.get("ready_timeout_s", 30))

addr = (t.get("address") or "127.0.0.1:50061").strip()
m = re.match(r"^(?:[^:]+:)?(\d+)$", addr)
port = int(m.group(1)) if m else 50061
emit("LOCAL_PORT",  port)
emit("REMOTE_PORT", port)
PY
}

# yaml → env (env override 우선: env 가 비어있는 키만 yaml 값으로 채움)
apply_yaml_to_env() {
  if [ ! -f "$PHASE2_SERVER_YAML" ]; then
    warn "yaml not found: $PHASE2_SERVER_YAML (env-only mode)"
    return
  fi
  local line k v
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      YAML_*=*) ;;
      *) continue ;;
    esac
    k="${line%%=*}"
    k="${k#YAML_}"
    v="${line#*=}"
    # strip outer single quotes (emit 가 KEY='...' 형식으로 출력)
    v="${v#\'}"; v="${v%\'}"
    if [ -z "${!k:-}" ]; then
      eval "$k=\$v"
      export "$k"
    fi
  done < <(read_yaml_settings)
}

# Default fallbacks — yaml 도 env 도 없을 때
set_defaults() {
  : "${REMOTE_HOST:=}"
  : "${SSH_ALIAS:=}"
  : "${SSH_PORT:=22}"
  : "${SSH_IDENTITY:=}"
  : "${REMOTE_PROJECT:=~/AutoDataCollector}"
  : "${SERVER_ENV_NAME:=lerobot}"
  : "${SERVER_GPU_ID:=0}"
  : "${SERVER_HOST:=0.0.0.0}"
  : "${SERVER_RECORDING_CONFIG:=pipeline_config/recording_config_ws3.yaml}"
  : "${SERVER_URDF:=assets/urdf/so101_robot4.urdf}"
  : "${TMUX_SESSION:=phase2_server}"
  : "${LOCAL_PORT:=50061}"
  : "${REMOTE_PORT:=50061}"
  : "${LOG_DIR:=/tmp/phase2_server_local}"
  : "${SETTLE_S:=30}"
  : "${REMOTE_GIT_PULL:=0}"
}

# ssh 명령에 붙일 옵션 배열.
# SSH_USING_ALIAS=1 일 때만 ~/.ssh/config 의 alias 항목을 그대로 신뢰 (port·key
# 등은 ssh_config 에 위임). 아니면 user@hostname 직접 연결로 -p / -i 명시.
build_ssh_opts() {
  SSH_OPTS=(-o StrictHostKeyChecking=accept-new)
  if [ "${SSH_USING_ALIAS:-0}" != "1" ]; then
    if [ -n "$SSH_PORT" ] && [ "$SSH_PORT" != "22" ]; then
      SSH_OPTS+=(-p "$SSH_PORT")
    fi
    if [ -n "$SSH_IDENTITY" ]; then
      SSH_OPTS+=(-i "$SSH_IDENTITY")
    fi
  fi
}

# ============================================================
# Step 0 — yaml 로딩 + defaults
# ============================================================
apply_yaml_to_env
set_defaults
build_ssh_opts

mkdir -p "$LOG_DIR"
TUNNEL_LOG="$LOG_DIR/tunnel.log"
TUNNEL_PID_FILE="$LOG_DIR/tunnel.pid"
SERVER_TAIL_LOG="$LOG_DIR/server_tail.log"

# ============================================================
# Subcommand: stop
# ============================================================
if [ "${1:-}" = "stop" ]; then
  bold "Stopping local tunnel + remote server"
  if [ -f "$TUNNEL_PID_FILE" ]; then
    pid="$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && info "tunnel killed (pid=$pid)"
    fi
    rm -f "$TUNNEL_PID_FILE"
  else
    info "no local tunnel pid file"
  fi
  if [ -n "$REMOTE_HOST" ]; then
    info "killing remote tmux session: $TMUX_SESSION"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
        "tmux kill-session -t '$TMUX_SESSION' 2>/dev/null || true; \
         pkill -f 'grpc_server.server' 2>/dev/null || true" \
      && info "remote stop signal sent" \
      || warn "remote stop ssh failed"
  else
    warn "REMOTE_HOST not set — skipping remote stop"
  fi
  exit 0
fi

# ============================================================
# Sanity
# ============================================================
if [ -z "$REMOTE_HOST" ]; then
  err "REMOTE_HOST 가 비어있음 — yaml 의 remote.ssh_alias 또는 remote.user/hostname 을 채우거나,"
  err "  REMOTE_HOST=user@host bash $0  형식으로 env override 하세요."
  err ""
  err "현재 yaml: $PHASE2_SERVER_YAML"
  err "  remote.ssh_alias    : '${SSH_ALIAS:-<empty>}'"
  err "  remote.user         : '${SSH_USER:-<empty>}'"
  err "  remote.hostname     : '${SSH_HOSTNAME:-<empty>}'"
  exit 1
fi

bold "Phase2 remote server launcher (yaml-driven)"
info "yaml      : $PHASE2_SERVER_YAML"
info "remote    : $REMOTE_HOST"
info "  ssh opts: ${SSH_OPTS[*]:-<none>}"
if [ -n "${SSH_ALIAS:-}" ] && [ "${SSH_ALIAS_REGISTERED:-0}" != "1" ] && [ "${SSH_USING_ALIAS:-0}" != "1" ]; then
  warn "  ssh_alias '$SSH_ALIAS' 는 ~/.ssh/config 에 미등록 → user@hostname 직접 연결로 fallback"
  warn "  영구 등록을 원하면 ~/.ssh/config 에 아래 블록을 추가:"
  warn "    Host $SSH_ALIAS"
  warn "        HostName $SSH_HOSTNAME"
  warn "        User $SSH_USER"
  [ -n "$SSH_IDENTITY" ] && warn "        IdentityFile $SSH_IDENTITY"
fi
info "project   : $REMOTE_PROJECT"
info "tmux ses  : $TMUX_SESSION"
info "tunnel    : localhost:$LOCAL_PORT  →  $REMOTE_HOST:$REMOTE_PORT"
info "log dir   : $LOG_DIR"

# ============================================================
# Step 1 — (옵션) 원격 git pull
# ============================================================
if [ "$REMOTE_GIT_PULL" = "1" ] || [ "$REMOTE_GIT_PULL" = "true" ]; then
  bold "step 1/4  remote git pull"
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
      "cd $REMOTE_PROJECT && git pull --rebase --autostash 2>&1" \
    | sed 's/^/    /'
else
  info "(skip git pull — set remote.git_pull_before_start: true 또는 REMOTE_GIT_PULL=1)"
fi

# ============================================================
# Step 2 — 원격 tmux 세션 위에 run_server.sh 띄움
# ============================================================
bold "step 2/4  start remote server"

session_exists=$(ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
    "tmux has-session -t '$TMUX_SESSION' 2>/dev/null && echo yes || echo no")

if [ "$session_exists" = "yes" ]; then
  info "remote tmux '$TMUX_SESSION' already running — reusing"
else
  ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash -s <<REMOTE_CMD
set -euo pipefail
cd "$REMOTE_PROJECT"
export ENV_NAME="$SERVER_ENV_NAME"
export GPU_ID="$SERVER_GPU_ID"
export HOST="$SERVER_HOST"
export PORT="$REMOTE_PORT"
export RECORDING_CONFIG="$SERVER_RECORDING_CONFIG"
export URDF="$SERVER_URDF"
if command -v tmux >/dev/null 2>&1; then
  # bash -lc 로 *login shell* 사용 — ~/.bashrc 의 conda init 등이 적용돼야
  # setup/run_server.sh 가 conda 를 찾을 수 있다. non-login 으로 띄우면
  # "conda not found on PATH" 로 즉시 죽음.
  tmux new-session -d -s '$TMUX_SESSION' \
    "bash -lc 'bash grpc_server/run_server.sh 2>&1 | tee /tmp/phase2_server.log'"
  echo "  remote: tmux session started"
else
  nohup bash -lc 'bash grpc_server/run_server.sh' > /tmp/phase2_server.log 2>&1 &
  echo "  remote: tmux not available — using nohup (pid=\$!)"
fi
REMOTE_CMD
fi

# ============================================================
# Step 3 — 로컬 SSH tunnel 띄움 (background)
# ============================================================
bold "step 3/4  start local SSH tunnel"

if [ -f "$TUNNEL_PID_FILE" ]; then
  old_pid="$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)"
  if [ -n "$old_pid" ] && kill -0 "$old_pid" 2>/dev/null; then
    info "killing stale tunnel pid=$old_pid"
    kill "$old_pid" 2>/dev/null || true
    sleep 0.5
  fi
fi

# nohup 은 외부 command 이므로 bash function `ssh` 를 보지 못함 — 명시적으로
# system binary + LD_LIBRARY_PATH 우회를 사용해야 conda env 의 OpenSSL ABI
# mismatch 를 피한다 (위 ssh() wrapper 와 동일한 회피).
nohup env -u LD_LIBRARY_PATH -u LD_PRELOAD "$_SYS_SSH" -N -T \
    "${SSH_OPTS[@]}" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L "$LOCAL_PORT:localhost:$REMOTE_PORT" \
    "$REMOTE_HOST" \
    > "$TUNNEL_LOG" 2>&1 &
echo $! > "$TUNNEL_PID_FILE"
sleep 1
if ! kill -0 "$(cat "$TUNNEL_PID_FILE")" 2>/dev/null; then
  err "tunnel failed to start. log:"
  tail -20 "$TUNNEL_LOG"
  exit 2
fi
info "tunnel pid=$(cat "$TUNNEL_PID_FILE")  log=$TUNNEL_LOG"

# ============================================================
# Step 4 — Ready RPC polling
# ============================================================
bold "step 4/4  poll Ready RPC (timeout ${SETTLE_S}s)"

start_ts=$(date +%s)
ok=0
while [ $(( $(date +%s) - start_ts )) -lt "$SETTLE_S" ]; do
  if python -m grpc_server.tools.check_ready --address "127.0.0.1:$LOCAL_PORT" \
        > "$SERVER_TAIL_LOG" 2>&1; then
    ok=1
    break
  fi
  printf "."
  sleep 2
done
echo

if [ "$ok" = "1" ]; then
  bold "READY"
  cat "$SERVER_TAIL_LOG"
  echo
  info "Client 측은 그대로 ws3.sh 실행 — yaml 의 127.0.0.1:$LOCAL_PORT 이 tunnel 됨"
  info "서버 정지: bash $0 stop"
  info "remote 서버 로그 tail: ssh ${SSH_OPTS[*]:-} $REMOTE_HOST 'tail -f /tmp/phase2_server.log'"
else
  warn "Ready 응답 없음 — 서버가 아직 모델 로딩 중일 수 있음."
  warn "remote 로그 확인:  ssh ${SSH_OPTS[*]:-} $REMOTE_HOST 'tail -50 /tmp/phase2_server.log'"
  warn "tunnel 로그:       tail -20 $TUNNEL_LOG"
  warn "tunnel 은 계속 살아있음. 모델 로딩 끝나면 client 자동 연결됨."
  exit 3
fi
