#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# launch_remote_server.sh — *yaml-driven* 원격 H100 launcher
#
# 로컬 터미널에서 원격 H100 의 grpc_server 를 띄우고, 동시에
# SSH port forwarding tunnel 을 열어 *로컬 client* 가 yaml 의
# ``transport.address: 127.0.0.1:<port>`` 그대로 호출할 수 있게 한다.
#
# 모든 연결 정보는 ``pipeline_config/phase2_config.yaml`` 의 ``remote:``
# 섹션에서 자동으로 읽어온다. ~/.ssh/config 를 따로 건드릴 필요 없음.
# (옛 phase2_server_infer_settings.yaml 는 2026-05-21 통합됨.)
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
PHASE2_SERVER_YAML="${PHASE2_SERVER_YAML:-$PROJ_ROOT/pipeline_config/phase2_config.yaml}"

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
#
# *ControlMaster 사용 안 함* — 옛 master 의 stale socket 이 *새 launcher 의
# tunnel 을 silent reject* 하는 race 가 너무 자주 발생. launcher 1회당 ssh 가
# 3~4번 (stop check, tmux start, tunnel) → sshd MaxStartups 10:30:60 안. 매번
# fresh handshake 가 더 robust.
build_ssh_opts() {
  SSH_OPTS=(
    -o StrictHostKeyChecking=accept-new
    -o ControlMaster=no
    -o ControlPath=none
  )
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

# ControlMaster socket 의 stale 상태 정리.
# 옛 master process 가 kill 된 후에도 ControlPath socket 파일이 *그대로 남아*,
# 새 ssh 가 ControlMaster=auto 로 *stale master* 에 연결 시도 → silent fail
# (tunnel.log 비어있음). 옛 master 가 *현재 살아있다면* 그 socket 은 유효하므로
# 함부로 지우지 않는다 — 살아있는 connection 으로 ssh -O check 검사.
sweep_stale_control_sockets() {
  local cm_dir="$LOG_DIR/cm"
  [ -d "$cm_dir" ] || return 0
  local removed=0
  shopt -s nullglob
  for sock in "$cm_dir"/*; do
    # 살아있는 master 인지 확인 (-O check). check 성공이면 valid → 건너뜀.
    if env -u LD_LIBRARY_PATH -u LD_PRELOAD "$_SYS_SSH" \
        -o "ControlPath=$sock" -O check placeholder >/dev/null 2>&1; then
      continue
    fi
    rm -f "$sock" || true
    removed=$((removed + 1))
  done
  shopt -u nullglob
  if [ "$removed" -gt 0 ]; then
    info "swept $removed stale ControlMaster socket(s) under $cm_dir"
  fi
  return 0
}

# 50061 점유 process 의 *모든* pid 정리. tunnel.pid 파일에 *기록된 pid* 뿐 아니라
# 다른 source (옛 background launcher, 사용자가 손수 띄운 tunnel 등) 의 잔재까지
# bind 충돌의 모든 후보를 sweep.
#
# 모든 외부 명령에 ``|| true`` — lsof 가 매치 없을 때 exit 1 을 던지면 set -e 가
# 함수 중간에 종료시켜 stop 분기 *silent fail* 의 원인이 됐다 (RCA 2026-05-21).
sweep_port_holders() {
  local port="${1:-$LOCAL_PORT}"
  if ! command -v lsof >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids=$(lsof -ti ":$port" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    info "killing port :$port holder(s): $pids"
    # SIGTERM 먼저, 1초 후 SIGKILL
    kill $pids 2>/dev/null || true
    sleep 1
    local survivors
    survivors=$(lsof -ti ":$port" 2>/dev/null || true)
    if [ -n "$survivors" ]; then
      kill -9 $survivors 2>/dev/null || true
    fi
  fi
  return 0
}

# ============================================================
# Subcommand: stop
# ============================================================
if [ "${1:-}" = "stop" ]; then
  # cleanup 도중에 외부 명령이 exit 1 을 던져도 *중간 종료 금지*. 모든 단계가
  # 실행돼야 원격 server / GPU 까지 풀린다.
  set +e
  bold "Stopping local tunnel + remote server"
  # 1) tunnel.pid 의 pid (있으면)
  if [ -f "$TUNNEL_PID_FILE" ]; then
    pid="$(cat "$TUNNEL_PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null && info "tunnel killed (pid=$pid)"
    fi
    rm -f "$TUNNEL_PID_FILE" 2>/dev/null || true
  fi
  # 2) 같은 port 의 다른 잔재 (옛 background launcher 등)
  sweep_port_holders "$LOCAL_PORT"
  # 3) ControlMaster sockets — 살아있는 master 만 남기고 stale 제거
  sweep_stale_control_sockets
  # 4) 모든 ControlMaster socket 강제 삭제 (stale check 없이 그냥 다 삭제).
  #    이번 stop 후 다음 launcher 가 fresh ssh 로 시작하도록.
  rm -rf "$LOG_DIR/cm" 2>/dev/null
  # 5) 원격 — *반드시* 도달해야 GPU 풀린다. 마지막 명령 ``true`` 로 exit 0 보장.
  if [ -n "$REMOTE_HOST" ]; then
    info "killing remote tmux session: $TMUX_SESSION + server process"
    if ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
        "tmux kill-session -t '$TMUX_SESSION' 2>/dev/null; \
         pkill -9 -f 'grpc_server.server' 2>/dev/null; \
         rm -f /tmp/phase2_server.log; \
         true" 2>/dev/null; then
      info "remote stop signal sent (tmux + server killed)"
    else
      warn "remote stop ssh exit != 0 (connect issue?). 수동:"
      warn "  ssh $REMOTE_HOST 'tmux kill-session -t $TMUX_SESSION; pkill -9 -f grpc_server.server'"
    fi
  else
    warn "REMOTE_HOST not set — skipping remote stop"
  fi
  exit 0
fi

# Start 분기 진입 직전에도 stale control socket sweep — 옛 background 가 띄운
# master 의 잔재로 새 ssh -L 이 silent fail 하는 race 를 방지.
sweep_stale_control_sockets

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

# conda 위치 탐지 + 원격 shell PATH 에 prepend.
# tmux 가 환경변수를 일부 못 상속하는 케이스가 있어, 탐지된 PATH 값을
# *그대로 tmux 명령 인자에 literal 로* 박아넣어 child shell 에서 보장.
CONDA_BIN=""
for d in \$HOME/miniconda3/bin \$HOME/anaconda3/bin /opt/conda/bin /opt/miniconda3/bin; do
    if [ -d "\$d" ]; then
        CONDA_BIN="\$d"
        break
    fi
done
if [ -z "\$CONDA_BIN" ]; then
    echo "  remote: ERROR — no conda installation found (miniconda3/anaconda3/opt)"
    exit 1
fi
export PATH="\$CONDA_BIN:\$PATH"
echo "  remote: conda PATH = \$CONDA_BIN"

export ENV_NAME="$SERVER_ENV_NAME"
export GPU_ID="$SERVER_GPU_ID"
export HOST="$SERVER_HOST"
export PORT="$REMOTE_PORT"
export RECORDING_CONFIG="$SERVER_RECORDING_CONFIG"
export URDF="$SERVER_URDF"

if command -v tmux >/dev/null 2>&1; then
  # tmux server 의 기존 환경(이전 옛 ENV_NAME 등)이 child 에 상속될 수 있어
  # 우리가 export 한 변수만으로는 *tmux child shell* 에 전달이 보장되지 않음.
  # 따라서 PATH/ENV_NAME/GPU_ID/HOST/PORT/RECORDING_CONFIG/URDF *모두* 를
  # tmux 명령 인자에 literal 로 박아 child shell 의 첫 줄에서 명시 export.
  tmux new-session -d -s '$TMUX_SESSION' \
      "PATH='\$PATH' ENV_NAME='\$ENV_NAME' GPU_ID='\$GPU_ID' HOST='\$HOST' PORT='\$PORT' RECORDING_CONFIG='\$RECORDING_CONFIG' URDF='\$URDF' bash grpc_server/run_server.sh 2>&1 | tee /tmp/phase2_server.log"
  echo "  remote: tmux session started (ENV_NAME=\$ENV_NAME, GPU_ID=\$GPU_ID)"
else
  nohup env PATH="\$PATH" ENV_NAME="\$ENV_NAME" GPU_ID="\$GPU_ID" \
      HOST="\$HOST" PORT="\$PORT" RECORDING_CONFIG="\$RECORDING_CONFIG" URDF="\$URDF" \
      bash grpc_server/run_server.sh > /tmp/phase2_server.log 2>&1 &
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
# Tunnel ssh — standalone (build_ssh_opts 가 이미 ControlMaster=no).
# ExitOnForwardFailure=yes 로 *port bind 실패 시 ssh 즉시 exit* (= 우리 polling
# 이 ssh 죽음을 빨리 감지).
nohup env -u LD_LIBRARY_PATH -u LD_PRELOAD "$_SYS_SSH" -N -T \
    "${SSH_OPTS[@]}" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -L "$LOCAL_PORT:localhost:$REMOTE_PORT" \
    "$REMOTE_HOST" \
    > "$TUNNEL_LOG" 2>&1 &
TUNNEL_PID=$!
echo "$TUNNEL_PID" > "$TUNNEL_PID_FILE"

# Polling — port 가 listen 시작하거나, ssh 가 죽거나, 15초 timeout.
# 옛 ``sleep 1 + kill -0`` 패턴은 standalone ssh 의 첫 fork 시간 race 와
# port-bind race 둘 다 잡지 못해 *살아있는 tunnel 도 dead 로 오판* 했다.
TUNNEL_WAIT_MAX=15
elapsed=0
tunnel_ok=0
while [ $elapsed -lt $TUNNEL_WAIT_MAX ]; do
  # ssh 죽음 즉시 감지
  if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
    err "tunnel ssh exited (pid=$TUNNEL_PID). log:"
    sed 's/^/    /' "$TUNNEL_LOG" 2>/dev/null | tail -20
    rm -f "$TUNNEL_PID_FILE"
    exit 2
  fi
  # port LISTEN 시작했는지 (= forward 실제로 bound)
  if command -v lsof >/dev/null 2>&1; then
    if lsof -ti ":$LOCAL_PORT" -sTCP:LISTEN >/dev/null 2>&1; then
      tunnel_ok=1
      break
    fi
  else
    # lsof 없으면 그냥 1초 대기 후 OK 로 간주 (fallback)
    sleep 1
    tunnel_ok=1
    break
  fi
  sleep 0.5
  elapsed=$((elapsed + 1))
done

if [ "$tunnel_ok" = "1" ]; then
  info "tunnel listening on localhost:$LOCAL_PORT (pid=$TUNNEL_PID) log=$TUNNEL_LOG"
else
  err "tunnel did not bind localhost:$LOCAL_PORT within ${TUNNEL_WAIT_MAX}s. log:"
  sed 's/^/    /' "$TUNNEL_LOG" 2>/dev/null | tail -20
  kill "$TUNNEL_PID" 2>/dev/null || true
  rm -f "$TUNNEL_PID_FILE"
  exit 2
fi

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
