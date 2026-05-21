#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# sync_artifacts.sh — yaml-driven artifact 동기화
#
# pipeline_config/phase2_config.yaml 의 다음 path 들을 *yaml.remote.*  대상에
# rsync 한다. 각 path 는 absolute 또는 repo-relative — 후자는 본 스크립트가
# project root 와 join 해 처리.
#
#   1. phase1_trained_vla_path                       (DCT-tuned VLA ckpt)
#   2. selector.skill_dct_parquet                    (P_phase1 build sidecar)
#   3. buffer.save_dir / buffer.filename             (server-local Vector DB)
#
# 사용:
#   bash grpc_server/sync_artifacts.sh                  # 한 줄
#   YAML=other.yaml bash grpc_server/sync_artifacts.sh  # 다른 yaml
#
# launch_remote_server.sh 가 step 0 으로 자동 호출 (SYNC_ARTIFACTS=1 default).
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
YAML="${YAML:-$PROJ_ROOT/pipeline_config/phase2_config.yaml}"

bold() { printf "\033[1;36m== %s ==\033[0m\n" "$*"; }
info() { printf "\033[36m  %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m!! %s\033[0m\n" "$*"; }
err()  { printf "\033[1;31mXX %s\033[0m\n" "$*" >&2; }

# system ssh — conda env 의 OpenSSL ABI mismatch 우회 (launcher 와 동일 패턴)
_SYS_SSH="$(command -v ssh || echo /usr/bin/ssh)"
sys_ssh() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD "$_SYS_SSH" "$@"
}

# Python helper: yaml → KEY=VAL emission
read_settings() {
  python - "$YAML" <<'PY'
import os, sys, yaml
from pathlib import Path

path = sys.argv[1]
try:
    y = yaml.safe_load(open(path)) or {}
except Exception as e:
    sys.stderr.write(f"yaml parse failed: {e}\n")
    sys.exit(1)

# remote — alias 우선 (단 ~/.ssh/config 에 등록된 경우만)
r = y.get("remote") or {}
alias = (r.get("ssh_alias") or "").strip()
user  = (r.get("user") or "").strip()
host  = (r.get("hostname") or "").strip()
proj  = (r.get("project_path") or "").strip()
port  = r.get("port", 22)
ident = os.path.expanduser((r.get("identity_file") or ""))

ssh_cfg = os.path.expanduser("~/.ssh/config")
alias_ok = False
if alias and os.path.exists(ssh_cfg):
    try:
        with open(ssh_cfg) as f:
            for line in f:
                s = line.strip()
                if s.lower().startswith("host ") or s.lower().startswith("host\t"):
                    if alias in s.split()[1:]:
                        alias_ok = True
                        break
    except Exception:
        alias_ok = False

if alias and alias_ok:
    target = alias
    using_alias = "1"
elif user and host:
    target = f"{user}@{host}"
    using_alias = "0"
elif alias:
    target = alias
    using_alias = "1"
else:
    target = ""
    using_alias = "0"

print(f"REMOTE_TARGET='{target}'")
print(f"REMOTE_PROJ='{proj}'")
print(f"REMOTE_PORT='{port}'")
print(f"REMOTE_IDENTITY='{ident}'")
print(f"REMOTE_USING_ALIAS='{using_alias}'")

# artifacts (repo-relative or absolute — 둘 다 처리)
def emit(label, p):
    p = (p or "").strip()
    print(f"ARTIFACT_{label}='{p}'")

emit("CKPT", y.get("phase1_trained_vla_path"))
sel = y.get("selector") or {}
emit("SIDECAR", sel.get("skill_dct_parquet"))
buf = y.get("buffer") or {}
buf_dir = (buf.get("save_dir") or "grpc_server/buffer").rstrip("/")
buf_file = buf.get("filename") or "server_skill_wise_vector_db.npz"
emit("VECTOR_DB", f"{buf_dir}/{buf_file}")
PY
}

eval "$(read_settings)"

if [ -z "${REMOTE_TARGET:-}" ] || [ -z "${REMOTE_PROJ:-}" ]; then
  err "remote.user/hostname/project_path 또는 ssh_alias 미설정 — yaml ($YAML) 확인."
  exit 1
fi

# ssh options
SSH_OPTS=(
  -o StrictHostKeyChecking=accept-new
  -o ControlMaster=no
  -o ControlPath=none
)
if [ "${REMOTE_USING_ALIAS:-0}" != "1" ]; then
  if [ -n "${REMOTE_PORT:-}" ] && [ "$REMOTE_PORT" != "22" ]; then
    SSH_OPTS+=(-p "$REMOTE_PORT")
  fi
  if [ -n "${REMOTE_IDENTITY:-}" ]; then
    SSH_OPTS+=(-i "$REMOTE_IDENTITY")
  fi
fi
# rsync 가 -e 로 ssh 호출 시 같은 옵션. rsync 부모 프로세스가 conda 의
# LD_LIBRARY_PATH/LD_PRELOAD 를 ssh child 로 상속하면 OpenSSL ABI mismatch
# (e.g., "Built against 30000020, you have 30600020") → rsync 자체를 env -u
# 로 wrap 해 child ssh 도 system lib 사용하도록.
RSYNC_SSH="$_SYS_SSH ${SSH_OPTS[*]}"
do_rsync() {
  env -u LD_LIBRARY_PATH -u LD_PRELOAD rsync "$@"
}

bold "Phase2 artifact sync (yaml-driven)"
info "yaml          : $YAML"
info "remote target : $REMOTE_TARGET"
info "remote project: $REMOTE_PROJ"

# repo-relative → local absolute
abspath() {
  case "$1" in
    "")    echo "" ;;
    /*)    echo "$1" ;;
    *)     echo "$PROJ_ROOT/$1" ;;
  esac
}

sync_one() {
  local label="$1" rel="$2"
  if [ -z "$rel" ]; then
    warn "$label SKIP — yaml 에 path 미설정"
    return 0
  fi
  local local_path remote_path remote_dir
  local_path="$(abspath "$rel")"
  remote_path="$REMOTE_PROJ/$rel"
  if [ ! -e "$local_path" ]; then
    warn "$label SKIP — local 에 없음: $local_path"
    return 0
  fi
  remote_dir="$(dirname "$remote_path")"
  # remote parent dir mkdir (--mkpath 의존 회피: rsync 3.2+ 필요)
  env -u LD_LIBRARY_PATH -u LD_PRELOAD "$_SYS_SSH" "${SSH_OPTS[@]}" "$REMOTE_TARGET" \
      "mkdir -p '$remote_dir'" || {
    err "$label remote mkdir 실패: $remote_dir"
    return 1
  }
  if [ -d "$local_path" ]; then
    info "$label DIR : $local_path → $REMOTE_TARGET:$remote_path/"
    do_rsync -avP -e "$RSYNC_SSH" "$local_path/" "$REMOTE_TARGET:$remote_path/"
  else
    info "$label FILE: $local_path → $REMOTE_TARGET:$remote_path"
    do_rsync -avP -e "$RSYNC_SSH" "$local_path" "$REMOTE_TARGET:$remote_path"
  fi
}

sync_one "ckpt"            "${ARTIFACT_CKPT:-}"
sync_one "sidecar parquet" "${ARTIFACT_SIDECAR:-}"
sync_one "vector DB"       "${ARTIFACT_VECTOR_DB:-}"

bold "artifacts sync done"
