#!/usr/bin/env bash
# server Ready RPC + buffer 상태 한 줄 확인.
# yaml 의 transport.address 자동 read — task 별로 port 다르므로.
#
# 사용:
#   bash grpc_server/server_ready.sh                   # 현재 branch 의 phase2_config.yaml
#   PHASE2_SERVER_YAML=.../phase2_config_pnp.yaml bash grpc_server/server_ready.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PHASE2_SERVER_YAML="${PHASE2_SERVER_YAML:-$PROJ_ROOT/pipeline_config/phase2_config.yaml}"

# conda env (grpcio + protobuf 정합). lerobot_cap 우선, miniconda3/anaconda3 자동 탐지.
for _p in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
    [ -f "$_p" ] && source "$_p" && break
done
conda activate "${CONDA_ENV:-lerobot_cap}" 2>/dev/null || true
cd "$PROJ_ROOT"

ADDR="$(python - "$PHASE2_SERVER_YAML" <<'PY'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])) or {}
print((cfg.get("transport") or {}).get("address") or "127.0.0.1:50061")
PY
)"

python - "$ADDR" <<'PY'
import sys, grpc
sys.path.insert(0, ".")
from grpc_server import preselective_pb2_grpc as g, preselective_pb2 as p
addr = sys.argv[1]
try:
    i = g.PreselectiveAcquirerStub(grpc.insecure_channel(addr)).Ready(p.Empty(), timeout=3)
    print(f"READY {addr}  buffer={i.buffer_total} skills={len(i.buffer_per_skill)}  ckpt={i.smolvla_checkpoint!r}  device={i.device}")
except grpc.RpcError as e:
    print(f"NOT READY {addr}  {e.code().name}")
PY
