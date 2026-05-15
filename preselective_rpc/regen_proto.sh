#!/usr/bin/env bash
# Regenerate _pb2.py + _pb2_grpc.py stubs from preselective.proto.
# Run from project root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJ_ROOT"

python -m grpc_tools.protoc \
  -I preselective_rpc \
  --python_out=preselective_rpc \
  --grpc_python_out=preselective_rpc \
  preselective_rpc/preselective.proto

# grpc_tools emits a flat `import preselective_pb2`, which fails inside a
# package. Patch the import to be relative.
sed -i 's/^import preselective_pb2 as preselective__pb2$/from . import preselective_pb2 as preselective__pb2/' \
  preselective_rpc/preselective_pb2_grpc.py

echo "[regen_proto] stubs refreshed in preselective_rpc/"
