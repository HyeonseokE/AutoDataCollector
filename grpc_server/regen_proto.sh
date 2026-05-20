#!/usr/bin/env bash
# Regenerate _pb2.py + _pb2_grpc.py stubs from preselective.proto.
# Run from project root.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJ_ROOT"

python -m grpc_tools.protoc \
  -I grpc_server \
  --python_out=grpc_server \
  --grpc_python_out=grpc_server \
  grpc_server/preselective.proto

# grpc_tools emits a flat `import preselective_pb2`, which fails inside a
# package. Patch the import to be relative.
sed -i 's/^import preselective_pb2 as preselective__pb2$/from . import preselective_pb2 as preselective__pb2/' \
  grpc_server/preselective_pb2_grpc.py

echo "[regen_proto] stubs refreshed in grpc_server/"
