#!/usr/bin/env bash
# ONE command to bring up the grpc_server gRPC server on the H100.
#
#   bash grpc_server/run_h100_server.sh
#
# It (1) runs the idempotent environment setup, (2) activates the conda env,
# (3) preflights the config, then (4) starts the server in the foreground
# (Ctrl+C to stop). Re-running is safe — setup skips whatever is already done.
#
# Optional env vars:
#   ENV_NAME          conda env                 (default: lerobot)
#   HOST / PORT       bind address              (default: 0.0.0.0 / 50061)
#   RECORDING_CONFIG  yaml the server reads      (default: pipeline_config/recording_config_ws3.yaml)
#   URDF              curobo URDF                (default: assets/urdf/so101_robot4.urdf)
#   SKIP_SETUP=1      skip step 1 (env already prepared — faster restart)

set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-50061}"
RECORDING_CONFIG="${RECORDING_CONFIG:-pipeline_config/recording_config_ws3.yaml}"
URDF="${URDF:-assets/urdf/so101_robot4.urdf}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_ROOT"

# Ignore ~/.local user-site packages so a stray lerobot/transformers there
# cannot shadow the vendored copy used by the server.
export PYTHONNOUSERSITE=1

bold() { printf "\033[1;36m== %s ==\033[0m\n" "$*"; }
err()  { printf "\033[1;31mXX %s\033[0m\n" "$*" >&2; }

# ── 1. one-time environment setup (idempotent) ───────────────────────────
if [[ "${SKIP_SETUP:-0}" == "1" ]]; then
  bold "step 1/4  environment setup  (SKIPPED — SKIP_SETUP=1)"
else
  bold "step 1/4  environment setup  (idempotent — skips what's done)"
  bash "$SCRIPT_DIR/setup_h100_server.sh"
fi

# ── 2. activate conda env ────────────────────────────────────────────────
bold "step 2/4  activate conda env: $ENV_NAME"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# ── 3. preflight: config + urdf must exist, preselective must be enabled ──
bold "step 3/4  preflight"
if [[ ! -f "$RECORDING_CONFIG" ]]; then
  err "recording config not found: $RECORDING_CONFIG"
  err "  copy it from the robot host (see setup step 8) or set RECORDING_CONFIG=..."
  exit 1
fi
if [[ ! -f "$URDF" ]]; then
  err "URDF not found: $URDF   (set URDF=... to override)"
  exit 1
fi
python - "$RECORDING_CONFIG" <<'PYEOF'
import sys, yaml
psf = (yaml.safe_load(open(sys.argv[1])) or {}).get("preselective_filter") or {}
if not bool(psf.get("enabled_forward", psf.get("enabled", False))):
    sys.exit("XX preselective_filter.enabled_forward must be true — "
             "the server has nothing to do otherwise.")
print("  config OK — preselective_filter.enabled_forward is true")
PYEOF

# ── 4. start the server (foreground; Ctrl+C to stop) ─────────────────────
bold "step 4/4  starting grpc_server server"
echo "  bind             : $HOST:$PORT"
echo "  recording-config : $RECORDING_CONFIG"
echo "  urdf             : $URDF"
echo "  (first start loads pi05 + curobo onto the GPU — give it a minute)"
echo
exec python -u -m grpc_server.server \
  --host "$HOST" --port "$PORT" \
  --recording-config "$RECORDING_CONFIG" \
  --urdf "$URDF"
