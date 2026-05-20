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

# ── 3. preflight: config + urdf must exist, phase2 grpc 활성화 확인 ──
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
# 옛 preflight 는 recording_config 의 ``preselective_filter.enabled_forward`` 를
# 검사. 그러나 phase2_server_infer_settings.yaml → phase2_config.yaml 통합 +
# recording_config_ws3.yaml 의 preselective_filter 섹션 제거 이후에는 그 키가
# 없음. 새 source-of-truth 는 ``phase2_config.yaml.skill_planner_transport.mode``
# (= "grpc" 일 때 server 가 동작해야 함).
PROJ_ROOT_PYEOF="$PROJ_ROOT" python - "$RECORDING_CONFIG" <<'PYEOF'
import os, sys, yaml
from pathlib import Path
proj_root = Path(os.environ.get("PROJ_ROOT_PYEOF", "."))
rec_cfg = yaml.safe_load(open(sys.argv[1])) or {}
ph2_path = proj_root / "pipeline_config" / "phase2_config.yaml"
ph2_cfg = {}
if ph2_path.exists():
    try:
        ph2_cfg = yaml.safe_load(open(ph2_path)) or {}
    except Exception as e:
        sys.exit(f"XX phase2_config.yaml parse error: {e}")
# 1) skill_planner_transport.mode == 'grpc' 우선
mode = ((ph2_cfg.get("skill_planner_transport") or {}).get("mode") or "").strip().lower()
if mode == "grpc":
    print("  config OK — phase2_config.yaml: skill_planner_transport.mode=grpc")
    sys.exit(0)
# 2) legacy fallback — recording_config 의 preselective_filter.enabled_forward
psf = rec_cfg.get("preselective_filter") or {}
if bool(psf.get("enabled_forward", psf.get("enabled", False))):
    print("  config OK — recording_config.preselective_filter.enabled_forward=true (legacy)")
    sys.exit(0)
sys.exit(
    "XX phase2 server 가 동작할 trigger 없음. 다음 중 하나가 필요:\n"
    "    (A) pipeline_config/phase2_config.yaml.skill_planner_transport.mode: grpc   ← 권장\n"
    "    (B) recording_config_*.yaml.preselective_filter.enabled_forward: true       ← legacy"
)
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
