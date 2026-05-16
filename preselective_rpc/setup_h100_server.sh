#!/usr/bin/env bash
# One-shot environment setup for the preselective_rpc gRPC server on H100.
#
# Idempotent — safe to re-run. Detects what's already in place and skips it.
#
# Run from the AutoDataCollector project root after `git clone`:
#   bash preselective_rpc/setup_h100_server.sh
#
# Optional env vars:
#   ENV_NAME        conda env to create/use (default: lerobot)
#   PY_VER          Python version          (default: 3.12)
#   CUROBO_REF      nvidia-curobo git ref   (default: main)
#   SKIP_ROBOT_CFG  set to 1 to skip curobo robot yml generation
#   ROBOT_IDS       robot ids to build curobo cfgs for (default: "0 1 2 3 4 6 8")

set -euo pipefail

ENV_NAME="${ENV_NAME:-lerobot}"
PY_VER="${PY_VER:-3.12}"
CUROBO_REF="${CUROBO_REF:-main}"
ROBOT_IDS="${ROBOT_IDS:-0 1 2 3 4 6 8}"

# Resolve project root (this script lives at preselective_rpc/setup_h100_server.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJ_ROOT"

# Ignore ~/.local user-site packages — the conda env must be self-contained.
# A stray `lerobot` / `transformers` in ~/.local otherwise shadows the
# vendored copy and breaks pi05 / smolvla loading.
export PYTHONNOUSERSITE=1

bold()  { printf "\033[1;36m== %s ==\033[0m\n" "$*"; }
info()  { printf "\033[36m  %s\033[0m\n" "$*"; }
warn()  { printf "\033[1;33m!! %s\033[0m\n" "$*"; }
err()   { printf "\033[1;31mXX %s\033[0m\n" "$*" >&2; }

# ──────────────────────────────────────────────────────────────────────
# 0. Sanity checks
# ──────────────────────────────────────────────────────────────────────
bold "0/9  preflight"

if ! command -v conda >/dev/null 2>&1; then
  err "conda not found on PATH. Install miniconda first."
  exit 1
fi
info "conda  : $(command -v conda)"

if ! command -v git >/dev/null 2>&1; then
  err "git not found on PATH."
  exit 1
fi
info "git    : $(git --version)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found — server boots fine but cuda calls will fail."
else
  info "GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
fi

if [[ ! -d "$PROJ_ROOT/lerobot" || ! -f "$PROJ_ROOT/lerobot/pyproject.toml" ]]; then
  err "Vendored lerobot/ missing under $PROJ_ROOT — bad clone?"
  exit 1
fi

# ──────────────────────────────────────────────────────────────────────
# 1. nvidia-curobo source clone (gitignored under src/, separate repo)
# ──────────────────────────────────────────────────────────────────────
bold "1/9  nvidia-curobo source"

CUROBO_DIR="$PROJ_ROOT/src/nvidia-curobo"
if [[ -d "$CUROBO_DIR/.git" ]]; then
  info "already present: $CUROBO_DIR"
else
  info "cloning NVlabs/curobo @ $CUROBO_REF → $CUROBO_DIR"
  mkdir -p "$PROJ_ROOT/src"
  git clone --depth 1 --branch "$CUROBO_REF" \
    https://github.com/NVlabs/curobo.git "$CUROBO_DIR"
fi

# ──────────────────────────────────────────────────────────────────────
# 2. conda env (create if missing)
# ──────────────────────────────────────────────────────────────────────
bold "2/9  conda env: $ENV_NAME (Python $PY_VER)"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  info "env exists; reusing"
else
  info "creating env"
  conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

conda activate "$ENV_NAME"
info "active : $(python --version) @ $CONDA_PREFIX"

# ──────────────────────────────────────────────────────────────────────
# 3. libstdc++ activation hook (CXXABI_1.3.15 for pinocchio / scipy)
# ──────────────────────────────────────────────────────────────────────
bold "3/9  LD_LIBRARY_PATH activation hook"

ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
HOOK="$ACT_DIR/preselective_libstdcxx.sh"
mkdir -p "$ACT_DIR"
if [[ ! -f "$HOOK" ]]; then
  cat > "$HOOK" <<'EOF'
#!/bin/bash
# Prepend conda env's libstdc++ so newer CXXABI symbols (CXXABI_1.3.15
# required by scipy / pinocchio) resolve to the conda copy, not /lib.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
  chmod +x "$HOOK"
  info "hook installed: $HOOK"
  conda deactivate && conda activate "$ENV_NAME"
else
  info "hook already present"
fi

# ──────────────────────────────────────────────────────────────────────
# 4. vendored lerobot (Python 3.12 required)
# ──────────────────────────────────────────────────────────────────────
bold "4/9  vendored lerobot/  (editable, with pi + smolvla extras)"

VENDORED_LEROBOT="$PROJ_ROOT/lerobot/src/lerobot"
lerobot_is_vendored() {
  python - 2>/dev/null <<PYEOF
import os, sys
try:
    import lerobot
except Exception:
    sys.exit(1)
got = os.path.realpath(os.path.dirname(lerobot.__file__))
sys.exit(0 if got == os.path.realpath("$VENDORED_LEROBOT") else 2)
PYEOF
}

if lerobot_is_vendored; then
  info "vendored lerobot active: $VENDORED_LEROBOT"
else
  warn "lerobot missing or shadowed by a non-vendored copy — reinstalling"
  # Loop: remove EVERY lerobot install (conda env + ~/.local can both have one).
  while pip uninstall -y lerobot >/dev/null 2>&1; do
    info "  removed a stray lerobot install"
  done
  info "pip install -e lerobot[pi,smolvla]  (pulls transformers 5.x, accelerate, …)"
  pip install -e "$PROJ_ROOT/lerobot[pi,smolvla]" --ignore-requires-python
  if ! lerobot_is_vendored; then
    err "lerobot STILL not the vendored copy — a stray install is shadowing it."
    err "  fix manually, then re-run this script:"
    err "    while pip uninstall -y lerobot 2>/dev/null; do :; done"
    err "    pip install -e lerobot[pi,smolvla] --ignore-requires-python"
    exit 1
  fi
  info "vendored lerobot installed"
fi

# ──────────────────────────────────────────────────────────────────────
# 5. nvidia-curobo (editable, no-deps to avoid pin/pinocchio collision)
# ──────────────────────────────────────────────────────────────────────
bold "5/9  nvidia-curobo  (editable, no-deps)"

if python -c "import curobo" 2>/dev/null; then
  info "curobo already importable"
else
  info "pip install cuda-core[cu12] + curobo runtime deps"
  pip install \
    "cuda-core[cu12]>=0.7" \
    trimesh viser warp-lang yourdfpy numpy-quaternion \
    importlib_resources setuptools_scm
  info "pip install -e nvidia-curobo --no-deps --no-build-isolation"
  pip install -e "$CUROBO_DIR" --no-deps --no-build-isolation
fi

# Smoke-test curobo MotionPlanner (catches missing runtime deps fast)
python - <<'PYEOF'
import curobo
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
print(f"  curobo @ {curobo.__file__}")
PYEOF

# ──────────────────────────────────────────────────────────────────────
# 6. gRPC + project deps
# ──────────────────────────────────────────────────────────────────────
bold "6/9  gRPC + project (lerobot_cap) editable"

if ! python -c "import grpc, grpc_tools" 2>/dev/null; then
  info "pip install grpcio grpcio-tools"
  pip install grpcio grpcio-tools
else
  info "grpcio already present"
fi

if ! python -c "import sys; sys.path.insert(0, '.'); from preselective_rpc.client import PreselectiveClient" 2>/dev/null; then
  info "pip install -e .  (lerobot_cap project metadata)"
  pip install -e "$PROJ_ROOT" --no-deps --no-build-isolation
fi

# Always-runs: refresh proto stubs in case .proto evolved post-clone.
if [[ -f "$PROJ_ROOT/preselective_rpc/preselective.proto" ]]; then
  info "regenerating proto stubs"
  bash "$PROJ_ROOT/preselective_rpc/regen_proto.sh"
fi

# ──────────────────────────────────────────────────────────────────────
# 7. curobo robot configs (so101_robot{N}.yml)
# ──────────────────────────────────────────────────────────────────────
if [[ "${SKIP_ROBOT_CFG:-0}" == "1" ]]; then
  bold "7/9  curobo robot configs  (SKIPPED — SKIP_ROBOT_CFG=1)"
else
  bold "7/9  curobo robot configs  (so101_robot{$ROBOT_IDS})"

  mkdir -p "$PROJ_ROOT/robot_configs/curobo"
  for r in $ROBOT_IDS; do
    out="$PROJ_ROOT/robot_configs/curobo/so101_robot${r}.yml"
    urdf="$PROJ_ROOT/assets/urdf/so101_robot${r}.urdf"
    if [[ -f "$out" ]]; then
      info "exists, skip: so101_robot${r}.yml"
      continue
    fi
    if [[ ! -f "$urdf" ]]; then
      warn "URDF missing, skip: $urdf"
      continue
    fi
    info "building: so101_robot${r}.yml (~30-60s)"
    python -m curobo.examples.getting_started.build_robot_model \
      --urdf "$urdf" \
      --asset-path "$PROJ_ROOT/assets/urdf/" \
      --output "$out" \
      --num-collision-samples 500 2>&1 \
      | grep -E "✓|✗|Saving|Fitted|Created|Error" | head -8
  done
fi

# ──────────────────────────────────────────────────────────────────────
# 8. recording_config_ws3.yaml stub (gitignored — must exist for server)
# ──────────────────────────────────────────────────────────────────────
bold "8/9  recording_config_ws3.yaml"

YAML="$PROJ_ROOT/pipeline_config/recording_config_ws3.yaml"
if [[ -f "$YAML" ]]; then
  info "already present: $YAML"
else
  warn "pipeline_config/recording_config_ws3.yaml missing"
  warn "  copy it from your robot host:"
  warn "    scp <robot_host>:<path>/AutoDataCollector/pipeline_config/recording_config_ws3.yaml \\"
  warn "        $YAML"
  warn "  the server reads preselective_filter + perturbation.skill sections only."
fi

# ──────────────────────────────────────────────────────────────────────
# 9. End-to-end import sanity (does NOT load SmolVLA / curobo planner)
# ──────────────────────────────────────────────────────────────────────
bold "9/9  import sanity"

python - <<'PYEOF'
import sys
sys.path.insert(0, '.')
import lerobot.policies.pi05.modeling_pi05        # pi05 reachable
import curobo                                     # curobo reachable
import grpc                                       # gRPC reachable
from preselective_rpc.client import PreselectiveClient
from preselective_rpc.server import PreselectiveAcquirerServicer
from preselective_filter.integration import GrpcPlannerClient, setup_preselective_filter
from preselective_filter import Selector, SelectorConfig
print("  all imports OK")
PYEOF

bold "DONE"
echo
echo "Next — start the server with ONE command:"
echo "  bash preselective_rpc/run_h100_server.sh"
echo
echo "(run_h100_server.sh re-runs this setup idempotently, then launches the"
echo " server. From the robot host, point the yaml's transport_address at this"
echo " machine and call ready() to verify connectivity.)"
