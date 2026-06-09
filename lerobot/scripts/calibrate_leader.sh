#!/usr/bin/env bash
# SO-101 Robot4 Leader Calibration Script
# Port: /dev/ttyACM1 (WS3 robot4 leader)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# -------- conda env --------
CONDA_ENV="${CONDA_ENV:-lerobot}"
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# -------- PYTHONPATH --------
ADC_ROOT="$(cd "$REPO_DIR/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:$ADC_ROOT:$ADC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

PORT="${PORT:-/dev/ttyACM1}"
ID="${ID:-so101_robot4_leader}"
TELEOP_TYPE="${TELEOP_TYPE:-so101_leader}"

echo "========================================="
echo "  SO-101 Robot4 Leader Calibration"
echo "  Port:  $PORT"
echo "  ID:    $ID"
echo "  Type:  $TELEOP_TYPE"
echo "========================================="

if [ -e "$PORT" ]; then
    sudo chmod 777 "$PORT"
else
    echo "[ERROR] Serial port not found: $PORT" >&2
    exit 1
fi

lerobot-calibrate \
    --teleop.type="$TELEOP_TYPE" \
    --teleop.port="$PORT" \
    --teleop.id="$ID"
