#!/usr/bin/env bash
# =============================================================================
#  Action Chunk EE Trajectory Visualizer
#
#  Usage:
#    # 기본 (matplotlib 팝업)
#    bash scripts/visualize_chunks.sh outputs/action_chunks/chunks_20260412_151957.npz
#
#    # 파일로 저장
#    bash scripts/visualize_chunks.sh outputs/action_chunks/chunks_20260412_151957.npz --output-dir outputs/plots
#
#    # FK 없이 joint angle만
#    bash scripts/visualize_chunks.sh outputs/action_chunks/chunks_20260412_151957.npz --no-fk
#
#    # URDF / joint names override
#    URDF=assets/urdf/so101_robot0.urdf bash scripts/visualize_chunks.sh chunks.npz
#
#  결과는 "$OUTPUT_DIR/<npz파일명>/" 하위에 저장됩니다.
#  예) NPZ_FILE=.../chunks_20260412_162511.npz → outputs/plots/chunks_20260412_162511/
#
#  생성되는 파일:
#    - joint_angles.png         조인트 각도 시계열
#    - ee_trajectory_3d.png     EE 3D 궤적 (4-view 정적 이미지)
#    - ee_displacement.png      청크별 EE 변위 시계열
#    - ee_trajectory_3d.html    EE 3D 궤적 (인터랙티브, 마우스 회전/줌/팬)
#
#  인터랙티브 HTML 열기 (실행 종료 시 실제 경로가 출력됩니다):
#    firefox outputs/plots/<npz이름>/ee_trajectory_3d.html &
# =============================================================================

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
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"

# -------- 설정 --------
URDF="${URDF:-$(cd "$REPO_DIR/.." && pwd)/assets/urdf/so101_robot2.urdf}"

# -------- npz 파일 경로 (직접 수정) --------
NPZ_FILE="${NPZ_FILE:-$REPO_DIR/outputs/action_chunks/chunks_20260415_142634.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/chunk_viz}"

# npz 파일 이름(확장자 제외)을 서브디렉토리로 자동 생성
NPZ_STEM="$(basename "$NPZ_FILE" .npz)"
FINAL_OUTPUT_DIR="$OUTPUT_DIR/$NPZ_STEM"
mkdir -p "$FINAL_OUTPUT_DIR"

# -------- 실행 --------
python -m chunk_visualizer.visualize \
    --npz "$NPZ_FILE" \
    --urdf "$URDF" \
    --output-dir "$FINAL_OUTPUT_DIR" \
    "$@"

# -------- 생성 파일 안내 --------
HTML_FILE="$FINAL_OUTPUT_DIR/ee_trajectory_3d.html"
if [[ -f "$HTML_FILE" ]]; then
    echo ""
    echo "=========================================="
    echo "  결과 위치: $FINAL_OUTPUT_DIR"
    echo "  인터랙티브 3D 시각화 (마우스로 회전):"
    echo "    firefox $HTML_FILE &"
    echo "    # 또는: xdg-open $HTML_FILE"
    echo "=========================================="
fi
