#!/usr/bin/env bash
# =============================================================================
#  Episode-level EE Trajectory Visualizer
#
#  데이터셋의 한 개 또는 여러 에피소드를 통째로 시각화. 한 에피소드 = 한 색상.
#  같은 에피소드 인덱스는 항상 같은 색상으로 렌더링됨 (deterministic).
#
#  Usage:
#    # 단일 에피소드
#    EPISODES=5 bash scripts/visualize_episodes.sh
#
#    # 범위 (1~10 inclusive)
#    EPISODES="1:10" bash scripts/visualize_episodes.sh
#
#    # 명시적 리스트 + 범위 혼합
#    EPISODES="0,3,7:9" bash scripts/visualize_episodes.sh
#
#    # 다른 데이터셋 (teleop 등 — skill 라벨 불필요)
#    REPO_ID=user/teleop_dataset REVISION=v3.0 EPISODES="0:4" \
#    URDF=assets/urdf/so101_robot1.urdf \
#    bash scripts/visualize_episodes.sh
#
#    # joint 소스 변경 (action 대신 observation.state)
#    JOINT_KEY=observation.state.radian_urdf0 EPISODES=2 \
#    bash scripts/visualize_episodes.sh
#
#  결과는 "$OUTPUT_DIR/<repo>_<rev>/<eptag>/" 디렉토리에 저장됩니다:
#    - ee_trajectory_3d.png     EE 3D 궤적 (4-view 정적 이미지)
#    - ee_trajectory_3d.html    EE 3D 궤적 (인터랙티브, 마우스 회전/줌/팬)
#    - ee_displacement.png      에피소드별 EE 변위 시계열
#    - joint_angles.png         조인트 각도 시계열 (에피소드별 색상)
#
#  eptag 예: ep05 (단일), ep01-10 (연속 범위), ep00_03_07 (불연속)
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
REPO_ID="${REPO_ID:-skkuprism/cap_pnp_100ep}"
REVISION="${REVISION:-v3.0}"
EPISODES="${EPISODES:-0}"
URDF="${URDF:-$(cd "$REPO_DIR/.." && pwd)/assets/urdf/so101_robot2.urdf}"
JOINT_KEY="${JOINT_KEY:-action.radian_urdf0}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/episode_viz}"

# -------- 출력 경로 결정 (Python의 _spec_to_tag와 동일 규칙) --------
REPO_TAG="$(echo "$REPO_ID" | tr '/' '_')_$REVISION"
EP_TAG="$(python -c "
import sys; sys.path.insert(0, '$REPO_DIR/scripts')
from episode_visualizer.visualize import parse_episodes_spec, _spec_to_tag
print(_spec_to_tag(parse_episodes_spec('$EPISODES')))
")"
FINAL_OUTPUT_DIR="$OUTPUT_DIR/$REPO_TAG/$EP_TAG"
mkdir -p "$FINAL_OUTPUT_DIR"

# -------- 실행 --------
python -m episode_visualizer.visualize \
    --repo-id "$REPO_ID" \
    --revision "$REVISION" \
    --episodes "$EPISODES" \
    --urdf "$URDF" \
    --joint-key "$JOINT_KEY" \
    --output-dir "$FINAL_OUTPUT_DIR" \
    "$@"

# -------- 결과 안내 --------
HTML_FILE="$FINAL_OUTPUT_DIR/ee_trajectory_3d.html"
if [[ -f "$HTML_FILE" ]]; then
    echo ""
    echo "=========================================="
    echo "  결과 위치: $FINAL_OUTPUT_DIR"
    echo "  인터랙티브 3D (마우스 회전/줌):"
    echo "    firefox $HTML_FILE &"
    echo "    # 또는: xdg-open $HTML_FILE"
    echo "=========================================="
fi
