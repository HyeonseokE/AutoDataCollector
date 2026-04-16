#!/usr/bin/env bash
# =============================================================================
#  Skill-segmented EE Trajectory Visualizer
#
#  HuggingFace 데이터셋의 한 에피소드를 로드해 skill 경계마다 색을 바꿔가며
#  EE 궤적을 4-view 3D로 그립니다.
#
#  Usage:
#    # 기본 (스크립트 상단 변수 사용)
#    bash scripts/visualize_skills.sh
#
#    # 에피소드만 바꾸기
#    EPISODE=3 bash scripts/visualize_skills.sh
#
#    # 다른 데이터셋 / 다른 로봇 URDF
#    REPO_ID=user/my_dataset REVISION=v3.0 \
#    URDF=assets/urdf/so101_robot0.urdf \
#    EPISODE=5 bash scripts/visualize_skills.sh
#
#    # skill 경계 기준 변경 (default: skill.natural_language)
#    SKILL_KEY=skill.type bash scripts/visualize_skills.sh
#
#    # joint 소스 변경 (default: action.radian_urdf0)
#    JOINT_KEY=observation.state.radian_urdf0 bash scripts/visualize_skills.sh
#
#    # CLI 인자 추가 전달
#    bash scripts/visualize_skills.sh --step-label-stride 50
#
#  결과는 "$OUTPUT_DIR/<repo>_<rev>/ep<NN>/" 디렉토리에 저장됩니다:
#    - ee_trajectory_3d.png     EE 3D 궤적 (4-view 정적 이미지)
#    - ee_trajectory_3d.html    EE 3D 궤적 (인터랙티브, 마우스 회전/줌/팬)
#    - ee_displacement.png      skill별 EE 변위 시계열
#    - joint_angles.png         조인트 각도 시계열 (skill별 색상)
#
#  인터랙티브 HTML 열기:
#    firefox outputs/skill_viz/<repo>_<rev>/ep<NN>/ee_trajectory_3d.html &
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
EPISODE="${EPISODE:-0}"
URDF="${URDF:-$(cd "$REPO_DIR/.." && pwd)/assets/urdf/so101_robot2.urdf}"
SKILL_KEY="${SKILL_KEY:-skill.natural_language}"
JOINT_KEY="${JOINT_KEY:-action.radian_urdf0}"

# -------- 출력 경로 자동 구성 --------
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/outputs/skill_viz}"
REPO_TAG="$(echo "$REPO_ID" | tr '/' '_')_$REVISION"
EPISODE_PADDED="$(printf '%02d' "$EPISODE")"
FINAL_OUTPUT_DIR="$OUTPUT_DIR/$REPO_TAG/ep${EPISODE_PADDED}"
mkdir -p "$FINAL_OUTPUT_DIR"

# -------- 실행 --------
python -m skill_visualizer.visualize \
    --repo-id "$REPO_ID" \
    --revision "$REVISION" \
    --episode "$EPISODE" \
    --urdf "$URDF" \
    --skill-key "$SKILL_KEY" \
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
