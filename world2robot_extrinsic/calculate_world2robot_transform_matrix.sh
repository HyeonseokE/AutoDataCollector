#!/bin/bash

# ============================================================
#   World-to-Robot 외부 캘리브레이션: 변환행렬 계산
# ============================================================
#
#   수집된 매칭점 쌍을 사용하여 Kabsch-Umeyama 알고리즘으로
#   world → robot base 변환행렬을 계산합니다.
#
# ============================================================

echo ""
echo "============================================================"
echo "World-to-Robot 외부 캘리브레이션: 변환행렬 계산"
echo "============================================================"
echo ""
echo "사용법: ./calculate_world2robot_transform_matrix.sh ROBOT_INDEX INPUT_JSON OUTPUT_DIR"
echo "  ROBOT_INDEX: 로봇 번호 (기본값: 2)"
echo "  INPUT_JSON:  매칭포인트 JSON 경로 (기본값: matching_points/robot{N}_matching_points.json)"
echo "  OUTPUT_DIR:  결과 저장 폴더 (기본값: extrinsics)"
echo ""
echo "예시:"
echo "  ./calculate_world2robot_transform_matrix.sh 2"
echo "  ./calculate_world2robot_transform_matrix.sh 2 matching_points/robot2_matching_points.json extrinsics"
echo "  ./calculate_world2robot_transform_matrix.sh 3 my_points.json my_output"
echo ""

# Arguments with defaults
ROBOT_INDEX=${1:-3}
INPUT_JSON=${2:-"matching_points/robot${ROBOT_INDEX}_matching_points.json"}
OUTPUT_DIR=${3:-"extrinsics"}

# Validate robot index
if [ "$ROBOT_INDEX" != "2" ] && [ "$ROBOT_INDEX" != "3" ]; then
    echo "오류: ROBOT_INDEX는 2 또는 3이어야 합니다"
    exit 1
fi

echo "로봇: $ROBOT_INDEX"
echo "입력 파일: $INPUT_JSON"
echo "출력 폴더: $OUTPUT_DIR"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Convert relative paths to absolute
if [[ ! "$INPUT_JSON" = /* ]]; then
    INPUT_JSON="$SCRIPT_DIR/$INPUT_JSON"
fi
if [[ ! "$OUTPUT_DIR" = /* ]]; then
    OUTPUT_DIR="$SCRIPT_DIR/$OUTPUT_DIR"
fi

# Run Python script
PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH" python3 "$SCRIPT_DIR/world_frame2robot_base_frame.py" \
    --robot "$ROBOT_INDEX" \
    --input "$INPUT_JSON" \
    --output-dir "$OUTPUT_DIR"
