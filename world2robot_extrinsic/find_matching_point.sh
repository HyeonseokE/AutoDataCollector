#!/bin/bash

# ============================================================
#   World-to-Robot 외부 캘리브레이션: 매칭점 수집
# ============================================================
#
#   한 번 실행할 때마다 하나의 매칭점 쌍을 수집합니다.
#   10번 이상 실행하여 충분한 점을 누적하세요.
#
#   처음부터 다시 시작하려면:
#     rm robot{N}_matching_points.json
#
# ============================================================

# Grant permission to all ttyACM ports
for port in /dev/ttyACM*; do
    if [ -e "$port" ]; then
        sudo chmod 777 "$port" 2>/dev/null
    fi
done

echo ""
echo "============================================================"
echo "World-to-Robot 외부 캘리브레이션: 매칭점 수집"
echo "============================================================"
echo ""
echo "사용법: ./find_matching_point.sh ROBOT_INDEX X Y Z"
echo "  ROBOT_INDEX: 로봇 번호 (기본값: 2)"
echo "  X:         World X 좌표 (기본값: 0.0)"
echo "  Y:         World Y 좌표 (기본값: 0.0)"
echo "  Z:         World Z 좌표 (기본값: 0.0)"
echo ""
echo "예시:"
echo "  ./find_matching_point.sh 2 0.10 0.00 0.05"
echo "  ./find_matching_point.sh 3 0.15 0.05 0.08"
echo ""
echo "처음부터 다시 시작하려면:"
echo "  rm robot{N}_matching_points.json"
echo ""

###################### 설정 ###########################
# Target position (positional arguments with defaults, 단위: meter)
# ex) 10cm = 0.1m
ROBOT_INDEX=${1:-2}
X=${2:-0.28}
Y=${3:--0.12}
Z=${4:-0.0}

# Validate robot index
if [ "$ROBOT_INDEX" != "2" ] && [ "$ROBOT_INDEX" != "3" ]; then
    echo "오류: ROBOT_INDEX는 2 또는 3이어야 합니다"
    exit 1
fi

echo "로봇: $ROBOT_INDEX"
echo "World 좌표: x=$X, y=$Y, z=$Z"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Run Python script
PYTHONPATH="$PROJECT_DIR/src:$PYTHONPATH" python3 "$SCRIPT_DIR/find_matching_point.py" \
    --robot "$ROBOT_INDEX" \
    --x "$X" \
    --y "$Y" \
    --z "$Z"
