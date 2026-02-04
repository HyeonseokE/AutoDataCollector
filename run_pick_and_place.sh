#!/bin/bash

# ============================================================
#   LeRobot Pick and Place Demo
# ============================================================
#
#   Pick and Place 동작 시연
#   skills/skills_lerobot.py의 execute_pick_and_place() 사용
#
# ============================================================

# Grant permission to all ttyACM ports
echo "Setting permissions for /dev/ttyACM* ports..."
for port in /dev/ttyACM*; do
    if [ -e "$port" ]; then
        sudo chmod 777 "$port"
        echo "  $port - OK"
    fi
done

echo ""
echo "============================================================"
echo "LeRobot Pick and Place Demo"
echo "============================================================"
echo ""
echo "동작 순서:"
echo "  0. Initial State로 이동"
echo "  1. 그리퍼 90도 회전"
echo "  2. Execute Pick (열기 → 접근 → 하강 → 닫기 → 들기)"
echo "  3. Execute Place (접근 → 하강 → 열기 → 들기)"
echo "  4. Initial State로 복귀"
echo "  5. Free State로 이동 (안전 주차)"
echo ""
echo "사용법: ./run_pick_and_place.sh [OPTIONS]"
echo ""
echo "로봇 선택:"
echo "  --robot N            로봇 인덱스 (2 또는 3, 기본값: 3)"
echo ""
echo "위치 옵션 (world 좌표계, 미터 단위):"
echo "  --pick-x X       Pick X 좌표 (기본값: 0.15)"
echo "  --pick-y Y       Pick Y 좌표 (기본값: 0.05)"
echo "  --pick-z Z       Pick Z 좌표 (기본값: 0.02)"
echo "  --place-x X      Place X 좌표 (기본값: 0.15)"
echo "  --place-y Y      Place Y 좌표 (기본값: -0.05)"
echo "  --place-z Z      Place Z 좌표 (기본값: 0.02)"
echo ""
echo "기타 옵션:"
echo "  --approach-height H  접근 높이 (기본값: 0.05m)"
echo "  --duration D         이동 시간 (기본값: 3.0초)"
echo "  --frame FRAME        좌표계 (기본값: world)"
echo "  --config FILE        로봇 설정 파일 (--robot과 함께 사용 불가)"
echo "  --dry-run            계획만 보여주기 (실행 안함)"
echo ""

# Default values
ROBOT_INDEX=${ROBOT_INDEX:-2}
FRAME=${FRAME:-world}
PICK_X=${PICK_X:-0.2}
PICK_Y=${PICK_Y:-0.0}
PICK_Z=${PICK_Z:-0.02}
PLACE_X=${PLACE_X:-0.25}
PLACE_Y=${PLACE_Y:-0.15}
PLACE_Z=${PLACE_Z:-0.02}
APPROACH_HEIGHT=${APPROACH_HEIGHT:-0.02}
DURATION=${DURATION:-3.0}

# Parse --robot argument
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --robot)
            ROBOT_INDEX="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# Set config based on robot index
CONFIG=${CONFIG:-robot_configs/robot/so101_robot${ROBOT_INDEX}.yaml}

echo "Using Robot: $ROBOT_INDEX"
echo "Config: $CONFIG"
echo ""

# Pass all arguments to Python script
PYTHONPATH=src:$PYTHONPATH python3 skills/skills_lerobot.py \
    --config "$CONFIG" \
    --frame "$FRAME" \
    --pick-x "$PICK_X" \
    --pick-y "$PICK_Y" \
    --pick-z "$PICK_Z" \
    --place-x "$PLACE_X" \
    --place-y "$PLACE_Y" \
    --place-z "$PLACE_Z" \
    --approach-height "$APPROACH_HEIGHT" \
    --duration "$DURATION" \
    "${EXTRA_ARGS[@]}"
