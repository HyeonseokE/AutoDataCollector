#!/bin/bash

# ============================================================
#   데이터 흐름 요약
#   소프트웨어 파이프라인: EE Pose → Hardware

#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  User Input: EE Position (x, y, z) in meters                                │
#   │  예: (0.25, 0.0, 0.15)                                                       │
#   └─────────────────────┬───────────────────────────────────────────────────────┘
#                         │
#                         ▼
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  [STEP 1] IK Solver (KinematicsEngine)                                      │
#   │  target_joints_rad = kinematics.inverse_kinematics_multi(target_position)   │
#   │  출력: 5개 관절 각도 (radians), 예: [0.1, -0.5, 0.8, 0.3, -0.2]              │
#   └─────────────────────┬───────────────────────────────────────────────────────┘
#                         │
#                         ▼
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  [STEP 2] Radians → Normalized (-100 to +100)                               │
#   │  calibration_limits.radians_to_normalized(target_joints_rad)                │
#   │  수식: normalized = (radians / half_range) * 100 + offset_normalized         │
#   │  출력: 예: [5.2, -25.1, 40.3, 15.2, -10.1]                                   │
#   └─────────────────────┬───────────────────────────────────────────────────────┘
#                         │
#                         ▼
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  [STEP 3] Normalized → Encoder (FeetechController._unnormalize)             │
#   │  수식: encoder = range_min + ((normalized + 100) / 200) * encoder_range     │
#   │  drive_mode 적용: drive_mode==1이면 normalized 부호 반전                      │
#   │  출력: 예: [2100, 1500, 2800, 2200, 1900]                                    │
#   └─────────────────────┬───────────────────────────────────────────────────────┘
#                         │
#                         ▼
#   ┌─────────────────────────────────────────────────────────────────────────────┐
#   │  [STEP 4] Hardware Command (write_2byte to ADDR_GOAL_POSITION)              │
#   │  motor._write_2byte(motor_id, ADDR_GOAL_POSITION, encoder_value)            │
#   │  출력: 모터가 해당 encoder 위치로 이동                                        │
#   └─────────────────────────────────────────────────────────────────────────────┘


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
echo "Move to Pose (IK + Trajectory Planning + Compensation)"
echo "============================================================"
echo ""
echo "좌표계:"
echo "  +X = 전방, +Y = 좌측, +Z = 상방"
echo "  원점 = base_link, 최대 도달거리: ~0.4m"
echo ""
echo "사용법: ./run_move_to_pose.sh X Y Z [OPTIONS]"
echo "  위치: X Y Z (미터)"
echo ""
echo "옵션:"
echo "  --frame FRAME       좌표계 지정 (기본: base_link, 또는 'world')"
echo "  --dry-run           계획만 수행, 실행 안함"
echo "  --no-initial-state  초기 상태 이동 생략"
echo "  --no-decel          종단 감속 비활성화"
echo "  --no-multi-ik       멀티-IK 비활성화 (현재 자세에서 단일 IK)"
echo "  --multi-ik-verbose  멀티-IK 탐색 상세 출력"
echo "  --no-compensation   적응형 보상 비활성화"
echo ""
echo "경로 추종 보상 (Adaptive Compensation):"
echo "  기본 활성화 - 궤적 실행 중 피드포워드 보정 적용"
echo "  - 방향 보상: 백래시/중력에 의한 방향별 오차 보정"
echo "  - 중력 LUT: 관절 각도에 따른 중력 처짐 보정"
echo "  COMP_FACTOR=auto    보상 계수 (auto = z 기반 적응형)"
echo "                      z<0.12m→2.0, 0.12-0.15m→1.75, ≥0.15m→1.5"
echo ""
echo "종단 감속 (End Deceleration):"
echo "  DECEL_START=0.6     감속 시작점 (0.6 = 마지막 40%)"
echo "  DECEL_STRENGTH=3.0  감속 강도 (2.0=보통, 3.0=강함)"
echo ""

# Target position
X=${1:-0.3}
Y=${2:--0.0}
Z=${3:-0.1}

# End deceleration parameters (can be overridden by environment variables)
DECEL_START=${DECEL_START:-0.6}
DECEL_STRENGTH=${DECEL_STRENGTH:-3.0}

# Adaptive compensation parameters (can be overridden by environment variables)
COMP_FACTOR=${COMP_FACTOR:-auto}

# Additional options
OPTIONS=""
USE_INITIAL_STATE="--via-initial-state"
USE_DECEL=""
USE_MULTI_IK="--multi-ik"
USE_MULTI_IK_VERBOSE=""
USE_COMPENSATION=""
USE_FRAME="--frame world"

# Parse arguments
NEXT_IS_FRAME=false
for arg in "$@"; do
    if $NEXT_IS_FRAME; then
        USE_FRAME="--frame $arg"
        NEXT_IS_FRAME=false
        continue
    fi
    case $arg in
        --frame)
            NEXT_IS_FRAME=true
            ;;
        --no-initial-state)
            USE_INITIAL_STATE=""
            ;;
        --dry-run)
            OPTIONS="$OPTIONS --dry-run"
            ;;
        --no-decel)
            USE_DECEL="--no-decel"
            ;;
        --no-multi-ik)
            USE_MULTI_IK=""
            ;;
        --multi-ik-verbose)
            USE_MULTI_IK_VERBOSE="--multi-ik-verbose"
            ;;
        --no-compensation)
            USE_COMPENSATION="--no-compensation"
            ;;
    esac
done

# Build deceleration args
DECEL_ARGS=""
DECEL_ARGS="$DECEL_ARGS --decel-start $DECEL_START"
DECEL_ARGS="$DECEL_ARGS --decel-strength $DECEL_STRENGTH"

# Build compensation args
COMP_ARGS=""
if [ "$COMP_FACTOR" != "auto" ]; then
    COMP_ARGS="--compensation-factor $COMP_FACTOR"
fi

echo "목표 위치: x=$X, y=$Y, z=$Z"

if [ -n "$USE_INITIAL_STATE" ]; then
    echo "모드: 초기 상태 경유 (안전한 IK)"
else
    echo "모드: 직접 이동 (초기 상태 생략)"
fi

if [ -n "$USE_DECEL" ]; then
    echo "종단 감속: 비활성화"
else
    echo "종단 감속: 활성화 (시작=$DECEL_START, 강도=$DECEL_STRENGTH)"
fi

if [ -n "$USE_MULTI_IK" ]; then
    echo "멀티-IK: 활성화 (여러 초기값 시도)"
else
    echo "멀티-IK: 비활성화 (단일 IK)"
fi

if [ -n "$USE_COMPENSATION" ]; then
    echo "경로 추종 보상: 비활성화"
else
    if [ "$COMP_FACTOR" = "auto" ]; then
        echo "경로 추종 보상: 활성화 (z-적응형)"
    else
        echo "경로 추종 보상: 활성화 (계수=$COMP_FACTOR)"
    fi
fi

if [ -n "$USE_FRAME" ]; then
    echo "좌표계: ${USE_FRAME#--frame }"
else
    echo "좌표계: base_link (기본)"
fi
echo ""

python scripts/move_to_pose.py \
    --config robot_configs/robot/so101_robot2.yaml \
    --x $X \
    --y $Y \
    --z $Z \
    --duration 3.0 \
    $USE_INITIAL_STATE \
    $USE_DECEL \
    $DECEL_ARGS \
    $USE_MULTI_IK \
    $USE_MULTI_IK_VERBOSE \
    $USE_COMPENSATION \
    $COMP_ARGS \
    $USE_FRAME \
    $OPTIONS
