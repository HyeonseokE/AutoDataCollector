#!/bin/bash
#
# Subgoal Prediction Model - Inference Script
#
# Usage:
#   ./models/run_inference.sh                     # Dry-run with default settings
#   ./models/run_inference.sh --execute           # Execute on robot
#   ./models/run_inference.sh --robot 2           # Use robot 2
#   ./models/run_inference.sh -i "custom instruction"
#

set -e

# ============================================================
# Configuration
# ============================================================
CHECKPOINT="models/subgoal_predictor/checkpoints/run_d64_l1/best.pt"
INSTRUCTION="pick up the red block and place it on the blue dish"
ROBOT_ID=3
DURATION=3.0
DRY_RUN=false
SAVE_VIS="results/pred.png"

# ============================================================
# Parse Arguments
# ============================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint|-c)
            CHECKPOINT="$2"
            shift 2
            ;;
        --instruction|-i)
            INSTRUCTION="$2"
            shift 2
            ;;
        --robot|-r)
            ROBOT_ID="$2"
            shift 2
            ;;
        --duration|-d)
            DURATION="$2"
            shift 2
            ;;
        --execute|-e)
            DRY_RUN=false
            shift
            ;;
        --save-vis|-s)
            SAVE_VIS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Subgoal Prediction Model - Inference Script"
            echo ""
            echo "Usage: ./models/run_inference.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -c, --checkpoint PATH    Model checkpoint (default: $CHECKPOINT)"
            echo "  -i, --instruction TEXT   Task instruction"
            echo "  -r, --robot ID           Robot ID: 2 or 3 (default: $ROBOT_ID)"
            echo "  -d, --duration SEC       Movement duration per subgoal (default: $DURATION)"
            echo "  -e, --execute            Execute on robot (default: dry-run)"
            echo "  -s, --save-vis PATH      Save visualization image"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  # Dry-run (predict only, no robot execution)"
            echo "  ./models/run_inference.sh"
            echo ""
            echo "  # Execute on robot 3"
            echo "  ./models/run_inference.sh --execute --robot 3"
            echo ""
            echo "  # Custom instruction"
            echo "  ./models/run_inference.sh -i \"pick up the blue cup\" --execute"
            echo ""
            echo "  # Save visualization"
            echo "  ./models/run_inference.sh --save-vis results/pred.png"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================
# Run Inference
# ============================================================
cd "$(dirname "$0")/.."

echo "============================================================"
echo "Subgoal Prediction Model - Inference"
echo "============================================================"
echo "Checkpoint:   $CHECKPOINT"
echo "Instruction:  $INSTRUCTION"
echo "Robot ID:     $ROBOT_ID"
echo "Duration:     $DURATION s"
echo "Mode:         $([ "$DRY_RUN" = true ] && echo 'DRY-RUN (no execution)' || echo 'EXECUTE')"
[ -n "$SAVE_VIS" ] && echo "Save vis:     $SAVE_VIS"
echo "============================================================"
echo ""

# Build command
CMD="python -m models.subgoal_predictor.robot_executor"
CMD="$CMD --checkpoint \"$CHECKPOINT\""
CMD="$CMD --instruction \"$INSTRUCTION\""
CMD="$CMD --robot $ROBOT_ID"
CMD="$CMD --duration $DURATION"

if [ "$DRY_RUN" = true ]; then
    CMD="$CMD --dry-run"
fi

if [ -n "$SAVE_VIS" ]; then
    CMD="$CMD --save-vis \"$SAVE_VIS\""
fi

# Execute
eval $CMD
