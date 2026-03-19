#!/bin/bash
# Move Robot to Recorded State
#
# Usage:
#   ./run_move_to_state.sh 0 initial          # robot0 -> initial state
#   ./run_move_to_state.sh 0 free             # robot0 -> free state
#   ./run_move_to_state.sh 0 initial free     # robot0 -> initial -> free (sequential)

cd "$(dirname "${BASH_SOURCE[0]}")"

python scripts/move_to_state.py "$@"
