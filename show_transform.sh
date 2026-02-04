#!/bin/bash

# Show world to base_link coordinate transformation
#
# Usage:
#   ./show_transform.sh X Y Z
#   ./show_transform.sh 0.25 0.0 0.15

X=${1:-0.0}
Y=${2:-0.0}
Z=${3:-0.0}

python scripts/show_transform.py --x $X --y $Y --z $Z
