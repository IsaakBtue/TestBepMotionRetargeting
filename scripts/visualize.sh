#!/bin/bash
# Visualize retargeted Booster T1 motion in MuJoCo viewer
#
# Usage:
#   ./visualize.sh [path/to/booster_t1_motion.pkl]
#
# If no argument is given, defaults to the standard output location.

set -euo pipefail

BEP="/home/isaak/BEP"
ROBOT_PKL="${1:-$BEP/output/retargeting/booster_t1_motion.pkl}"
T1_ROOT="$BEP/src/retargeting"

if [ ! -f "$ROBOT_PKL" ]; then
    echo "Error: motion file not found: $ROBOT_PKL"
    echo "Run ./pipeline.sh (or ./retarget_direct.sh) first."
    exit 1
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate HumanoidDataGeneration

export PYTHONPATH="$T1_ROOT:${PYTHONPATH:-}"

echo "Opening MuJoCo viewer for: $ROBOT_PKL"
python "$T1_ROOT/scripts/vis_robot_motion.py" \
    --robot booster_t1 \
    --robot_motion_path "$ROBOT_PKL"
