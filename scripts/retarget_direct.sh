#!/bin/bash
# Retarget: MoSh++ SMPL-X PKL -> retargeted Booster T1 motion PKL
#
# This is the default retarget path for marker-based mocap in this repo:
#   MoSh++ already produces SMPL-X parameters, so we retarget directly from
#   `smplx_body_mesh_all_frames.pkl` without any GVHMR/video-specific format shims.
#
# Usage:
#   ./retarget_direct.sh [path/to/smplx_body_mesh_all_frames.pkl]
#
# Output:
#   output/retargeting/booster_t1_motion.pkl (default)
#
# You can override the output dir via:
#   RETARGET_DIR_OVERRIDE=/some/dir ./retarget_direct.sh <pkl>

set -euo pipefail

BEP="/home/isaak/BEP"
SMPLX_MESH_PKL="${1:-$BEP/output/soma_mosh39/smplx_body_mesh_all_frames.pkl}"
RETARGET_DIR="${RETARGET_DIR_OVERRIDE:-$BEP/output/retargeting}"
T1_ROOT="$BEP/src/retargeting"
ROBOT_PKL="$RETARGET_DIR/booster_t1_motion.pkl"

if [ ! -f "$SMPLX_MESH_PKL" ]; then
    echo "Error: input file not found: $SMPLX_MESH_PKL"
    exit 1
fi

mkdir -p "$RETARGET_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate HumanoidDataGeneration

export PYTHONPATH="$T1_ROOT:${PYTHONPATH:-}"

echo "Retargeting to Booster T1 (direct MoSh++ path) ..."
python "$T1_ROOT/scripts/mosh_to_robot.py" \
    --smplx_mesh_pkl "$SMPLX_MESH_PKL" \
    --robot booster_t1 \
    --rate_limit \
    --save_path "$ROBOT_PKL"

echo ""
echo "Done. Output:"
echo "  $ROBOT_PKL"
echo ""
echo "Run ./visualize.sh to view the robot motion."

