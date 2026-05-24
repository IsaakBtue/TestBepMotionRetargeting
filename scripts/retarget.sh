#!/bin/bash
# Convert SMPL-X MoSh++ output -> retargeted Booster T1 motion PKL
#
# Usage:
#   ./retarget.sh [path/to/smplx_body_mesh_all_frames.pkl]
#
# If no argument is given, defaults to the standard output location.

set -euo pipefail

BEP="/home/isaak/BEP"
SMPLX_MESH_PKL="${1:-$BEP/output/soma_mosh39/smplx_body_mesh_all_frames.pkl}"
RETARGET_DIR="$BEP/output/retargeting"
T1_ROOT="$BEP/src/retargeting"
GVHMR_PT="$RETARGET_DIR/gvhmr_pred_from_smplx.pt"
ROBOT_PKL="$RETARGET_DIR/booster_t1_motion.pkl"

if [ ! -f "$SMPLX_MESH_PKL" ]; then
    echo "Error: input file not found: $SMPLX_MESH_PKL"
    exit 1
fi

mkdir -p "$RETARGET_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate HumanoidDataGeneration

export PYTHONPATH="$T1_ROOT:${PYTHONPATH:-}"

echo "Converting SMPL-X PKL -> GVHMR-style .pt ..."
SMPLX_MESH_PKL_ENV="$SMPLX_MESH_PKL" GVHMR_PT_ENV="$GVHMR_PT" \
python - <<'PY'
import os, pickle, numpy as np, torch

in_pkl = os.environ["SMPLX_MESH_PKL_ENV"]
out_pt = os.environ["GVHMR_PT_ENV"]

with open(in_pkl, "rb") as f:
    d = pickle.load(f)

poses = np.asarray(d["poses"], dtype=np.float32)   # (T,165)
trans = np.asarray(d["trans"], dtype=np.float32)   # (T,3)
betas = np.asarray(d["betas"], dtype=np.float32)   # (T,16)

smpl_params_global = {
    "body_pose": torch.from_numpy(poses[:, 3:66]),
    "global_orient": torch.from_numpy(poses[:, 0:3]),
    "transl": torch.from_numpy(trans),
    "betas": torch.from_numpy(betas.mean(axis=0)[:10]).unsqueeze(0),
}

torch.save({"smpl_params_global": smpl_params_global}, out_pt)
print("Wrote:", out_pt)
PY

echo "Retargeting to Booster T1 ..."
python "$T1_ROOT/scripts/gvhmr_to_robot.py" \
    --gvhmr_pred_file "$GVHMR_PT" \
    --robot booster_t1 \
    --rate_limit \
    --save_path "$ROBOT_PKL"

echo ""
echo "Done. Outputs:"
echo "  $GVHMR_PT"
echo "  $ROBOT_PKL"
echo ""
echo "Run ./visualize.sh to view the robot motion."
