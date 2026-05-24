#!/bin/bash
# G1 .pt → SMPL-X → Booster T1 retargeting pipeline
#
# Usage:
#   ./scripts/pipeline_g1_pt_to_t1.sh --pt <path/to/motion.pt> [--fps <N>]
#
# --fps  Source capture frame rate in Hz (default: 30).
#        Common values: 30 (real-time capture), 50 (IsaacLab RL), 100 (high-speed).
#        This controls the temporal alignment inside the retargeter.
#
# Output (in output/g1_<stem>/):
#   smplx_body_mesh_all_frames.pkl     (intermediate SMPL-X)
#   smplx_body_mesh_vertices.pkl       (SMPL-X mesh vertices)
#   smplx_body_animation.gif           (SMPL-X body animation GIF)
#   retargeting/<stem>_booster.pkl     (retargeted Booster T1 motion)
#
# Then run ./visualize.sh to open the MuJoCo viewer.

set -euo pipefail

BEP="/home/isaak/BEP"
PT_PATH=""
FPS=30
FORCE_MOSH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pt)         PT_PATH="$2"; shift 2 ;;
        --fps)        FPS="$2"; shift 2 ;;
        --force-mosh) FORCE_MOSH="--force"; shift ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 --pt <file.pt> [--fps N] [--force-mosh]"; exit 1 ;;
    esac
done

if [ -z "$PT_PATH" ]; then
    echo "Usage: $0 --pt <path/to/motion.pt> [--fps N]"
    exit 1
fi
if [ ! -f "$PT_PATH" ]; then
    echo "Error: .pt file not found: $PT_PATH"
    exit 1
fi

STEM=$(basename "${PT_PATH%.pt}")
OUTPUT_DIR="$BEP/output/g1_$STEM"
SMPLX_PKL="$OUTPUT_DIR/smplx_body_mesh_all_frames.pkl"
MESH_PKL="$OUTPUT_DIR/smplx_body_mesh_vertices.pkl"
BODY_GIF="$OUTPUT_DIR/smplx_body_animation.gif"
RETARGET_DIR="$OUTPUT_DIR/retargeting"
ROBOT_PKL="$RETARGET_DIR/${STEM}_booster.pkl"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

# ── Step 1: G1 .pt → SMPL-X PKL (via MoSh++) ────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " Step 1: G1 .pt → SMPL-X (MoSh++ via 17 link markers)"
echo "════════════════════════════════════════════════════"
# Uses g1_pt_to_mosh.py (MoSh++) instead of the old g1_pt_to_smplx_pkl.py.
# The analytical approach had ~50cm arm error due to URDF pre-rotation offsets
# (see MAPPING_ANALYSIS section 10). MoSh++ fits SMPL-X directly to the 17 G1
# link_position markers, bypassing the broken coordinate-frame conversion.
conda activate soma
python "$BEP/src/pipeline/g1_pt_to_mosh.py" \
    --pt "$PT_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --fps "$FPS" \
    $FORCE_MOSH
echo "→ $SMPLX_PKL"

# ── Step 2: SMPL-X PKL → mesh vertices + body GIF ───────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " Step 2: SMPL-X → mesh vertices + body animation GIF"
echo "════════════════════════════════════════════════════"
conda activate soma
python "$BEP/src/pipeline/g1_smplx_to_mesh_pkl.py" \
    --input  "$SMPLX_PKL" \
    --output "$MESH_PKL"
python "$BEP/src/pipeline/create_smplx_gif.py" \
    --mesh "$MESH_PKL" \
    --out  "$BODY_GIF"
echo "→ $BODY_GIF"

# ── Step 3: SMPL-X → Booster T1 ─────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " Step 3: Retarget SMPL-X → Booster T1"
echo "════════════════════════════════════════════════════"
conda activate HumanoidDataGeneration
mkdir -p "$RETARGET_DIR"
export PYTHONPATH="$BEP/src/retargeting:${PYTHONPATH:-}"
python "$BEP/src/retargeting/scripts/mosh_to_robot.py" \
    --smplx_mesh_pkl "$SMPLX_PKL" \
    --robot booster_t1 \
    --mocap_fps "$FPS" \
    --no_viewer \
    --save_path "$ROBOT_PKL"
echo "→ $ROBOT_PKL"

echo ""
echo "════════════════════════════════════════════════════"
echo " Done!"
echo "  Body GIF : $BODY_GIF"
echo "  Robot PKL: $ROBOT_PKL"
echo "  Run: ./visualize.sh $ROBOT_PKL"
echo "════════════════════════════════════════════════════"
