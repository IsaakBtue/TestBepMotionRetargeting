#!/bin/bash
# Full pipeline: raw CSV -> filter/fill -> MoSh++ -> SMPL-X -> retargeted Booster T1
#
# Usage:
#   ./pipeline.sh --csv <path/to/raw_markers.csv> [--skip-smooth] [--compare]
#
# Outputs:
#   - MoSh++ / SMPL-X outputs:
#       output/soma_mosh39/ (default) or output/soma_mosh39_raw/ + output/soma_mosh39_smoothed/ (when --compare)
#   - Retargeted robot motion:
#       output/retargeting/ (default) or output/retargeting_raw/ + output/retargeting_smoothed/ (when --compare)
#
# Then run ./visualize.sh to open the MuJoCo viewer.

set -euo pipefail

# ── parse args ───────────────────────────────────────────────────────────────
CSV=""
SKIP_SMOOTH=""
COMPARE="0"
FORCE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --csv) CSV="$2"; shift 2 ;;
        --skip-smooth) SKIP_SMOOTH="--skip-smooth"; shift 1 ;;
        --compare) COMPARE="1"; shift 1 ;;
        --force) FORCE="--force"; shift 1 ;;
        *) echo "Unknown argument: $1"; echo "Usage: $0 --csv <path/to/raw.csv> [--skip-smooth] [--compare] [--force]"; exit 1 ;;
    esac
done

if [ -z "$CSV" ]; then
    echo "Usage: $0 --csv <path/to/raw.csv> [--skip-smooth] [--compare] [--force]"
    exit 1
fi

if [ ! -f "$CSV" ]; then
    echo "Error: CSV not found: $CSV"
    exit 1
fi

# ── paths ────────────────────────────────────────────────────────────────────
BEP="/home/isaak/BEP"
PIPELINE="$BEP/src/pipeline"
SCRIPTS="$BEP/scripts"

CSV_FILLED="${CSV%.csv}_filled39.csv"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

# ── Step 0: filter & fill gaps in raw CSV ────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo " Step 0: Filter & fill CSV"
echo "════════════════════════════════════════════════════"
conda activate soma
cd "$BEP"
python "$PIPELINE/filter_fill_39markers.py" "$CSV" --out "$CSV_FILLED"
echo "→ $CSV_FILLED"

run_variant () {
    local variant_name="$1"   # default/raw/smoothed (used for display only)
    local output_dir="$2"
    local retarget_dir="$3"
    local smooth_arg="${4:-}" # "" or "--skip-smooth"

    local smplx_pkl="$output_dir/smplx_body_mesh_all_frames.pkl"

    echo ""
    echo "════════════════════════════════════════════════════"
    echo " Step 1 (${variant_name}): MoSh++ -> SMPL-X"
    echo "════════════════════════════════════════════════════"
    python "$PIPELINE/run_39marker_soma_mosh.py" \
        --csv "$CSV_FILLED" \
        --output-dir "$output_dir" \
        $smooth_arg $FORCE

    echo ""
    echo "════════════════════════════════════════════════════"
    echo " Step 2 (${variant_name}): Retarget to Booster T1 (direct PKL path)"
    echo "════════════════════════════════════════════════════"
    RETARGET_DIR_OVERRIDE="$retarget_dir" "$SCRIPTS/retarget_direct.sh" "$smplx_pkl"
}

if [ "$COMPARE" = "1" ]; then
    run_variant "raw" "$BEP/output/soma_mosh39_raw" "$BEP/output/retargeting_raw" "--skip-smooth"
    run_variant "smoothed" "$BEP/output/soma_mosh39_smoothed" "$BEP/output/retargeting_smoothed" ""
else
    run_variant "default" "$BEP/output/soma_mosh39" "$BEP/output/retargeting" "$SKIP_SMOOTH"
fi

echo ""
echo "════════════════════════════════════════════════════"
echo " Done! Run ./visualize.sh to open the MuJoCo viewer"
echo "════════════════════════════════════════════════════"
