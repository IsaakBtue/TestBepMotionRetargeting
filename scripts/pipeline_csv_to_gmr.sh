#!/bin/bash
# TEST pipeline: (filled) 39-marker CSV -> GMR -> retargeted Booster T1 motion
#
# This intentionally skips MoSh++ / SMPL-X fitting. It approximates a small set
# of SMPL-X joint frames from sparse markers (positions + identity rotations),
# then uses GMR IK to produce a robot motion.
#
# Usage:
#   ./pipeline_csv_to_gmr.sh --csv <path/to/*_filled39.csv> [--units m|mm] [--no-viewer]
#
# Output:
#   output/csv_to_gmr/<name>__csv_to_gmr__booster_t1.pkl

set -euo pipefail

CSV=""
UNITS="auto"
NO_VIEWER=""
MOSH_ROTATE_FLAG=""
YUP_TO_ZUP_FLAG=""
FORWARD_FROM="mb_to_mf"
YAW180=""
ORIENT_MODE="segments"
NO_LR_SWAP=""
SHOW_HUMAN_NAMES=""
IK_PROFILE="csv_pos"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv) CSV="$2"; shift 2 ;;
    --units) UNITS="$2"; shift 2 ;;
    --no-mosh-rotate) MOSH_ROTATE_FLAG="--no-mosh-rotate"; shift 1 ;;
    --no-yup-to-zup) YUP_TO_ZUP_FLAG="--no-yup-to-zup"; shift 1 ;;
    --forward-from) FORWARD_FROM="$2"; shift 2 ;;
    --yaw180) YAW180="--yaw180"; shift 1 ;;
    --orient-mode) ORIENT_MODE="$2"; shift 2 ;;
    --no-lr-swap) NO_LR_SWAP="--no-lr-swap"; shift 1 ;;
    --show-human-names) SHOW_HUMAN_NAMES="--show-human-names"; shift 1 ;;
    --ik-profile) IK_PROFILE="$2"; shift 2 ;;
    --no-viewer) NO_VIEWER="--no_viewer"; shift 1 ;;
    *) echo "Unknown argument: $1"; echo "Usage: $0 --csv <file> [--units m|mm|auto] [--no-mosh-rotate] [--no-yup-to-zup] [--forward-from mb_to_mf|mf_to_mb] [--no-viewer]"; exit 1 ;;
  esac
done

if [ -z "$CSV" ]; then
  echo "Usage: $0 --csv <file> [--units m|mm|auto] [--no-mosh-rotate] [--no-yup-to-zup] [--forward-from mb_to_mf|mf_to_mb] [--no-viewer]"
  exit 1
fi

if [ ! -f "$CSV" ]; then
  echo "Error: CSV not found: $CSV"
  exit 1
fi

BEP="/home/isaak/BEP"
PIPELINE="$BEP/src/pipeline"
OUT_DIR="$BEP/output/csv_to_gmr"
T1_ROOT="$BEP/src/retargeting"

mkdir -p "$OUT_DIR"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate HumanoidDataGeneration

export PYTHONPATH="$T1_ROOT:${PYTHONPATH:-}"

cd "$BEP"
python "$PIPELINE/csv_to_gmr.py" \
  --csv "$CSV" \
  --robot booster_t1 \
  --out "$OUT_DIR/$(basename "${CSV%.csv}")__csv_to_gmr__booster_t1.pkl" \
  --units "$UNITS" \
  --forward-from "$FORWARD_FROM" \
  $YAW180 \
  --orient-mode "$ORIENT_MODE" \
  --ik-profile "$IK_PROFILE" \
  $NO_LR_SWAP \
  $SHOW_HUMAN_NAMES \
  $MOSH_ROTATE_FLAG \
  $YUP_TO_ZUP_FLAG \
  $NO_VIEWER

