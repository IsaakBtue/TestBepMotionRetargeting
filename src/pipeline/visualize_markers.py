#!/usr/bin/env python3
"""
Visualize 39 marker positions from the cleaned CSV in 3D.

Reads one frame (default: frame 0) and plots every marker as a coloured dot
with its label, grouped by body region. Saves a PNG and shows interactively.

Usage (from repo root, any env with matplotlib/numpy):
  python src/pipeline/visualize_markers.py --csv data/Take8cleaned_filled39.csv
  python src/pipeline/visualize_markers.py --csv data/Take8cleaned_filled39.csv --frame 100
  python src/pipeline/visualize_markers.py --csv data/Take8cleaned_filled39.csv --out docs/marker_layout.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

# ── Body-region grouping ──────────────────────────────────────────────────────
# fmt: off
REGIONS: dict[str, list[str]] = {
    "Head":       ["LFHD", "RFHD", "LBHD", "RBHD", "ARIEL"],
    "Torso":      ["CLAV", "STRN", "C7", "T10"],
    "Shoulder":   ["LFSH", "RFSH"],
    "Upper arm":  ["LUPA", "RUPA"],
    "Elbow/wrist":["LELB", "RELB", "LIWR", "RIWR"],
    "Waist front":["LFWT", "RFWT", "MFWT"],
    "Waist back": ["LBWT", "RBWT", "MBWT"],
    "Thigh":      ["LTHI", "RTHI"],
    "Knee":       ["LKNE", "RKNE"],
    "Shin":       ["LSHN", "RSHN"],
    "Ankle":      ["LANK", "RANK"],
    "Heel/foot":  ["LHEE", "RHEE", "LMT1", "RMT1", "LMT5", "RMT5", "LTOE", "RTOE"],
}
# fmt: on

# Friendly full-name labels shown in the plot
LABEL_NAMES: dict[str, str] = {
    "LFHD": "L Front Head",    "RFHD": "R Front Head",
    "LBHD": "L Back Head",     "RBHD": "R Back Head",
    "ARIEL": "Top of Head",
    "CLAV": "Clavicle",        "STRN": "Sternum",
    "C7":   "C7 (neck base)",  "T10":  "T10 (mid-back)",
    "LFSH": "L Front Shoulder","RFSH": "R Front Shoulder",
    "LUPA": "L Upper Arm",     "RUPA": "R Upper Arm",
    "LELB": "L Elbow",         "RELB": "R Elbow",
    "LIWR": "L Inner Wrist",   "RIWR": "R Inner Wrist",
    "LFWT": "L Front Waist",   "RFWT": "R Front Waist",
    "MFWT": "Mid Front Waist ★","MBWT": "Mid Back Waist ★",
    "LBWT": "L Back Waist",    "RBWT": "R Back Waist",
    "LTHI": "L Thigh",         "RTHI": "R Thigh",
    "LKNE": "L Knee",          "RKNE": "R Knee",
    "LSHN": "L Shin",          "RSHN": "R Shin",
    "LANK": "L Ankle",         "RANK": "R Ankle",
    "LHEE": "L Heel",          "RHEE": "R Heel",
    "LMT1": "L MT1 (inner toe)","RMT1": "R MT1 (inner toe)",
    "LMT5": "L MT5 (outer toe)","RMT5": "R MT5 (outer toe)",
    "LTOE": "L Toe tip",       "RTOE": "R Toe tip",
}

# Colour per region (matplotlib named colours)
REGION_COLORS = {
    "Head":        "#e67e22",
    "Torso":       "#2980b9",
    "Shoulder":    "#8e44ad",
    "Upper arm":   "#c0392b",
    "Elbow/wrist": "#e74c3c",
    "Waist front": "#27ae60",
    "Waist back":  "#16a085",
    "Thigh":       "#f39c12",
    "Knee":        "#d35400",
    "Shin":        "#e67e22",
    "Ankle":       "#7f8c8d",
    "Heel/foot":   "#2c3e50",
}


def read_frame(csv_path: Path, frame_idx: int) -> dict[str, np.ndarray]:
    """Return {marker_name: np.array([x, y, z])} for one frame."""
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        lines = f.readlines()

    row = lines[frame_idx].strip().split(",")
    data: dict[str, np.ndarray] = {}
    col = 0
    while col < len(header):
        h = header[col]
        if h.endswith("_x") and col + 2 < len(header):
            name = h[:-2]
            try:
                xyz = np.array([float(row[col]), float(row[col + 1]), float(row[col + 2])])
                data[name] = xyz
            except ValueError:
                pass
            col += 3
        else:
            col += 1
    return data


def plot_markers(
    marker_data: dict[str, np.ndarray],
    frame_idx: int,
    out_path: Path | None,
) -> None:
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Build reverse lookup: marker → region
    marker_to_region = {m: r for r, ms in REGIONS.items() for m in ms}

    # Find axis limits for equal aspect
    coords = np.array(list(marker_data.values()))
    centre = coords.mean(axis=0)
    span = (coords.max(axis=0) - coords.min(axis=0)).max() * 0.6

    plotted_regions: set[str] = set()

    for name, xyz in marker_data.items():
        region = marker_to_region.get(name, "Other")
        color = REGION_COLORS.get(region, "#95a5a6")
        label_str = region if region not in plotted_regions else "_nolegend_"
        plotted_regions.add(region)

        # Larger dot for MFWT and MBWT to highlight them
        size = 120 if name in ("MFWT", "MBWT") else 60
        marker_sym = "^" if name in ("MFWT", "MBWT") else "o"

        # Y-up: plot as (X, Z, Y) so Z is depth and matplotlib Y = height
        ax.scatter(xyz[0], xyz[2], xyz[1],
                   c=color, s=size, marker=marker_sym,
                   label=label_str, depthshade=True, alpha=0.9)

        display_name = LABEL_NAMES.get(name, name)
        ax.text(xyz[0], xyz[2], xyz[1], f" {display_name}",
                fontsize=6.5, color=color, va="center")

    ax.set_xlabel("X (lateral)")
    ax.set_ylabel("Z (depth)")
    ax.set_zlabel("Y (height ↑)")
    ax.set_xlim(centre[0] - span, centre[0] + span)
    ax.set_ylim(centre[2] - span, centre[2] + span)
    ax.set_zlim(centre[1] - span, centre[1] + span)
    ax.view_init(elev=15, azim=-75)
    ax.set_title(
        f"39-marker layout — frame {frame_idx}\n"
        "★ = MFWT / MBWT (belly & lower-back, triangles) — potentially redundant with the 4 waist corners",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=8, markerscale=1.2, framealpha=0.7)
    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {out_path}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize 39-marker layout from cleaned CSV"
    )
    parser.add_argument("--csv", required=True, help="Path to cleaned 39-marker CSV")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default 0)")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: show interactively)")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    out_path = Path(args.out) if args.out else None
    marker_data = read_frame(csv_path, args.frame)

    # ── Audit: which markers are in data vs known to MoSh ────────────────────
    all_known = {m for ms in REGIONS.values() for m in ms}
    in_csv = set(marker_data)
    missing_from_known = in_csv - all_known
    missing_from_csv = all_known - in_csv
    print(f"Markers in CSV        : {len(in_csv)}  → {sorted(in_csv)}")
    print(f"All known to MoSh++   : {len(all_known)} (all 39 map to valid SMPL-X verts)")
    if missing_from_known:
        print(f"  In CSV but not in region map : {sorted(missing_from_known)}")
    if missing_from_csv:
        print(f"  In region map but not in CSV : {sorted(missing_from_csv)}")
    print()
    print("Waist ring (6 markers):")
    print("  Front: LFWT, MFWT★ (belly), RFWT")
    print("  Back:  LBWT, MBWT★ (lower-back), RBWT")
    print()
    print("MFWT and MBWT ARE known to MoSh++ and will be used.")
    print("They add one extra constraint between the L/R corner pairs.")
    print("Remove them only if they cause occlusion or labelling errors.")

    plot_markers(marker_data, args.frame, out_path)


if __name__ == "__main__":
    main()
