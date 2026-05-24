#!/usr/bin/env python3
"""
Convert a Motive/CSV export (e.g. Take12502.csv) to C3D for use with SOMA + MoSh++.

SOMA expects: unlabeled mocap point cloud (C3D) in folder structure
  support_files/evaluation_mocaps/original/<dataset_name>/<subject_name>/<sequence>.c3d

This script writes C3D in millimeters (standard). Use mocap.unit: 'mm' when running SOMA.
If your CSV is in meters (e.g. "Length Units,Meters" in header), we convert to mm on write.

Usage:
  python csv_to_c3d.py Take12502.csv -o Custom24/subject1/Take12502.c3d
  # Then place Custom24 under SOMA's support_files/evaluation_mocaps/original/
"""

import argparse
import csv
import os
import re

import numpy as np


def parse_header_for_rate(path: str) -> float:
    """Parse frame rate from CSV header (Capture Frame Rate or Export Frame Rate)."""
    with open(path, "r") as f:
        first = f.readline()
    m = re.search(r"(?:Export|Capture) Frame Rate,([\d.]+)", first)
    return float(m.group(1)) if m else 100.0


def _is_cleaned_format(path: str) -> bool:
    """Return True if this CSV is a cleaned _filled39 file (Frame,Time,NAME_x,...)."""
    with open(path, "r") as f:
        header = f.readline().strip()
    return header.startswith("Frame,Time,") and header.split(",")[2].endswith("_x")


def find_data_start(path: str) -> int:
    """Return 0-based row index where the data header row starts."""
    with open(path, "r") as f:
        for i, line in enumerate(f):
            stripped = line.strip()
            if stripped.startswith("Frame,Time (Seconds),") or stripped.startswith("Frame,Time,"):
                return i
    raise ValueError("Could not find data header row in CSV")


def read_markers_csv(path: str):
    """
    Read marker data from CSV. Returns (markers, frame_rate, labels).
    markers: (n_frames, n_markers, 3) in METERS; missing data as nan.

    Handles two formats:
    - Raw Motive export (5-row header, repeated X/Y/Z columns, generic M1..MN labels)
    - Cleaned _filled39 CSV (Frame,Time,LMT5_x,LMT5_y,LMT5_z,... — labels from column names)
    """
    frame_rate = parse_header_for_rate(path)
    cleaned = _is_cleaned_format(path)
    data_start = find_data_start(path)

    all_rows = []
    with open(path, "r") as f:
        r = csv.reader(f)
        for i, row in enumerate(r):
            if i < data_start:
                continue
            all_rows.append(row)

    if not all_rows:
        raise ValueError("No rows found after header")

    header_row = all_rows[0]

    if cleaned:
        # Columns: Frame, Time, NAME_x, NAME_y, NAME_z, NAME_x, ...
        # Extract one label per marker triplet from the _x column names
        labels = []
        for col in header_row[2:]:
            if col.endswith("_x"):
                labels.append(col[:-2])  # strip "_x" suffix

        n_markers = len(labels)
        data_rows = [row for row in all_rows[1:] if row and row[0].strip() and row[0].strip() != "Frame"]
        n_frames = len(data_rows)
        markers = np.full((n_frames, n_markers, 3), np.nan, dtype=np.float64)

        for t, row in enumerate(data_rows):
            for mi in range(n_markers):
                base = 2 + mi * 3
                if base + 2 >= len(row):
                    break
                try:
                    x = float(row[base])   if row[base].strip()   else np.nan
                    y = float(row[base+1]) if row[base+1].strip() else np.nan
                    z = float(row[base+2]) if row[base+2].strip() else np.nan
                    markers[t, mi] = (x, y, z)
                except ValueError:
                    pass
    else:
        # Raw Motive format — skip header row, use generic labels
        data_rows = []
        for row in all_rows[1:]:
            if not row or row[0].strip() == "":
                continue
            try:
                int(row[0])
            except (ValueError, IndexError):
                continue
            data_rows.append(row)

        if not data_rows:
            raise ValueError("No data rows found")

        n_cols = len(data_rows[0])
        n_markers = (n_cols - 2) // 3
        if n_markers <= 0:
            raise ValueError(f"Not enough columns for markers: n_cols={n_cols}")

        n_frames = len(data_rows)
        markers = np.full((n_frames, n_markers, 3), np.nan, dtype=np.float64)
        labels = [f"M{i+1}" for i in range(n_markers)]

        for t, row in enumerate(data_rows):
            for mi in range(n_markers):
                base = 2 + mi * 3
                if base + 2 >= len(row):
                    break
                try:
                    x = float(row[base])   if row[base].strip()   else np.nan
                    y = float(row[base+1]) if row[base+1].strip() else np.nan
                    z = float(row[base+2]) if row[base+2].strip() else np.nan
                    markers[t, mi] = (x, y, z)
                except ValueError:
                    pass

    return markers, frame_rate, labels


def write_c3d(markers: "np.ndarray", labels: list, out_path: str, frame_rate: float, units_m: bool = True):
    """
    Write C3D file. markers: (n_frames, n_markers, 3).
    If units_m=True, markers are in meters and we convert to mm for C3D (standard).
    """
    try:
        import ezc3d
    except ImportError:
        raise ImportError("Install ezc3d: pip install ezc3d (or use SOMA/MoSh++ env)")

    if units_m:
        markers = markers * 1000.0  # m -> mm

    # C3D: points shape (4, n_points, n_frames) - 4 = x,y,z, residual
    n_frames, n_pts, _ = markers.shape
    pts = np.asarray(markers, dtype=np.float64)
    residual = np.ones((n_frames, n_pts))
    residual[np.isnan(pts).any(axis=-1)] = -1
    pts = np.nan_to_num(pts, nan=0.0)
    pts_4 = np.concatenate([pts, np.zeros((n_frames, n_pts, 1))], axis=-1)  # (T, P, 4)
    data_points = pts_4.transpose(2, 1, 0)  # (4, P, T)
    data_residuals = residual.transpose(1, 0)[None, :, :]  # (1, P, T)

    c = ezc3d.c3d()
    c["parameters"]["POINT"]["RATE"]["value"] = [frame_rate]
    c["parameters"]["POINT"]["LABELS"]["value"] = labels
    c["data"]["points"] = data_points
    if "meta_points" not in c["data"]:
        c["data"]["meta_points"] = {}
    c["data"]["meta_points"]["residuals"] = data_residuals
    c.write(out_path)


def main():
    import numpy as np
    global np
    parser = argparse.ArgumentParser(description="Convert CSV export to C3D for SOMA/MoSh++.")
    parser.add_argument("csv_path", help="Path to CSV (e.g. Take12502.csv)")
    parser.add_argument("-o", "--output", required=True, help="Output C3D path (e.g. Custom24/subject1/Take12502.c3d)")
    parser.add_argument("--units-m", action="store_true", default=True, help="CSV coordinates in meters (default)")
    parser.add_argument("--units-mm", action="store_true", help="CSV coordinates in millimeters")
    args = parser.parse_args()

    units_m = not args.units_mm
    markers, frame_rate, labels = read_markers_csv(args.csv_path)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_c3d(markers, labels, args.output, frame_rate, units_m=units_m)
    print(f"Wrote {args.output} | {markers.shape[0]} frames, {markers.shape[1]} markers, {frame_rate} Hz")
    if units_m:
        print("C3D written in mm. Use mocap.unit: 'mm' in SOMA.")


if __name__ == "__main__":
    main()
