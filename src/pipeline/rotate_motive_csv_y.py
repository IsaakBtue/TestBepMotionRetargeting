#!/usr/bin/env python3
"""
Rotate all marker positions in a raw Motive CSV around the vertical (Y) axis.

Motive export is Y-up (X, Y, Z columns per marker). Rotation is applied to every
(X, Y, Z) triplet after Frame and Time; header rows are copied unchanged.

Default --degrees -90 is “90° to the right” in plan view (clockwise when looking
down from +Y), matching a common “turn the capture to align with +X” experiment:

  x' = cos(θ)·x + sin(θ)·z
  z' = -sin(θ)·x + cos(θ)·z
  y' = y

Examples (from repo root, soma env):

  python -m src.pipeline.rotate_motive_csv_y data/overlay.csv --out data/overlay_rY90_right.csv
  python -m src.pipeline.rotate_motive_csv_y data/overlay.csv --degrees 90 --out data/overlay_rY90_left.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import pathlib
from typing import List


def _find_frame_row(rows: List[List[str]]) -> int:
    for i, row in enumerate(rows):
        if row and str(row[0]).strip() == "Frame":
            return i
    raise ValueError("Could not find a row whose first cell is 'Frame'.")


def rotate_triplet(x: float, y: float, z: float, deg: float) -> tuple[float, float, float]:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    xn = c * x + s * z
    zn = -s * x + c * z
    return xn, y, zn


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", type=pathlib.Path, help="Raw Motive CSV (multi-line header).")
    p.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="Output path (default: <stem>_rY<degrees>.csv next to input).",
    )
    p.add_argument(
        "--degrees",
        type=float,
        default=-90.0,
        help="Rotation around +Y in degrees (default -90 = 90° right in plan view).",
    )
    args = p.parse_args()

    inp = args.csv_path.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    out = args.out
    if out is None:
        out = inp.with_name(f"{inp.stem}_rY{args.degrees:g}.csv")
    else:
        out = out.expanduser().resolve()

    with inp.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    frame_i = _find_frame_row(rows)
    header_len = len(rows[frame_i])
    out_rows: List[List[str]] = [list(r) for r in rows[: frame_i + 1]]

    for r in rows[frame_i + 1 :]:
        if not r or all(c == "" for c in r):
            out_rows.append(r)
            continue
        row = list(r)
        while len(row) < header_len:
            row.append("")
        new_row = row[:2]
        i = 2
        while i + 2 < len(row):
            xs, ys, zs = row[i], row[i + 1], row[i + 2]
            try:
                x = float(xs) if xs not in ("", None) else float("nan")
                y = float(ys) if ys not in ("", None) else float("nan")
                z = float(zs) if zs not in ("", None) else float("nan")
            except ValueError:
                new_row.extend([xs, ys, zs])
                i += 3
                continue
            if math.isnan(x) or math.isnan(y) or math.isnan(z):
                new_row.extend([xs, ys, zs])
            else:
                xn, yn, zn = rotate_triplet(x, y, z, args.degrees)
                new_row.extend([f"{xn:.6f}", f"{yn:.6f}", f"{zn:.6f}"])
            i += 3
        # append any trailing incomplete triplet unchanged
        if i < len(row):
            new_row.extend(row[i:])
        out_rows.append(new_row)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        for row in out_rows:
            w.writerow(row)

    print(f"Wrote {out} (Ry({args.degrees:g}°) on marker XYZ, Frame/Time unchanged)")


if __name__ == "__main__":
    main()
