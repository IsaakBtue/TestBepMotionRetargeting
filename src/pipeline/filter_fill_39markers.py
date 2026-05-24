import argparse
import csv
import pathlib
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

"""
Canonical 39-marker layout (names only; Motive maps by <template>:Marker#).

lower39final — rigid indices 1..19 (display L1..L19):
  1 LMT5 … 19 RFWT  (no MBWT on lower when upper exports 20 markers; see below)

upper39final — rigid indices 1..20 on the upper / High template (e.g. High39V2Humanoid):
  1 LBWT … 19 ARIEL  20 MBWT

Some older takes only expose MBWT on Lower:Marker20 and have 19 upper markers; then
we still map Lower:Marker20 -> MBWT (see process_file).

Display U1..U20 in visualize_soma_markers_on_smplx.py (MBWT at U2 when using standard
mapping; data from MBWT column). For plot-only label corrections on filled CSVs, see
docs/Visualization/vis_raw.py VIS_LABEL_REMAP.

L/R swap mode: LOWER_NAMES and UPPER_NAMES exchange left/right SOMA labels at each
paired Motive index (midline markers unchanged). Regenerate *_filled39.csv after edits.
"""

# Lower rigid: L/R swapped vs original (paired indices exchange L↔R SOMA tags).
LOWER_NAMES: Dict[int, str] = {
    1: "RMT5",
    2: "RTOE",
    3: "RMT1",
    4: "RHEE",
    5: "RANK",
    6: "RSHN",
    7: "LMT1",
    8: "LTOE",
    9: "LMT5",
    10: "LHEE",
    11: "LANK",
    12: "LSHN",
    13: "LKNE",
    14: "RKNE",
    15: "LTHI",
    16: "RTHI",
    17: "MFWT",
    18: "RFWT",
    19: "LFWT",
}

# Upper body: Motive Marker1..Marker20 on High/upper assets.
# High39V2Humanoid lab layout with L/R labels swapped vs the original calibration
# (use when Motive rigid indices match the suit but left/right names were mirrored).
UPPER_NAMES: Dict[int, str] = {
    1: "RBWT",
    2: "MBWT",
    3: "LBWT",
    4: "STRN",
    5: "T10",
    6: "CLAV",
    7: "RIWR",
    8: "RELB",
    9: "RUPA",
    10: "RFSH",
    11: "C7",
    12: "LFSH",
    13: "LUPA",
    14: "LELB",
    15: "LIWR",
    16: "RFHD",
    17: "LFHD",
    18: "LBHD",
    19: "RBHD",
    20: "ARIEL",
}


def _name_row_has_marker(name_row: List[str], n_cols: int, template: str, idx: int) -> bool:
    token = f"{template}:Marker{idx}"
    for col_idx, cell in enumerate(name_row):
        if col_idx >= n_cols:
            break
        if isinstance(cell, str) and cell.strip() == token:
            return True
    return False


def detect_template_names(name_row: List[str], n_cols: int) -> Tuple[str, str]:
    """
    Detect template prefixes from Name row entries like '<template>:Marker12'.

    Returns:
      (lower_template_name, upper_template_name)
    using this rule:
      - template containing Marker20 -> lower (20 markers)
      - template containing markers up to 19 only -> upper (19 markers)
    """
    pattern = re.compile(r"^(.*):Marker(\d+)$")
    template_markers: Dict[str, set[int]] = {}

    for col_idx, cell in enumerate(name_row):
        if col_idx >= n_cols:
            break
        m = pattern.match(cell.strip()) if isinstance(cell, str) else None
        if not m:
            continue
        tmpl = m.group(1)
        marker_idx = int(m.group(2))
        template_markers.setdefault(tmpl, set()).add(marker_idx)

    if not template_markers:
        raise ValueError("Could not detect template marker labels in Name row.")

    # Prefer explicit textual hints when available (most reliable for custom
    # takes where marker counts can be swapped between templates).
    lower_text = [t for t in template_markers if "lower" in t.lower()]
    upper_text = [t for t in template_markers if "upper" in t.lower() or "high" in t.lower()]
    if len(lower_text) == 1 and len(upper_text) == 1:
        return lower_text[0], upper_text[0]

    lower_candidates = [t for t, nums in template_markers.items() if 20 in nums]
    upper_candidates = [t for t, nums in template_markers.items() if 20 not in nums and 19 in nums]
    if len(lower_candidates) == 1 and len(upper_candidates) == 1:
        return lower_candidates[0], upper_candidates[0]

    # Fallback to legacy names if present.
    if "lower39final" in template_markers and "upper39final" in template_markers:
        return "lower39final", "upper39final"

    raise ValueError(
        "Could not uniquely detect lower/upper templates from Name row. "
        f"Detected templates: {sorted(template_markers.keys())}"
    )


def find_header_rows(rows: List[List[str]]) -> Tuple[int, int]:
    """Return indices of the 'Name' row and the 'Frame' row."""
    name_idx = None
    frame_idx = None
    for i, row in enumerate(rows):
        vals = [str(v) for v in row]
        if name_idx is None and "Name" in vals:
            name_idx = i
        if frame_idx is None and "Frame" in vals:
            frame_idx = i
    if name_idx is None or frame_idx is None:
        raise ValueError("Could not locate 'Name' and/or 'Frame' rows in CSV.")
    return name_idx, frame_idx


def process_file(input_path: pathlib.Path, output_path: pathlib.Path) -> None:
    # ── 1. Read all rows with csv.reader (handles variable-width rows) ──────
    with input_path.open(newline="") as f:
        rows: List[List[str]] = list(csv.reader(f))

    name_idx, frame_idx = find_header_rows(rows)

    name_row = rows[name_idx]    # e.g. ["", "Name", "lower39final:Marker1", ...]
    header_row = rows[frame_idx] # e.g. ["Frame", "Time (Seconds)", "X", "Y", "Z", ...]
    n_cols = len(header_row)

    # ── 2. Build a numpy array for all data rows, padded to n_cols ──────────
    data_rows = rows[frame_idx + 1:]
    n_rows = len(data_rows)

    # Fill a string array with empty strings, then copy each row in.
    raw = np.full((n_rows, n_cols), "", dtype=object)
    for r, row in enumerate(data_rows):
        length = min(len(row), n_cols)
        raw[r, :length] = row[:length]

    # ── 3. Identify 39-marker column positions from the Name row ─────────────
    # The name_row may be shorter or longer than header_row; we only care about
    # columns that exist in both, so cap at n_cols.
    lower_template, upper_template = detect_template_names(name_row, n_cols)
    print(f"Detected templates: lower='{lower_template}', upper='{upper_template}'")

    label_to_human: Dict[str, str] = {}
    for k, name in LOWER_NAMES.items():
        label_to_human[f"{lower_template}:Marker{k}"] = name
    for k, name in UPPER_NAMES.items():
        label_to_human[f"{upper_template}:Marker{k}"] = name

    # Collect (human_name, col_idx) for the first occurrence of each marker
    # label in the name row (each marker appears 3× for X/Y/Z).
    seen: Dict[str, int] = {}  # label -> first col index
    for col_idx, cell in enumerate(name_row):
        if col_idx >= n_cols:
            break
        if cell in label_to_human and cell not in seen:
            seen[cell] = col_idx

    # ── 4. Extract, convert to float, fill gaps, collect output columns ──────
    out_cols: Dict[str, object] = {}
    out_cols["Frame"] = pd.Series(pd.to_numeric(raw[:, 0], errors="coerce"), dtype=float)
    out_cols["Time"]  = pd.Series(pd.to_numeric(raw[:, 1], errors="coerce"), dtype=float)

    missing_report = []
    for label, base_idx in seen.items():
        human_name = label_to_human[label]
        for coord, col_idx in zip(("x", "y", "z"),
                                  (base_idx, base_idx + 1, base_idx + 2)):
            if col_idx >= n_cols:
                print(f"  WARNING: {human_name}_{coord} at col {col_idx} "
                      f"is beyond data width ({n_cols}); filling with NaN.")
                out_cols[f"{human_name}_{coord}"] = np.full(n_rows, np.nan)
                continue

            series = pd.Series(
                pd.to_numeric(raw[:, col_idx], errors="coerce"), dtype=float
            )
            n_missing = int(series.isna().sum())
            if n_missing:
                missing_report.append(
                    f"  {human_name}_{coord}: {n_missing}/{n_rows} frames missing"
                )

            # Fill: use the NEXT available frame first (bfill),
            # then fall back to previous (ffill) for gaps at the end.
            series_filled = series.bfill().ffill()
            out_cols[f"{human_name}_{coord}"] = series_filled.values

    # ── 5. Report gaps ────────────────────────────────────────────────────────
    if missing_report:
        print(f"Gap report for {input_path.name}:")
        for line in missing_report:
            print(line)
    else:
        print("No missing values found.")

    # ── 6. Write output ───────────────────────────────────────────────────────
    out = pd.DataFrame(out_cols)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Wrote {len(out)} frames, {len(out.columns)} columns → {output_path}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the 39-marker set (lower+upper) from a Motive CSV "
            "and fill gaps: use the next available frame first, then the "
            "previous one if no future frame has data."
        )
    )
    parser.add_argument(
        "csv_path",
        help="Path to the Motive CSV (e.g. Take8cleaned.csv).",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help=(
            "Output CSV path. Defaults to <input>_filled39.csv "
            "next to the input file."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    input_path = pathlib.Path(args.csv_path)
    output_path = (
        pathlib.Path(args.out_path)
        if args.out_path
        else input_path.with_name(f"{input_path.stem}_filled39.csv")
    )
    process_file(input_path, output_path)


if __name__ == "__main__":
    main()
