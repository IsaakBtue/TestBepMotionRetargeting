#!/usr/bin/env python3
"""
G1 .pt → MoSh++ → smplx_body_mesh_all_frames.pkl

Converts Unitree G1 link positions (17 links, world Z-up) into a C3D file
with SOMA-compatible marker names, then runs MoSh++ to fit SMPL-X.

This REPLACES the broken analytical g1_pt_to_smplx_pkl.py approach:
  - Manual joint angle → SMPL axis mapping fails because the G1's URDF
    shoulder pre-rotations (±16°) mean G1 roll (+74°) puts the arm nearly
    horizontal, but our R_g2s similarity transform maps it to a large upward
    arm rotation in SMPL space (diagnostic: Y error 43cm at elbow).
  - Using link_positions as virtual markers and letting MoSh++ optimize the
    SMPL-X fit is more robust and frame-correct.

Coordinate conversion: G1 world (X=east, Y=north/forward, Z=up) → MoSh input
  (X=forward, Y=up, Z=right): apply [[0,1,0],[0,0,1],[1,0,0]].
MoSh then internally applies Ry(−90°) to orient the body to SMPL (+Z forward).

Usage (soma env):
    conda activate soma
    python src/pipeline/g1_pt_to_mosh.py \\
        --pt ConvertData/export/motion_dataset/lefthand.pt \\
        --output-dir output/g1_lefthand \\
        [--fps 30]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import pickle
from pathlib import Path

import numpy as np
import torch

BEP_ROOT = Path(__file__).resolve().parents[2]

# ── Coordinate transform G1 world (Z-up) → MoSh input (Y-up, facing +X) ────
# G1: X=east(right), Y=north(forward), Z=up
# MoSh input: X=forward, Y=up, Z=right
# Robot faces +Y_world, robot's right = +X_world
R_G1_TO_MOSH = np.array([
    [0, 1, 0],   # mosh_X = g1_Y (forward)
    [0, 0, 1],   # mosh_Y = g1_Z (up)
    [1, 0, 0],   # mosh_Z = g1_X (right)
], dtype=np.float64)

# G1 link index → SOMA marker name (SMPL anatomical convention: L*=person left, R*=right)
# Mapping determined from link Z-heights and anatomical positions.
# All 17 G1 links are used:
#   links 0–9, 11–13, 15–16 → original 15 markers
#   links 10, 14             → LBSH/RBSH (left/right back shoulder, added after vis_raw
#                              confirmed only 15 markers were being exported; see MAPPING_ANALYSIS
#                              section 12 for why these two were skipped initially)
G1_LINK_TO_SOMA: dict[int, str] = {
    0:  "MFWT",   # pelvis (midline front waist)
    1:  "LTHI",   # left_hip_yaw_link        → left thigh
    2:  "LKNE",   # left_knee_link           → left knee
    3:  "LANK",   # left_ankle_roll_link     → left ankle
    4:  "RTHI",   # right_hip_yaw_link       → right thigh
    5:  "RKNE",   # right_knee_link          → right knee
    6:  "RANK",   # right_ankle_roll_link    → right ankle
    7:  "ARIEL",  # head link                → top of head
    8:  "STRN",   # torso/waist_yaw_link     → sternum
    9:  "LFSH",   # left_sho_pitch_link      → left front shoulder
    10: "LBSH",   # left_sho_roll_link       → left back shoulder (lateral shoulder joint)
    11: "LUPA",   # left_sho_yaw_link        → left upper arm
    12: "LELB",   # left_elbow_link          → left elbow
    13: "RFSH",   # right_sho_pitch_link     → right front shoulder
    14: "RBSH",   # right_sho_roll_link      → right back shoulder (lateral shoulder joint)
    15: "RUPA",   # right_sho_yaw_link       → right upper arm
    16: "RELB",   # right_elbow_link         → right elbow
}

DATASET_NAME = "Custom39"
SUBJECT_NAME = "subject1"
SOMA_DATA_ID = "OC_05_G_03_real_000_synt_100"


def _ensure_sys_path() -> None:
    for p in [
        str(BEP_ROOT / "moshpp" / "src"),
        str(BEP_ROOT / "soma" / "src"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)


def pt_to_c3d(pt_path: Path, out_c3d: Path, fps: float = 30.0) -> None:
    """Convert G1 link_position data → C3D with SOMA marker labels."""
    _ensure_sys_path()
    sys.path.insert(0, str(Path(__file__).parent))
    from csv_to_c3d import write_c3d

    d = torch.load(pt_path, map_location="cpu")
    link_pos = d["link_position"].numpy()   # (T, 17, 3) in G1 world frame (meters)
    T = link_pos.shape[0]

    if "frame_rate" in d:
        fps = float(d["frame_rate"])
    elif "fps" in d:
        fps = float(d["fps"])

    link_indices = sorted(G1_LINK_TO_SOMA.keys())
    labels = [G1_LINK_TO_SOMA[i] for i in link_indices]
    N = len(link_indices)

    markers = np.zeros((T, N, 3), dtype=np.float64)
    for j, link_idx in enumerate(link_indices):
        # G1 world positions (meters) → MoSh input frame (meters)
        markers[:, j, :] = (R_G1_TO_MOSH @ link_pos[:, link_idx, :].T).T

    out_c3d.parent.mkdir(parents=True, exist_ok=True)
    write_c3d(markers, labels, str(out_c3d), fps, units_m=True)
    print(f"[g1_pt_to_mosh] C3D written: {out_c3d}  ({T} frames, {N} markers @ {fps:.0f} Hz)")

    # Also write a vis_raw-compatible CSV (Frame, Time, NAME_x, NAME_y, NAME_z, ...)
    import pandas as pd
    csv_path = out_c3d.with_suffix(".csv")
    row: dict = {"Frame": list(range(T)), "Time": [i / fps for i in range(T)]}
    for j, name in enumerate(labels):
        row[f"{name}_x"] = markers[:, j, 0]
        row[f"{name}_y"] = markers[:, j, 1]
        row[f"{name}_z"] = markers[:, j, 2]
    pd.DataFrame(row).to_csv(csv_path, index=False)
    print(f"[g1_pt_to_mosh] CSV written:  {csv_path}")


def _find_stageii(search_root: Path, seq_name: str) -> Path | None:
    """Recursively find any stageii pkl for this sequence under search_root."""
    matches = sorted(search_root.rglob(f"{seq_name}*stageii*.pkl"))
    return matches[0] if matches else None


def run_mosh(soma_work: Path, c3d_path: Path, force: bool = False) -> Path | None:
    """Run MoSh++ on the G1 C3D. Returns path to stageii.pkl."""
    _ensure_sys_path()
    from moshpp.mosh_head import MoSh, run_moshpp_once

    seq_name  = re.sub(r"\s+", "_", c3d_path.stem)
    eval_dir  = (soma_work / "training_experiments" / "V48_02_SuperSet"
                 / SOMA_DATA_ID / "evaluations")
    tracklet_dir = eval_dir / "mosh_results_tracklet"
    # MoSh mirrors the C3D path inside tracklet_dir; use recursive glob to find it.
    work_base_dir = tracklet_dir

    if force:
        existing = _find_stageii(tracklet_dir, seq_name)
        if existing:
            for p in existing.parent.glob(f"{seq_name}*"):
                p.unlink()
            print(f"[g1_pt_to_mosh] Deleted prior MoSh results for {seq_name}")
        # Also delete the smplx.json marker layout next to the C3D so MoSh++ rebuilds it
        # for the new marker set rather than loading the stale one from the previous run.
        for layout in c3d_path.parent.glob(f"{seq_name}*.json"):
            layout.unlink()
            print(f"[g1_pt_to_mosh] Deleted stale marker layout: {layout.name}")

    existing = _find_stageii(tracklet_dir, seq_name)
    if existing:
        print(f"[g1_pt_to_mosh] MoSh result exists, skipping: {existing}")
        return existing

    tracklet_dir.mkdir(parents=True, exist_ok=True)
    support_base_dir = soma_work / "support_files"

    mosh_job = {
        "mocap": {
            "fname": str(c3d_path),
            "unit": "mm",
            "subject_name": SUBJECT_NAME,
            "rotate": [0, -90, 0],
            "only_markers": None,
        },
        "surface_model": {"gender": "neutral"},
        "dirs": {
            "work_base_dir": str(work_base_dir),
            "support_base_dir": str(support_base_dir),
        },
        "moshpp": {
            "perseq_mosh_stagei": True,
            "stagei_frame_picker": {
                "stagei_mocap_fnames": [str(c3d_path)],
                "least_avail_markers": 0.7,
                "type": "random_strict",
            },
        },
        "opt_settings": {
            "extra_initial_rigid_adjustment": True,
        },
    }

    cfg = MoSh.prepare_cfg(mosh_job)
    run_moshpp_once(cfg)

    stageii = _find_stageii(tracklet_dir, seq_name)
    if stageii:
        print(f"[g1_pt_to_mosh] MoSh++ done → {stageii}")
        return stageii

    print(f"[g1_pt_to_mosh] WARNING: MoSh++ finished but stageii not found under {tracklet_dir}")
    return None


def stageii_to_smplx_pkl(stageii_pkl: Path, output_pkl: Path) -> None:
    """Convert MoSh stageii.pkl → smplx_body_mesh_all_frames.pkl format."""
    with open(stageii_pkl, "rb") as f:
        data = pickle.load(f)

    # MoSh stageii stores: poses (T,165), trans (T,3), betas (T,16 or 300)
    poses = np.array(data["fullpose"])   # (T, 165) or similar
    trans = np.array(data["trans"])      # (T, 3)

    # betas may be per-frame or single; normalise to (T, 16)
    betas_raw = np.array(data.get("betas", np.zeros((poses.shape[0], 16))))
    if betas_raw.ndim == 1:
        betas_raw = np.tile(betas_raw, (poses.shape[0], 1))
    # Truncate or pad to 16
    T = poses.shape[0]
    betas = np.zeros((T, 16), dtype=np.float64)
    n = min(betas_raw.shape[1], 16)
    betas[:, :n] = betas_raw[:, :n]

    # Ensure float64
    poses = poses.astype(np.float64)
    trans = trans.astype(np.float64)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump({"poses": poses, "trans": trans, "betas": betas}, f)
    print(f"[g1_pt_to_mosh] {T} frames → {output_pkl}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt",         required=True, type=Path)
    parser.add_argument("--output-dir", default=None,  type=Path)
    parser.add_argument("--fps",        type=float, default=30.0)
    parser.add_argument("--force",      action="store_true",
                        help="Delete prior MoSh results and re-run")
    parser.add_argument("--soma-work",  default=None, type=Path,
                        help="soma_work directory (default: BEP/soma_work)")
    args = parser.parse_args()

    soma_work  = args.soma_work or (BEP_ROOT / "soma_work")
    output_dir = args.output_dir or (BEP_ROOT / "output" / f"g1_{args.pt.stem}")
    output_dir.mkdir(parents=True, exist_ok=True)

    stem  = args.pt.stem
    c3d   = output_dir / f"{stem}_g1markers.c3d"
    out_pkl = output_dir / "smplx_body_mesh_all_frames.pkl"

    # ── Step 1: G1 link positions → C3D ──────────────────────────────────────
    print("\n[g1_pt_to_mosh] Step 1: .pt → C3D")
    pt_to_c3d(args.pt, c3d, fps=args.fps)

    # ── Step 2: Run MoSh++ ───────────────────────────────────────────────────
    print("\n[g1_pt_to_mosh] Step 2: MoSh++")
    stageii = run_mosh(soma_work, c3d, force=args.force)
    if stageii is None:
        print("[g1_pt_to_mosh] MoSh++ failed. Check logs above.")
        sys.exit(1)

    # ── Step 3: Convert stageii.pkl → our pkl format ─────────────────────────
    print("\n[g1_pt_to_mosh] Step 3: stageii → smplx_body_mesh_all_frames.pkl")
    stageii_to_smplx_pkl(stageii, out_pkl)

    print(f"\n[g1_pt_to_mosh] Done → {out_pkl}")
    print("Next: conda activate soma && python src/pipeline/g1_smplx_to_mesh_pkl.py "
          f"--input {out_pkl} --output {output_dir / 'smplx_body_mesh_vertices.pkl'}")


if __name__ == "__main__":
    main()
