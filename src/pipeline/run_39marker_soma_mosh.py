#!/usr/bin/env python3
from __future__ import annotations

"""
39-marker pipeline: cleaned CSV → C3D → MoSh++ → SMPL-X mesh.

Labels are already known (column names in the cleaned CSV), so SOMA is skipped.
MoSh++ auto-creates the marker layout from the C3D labels using its internal
all_marker_vids database — all 37 observed labels map to valid SMPL-X vertices.

Coordinate system note:
  Our Motive data is Y-up (Y = height). The person faces +X (front markers
  like CLAV have larger X than back markers like C7). SMPL-X is Y-up and its
  template default has the body facing +Z.

  The rotation_tools.py Ry matrix for angle θ is the standard:
    [cosθ, 0, sinθ; 0, 1, 0; -sinθ, 0, cosθ]
  For Ry(−90°): x_new = −z_old, y_new = y_old, z_new = +x_old.
  This maps the person's +X (front) → +Z in the rotated frame, matching
  SMPL-X's convention. Using +90° instead maps +X → −Z (a 180° front/back
  flip that puts CLAV behind the pelvis and C7 in front of it).

  extra_initial_rigid_adjustment = True is also required. After Procrustes
  (which alone consistently gets confused by walking frames and finds a
  ~180°-flipped degenerate solution), this extra gradient-descent step
  re-optimizes only the root orientation and translation to minimise marker
  residuals directly. Together these two settings give 18–21mm per-marker
  RMSE across all 275 frames — within MoSh++ quality range.

Usage (from repo root, soma env active):
  cd /home/isaak/BEP
  conda activate soma

  python src/pipeline/run_39marker_soma_mosh.py --csv data/Take8cleaned_filled39.csv

  # Skip MoSh++ if already ran, jump straight to visualization:
  python src/pipeline/run_39marker_soma_mosh.py --csv data/Take8cleaned_filled39.csv --skip-mosh

  # Visualize SOMA markers on SMPL-X T-pose after running:
  python docs/Visualization/visualize_soma_markers_on_smplx.py --tpose -i --out_dir output/soma_mosh39
"""

import pickle
import sys
import os
import argparse
import re
from pathlib import Path

import numpy as np

BEP_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOMA_WORK = BEP_ROOT / "soma_work"
DATASET_NAME = "Custom39"
SUBJECT_NAME = "subject1"
SOMA_DATA_ID = "OC_05_G_03_real_000_synt_100"


def log(msg: str) -> None:
    print(f"[run_39marker] {msg}")


def ensure_sys_path() -> None:
    soma_src = BEP_ROOT / "soma" / "src"
    mosh_src = BEP_ROOT / "moshpp" / "src"
    for p in (soma_src, mosh_src):
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _swap_lr(name: str) -> str:
    """Swap Left/Right prefix on bilateral marker names (e.g. LKNE→RKNE, RHEE→LHEE).
    Midline markers (MFWT, MBWT, C7, CLAV, STRN, T10, ARIEL, …) are unaffected
    because they don't start with L or R.
    """
    if name.startswith('L'):
        return 'R' + name[1:]
    if name.startswith('R'):
        return 'L' + name[1:]
    return name


def csv_to_c3d_39(csv_path: Path, out_c3d: Path) -> list[str]:
    """Convert cleaned 39-marker CSV to C3D. Returns the list of labels.

    Marker L/R sides in the Motive export are anatomically mirrored relative to
    MoSh++'s SMPL-X convention, so every bilateral label is swapped here
    (LKNE→RKNE, RHEE→LHEE, etc.) before writing the C3D.
    Midline markers (C7, CLAV, STRN, T10, ARIEL, MFWT, MBWT) are unchanged.
    """
    ensure_sys_path()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from csv_to_c3d import read_markers_csv, write_c3d

    markers, frame_rate, labels = read_markers_csv(str(csv_path))

    # Swap L/R labeling to match MoSh++ / SMPL-X anatomical convention
    swapped = [_swap_lr(lbl) for lbl in labels]
    log(f"L/R label swap: {[(o, s) for o, s in zip(labels, swapped) if o != s]}")

    out_c3d.parent.mkdir(parents=True, exist_ok=True)
    write_c3d(markers, swapped, str(out_c3d), frame_rate, units_m=True)
    log(f"C3D written: {out_c3d} ({markers.shape[0]} frames, {len(swapped)} markers @ {frame_rate} Hz)")
    return swapped


def run_mosh(soma_work: Path, c3d_path: Path, skip: bool, force: bool = False) -> Path | None:
    ensure_sys_path()
    from moshpp.mosh_head import MoSh, run_moshpp_once

    seq_name = c3d_path.stem
    # MoSh normalizes spaces in mocap basenames to underscores when writing outputs.
    # Support both forms so CSV names with spaces still work end-to-end.
    seq_name_normalized = re.sub(r"\s+", "_", seq_name)
    eval_dir = (soma_work / "training_experiments" / "V48_02_SuperSet"
                / SOMA_DATA_ID / "evaluations")
    mosh_dir = eval_dir / "mosh_results_tracklet" / DATASET_NAME / SUBJECT_NAME
    stageii_candidates = [
        mosh_dir / f"{seq_name}_stageii.pkl",
        mosh_dir / f"{seq_name_normalized}_stageii.pkl",
    ]
    stagei_candidates = [
        mosh_dir / f"{seq_name}_stagei.pkl",
        mosh_dir / f"{seq_name_normalized}_stagei.pkl",
    ]

    def _first_existing(candidates: list[Path]) -> Path | None:
        for p in candidates:
            if p.exists():
                return p
        return None

    if force:
        for p in stageii_candidates + stagei_candidates:
            if p.exists():
                p.unlink()
                log(f"Deleted checkpoint: {p}")

    existing_stageii = _first_existing(stageii_candidates)
    if skip and existing_stageii is not None:
        log(f"Skip MoSh++: using existing {existing_stageii}")
        return existing_stageii

    support_base_dir = soma_work / "support_files"
    mosh_dir.mkdir(parents=True, exist_ok=True)

    # Do NOT set dirs.marker_layout — MoSh++ auto-creates it from the C3D
    # labels using its internal all_marker_vids['smplx'] database.
    # All 37 of our observed labels map to valid SMPL-X vertex IDs there.
    mosh_job = {
        "mocap": {
            "fname": str(c3d_path),
            "unit": "mm",
            "subject_name": SUBJECT_NAME,
            "rotate": [0, -90, 0],
            "only_markers": None,
        },
        "surface_model": {"gender": "male"},
        "dirs": {
            "work_base_dir": str(mosh_dir.parent.parent),
            "support_base_dir": str(support_base_dir),
        },
        "moshpp": {
            "perseq_mosh_stagei": True,
            "stagei_frame_picker": {
                "stagei_mocap_fnames": [str(c3d_path)],
                "least_avail_markers": 0.9,
                "type": "random_strict",
            },
        },
        "opt_settings": {
            "extra_initial_rigid_adjustment": True,
        },
    }

    cur_mosh_cfg = MoSh.prepare_cfg(mosh_job)
    run_moshpp_once(cur_mosh_cfg)

    existing_stageii = _first_existing(stageii_candidates)
    if existing_stageii is not None:
        return existing_stageii

    log(f"MoSh++ finished but stageii pkl not found; checked: {stageii_candidates}")
    return None


def smooth_mosh_output(stageii_pkl: Path) -> None:
    """
    Post-process MoSH stageii.pkl to fix wild joint angles caused by axis-angle
    gimbal lock and optimization instability.

    Converts each joint's axis-angle to quaternions (no ambiguity), ensures sign
    continuity across frames, applies Gaussian smoothing, then converts back.
    Overwrites the pkl in place.
    """
    try:
        from scipy.ndimage import gaussian_filter1d
        from scipy.spatial.transform import Rotation as R
    except ImportError:
        log("scipy not available — skipping pose smoothing.")
        return

    with open(stageii_pkl, "rb") as f:
        data = pickle.load(f)

    fp = np.array(data["fullpose"], dtype=np.float64)  # (T, 165)
    T, D = fp.shape
    n_joints = D // 3
    smoothed = fp.copy()

    # Sigma in frames — 5 frames = 50 ms at 100 fps; large enough to kill noise/flips
    sigma = 5.0

    for j in range(n_joints):
        aa = fp[:, j * 3:(j + 1) * 3]  # (T, 3) axis-angle

        # Skip joints with zero motion — nothing to smooth
        if aa.std() < 1e-6:
            continue

        # axis-angle → quaternion (handles magnitude correctly)
        rots = R.from_rotvec(aa)
        quats = rots.as_quat()  # (T, 4) xyzw

        # Ensure quaternion sign continuity: if consecutive quaternions are
        # on opposite hemispheres, flip to keep them consistent.
        for t in range(1, T):
            if np.dot(quats[t], quats[t - 1]) < 0:
                quats[t] = -quats[t]

        # Gaussian-smooth each quaternion component independently
        sq = np.stack([
            gaussian_filter1d(quats[:, c], sigma=sigma) for c in range(4)
        ], axis=1)

        # Re-normalise to unit quaternions
        norms = np.linalg.norm(sq, axis=1, keepdims=True)
        sq /= np.maximum(norms, 1e-8)

        # quaternion → axis-angle
        smoothed[:, j * 3:(j + 1) * 3] = R.from_quat(sq).as_rotvec()

    data["fullpose"] = smoothed
    # Smooth global translation too (removes jitter without changing motion)
    trans = np.array(data["trans"], dtype=np.float64)
    data["trans"] = np.stack([
        gaussian_filter1d(trans[:, c], sigma=sigma) for c in range(3)
    ], axis=1)

    with open(stageii_pkl, "wb") as f:
        pickle.dump(data, f)
    log(f"Smoothed poses written to {stageii_pkl}")


def run_visualization(stageii_pkl: Path, output_dir: Path) -> None:
    import subprocess
    vis_script = BEP_ROOT / "src" / "pipeline" / "visualize_smplx_body.py"
    if not vis_script.exists():
        log(f"Visualization script not found: {vis_script}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["STAGEII"] = str(stageii_pkl)
    env["OUT_DIR"] = str(output_dir)
    support = BEP_ROOT / "moshpp" / "support_data"
    if support.exists():
        env["SUPPORT_BASE_DIR"] = str(support)
    log(f"Running visualization → {output_dir}")
    subprocess.run([sys.executable, str(vis_script)], env=env,
                   cwd=str(BEP_ROOT / "src" / "pipeline"), check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="39-marker cleaned CSV → C3D → MoSh++ → SMPL-X visualization"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to cleaned 39-marker CSV (e.g. Take8cleaned_filled39.csv)"
    )
    parser.add_argument(
        "--soma-work", default=None,
        help=f"SOMA work root (default: {DEFAULT_SOMA_WORK})"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Visualization output dir (default: output/soma_mosh39)"
    )
    parser.add_argument(
        "--skip-mosh", action="store_true",
        help="Skip MoSh++ (use existing stageii pkl)"
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip visualization step"
    )
    parser.add_argument(
        "--skip-smooth", action="store_true",
        help="Skip quaternion Gaussian pose smoothing (keep raw MoSh++ poses)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing stagei/stageii checkpoints and re-run MoSh++ from scratch"
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    soma_work = Path(args.soma_work) if args.soma_work else DEFAULT_SOMA_WORK
    output_dir = (Path(args.output_dir) if args.output_dir
                  else BEP_ROOT / "output" / "soma_mosh39")

    if not csv_path.exists():
        log(f"CSV not found: {csv_path}")
        sys.exit(1)

    seq_name = csv_path.stem
    log(f"Sequence: {seq_name}")

    # 1) CSV → C3D (labels extracted from column names)
    eval_dir = (soma_work / "training_experiments" / "V48_02_SuperSet"
                / SOMA_DATA_ID / "evaluations")
    c3d_dir = eval_dir / "soma_labeled_mocap_tracklet" / DATASET_NAME / SUBJECT_NAME
    c3d_dir.mkdir(parents=True, exist_ok=True)
    c3d_path = c3d_dir / f"{seq_name}.c3d"

    settings = c3d_dir / "settings.json"
    if not settings.exists():
        settings.write_text('{"gender": "male"}\n')

    csv_to_c3d_39(csv_path, c3d_path)

    # 2) MoSh++
    stageii_pkl = run_mosh(soma_work, c3d_path, args.skip_mosh, force=args.force)
    if stageii_pkl is None:
        sys.exit(1)

    # 2b) Post-process: smooth poses to fix gimbal-lock / optimizer instability
    if args.skip_smooth:
        log("Skip smoothing: keeping raw MoSh++ poses (axis-angle).")
    else:
        smooth_mosh_output(stageii_pkl)

    # 3) Visualization
    if not args.skip_viz:
        run_visualization(stageii_pkl, output_dir)
    else:
        log(f"Skip viz. To visualize:\n"
            f"  STAGEII={stageii_pkl} OUT_DIR={output_dir} "
            f"python src/pipeline/visualize_smplx_body.py")

    log("Done.")


if __name__ == "__main__":
    main()
