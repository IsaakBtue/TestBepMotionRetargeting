#!/usr/bin/env python3
"""
Convert Unitree G1 .pt motion files to Booster T1 format using GMR IK.

Key corrections applied vs. a naive feed-through:
  1. -90° Z rotation: G1 faces +Y (left=-X), GMR/T1 expects forward=+X (left=+Y).
  2. Z normalization: G1 foot links float at z~0.46 m; subtract per-clip min so
     feet sit at z=0.05 m in GMR frame (consistent with T1 standing on ground).
  3. Custom IK config (g1_to_t1.json): rot offsets = identity [1,0,0,0] so the
     upright constraint keeps T1 upright instead of flipping it; scale = 0.737
     calibrated to match T1 Waist height (0.527 m) from G1 pelvis (0.715 m norm).
  4. Identity quaternions: G1 world-frame orientations are meaningless to GMR;
     positions carry all motion information.

Usage:
  python g1_to_t1_gmr.py                          # converts lefthand.pt
  python g1_to_t1_gmr.py --all                    # converts all 6 clips
  python g1_to_t1_gmr.py --input foo.pt --output bar.pkl
"""
import sys, os, pickle, argparse
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, "/home/isaak/GMR")
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import IK_CONFIG_DICT

# Register custom G1→T1 config so GMR's constructor can find it
_HERE = Path(__file__).parent
IK_CONFIG_DICT["g1_gmr"] = {"booster_t1": str(_HERE / "g1_to_t1.json")}

# ---------------------------------------------------------------------------

DATASET_DIR = Path("/home/isaak/BEP/ConvertData/export/motion_dataset")
OUTPUT_DIR  = Path(_HERE / "output")

# G1 link index → SMPL-X body name (12 bodies used by g1_to_t1.json)
G1_LINK_TO_SMPLX = {
    0:  "pelvis",
    1:  "left_hip",
    2:  "left_knee",
    3:  "left_foot",
    4:  "right_hip",
    5:  "right_knee",
    6:  "right_foot",
    8:  "spine3",
    10: "left_shoulder",
    12: "left_elbow",
    14: "right_shoulder",
    16: "right_elbow",
}

# G1 faces +Y (left=-X, right=+X).  GMR/T1 expects forward=+X (left=+Y, right=-Y).
# Rotate -90° around Z: (x,y,z) -> (y, -x, z)
R_G1_TO_GMR = np.array([[0, 1, 0],
                         [-1, 0, 0],
                         [0, 0, 1]], dtype=np.float64)

FOOT_TARGET_Z = 0.05   # feet sit this high above GMR ground plane


def load_g1_pt(pt_file):
    data = torch.load(pt_file, map_location="cpu")
    return data["link_position"].numpy()   # (F, 17, 3)


def normalize_z(positions):
    """Shift all Z so the lowest foot across all frames sits at FOOT_TARGET_Z."""
    foot_z = np.minimum(positions[:, 3, 2], positions[:, 6, 2])  # left/right foot
    shift = foot_z.min() - FOOT_TARGET_Z
    out = positions.copy()
    out[:, :, 2] -= shift
    return out


def build_frame(pos_frame):
    """One frame: apply rotation, return human_data dict with identity quats."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    return {
        name: (R_G1_TO_GMR @ pos_frame[idx], identity.copy())
        for idx, name in G1_LINK_TO_SMPLX.items()
    }


def convert(pt_file, output_path, fps=30):
    positions = load_g1_pt(pt_file)
    positions = normalize_z(positions)
    n_frames  = positions.shape[0]
    print(f"  {n_frames} frames  →  {output_path}")

    retarget = GMR(src_human="g1_gmr", tgt_robot="booster_t1", verbose=False)

    qpos_list = []
    for i in range(n_frames):
        qpos = retarget.retarget(build_frame(positions[i]))
        qpos_list.append(qpos.copy())
        if (i + 1) % 30 == 0:
            print(f"    frame {i+1}/{n_frames}")

    qpos_arr = np.array(qpos_list)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    motion_data = {
        "fps":            fps,
        "root_pos":       qpos_arr[:, :3],
        "root_rot":       qpos_arr[:, 3:7][:, [1, 2, 3, 0]],  # wxyz→xyzw
        "dof_pos":        qpos_arr[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(output_path, "wb") as f:
        pickle.dump(motion_data, f)
    print(f"  Saved  →  {output_path}")
    return motion_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=str(DATASET_DIR / "lefthand.pt"))
    parser.add_argument("--output", default=str(OUTPUT_DIR  / "lefthand_booster_t1.pkl"))
    parser.add_argument("--fps",    type=int, default=30)
    parser.add_argument("--all",    action="store_true")
    args = parser.parse_args()

    if args.all:
        for pt_file in sorted(DATASET_DIR.glob("*.pt")):
            out = OUTPUT_DIR / f"{pt_file.stem}_booster_t1.pkl"
            print(f"\n{pt_file.name}")
            convert(pt_file, out, fps=args.fps)
    else:
        convert(args.input, args.output, fps=args.fps)
