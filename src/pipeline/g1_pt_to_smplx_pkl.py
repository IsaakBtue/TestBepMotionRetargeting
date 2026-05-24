#!/usr/bin/env python3
"""
Convert Unitree G1 .pt motion file → smplx_body_mesh_all_frames.pkl
for use with retarget_direct.sh / mosh_to_robot.py.

Coordinate systems
------------------
G1 world frame (MuJoCo Z-up): X = east, Y = north (actual forward in dataset), Z = up.
NOTE: in this dataset the robot faces world +Y (≈85° yaw from canonical +X).
SMPL (Y-up, T-pose facing +Z): X = person's LEFT, Y = up, Z = forward.

R_pos maps G1 world positions → SMPL positions (robot faces +Y so world +X = robot right):
  smpl_x = -world_x  (G1 east is robot's right; person's left = world west = -X)
  smpl_y =  world_z  (G1 up → SMPL up)
  smpl_z =  world_y  (G1 north = forward → SMPL forward)

R_g2s maps G1 local joint frame (X=fwd, Y=left, Z=up) → SMPL (X=left, Y=up, Z=fwd):
  smpl_x = +g1_y  (G1 left → SMPL left)
  smpl_y =  g1_z  (G1 up   → SMPL up)
  smpl_z =  g1_x  (G1 fwd  → SMPL fwd)
Both frames are right-handed, so det(R_g2s) = +1 (proper rotation).

Joint-angle mapping
-------------------
G1 joint axes (from g1_23.urdf, all in the parent-link local frame):
  pitch → Y-axis
  roll  → X-axis
  yaw   → Z-axis

Each G1 joint rotation is R(axis, angle).  Multi-DOF joints (hip, shoulder)
are composed left-to-right in joint-chain order, then the combined rotation
matrix is brought into the SMPL frame via  R_combined @ R_g1 @ R_combined.T.

SMPL-X body_pose index → joint name (first 21 body joints, offset 1):
  0: L_Hip    1: R_Hip    2: Spine1
  3: L_Knee   4: R_Knee   5: Spine2
  6: L_Ankle  7: R_Ankle  8: Spine3
  9: L_Foot  10: R_Foot  11: Neck
 12: L_Collar 13: R_Collar 14: Head
 15: L_Shoulder 16: R_Shoulder
 17: L_Elbow   18: R_Elbow
 19: L_Wrist   20: R_Wrist

In the flat poses vector (55 joints × 3 = 165):
  poses[j*3 : (j+1)*3]  for j = 0…54
  j=0 → global orient, j=1 → L_Hip, …

Usage (HumanoidDataGeneration env):
  conda activate HumanoidDataGeneration
  python src/pipeline/g1_pt_to_smplx_pkl.py \\
      --input-pt ConvertData/export/motion_dataset/lefthand.pt \\
      --output-dir output/g1_lefthand \\
      [--fps 30]
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# ---------------------------------------------------------------------------
# Coordinate-system transforms
#
# R_g2s: G1 local frame (X=fwd, Y=left, Z=up) → SMPL (X=left, Y=up, Z=fwd).
#   det = +1 (proper rotation, both frames are right-handed).
#   Used as a similarity transform for joint rotations: R_smpl = R_g2s @ R_g1 @ R_g2s.T
#
# R_pos: G1 world positions → SMPL positions.
#   Robot faces world +Y → world +X is robot's right, -X is robot's left.
#   SMPL +X = person's left = world -X; SMPL +Y = up = world +Z; SMPL +Z = fwd = world +Y.
#
# R_canonical: world-yaw rotation of frame 0, removed from R_base before applying R_g2s,
#   so that global_orient ≈ 0 when the robot is upright in its starting pose.
# ---------------------------------------------------------------------------
R_g2s = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
], dtype=np.float64)

R_pos = np.array([
    [-1, 0, 0],
    [0,  0, 1],
    [0,  1, 0],
], dtype=np.float64)


def _chain_to_smplx(angles: list[float], axes: list[np.ndarray]) -> np.ndarray:
    """Compose (axis, angle) pairs in G1 local frame → axis-angle in SMPL frame."""
    rot_g1 = R.identity()
    for axis, angle in zip(axes, angles):
        rot_g1 = rot_g1 * R.from_rotvec(axis * angle)
    mat_smpl = R_g2s @ rot_g1.as_matrix() @ R_g2s.T
    return R.from_matrix(mat_smpl).as_rotvec()


# G1 joint axes (URDF parent-link frame)
_Y = np.array([0, 1, 0], dtype=np.float64)  # pitch
_X = np.array([1, 0, 0], dtype=np.float64)  # roll
_Z = np.array([0, 0, 1], dtype=np.float64)  # yaw


def _convert_frame(
    base_pos: np.ndarray,          # (3,) Z-up G1 world position
    base_quat_xyzw: np.ndarray,    # (4,) xyzw G1 world orientation
    jp: np.ndarray,                # (21,) joint angles (radians)
    R_canonical: np.ndarray,       # (3,3) canonical world yaw to subtract
) -> tuple[np.ndarray, np.ndarray]:
    """Return (poses (165,), trans (3,)) in SMPL Y-up frame."""
    poses = np.zeros(165, dtype=np.float64)

    # --- Root translation: G1 world (x,y,z) → SMPL (x, z, y) ---
    trans = R_pos @ base_pos

    # --- Root orientation: remove canonical world yaw, then apply R_g2s ---
    R_base = R.from_quat(base_quat_xyzw)
    R_rel = R_base.as_matrix() @ R_canonical.T   # residual rotation after yaw removal
    mat_smpl = R_g2s @ R_rel @ R_g2s.T
    poses[0:3] = R.from_matrix(mat_smpl).as_rotvec()   # global orient

    # --- Body joints ---
    # G1 joint_position order (from joint_id.txt):
    #  0  left_hip_pitch  (Y)   6  right_hip_pitch  (Y)  12 waist_yaw  (Z)
    #  1  left_hip_roll   (X)   7  right_hip_roll   (X)  13 l_sho_pitch(Y)
    #  2  left_hip_yaw    (Z)   8  right_hip_yaw    (Z)  14 l_sho_roll (X)
    #  3  left_knee       (Y)   9  right_knee       (Y)  15 l_sho_yaw  (Z)
    #  4  l_ankle_pitch   (Y)  10  r_ankle_pitch    (Y)  16 l_elbow    (Y)
    #  5  l_ankle_roll    (X)  11  r_ankle_roll     (X)  17 r_sho_pitch(Y)
    #                                                     18 r_sho_roll (X)
    #                                                     19 r_sho_yaw  (Z)
    #                                                     20 r_elbow    (Y)

    # SMPL joint j → poses[j*3:(j+1)*3]  (j=1…21 for body joints)

    # L_Hip (j=1)
    poses[3:6] = _chain_to_smplx([jp[0], jp[1], jp[2]], [_Y, _X, _Z])
    # R_Hip (j=2)
    poses[6:9] = _chain_to_smplx([jp[6], jp[7], jp[8]], [_Y, _X, _Z])
    # Spine1 (j=3) — waist yaw only
    poses[9:12] = _chain_to_smplx([jp[12]], [_Z])
    # L_Knee (j=4)
    poses[12:15] = _chain_to_smplx([jp[3]], [_Y])
    # R_Knee (j=5)
    poses[15:18] = _chain_to_smplx([jp[9]], [_Y])
    # L_Ankle (j=7)
    poses[21:24] = _chain_to_smplx([jp[4], jp[5]], [_Y, _X])
    # R_Ankle (j=8)
    poses[24:27] = _chain_to_smplx([jp[10], jp[11]], [_Y, _X])
    # L_Shoulder (j=16)
    poses[48:51] = _chain_to_smplx([jp[13], jp[14], jp[15]], [_Y, _X, _Z])
    # R_Shoulder (j=17)
    poses[51:54] = _chain_to_smplx([jp[17], jp[18], jp[19]], [_Y, _X, _Z])
    # L_Elbow (j=18)
    poses[54:57] = _chain_to_smplx([jp[16]], [_Y])
    # R_Elbow (j=19)
    poses[57:60] = _chain_to_smplx([jp[20]], [_Y])
    # Remaining joints (Spine2/3, Foot, Neck, Collar, Head, Wrist) → zero

    return poses, trans


def convert(
    input_pt: Path,
    output_dir: Path,
    fps: float = 30.0,
) -> Path:
    data = torch.load(input_pt, map_location="cpu")
    joint_pos = data["joint_position"].numpy()   # (T, 21)
    base_pos  = data["base_position"].numpy()    # (T, 3)
    base_pose = data["base_pose"].numpy()        # (T, 4) xyzw

    # Use fps stored in .pt if available
    if "frame_rate" in data:
        fps = float(data["frame_rate"])
    elif "fps" in data:
        fps = float(data["fps"])

    T = joint_pos.shape[0]
    all_poses = np.zeros((T, 165), dtype=np.float64)
    all_trans = np.zeros((T, 3),   dtype=np.float64)

    # Extract canonical world yaw from first frame (the dataset's forward direction).
    # Subtracting this makes global_orient = 0 when the robot is upright and facing forward,
    # matching the SMPL convention expected by mosh_to_robot.py.
    fwd0 = R.from_quat(base_pose[0]).apply([1.0, 0.0, 0.0])  # where G1 local X points
    yaw0 = float(np.arctan2(fwd0[1], fwd0[0]))
    R_canonical = R.from_euler("z", yaw0).as_matrix()

    for t in range(T):
        all_poses[t], all_trans[t] = _convert_frame(
            base_pos[t], base_pose[t], joint_pos[t], R_canonical
        )

    betas = np.zeros((T, 16), dtype=np.float64)   # neutral shape

    output_dir.mkdir(parents=True, exist_ok=True)
    out_pkl = output_dir / "smplx_body_mesh_all_frames.pkl"
    with open(out_pkl, "wb") as f:
        pickle.dump(
            {
                "poses": all_poses,  # (T, 165) float64
                "trans": all_trans,  # (T,   3) float64
                "betas": betas,      # (T,  16) float64
            },
            f,
        )

    print(f"[g1_pt_to_smplx_pkl] Saved {T} frames ({fps:.0f} fps) → {out_pkl}")
    return out_pkl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Unitree G1 .pt motion → smplx_body_mesh_all_frames.pkl"
    )
    parser.add_argument(
        "--input-pt", required=True, type=Path,
        help="Path to G1 .pt motion file (e.g. lefthand.pt)"
    )
    parser.add_argument(
        "--output-dir", default=None, type=Path,
        help="Output directory (default: output/g1_<stem>)"
    )
    parser.add_argument(
        "--fps", type=float, default=30.0,
        help="Capture frame rate in Hz used by mosh_to_robot.py (default: 30)"
    )
    args = parser.parse_args()

    if args.output_dir is None:
        bep = Path(__file__).resolve().parents[2]
        args.output_dir = bep / "output" / f"g1_{args.input_pt.stem}"

    out = convert(args.input_pt, args.output_dir, fps=args.fps)
    print(f"[g1_pt_to_smplx_pkl] Done. PKL: {out}")
    print()
    print("Next step – retarget to Booster T1:")
    print(f"  RETARGET_DIR_OVERRIDE={args.output_dir}/retargeting \\")
    print(f"  ./scripts/retarget_direct.sh {out}")


if __name__ == "__main__":
    main()
