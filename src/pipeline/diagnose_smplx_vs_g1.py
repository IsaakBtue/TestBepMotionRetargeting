#!/usr/bin/env python3
"""
Diagnostic: compare SMPL-X joint positions (from our pkl) against G1 link_position
ground truth (from lefthand.pt), to identify axis mapping errors in g1_pt_to_smplx_pkl.py.

Run in soma conda env:
    conda activate soma
    python src/pipeline/diagnose_smplx_vs_g1.py

Coordinate conventions:
  G1 world: Z-up, X=east, Y=north (forward for this dataset)
  SMPL:     Y-up, X=person left, Z=person forward

R_pos: G1 world pos → SMPL pos
  smpl_x = -world_x
  smpl_y =  world_z
  smpl_z =  world_y
"""
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R

BEP = Path(__file__).resolve().parents[2]
PKL_PATH  = BEP / "output/g1_lefthand/smplx_body_mesh_all_frames.pkl"
PT_PATH   = BEP / "ConvertData/export/motion_dataset/lefthand.pt"
MODEL_DIR = BEP / "body_models"

# G1 world → SMPL position transform
R_pos = np.array([
    [-1, 0, 0],
    [ 0, 0, 1],
    [ 0, 1, 0],
], dtype=np.float64)

# SMPL-X body joint names (index 0..21)
SMPL_JOINT_NAMES = [
    "pelvis",
    "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist",
]

# G1 link_position index → link name (17 links from IsaacLab export)
# Determined from URDF and Z-height analysis
G1_LINK_NAMES = [
    "pelvis",                  # 0
    "left_hip_yaw_link",       # 1  (distal of left hip chain)
    "left_knee_link",          # 2
    "left_ankle_roll_link",    # 3  (foot end-effector)
    "right_hip_yaw_link",      # 4
    "right_knee_link",         # 5
    "right_ankle_roll_link",   # 6
    "torso_link",              # 7
    "waist_yaw_link",          # 8
    "left_shoulder_pitch_link",# 9
    "left_shoulder_roll_link", # 10
    "left_shoulder_yaw_link",  # 11
    "left_elbow_link",         # 12
    "right_shoulder_pitch_link",# 13
    "right_shoulder_roll_link",# 14
    "right_shoulder_yaw_link", # 15
    "right_elbow_link",        # 16
]

# Pairs: (SMPL joint index, G1 link index, label)
COMPARE_PAIRS = [
    (1,  1,  "L_Hip   ↔ left_hip_yaw_link"),
    (2,  4,  "R_Hip   ↔ right_hip_yaw_link"),
    (4,  2,  "L_Knee  ↔ left_knee_link"),
    (5,  5,  "R_Knee  ↔ right_knee_link"),
    (7,  3,  "L_Ankle ↔ left_ankle_roll_link"),
    (8,  6,  "R_Ankle ↔ right_ankle_roll_link"),
    (16, 11, "L_Shoul ↔ left_shoulder_yaw_link"),
    (17, 15, "R_Shoul ↔ right_shoulder_yaw_link"),
    (18, 12, "L_Elbow ↔ left_elbow_link"),
    (19, 16, "R_Elbow ↔ right_elbow_link"),
]


def load_smplx_joints(pkl_path: Path, model_dir: Path):
    """Run SMPL-X forward pass on our poses → joint positions (T, 22, 3) Y-up."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    poses = data["poses"]   # (T, 165)
    trans = data["trans"]   # (T, 3)
    betas = data["betas"]   # (T, 16)
    T = poses.shape[0]

    body_model = smplx.create(
        str(model_dir), model_type="smplx", gender="neutral",
        num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T,
    )

    with torch.no_grad():
        out = body_model(
            global_orient=torch.tensor(poses[:, 0:3],  dtype=torch.float32),
            body_pose    =torch.tensor(poses[:, 3:66], dtype=torch.float32),
            betas        =torch.tensor(betas,          dtype=torch.float32),
            transl       =torch.tensor(trans,          dtype=torch.float32),
        )

    # joints: (T, J, 3)  — first 22 are the body joints
    joints = out.joints.numpy()[:, :22, :]
    return joints   # Y-up SMPL world frame


def load_g1_links(pt_path: Path):
    """Load G1 link positions (T, 17, 3) Z-up world frame."""
    import sys
    sys.path.insert(0, str(BEP))  # for torch
    d = torch.load(pt_path, map_location="cpu")
    link_pos = d["link_position"].numpy()   # (T, 17, 3)
    base_pos = d["base_position"].numpy()   # (T, 3)
    return link_pos, base_pos


def g1_to_smpl_pos(pos_world: np.ndarray) -> np.ndarray:
    """Convert G1 world position(s) → SMPL frame. Supports (..., 3)."""
    return (R_pos @ pos_world.T).T


def print_frame_comparison(t: int, smplx_joints, g1_links, g1_base):
    """Print absolute positions and errors for frame t."""
    print(f"\n{'='*70}")
    print(f" Frame {t}")
    print(f"{'='*70}")

    # G1 root in SMPL frame
    g1_root_smpl = g1_to_smpl_pos(g1_links[t, 0])   # pelvis link
    smplx_root   = smplx_joints[t, 0]                 # SMPL pelvis

    print(f"  Root — SMPL: {smplx_root}  G1→SMPL: {g1_root_smpl}")

    print(f"\n  {'Joint':<30} {'SMPL pos':>36} {'G1→SMPL pos':>36} {'error (m)':>10}")
    print(f"  {'-'*30} {'-'*36} {'-'*36} {'-'*10}")

    for smpl_idx, g1_idx, label in COMPARE_PAIRS:
        sp = smplx_joints[t, smpl_idx]
        gp = g1_to_smpl_pos(g1_links[t, g1_idx])
        err = np.linalg.norm(sp - gp)
        print(f"  {label:<30} [{sp[0]:+6.3f} {sp[1]:+6.3f} {sp[2]:+6.3f}]   "
              f"[{gp[0]:+6.3f} {gp[1]:+6.3f} {gp[2]:+6.3f}]  {err:>8.3f}")


def print_relative_comparison(t: int, smplx_joints, g1_links):
    """Compare joint positions relative to root (root-subtracted)."""
    print(f"\n--- Frame {t}: positions relative to root ---")

    smplx_root = smplx_joints[t, 0]
    g1_root_smpl = g1_to_smpl_pos(g1_links[t, 0])

    print(f"  {'Joint':<30} {'SMPL (rel)':>36} {'G1→SMPL (rel)':>36} {'err':>8}")
    print(f"  {'-'*30} {'-'*36} {'-'*36} {'-'*8}")

    for smpl_idx, g1_idx, label in COMPARE_PAIRS:
        sp_rel = smplx_joints[t, smpl_idx] - smplx_root
        gp_rel = g1_to_smpl_pos(g1_links[t, g1_idx]) - g1_root_smpl
        err = np.linalg.norm(sp_rel - gp_rel)
        print(f"  {label:<30} [{sp_rel[0]:+6.3f} {sp_rel[1]:+6.3f} {sp_rel[2]:+6.3f}]   "
              f"[{gp_rel[0]:+6.3f} {gp_rel[1]:+6.3f} {gp_rel[2]:+6.3f}]  {err:>6.3f}")


def print_axis_analysis(t: int, smplx_joints, g1_links):
    """Per-axis breakdown to spot which axis is wrong."""
    print(f"\n--- Frame {t}: per-axis signed error (SMPL − G1→SMPL) relative to root ---")
    print(f"  positive error → our SMPL value is TOO LARGE on that axis")

    smplx_root   = smplx_joints[t, 0]
    g1_root_smpl = g1_to_smpl_pos(g1_links[t, 0])

    print(f"  {'Joint':<30} {'err_X':>8} {'err_Y':>8} {'err_Z':>8}  note")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")

    for smpl_idx, g1_idx, label in COMPARE_PAIRS:
        sp_rel = smplx_joints[t, smpl_idx] - smplx_root
        gp_rel = g1_to_smpl_pos(g1_links[t, g1_idx]) - g1_root_smpl
        diff   = sp_rel - gp_rel
        # note: X=left, Y=up, Z=fwd in SMPL
        note = ""
        if abs(diff[0]) > 0.05: note += f"X{'↑' if diff[0]>0 else '↓'} "
        if abs(diff[1]) > 0.05: note += f"Y{'↑' if diff[1]>0 else '↓'} "
        if abs(diff[2]) > 0.05: note += f"Z{'↑' if diff[2]>0 else '↓'} "
        print(f"  {label:<30} {diff[0]:>+8.3f} {diff[1]:>+8.3f} {diff[2]:>+8.3f}  {note}")


def main():
    print("Loading SMPL-X joint positions...")
    smplx_joints = load_smplx_joints(PKL_PATH, MODEL_DIR)
    print(f"  shape: {smplx_joints.shape}  (T, 22, 3)")

    print("Loading G1 link positions...")
    g1_links, g1_base = load_g1_links(PT_PATH)
    print(f"  link_position shape: {g1_links.shape}")

    T = smplx_joints.shape[0]
    print(f"  Frames: {T}")

    # Check a few key frames
    frames_to_check = [0, T//4, T//2, 3*T//4, T-1]

    for t in frames_to_check:
        print_relative_comparison(t, smplx_joints, g1_links)

    # Detailed axis analysis for most informative frame (mid-motion)
    print("\n" + "="*70)
    print(" AXIS ANALYSIS (mid-motion frame)")
    print("="*70)
    print_axis_analysis(T//2, smplx_joints, g1_links)

    # Find worst frame per joint
    print("\n" + "="*70)
    print(" MEAN ERROR PER JOINT (root-relative) across all frames")
    print("="*70)
    smplx_root_all = smplx_joints[:, 0:1, :]  # (T,1,3)
    g1_root_smpl_all = np.stack([g1_to_smpl_pos(g1_links[t, 0]) for t in range(T)])[:, np.newaxis, :]

    smplx_rel = smplx_joints - smplx_root_all
    g1_rel = np.stack([g1_to_smpl_pos(g1_links[t]) for t in range(T)]) - g1_root_smpl_all[:, 0:1, :]

    print(f"  {'Joint':<30} {'mean_err':>10} {'max_err':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for smpl_idx, g1_idx, label in COMPARE_PAIRS:
        diffs = smplx_rel[:, smpl_idx, :] - g1_rel[:, g1_idx, :]
        errs  = np.linalg.norm(diffs, axis=1)
        print(f"  {label:<30} {errs.mean():>10.3f} {errs.max():>10.3f}")

    # Also show what swap/mirror would fix things
    print("\n" + "="*70)
    print(" AXIS SWAP DIAGNOSIS")
    print(" Checking if swapping SMPL X↔Z or negating any axis reduces error")
    print("="*70)
    t = T // 2
    for smpl_idx, g1_idx, label in COMPARE_PAIRS[:4]:  # legs only
        sp = smplx_rel[t, smpl_idx]
        gp = g1_rel[t, g1_idx]
        print(f"\n  {label}:")
        print(f"    SMPL (rel):    X={sp[0]:+.3f}  Y={sp[1]:+.3f}  Z={sp[2]:+.3f}")
        print(f"    G1→SMPL (rel): X={gp[0]:+.3f}  Y={gp[1]:+.3f}  Z={gp[2]:+.3f}")
        print(f"    As-is error:   {np.linalg.norm(sp-gp):.3f}")
        # Try X↔Z swap
        sp_xz = np.array([sp[2], sp[1], sp[0]])
        print(f"    X↔Z swap err:  {np.linalg.norm(sp_xz-gp):.3f}")
        # Try negate X
        sp_nx = np.array([-sp[0], sp[1], sp[2]])
        print(f"    neg-X err:     {np.linalg.norm(sp_nx-gp):.3f}")
        # Try negate Z
        sp_nz = np.array([sp[0], sp[1], -sp[2]])
        print(f"    neg-Z err:     {np.linalg.norm(sp_nz-gp):.3f}")
        # Try negate Y
        sp_ny = np.array([sp[0], -sp[1], sp[2]])
        print(f"    neg-Y err:     {np.linalg.norm(sp_ny-gp):.3f}")


if __name__ == "__main__":
    main()
