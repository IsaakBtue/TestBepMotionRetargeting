#!/usr/bin/env python3
"""
Check actual G1 joint angles at key frames and compare SMPL output
for shoulder/elbow specifically.
Run: conda activate soma && python src/pipeline/diagnose_joints.py
"""
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch
from scipy.spatial.transform import Rotation as R

BEP       = Path(__file__).resolve().parents[2]
PT_PATH   = BEP / "ConvertData/export/motion_dataset/lefthand.pt"
PKL_PATH  = BEP / "output/g1_lefthand/smplx_body_mesh_all_frames.pkl"
MODEL_DIR = BEP / "body_models"

R_pos = np.array([[-1,0,0],[0,0,1],[0,1,0]], dtype=np.float64)
R_g2s = np.array([[0,1,0],[0,0,1],[1,0,0]], dtype=np.float64)

_Y = np.array([0,1,0], dtype=np.float64)
_X = np.array([1,0,0], dtype=np.float64)
_Z = np.array([0,0,1], dtype=np.float64)


def chain_to_smplx(angles, axes):
    rot_g1 = R.identity()
    for axis, angle in zip(axes, angles):
        rot_g1 = rot_g1 * R.from_rotvec(axis * angle)
    mat_smpl = R_g2s @ rot_g1.as_matrix() @ R_g2s.T
    return R.from_matrix(mat_smpl).as_rotvec()


def g1_link_smpl_rel(link_pos_world, root_pos_world):
    return R_pos @ (link_pos_world - root_pos_world)


def forward_smplx(poses_np, betas_np, trans_np):
    T = poses_np.shape[0]
    model = smplx.create(str(MODEL_DIR), model_type="smplx", gender="neutral",
                         num_betas=16, use_pca=False, flat_hand_mean=True, batch_size=T)
    with torch.no_grad():
        out = model(
            global_orient=torch.tensor(poses_np[:,0:3], dtype=torch.float32),
            body_pose    =torch.tensor(poses_np[:,3:66], dtype=torch.float32),
            betas        =torch.tensor(betas_np,         dtype=torch.float32),
            transl       =torch.tensor(trans_np,         dtype=torch.float32),
        )
    return out.joints.numpy()[:, :22, :]


def main():
    d = torch.load(PT_PATH, map_location="cpu")
    jp = d["joint_position"].numpy()   # (T, 21)
    lp = d["link_position"].numpy()    # (T, 17, 3)
    bp = d["base_position"].numpy()    # (T, 3)

    with open(PKL_PATH, "rb") as f:
        pkl = pickle.load(f)

    smplx_joints = forward_smplx(pkl["poses"], pkl["betas"], pkl["trans"])

    T = jp.shape[0]
    # Peak left-arm frame (frame with largest L_shoulder angle magnitude)
    l_sho_mag = np.linalg.norm(jp[:, 13:16], axis=1)
    t_peak = int(np.argmax(l_sho_mag))

    for t in [0, t_peak]:
        print(f"\n{'='*60}")
        print(f" Frame {t} (peak shoulder frame = {t_peak})")
        print(f"{'='*60}")

        print(f"\n G1 LEFT ARM joint angles (deg):")
        print(f"   L_sho_pitch (Y): {np.degrees(jp[t,13]):+8.2f}")
        print(f"   L_sho_roll  (X): {np.degrees(jp[t,14]):+8.2f}")
        print(f"   L_sho_yaw   (Z): {np.degrees(jp[t,15]):+8.2f}")
        print(f"   L_elbow     (Y): {np.degrees(jp[t,16]):+8.2f}")

        print(f"\n G1 RIGHT ARM joint angles (deg):")
        print(f"   R_sho_pitch (Y): {np.degrees(jp[t,17]):+8.2f}")
        print(f"   R_sho_roll  (X): {np.degrees(jp[t,18]):+8.2f}")
        print(f"   R_sho_yaw   (Z): {np.degrees(jp[t,19]):+8.2f}")
        print(f"   R_elbow     (Y): {np.degrees(jp[t,20]):+8.2f}")

        # G1 link positions converted to SMPL relative to root
        root_smpl = R_pos @ bp[t]
        smplx_root = smplx_joints[t, 0]

        print(f"\n G1 link → SMPL (relative to G1 pelvis, converted with R_pos):")
        for idx, name in [(11,"L_shoulder_yaw_link"), (12,"L_elbow_link"),
                          (15,"R_shoulder_yaw_link"), (16,"R_elbow_link")]:
            rel = g1_link_smpl_rel(lp[t, idx], bp[t])
            print(f"   link[{idx:2d}] {name:<22}: X={rel[0]:+.3f}  Y={rel[1]:+.3f}  Z={rel[2]:+.3f}")

        print(f"\n SMPL-X joints (relative to SMPL pelvis):")
        for jidx, name in [(16,"L_Shoulder"), (18,"L_Elbow"),
                           (17,"R_Shoulder"), (19,"R_Elbow")]:
            rel = smplx_joints[t, jidx] - smplx_root
            print(f"   joint[{jidx:2d}] {name:<12}: X={rel[0]:+.3f}  Y={rel[1]:+.3f}  Z={rel[2]:+.3f}")

        # What SMPL shoulder rotation we computed from G1:
        l_sho_rv = chain_to_smplx([jp[t,13], jp[t,14], jp[t,15]], [_Y, _X, _Z])
        r_sho_rv = chain_to_smplx([jp[t,17], jp[t,18], jp[t,19]], [_Y, _X, _Z])
        l_elb_rv = chain_to_smplx([jp[t,16]], [_Y])
        r_elb_rv = chain_to_smplx([jp[t,20]], [_Y])
        print(f"\n Computed SMPL rotations from G1 joint angles:")
        print(f"   L_shoulder rotvec (deg): {np.degrees(l_sho_rv)}")
        print(f"   R_shoulder rotvec (deg): {np.degrees(r_sho_rv)}")
        print(f"   L_elbow    rotvec (deg): {np.degrees(l_elb_rv)}")
        print(f"   R_elbow    rotvec (deg): {np.degrees(r_elb_rv)}")

        # Test: what if shoulder roll sign is NEGATED?
        print(f"\n Test: NEGATED shoulder roll angle:")
        l_sho_neg = chain_to_smplx([jp[t,13], -jp[t,14], jp[t,15]], [_Y, _X, _Z])
        r_sho_neg = chain_to_smplx([jp[t,17], -jp[t,18], jp[t,19]], [_Y, _X, _Z])
        print(f"   L_shoulder (neg roll) rotvec (deg): {np.degrees(l_sho_neg)}")
        print(f"   R_shoulder (neg roll) rotvec (deg): {np.degrees(r_sho_neg)}")

    # Check T-pose: what does SMPL give with zero arm angles?
    print(f"\n{'='*60}")
    print(" SMPL T-pose test: zero pose → shoulder and elbow positions")
    print(f"{'='*60}")
    zero_poses = np.zeros((1, 165))
    zero_betas = np.zeros((1, 16))
    zero_trans = np.zeros((1, 3))
    tpose_joints = forward_smplx(zero_poses, zero_betas, zero_trans)
    root = tpose_joints[0, 0]
    for jidx, name in [(16,"L_Shoulder"),(18,"L_Elbow"),(17,"R_Shoulder"),(19,"R_Elbow")]:
        rel = tpose_joints[0, jidx] - root
        print(f"  {name:<12}: X={rel[0]:+.3f}  Y={rel[1]:+.3f}  Z={rel[2]:+.3f}")

    # Test specific rotation to match G1 natural position
    print(f"\n{'='*60}")
    print(" Finding shoulder roll to make arms hang (match G1 link pos at frame 0)")
    print(f"{'='*60}")
    # G1 left arm: shoulder_yaw_link at [0.298, 0.258, 0.088], elbow at [0.499, 0.211, 0.229]
    # From T-pose, need to find what rotation produces elbow at ~[0.499, 0.211, 0.229]
    print(f"  G1 L_elbow target (relative to root): X=+0.499  Y=+0.211  Z=+0.229")
    print(f"  SMPL T-pose L_elbow:                  ", end="")
    tpose_elb = tpose_joints[0, 18] - tpose_joints[0, 0]
    print(f"X={tpose_elb[0]:+.3f}  Y={tpose_elb[1]:+.3f}  Z={tpose_elb[2]:+.3f}")

    # Try different roll angles and see which gives Y ≈ 0.211
    print(f"\n  Scanning roll offset for L_elbow Y ≈ 0.211:")
    for roll_deg in range(-90, 91, 10):
        poses = np.zeros((1, 165))
        # L_shoulder = poses[48:51] (j=16)
        roll_rv = chain_to_smplx([0, np.radians(roll_deg), 0], [_Y, _X, _Z])
        poses[0, 48:51] = roll_rv
        jnts = forward_smplx(poses, zero_betas, zero_trans)
        elb_rel = jnts[0, 18] - jnts[0, 0]
        print(f"    roll={roll_deg:+4d}°: L_elbow X={elb_rel[0]:+.3f} Y={elb_rel[1]:+.3f} Z={elb_rel[2]:+.3f}")


if __name__ == "__main__":
    main()
