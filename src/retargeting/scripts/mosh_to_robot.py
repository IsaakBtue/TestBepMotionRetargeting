import argparse
import pathlib
import os
import pickle
import time

import numpy as np
import smplx
import torch

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import get_gvhmr_data_offline_fast

from rich import print

# ── Tunable defaults ──────────────────────────────────────────────────────────
DEFAULT_MOCAP_FPS = 100   # Motive capture rate
DEFAULT_TGT_FPS   = 30    # Retargeting output FPS
DEFAULT_ROBOT     = "booster_t1"
SMPLX_FOLDER      = pathlib.Path("/home/isaak/BEP/body_models")
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Retarget MoSh++ SMPL-X PKL directly to a robot "
                    "(uses get_smplx_data_offline_fast — no GVHMR coordinate rotation)."
    )
    parser.add_argument(
        "--smplx_mesh_pkl",
        help="Path to smplx_body_mesh_all_frames.pkl produced by the MoSh++ pipeline.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--robot",
        choices=[
            "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
            "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
            "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
            "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
            "tienkung", "fourier_gr3",
        ],
        default=DEFAULT_ROBOT,
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the retargeted robot motion (.pkl).",
    )
    parser.add_argument(
        "--loop",
        default=False,
        action="store_true",
        help="Loop the motion in the viewer.",
    )
    parser.add_argument(
        "--record_video",
        default=False,
        action="store_true",
        help="Record a video of the viewer.",
    )
    parser.add_argument(
        "--rate_limit",
        default=False,
        action="store_true",
        help="Limit viewer rate to match human motion FPS.",
    )
    parser.add_argument(
        "--tgt_fps",
        type=int,
        default=DEFAULT_TGT_FPS,
        help=f"Target FPS for retargeting (default: {DEFAULT_TGT_FPS}).",
    )
    parser.add_argument(
        "--mocap_fps",
        type=int,
        default=DEFAULT_MOCAP_FPS,
        help=f"Source mocap capture rate in Hz (default: {DEFAULT_MOCAP_FPS}).",
    )
    parser.add_argument(
        "--no_viewer",
        default=False,
        action="store_true",
        help="Skip the MuJoCo viewer (save-only mode).",
    )

    args = parser.parse_args()

    # ── Load MoSh++ SMPL-X PKL ───────────────────────────────────────────────
    with open(args.smplx_mesh_pkl, "rb") as f:
        d = pickle.load(f)

    poses    = np.asarray(d["poses"], dtype=np.float32)   # (T, 165)
    trans    = np.asarray(d["trans"], dtype=np.float32)   # (T, 3)
    betas_raw = np.asarray(d["betas"], dtype=np.float32)  # (T, 16) or (16,)

    # Normalise betas to 1-D mean across frames
    betas_1d = betas_raw.mean(axis=0) if betas_raw.ndim == 2 else betas_raw
    betas_1d = betas_1d[:16]

    num_frames = poses.shape[0]
    print(f"Loaded {num_frames} frames from {args.smplx_mesh_pkl}")

    smplx_data = {
        "pose_body":        poses[:, 3:66],          # (T, 63) body joints
        "root_orient":      poses[:, 0:3],            # (T, 3)  global orient
        "trans":            trans,                    # (T, 3)
        "betas":            betas_1d,                 # (16,)
        "mocap_frame_rate": torch.tensor(args.mocap_fps),
    }

    # ── Build SMPL-X body model ───────────────────────────────────────────────
    body_model = smplx.create(
        SMPLX_FOLDER,
        "smplx",
        gender="male",
        use_pca=False,
        num_betas=16,
        num_expression_coeffs=10,
    )

    betas_tensor = torch.tensor(smplx_data["betas"]).float().view(1, -1).expand(num_frames, -1)
    smplx_output = body_model(
        betas=betas_tensor,
        global_orient=torch.tensor(smplx_data["root_orient"]).float(),
        body_pose=torch.tensor(smplx_data["pose_body"]).float(),
        transl=torch.tensor(smplx_data["trans"]).float(),
        left_hand_pose=torch.zeros(num_frames, 45).float(),
        right_hand_pose=torch.zeros(num_frames, 45).float(),
        jaw_pose=torch.zeros(num_frames, 3).float(),
        leye_pose=torch.zeros(num_frames, 3).float(),
        reye_pose=torch.zeros(num_frames, 3).float(),
        expression=torch.zeros(num_frames, 10).float(),
        return_full_pose=True,
    )

    actual_human_height = 1.66 + 0.1 * float(betas_1d[0])
    print(f"Estimated human height: {actual_human_height:.3f} m")

    # ── FPS alignment + Y-up → Z-up coordinate conversion ────────────────────
    # MoSh++ / SMPL-X uses Y as the vertical axis; MuJoCo / GMR expects Z-up.
    # get_gvhmr_data_offline_fast applies the same SLERP interpolation as
    # get_smplx_data_offline_fast but also rotates all joint positions and
    # orientations from Y-up to Z-up before handing them to the retargeter.
    smplx_data_frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.tgt_fps
    )
    print(f"Aligned to {aligned_fps:.1f} FPS → {len(smplx_data_frames)} frames")

    # ── Initialise retargeting system ────────────────────────────────────────
    retarget = GMR(
        actual_human_height=actual_human_height,
        src_human="smplx",
        tgt_robot=args.robot,
    )

    if not args.no_viewer:
        robot_motion_viewer = RobotMotionViewer(
            robot_type=args.robot,
            motion_fps=aligned_fps,
            transparent_robot=0,
            record_video=args.record_video,
            video_path=(
                f"videos/{args.robot}_"
                f"{pathlib.Path(args.smplx_mesh_pkl).stem}.mp4"
            ),
        )

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    fps_counter    = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    i = 0
    while True:
        if args.loop:
            i = (i + 1) % len(smplx_data_frames)
        else:
            i += 1
            if i >= len(smplx_data_frames):
                break

        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            print(f"Actual rendering FPS: {fps_counter / (current_time - fps_start_time):.2f}")
            fps_counter    = 0
            fps_start_time = current_time

        smplx_frame = smplx_data_frames[i]
        qpos = retarget.retarget(smplx_frame)

        if not args.no_viewer:
            robot_motion_viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                human_pos_offset=np.array([0.0, 0.0, 0.0]),
                show_human_body_name=False,
                rate_limit=args.rate_limit,
            )

        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        root_pos = np.array([q[:3]          for q in qpos_list])
        root_rot = np.array([q[3:7][[1,2,3,0]] for q in qpos_list])  # wxyz → xyzw
        dof_pos  = np.array([q[7:]          for q in qpos_list])

        motion_data = {
            "fps":           aligned_fps,
            "root_pos":      root_pos,
            "root_rot":      root_rot,
            "dof_pos":       dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    if not args.no_viewer:
        robot_motion_viewer.close()
