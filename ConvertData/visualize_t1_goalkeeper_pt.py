#!/usr/bin/env python3
"""
Play back a converted Booster T1 goalkeeper-style .pt (from convert_g1_pt_to_t1) in the GMR MuJoCo viewer.

Expects torch.save dict keys: base_position (T,3), base_pose (T,4) xyzw, joint_position (T,23)
in the same column order as manual_g1_to_t1_23_mapping.json target_joint_order.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

BEP_ROOT = Path(__file__).resolve().parents[1]
_CONVERT_DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(_CONVERT_DATA))
sys.path.insert(0, str(BEP_ROOT / "src" / "retargeting"))

from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer  # noqa: E402

from goalkeeper_viewer_shared import ROBOT_MOTION_VIEWER_KWARGS  # noqa: E402


def to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion-pt",
        type=Path,
        default=BEP_ROOT
        / "ConvertData/datasets/t1_from_g1_goalkeeper/motions/leftjump.pt",
        help="Converted T1 .pt file",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override FPS (default: from file or 30)")
    parser.add_argument("--no-loop", action="store_true", help="Stop after last frame")
    parser.add_argument("--validate-only", action="store_true", help="Load model + first frame, exit")
    args = parser.parse_args()

    if not args.motion_pt.is_file():
        raise FileNotFoundError(args.motion_pt)

    data = torch.load(args.motion_pt, map_location="cpu")
    root_pos = to_numpy(data["base_position"])
    root_quat_xyzw = to_numpy(data["base_pose"])
    dof_pos = to_numpy(data["joint_position"])
    fps = float(args.fps) if args.fps is not None else float(data.get("frame_rate", data.get("fps", 30.0)))

    if root_quat_xyzw.ndim != 2 or root_quat_xyzw.shape[1] != 4:
        raise ValueError("base_pose must be (T, 4) xyzw quaternions")
    if dof_pos.ndim != 2 or dof_pos.shape[1] != 23:
        raise ValueError(f"joint_position must be (T, 23), got {dof_pos.shape}")

    root_rot_wxyz = quat_xyzw_to_wxyz(root_quat_xyzw)
    n_frames = min(len(root_pos), len(root_rot_wxyz), len(dof_pos))

    if args.validate_only:
        print(f"[ok] Would play {n_frames} frames at {fps} Hz (validate-only).")
        return

    if not os.environ.get("DISPLAY"):
        print("[warn] DISPLAY is not set; cannot open MuJoCo viewer. On a desktop, run from a GUI session.")
        print(f"[ok] Loaded {n_frames} frames at {fps} Hz — data OK.")
        return

    viewer = RobotMotionViewer(robot_type="booster_t1", motion_fps=fps, **ROBOT_MOTION_VIEWER_KWARGS)

    print(f"[info] Playing Booster T1: {args.motion_pt} ({n_frames} frames @ {fps} Hz)")
    print("[info] Close the viewer window to stop.")

    try:
        idx = 0
        while True:
            viewer.step(
                root_pos[idx],
                root_rot_wxyz[idx],
                dof_pos[idx],
                rate_limit=True,
                follow_camera=True,
            )
            idx += 1
            if idx >= n_frames:
                if args.no_loop:
                    break
                idx = 0
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
