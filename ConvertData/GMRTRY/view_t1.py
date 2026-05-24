#!/usr/bin/env python3
"""
Simple MuJoCo viewer for T1 retargeted motion pkl files.
Usage:  python view_t1.py output/lefthand_booster_t1.pkl
"""
import sys
import time
import pickle
import argparse
import numpy as np
import mujoco
import mujoco.viewer

T1_XML = "/home/isaak/GMR/assets/booster_t1/T1_serial.xml"


def load_pkl(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    fps      = d["fps"]
    root_pos = d["root_pos"]                          # (F, 3)
    root_rot = d["root_rot"][:, [3, 0, 1, 2]]        # xyzw → wxyz for MuJoCo
    dof_pos  = d["dof_pos"]                           # (F, N)
    return fps, root_pos, root_rot, dof_pos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("motion_file")
    parser.add_argument("--loop", action="store_true", default=True)
    args = parser.parse_args()

    fps, root_pos, root_rot, dof_pos = load_pkl(args.motion_file)
    n_frames  = root_pos.shape[0]
    dt        = 1.0 / fps

    model = mujoco.MjModel.from_xml_path(T1_XML)
    data  = mujoco.MjData(model)

    print(f"Loaded {n_frames} frames @ {fps} fps  |  {dof_pos.shape[1]} DOFs")
    print("Press ESC to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            t0 = time.time()

            data.qpos[:3]  = root_pos[frame]
            data.qpos[3:7] = root_rot[frame]
            data.qpos[7:]  = dof_pos[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()

            frame = (frame + 1) % n_frames
            elapsed = time.time() - t0
            remaining = dt - elapsed
            if remaining > 0:
                time.sleep(remaining)
