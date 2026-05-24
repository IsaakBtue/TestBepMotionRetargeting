#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create a GIF animation of the SMPL-X body mesh sequence from a MoSh++ stageii pkl.

Rendering approach mirrors SOMA's parameters_to_mesh.py + mesh_to_video_standard.py:
  - BodyModel → mesh vertices (Y-up, world space)
  - Apply Rx(-90°) to convert Y-up → Z-up, matching the rotation SOMA applies
    before passing meshes to Blender
  - Use pyrender with EGL backend (proper Z-buffer, lighting, smooth shading)
    instead of Blender, since we don't need photorealistic quality

Usage (from repo root, soma env active):
    python src/pipeline/create_smplx_gif.py \
        --mesh output/soma_mosh39/smplx_body_mesh_all_frames.pkl \
        --out  output/soma_mosh39/smplx_body_animation.gif
"""
import os
# Must be set before importing pyrender/OpenGL
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import pickle
import sys
from pathlib import Path

_BEP_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_MESH = _BEP_ROOT / "output" / "soma_mosh39" / "smplx_body_mesh_all_frames.pkl"
_DEFAULT_GIF = _BEP_ROOT / "output" / "soma_mosh39" / "smplx_body_animation.gif"

import imageio
import numpy as np
import pyrender
import trimesh
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Geometry helpers (mirrors SOMA's rotate_points_xyz usage)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scene / camera helpers
# ---------------------------------------------------------------------------

def _look_at(eye: np.ndarray, target: np.ndarray,
             up: np.ndarray = None) -> np.ndarray:
    """
    Build a 4×4 camera-to-world pose matrix (OpenGL/pyrender convention:
    camera looks along -Z, +Y is up in camera space).

    right   = cross(forward, up)   → right-handed +X
    true_up = cross(right, forward) → reorthonormalised +Y
    col 2   = -forward             → camera -Z axis points along world forward
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = right
    mat[:3, 1] = true_up
    mat[:3, 2] = -forward
    mat[:3, 3] = eye
    return mat


def add_lights_and_camera(scene: pyrender.Scene,
                          centroid: np.ndarray,
                          height: float) -> None:
    """
    Three-point lighting + frontal perspective camera for Y-up SMPL-X world space.

    SMPL-X convention: Y = height, person faces +Z.
    Camera is placed in front of the person (+Z side) at mid-body height,
    looking toward the centroid. No vertex rotation is needed.
    """
    dist = max(2.5, height * 1.6)

    # Key light — front-right, slightly above
    key_pose = _look_at(
        eye=centroid + np.array([dist * 0.6, height * 0.4, dist]),
        target=centroid)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0),
              pose=key_pose)

    # Fill light — front-left
    fill_pose = _look_at(
        eye=centroid + np.array([-dist * 0.5, height * 0.1, dist * 0.8]),
        target=centroid)
    scene.add(pyrender.DirectionalLight(color=[0.9, 0.9, 1.0], intensity=2.0),
              pose=fill_pose)

    # Rim / back light
    back_pose = _look_at(
        eye=centroid + np.array([0.0, height * 0.3, -dist * 1.2]),
        target=centroid)
    scene.add(pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=1.5),
              pose=back_pose)

    # Camera — placed in front (+Z), looking back at person
    cam_eye = centroid + np.array([0.0, height * 0.05, dist * 1.05])
    cam_pose = _look_at(eye=cam_eye, target=centroid)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(42), znear=0.05, zfar=50.0)
    scene.add(camera, pose=cam_pose)


def build_scene(vertices: np.ndarray, faces: np.ndarray,
                smooth: bool = True) -> pyrender.Scene:
    """Build a pyrender Scene for one frame."""
    tri = trimesh.Trimesh(vertices=vertices, faces=faces,
                          process=False)
    if smooth:
        tri = trimesh.smoothing.filter_laplacian(tri, iterations=1)

    mesh_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.65, 0.74, 0.86, 1.0],  # light blue — matches SOMA default
        metallicFactor=0.0,
        roughnessFactor=0.6,
        smooth=smooth,
    )
    py_mesh = pyrender.Mesh.from_trimesh(tri, material=mesh_material, smooth=smooth)

    scene = pyrender.Scene(
        bg_color=[0.95, 0.95, 0.95, 1.0],
        ambient_light=[0.35, 0.35, 0.35],
    )
    scene.add(py_mesh)
    return scene


# ---------------------------------------------------------------------------
# Main rendering loop
# ---------------------------------------------------------------------------

def create_gif_from_mesh(mesh_path: str, output_gif_path: str,
                         fps: int = 15,
                         max_frames: int = None,
                         downsample: int = 1,
                         width: int = 640,
                         height: int = 720) -> str:
    logger.info(f"Loading mesh data from {mesh_path}...")
    with open(mesh_path, "rb") as f:
        mesh_data = pickle.load(f)

    vertices_all = mesh_data["vertices"]   # (T, V, 3)  — Y-up world space
    faces = mesh_data["faces"]             # (F, 3)

    total_frames = len(vertices_all)
    logger.info(f"Loaded {total_frames} frames | "
                f"{vertices_all[0].shape[0]} verts | {len(faces)} faces")

    # Frame selection
    frame_indices = np.arange(0, total_frames, max(1, downsample))
    if max_frames is not None:
        frame_indices = frame_indices[:max_frames]
    logger.info(f"Rendering {len(frame_indices)} frames at {fps} fps…")

    # Compute stable scene bounds from the whole sequence so the camera
    # doesn't jump between frames.
    # SMPL-X is Y-up (Y = height). No rotation is applied to the vertices —
    # pyrender works fine in Y-up world space. The camera is placed at +Z
    # (in front of the person who faces +Z) looking back toward the centroid.
    all_verts_sample = vertices_all[frame_indices[::max(1, len(frame_indices)//20)]]
    centroid = all_verts_sample.reshape(-1, 3).mean(axis=0)
    body_height = (all_verts_sample.reshape(-1, 3)[:, 1].max() -
                   all_verts_sample.reshape(-1, 3)[:, 1].min())

    renderer = pyrender.OffscreenRenderer(width, height)
    frames_out = []

    try:
        for frame_idx in tqdm(frame_indices, desc="Rendering"):
            verts = vertices_all[frame_idx]   # keep in native Y-up SMPL-X space

            scene = build_scene(verts, faces, smooth=True)
            add_lights_and_camera(scene, centroid, body_height)

            color, _ = renderer.render(scene,
                                       flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            frames_out.append(color)
    finally:
        renderer.delete()

    logger.info(f"Saving GIF to {output_gif_path}…")
    duration_ms = int(1000 / fps)
    imageio.mimsave(output_gif_path, frames_out, duration=duration_ms, loop=0)
    logger.success(f"GIF saved ({len(frames_out)} frames, {fps} fps, "
                   f"{duration_ms} ms/frame)")
    return output_gif_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Render SMPL-X mesh sequence → GIF using pyrender (EGL)")
    parser.add_argument(
        "--mesh",
        default=str(_DEFAULT_MESH),
        help="Path to smplx_body_mesh_all_frames.pkl")
    parser.add_argument(
        "--out",
        default=str(_DEFAULT_GIF),
        help="Output GIF path")
    parser.add_argument("--fps",        type=int, default=15)
    parser.add_argument("--downsample", type=int, default=1,
                        help="Use every Nth frame (1 = all frames)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--width",  type=int, default=640)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    if not os.path.exists(args.mesh):
        logger.error(f"Mesh file not found: {args.mesh}")
        sys.exit(1)

    create_gif_from_mesh(
        mesh_path=args.mesh,
        output_gif_path=args.out,
        fps=args.fps,
        max_frames=args.max_frames,
        downsample=args.downsample,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
