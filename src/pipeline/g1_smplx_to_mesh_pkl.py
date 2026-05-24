#!/usr/bin/env python3
"""
Convert g1_pt_to_smplx_pkl.py output (poses/trans/betas) → vertices/faces pkl
compatible with create_smplx_gif.py.

Usage (soma env):
    conda activate soma
    python src/pipeline/g1_smplx_to_mesh_pkl.py \
        --input  output/g1_lefthand/smplx_body_mesh_all_frames.pkl \
        --output output/g1_lefthand/smplx_body_mesh_vertices.pkl
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import smplx
import torch


_MODEL_PATH = Path("/home/isaak/BEP/body_models")


def convert(input_pkl: Path, output_pkl: Path) -> None:
    with open(input_pkl, "rb") as f:
        data = pickle.load(f)

    poses = data["poses"]   # (T, 165)
    trans = data["trans"]   # (T, 3)
    betas = data["betas"]   # (T, 16)
    T = poses.shape[0]

    body_model = smplx.create(
        str(_MODEL_PATH),
        model_type="smplx",
        gender="neutral",
        num_betas=16,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=T,
    )

    global_orient = torch.tensor(poses[:, 0:3], dtype=torch.float32)
    body_pose     = torch.tensor(poses[:, 3:66], dtype=torch.float32)
    betas_t       = torch.tensor(betas, dtype=torch.float32)
    transl        = torch.tensor(trans, dtype=torch.float32)

    with torch.no_grad():
        out = body_model(
            global_orient=global_orient,
            body_pose=body_pose,
            betas=betas_t,
            transl=transl,
        )

    vertices = out.vertices.numpy()   # (T, V, 3)
    faces    = body_model.faces       # (F, 3)

    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_pkl, "wb") as f:
        pickle.dump(
            {"vertices": vertices, "faces": faces,
             "poses": poses, "trans": trans, "betas": betas},
            f,
        )
    print(f"[g1_smplx_to_mesh_pkl] {T} frames → {output_pkl}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",  required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
