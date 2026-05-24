t#!/usr/bin/env python3
"""Manual joint mapping converter: G1 motion_dataset .pt -> Booster T1 23-DoF .pt."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_joint_names(joint_id_path: Path) -> list[str]:
    names: list[str] = []
    for raw_line in joint_id_path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        names.append(parts[1].strip())
    return names


def to_numpy(arr) -> np.ndarray:
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def convert_g1_pt_to_t1(
    *,
    input_pt: Path,
    output_pt: Path,
    joint_id: Path,
    mapping_json: Path,
    strict: bool = False,
) -> None:
    """Load one G1 .pt, apply manual joint map, save T1-layout .pt."""
    source_dict = torch.load(input_pt, map_location="cpu")
    source_joint_names = load_joint_names(joint_id)
    source_joint_position = to_numpy(source_dict["joint_position"])
    source_joint_velocity = (
        to_numpy(source_dict["joint_velocity"])
        if "joint_velocity" in source_dict and source_dict["joint_velocity"] is not None
        else None
    )

    source_cols = min(source_joint_position.shape[1], len(source_joint_names))
    source_joint_names = source_joint_names[:source_cols]
    source_joint_position = source_joint_position[:, :source_cols]
    if source_joint_velocity is not None:
        source_joint_velocity = source_joint_velocity[:, :source_cols]

    cfg = json.loads(mapping_json.read_text())
    target_joint_order = cfg["target_joint_order"]
    mappings = cfg["mappings"]

    source_index = {name: idx for idx, name in enumerate(source_joint_names)}
    target_index = {name: idx for idx, name in enumerate(target_joint_order)}

    frames = source_joint_position.shape[0]
    out_pos = np.zeros((frames, len(target_joint_order)), dtype=source_joint_position.dtype)
    out_vel = (
        np.zeros((frames, len(target_joint_order)), dtype=source_joint_velocity.dtype)
        if source_joint_velocity is not None
        else None
    )

    for entry in mappings:
        src = entry["source"]
        tgt = entry["target"]
        scale = float(entry.get("scale", 1.0))
        offset = float(entry.get("offset", 0.0))
        if src not in source_index:
            print(f"[warn] Source joint not found: {src}")
            continue
        if tgt not in target_index:
            print(f"[warn] Target joint not found: {tgt}")
            continue
        s = source_index[src]
        t = target_index[tgt]
        out_pos[:, t] = source_joint_position[:, s] * scale + offset
        if out_vel is not None:
            out_vel[:, t] = source_joint_velocity[:, s] * scale

    unmapped_target = [
        name
        for name in target_joint_order
        if not np.any(np.abs(out_pos[:, target_index[name]]) > 0.0)
    ]
    if unmapped_target:
        print("[warn] Target joints still all-zero (check mapping):", ", ".join(unmapped_target))
        if strict:
            raise ValueError("Strict mode: some target joints are unmapped/all-zero.")

    out_dict = dict(source_dict)
    out_dict["joint_position"] = torch.as_tensor(out_pos)
    if out_vel is not None:
        out_dict["joint_velocity"] = torch.as_tensor(out_vel)

    out_dict["target_joint_order"] = target_joint_order
    out_dict["retarget_metadata"] = {
        "source_robot": "unitree_g1",
        "target_robot": "booster_t1",
        "target_dof": len(target_joint_order),
        "mapping_json": str(mapping_json),
    }

    output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_dict, output_pt)
    print(f"[ok] Saved converted motion to: {output_pt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pt", type=Path, required=True, help="Input G1 .pt file")
    parser.add_argument("--output-pt", type=Path, required=True, help="Output T1 .pt file")
    parser.add_argument(
        "--joint-id",
        type=Path,
        default=Path("/home/isaak/BEP/ConvertData/export/motion_dataset/joint_id.txt"),
        help="Path to source joint_id.txt",
    )
    parser.add_argument(
        "--mapping-json",
        type=Path,
        default=Path("/home/isaak/BEP/ConvertData/manual_g1_to_t1_23_mapping.json"),
        help="Manual mapping JSON file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any target joint has no mapping.",
    )
    args = parser.parse_args()

    convert_g1_pt_to_t1(
        input_pt=args.input_pt,
        output_pt=args.output_pt,
        joint_id=args.joint_id,
        mapping_json=args.mapping_json,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
