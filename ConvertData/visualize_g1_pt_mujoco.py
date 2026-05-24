#!/usr/bin/env python3
"""
Visualize Unitree G1 motion_dataset .pt in MuJoCo using the same RobotMotionViewer
pattern as visualize_t1_goalkeeper_pt.py (camera, passive viewer, step loop).

By default, merges a Booster-style scene (checker floor, lights, sky) with the robot:
the runtime URDF is compiled to MJCF (mj_saveLastXML), then combined with
`assets/booster_like_goalkeeper_scene.xml`. Pass --no-booster-scene for URDF-only preview.
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
import tempfile
import textwrap
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

BEP_ROOT = Path(__file__).resolve().parents[1]
_CONVERT_DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(_CONVERT_DATA))
sys.path.insert(0, str(BEP_ROOT / "src" / "retargeting"))

from general_motion_retargeting.robot_motion_viewer import RobotMotionViewer  # noqa: E402

from goalkeeper_viewer_shared import (  # noqa: E402
    BOOSTER_T1_VISUAL_REFERENCE_XML,
    ROBOT_MOTION_VIEWER_KWARGS,
)

# Booster T1-style floor / lights / sky (see T1_serial.xml “setup scene”); merged via wrapper MJCF.
BOOSTER_LIKE_SCENE_XML = _CONVERT_DATA / "assets" / "booster_like_goalkeeper_scene.xml"


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


def reorder_quat(quat: np.ndarray, order: str) -> np.ndarray:
    if order == "wxyz":
        return quat
    if order == "xyzw":
        return quat[[3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion order: {order}")


def pack_hinge_dof_vector(
    model: mj.MjModel, joint_names: list[str], values: np.ndarray, *, base_qpos: int = 7
) -> np.ndarray:
    """Map named dataset joints into qpos[7:] flat vector (hinges, one scalar each)."""
    out = np.zeros(model.nq - base_qpos, dtype=np.float64)
    n = min(len(joint_names), len(values))
    for i in range(n):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, joint_names[i])
        if jid < 0:
            continue
        adr = model.jnt_qposadr[jid]
        if adr < base_qpos:
            continue
        idx = adr - base_qpos
        if 0 <= idx < len(out):
            out[idx] = float(values[i])
    return out


def _default_g1_urdf() -> Path:
    candidates = [
        Path("/home/isaak/BEP/ConvertData/export/unitreeg1/urdf/g1_23.urdf"),
        Path("/home/isaak/BEP/ConvertData/export/unitree_g1/urdf/g1_23.urdf"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return candidates[-1]


def _iter_mesh_filenames(urdf_text: str) -> list[str]:
    root = ET.fromstring(urdf_text)
    out: list[str] = []
    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename")
        if fn:
            out.append(fn)
    return out


def _mesh_paths_missing(urdf_path: Path, urdf_text: str) -> list[str]:
    urdf_dir = urdf_path.parent.resolve()
    missing: list[str] = []
    for fn in _iter_mesh_filenames(urdf_text):
        candidate = (urdf_dir / fn).resolve()
        if not candidate.is_file():
            missing.append(str(candidate))
    return missing


def _strip_mesh_visual_collision(urdf_text: str) -> str:
    root = ET.fromstring(urdf_text)
    removed = 0
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for elem in list(link.findall(tag)):
                geom = elem.find("geometry")
                if geom is not None and geom.find("mesh") is not None:
                    link.remove(elem)
                    removed += 1
    if removed:
        print(f"[info] Stripped {removed} URDF <visual>/<collision> blocks that referenced meshes.")
    return ET.tostring(root, encoding="unicode")


def _absolutize_mesh_paths(urdf_text: str, urdf_dir: Path) -> str:
    root = ET.fromstring(urdf_text)
    changed = 0
    for mesh in root.iter("mesh"):
        fn = mesh.attrib.get("filename")
        if not fn:
            continue
        abs_path = (urdf_dir / fn).resolve()
        mesh.set("filename", str(abs_path))
        changed += 1
    if changed:
        print(f"[info] Rewrote {changed} mesh path(s) to absolute files (for out-of-tree runtime URDF).")
    return ET.tostring(root, encoding="unicode")


def _normalize_meshdir_for_relative_mesh_filenames(urdf_text: str) -> str:
    root = ET.fromstring(urdf_text)
    compiler = root.find("./mujoco/compiler")
    if compiler is None:
        return urdf_text
    meshdir = compiler.attrib.get("meshdir")
    if meshdir != "../meshes":
        compiler.set("meshdir", "../meshes")
        print(f"[info] Normalized MuJoCo compiler meshdir from {meshdir!r} to '../meshes'")
    return ET.tostring(root, encoding="unicode")


def _uncomment_floating_base_block(urdf_text: str) -> str:
    needle = "<!-- [CAUTION] uncomment when convert to mujoco -->"
    if needle not in urdf_text:
        return urdf_text
    start = urdf_text.find(needle)
    if start == -1:
        return urdf_text
    rest = urdf_text[start + len(needle) :]
    end_marker = "-->"
    end_rel = rest.find(end_marker)
    if end_rel == -1:
        return urdf_text
    block = rest[: end_rel + len(end_marker)]
    inner = block
    if inner.startswith("\n"):
        inner = inner[1:]
    inner = inner.replace("<!--", "").replace("-->", "")
    inner_stripped = textwrap.dedent(inner).strip("\n")
    new_text = urdf_text[:start] + inner_stripped + urdf_text[start + len(needle) + end_rel + len(end_marker) :]
    if "floating_base_joint" not in new_text:
        print("[warn] floating_base_joint not found after uncomment attempt.")
    else:
        print("[info] Enabled floating_base_joint block for root motion from the .pt file.")
    return new_text


def _prepare_runtime_urdf(
    urdf_path: Path,
    *,
    allow_missing_meshes: bool,
    enable_floating_base: bool,
    keep_runtime_urdf: bool,
) -> Path:
    text = urdf_path.read_text()
    if enable_floating_base:
        text = _uncomment_floating_base_block(text)
    text = _normalize_meshdir_for_relative_mesh_filenames(text)

    missing = _mesh_paths_missing(urdf_path, text)
    if missing:
        print(f"[warn] Missing {len(missing)} mesh file(s). First missing:\n  {missing[0]}")
        if not allow_missing_meshes:
            raise FileNotFoundError(
                "G1 URDF references STL meshes that are not present next to the URDF. "
                "Copy `.../g1/meshes/` into your export, or rerun with "
                "`--allow-missing-meshes` to strip mesh visuals/collisions for preview."
            )
        text = _strip_mesh_visual_collision(text)

    urdf_dir = urdf_path.parent.resolve()
    runtime_name = f".g1_runtime_{uuid.uuid4().hex}_{urdf_path.name}"
    primary = urdf_dir / runtime_name

    def _write(path: Path) -> None:
        path.write_text(text, encoding="utf-8")

    try:
        _write(primary)
        path = primary
    except OSError as exc:
        print(f"[warn] Could not write runtime URDF next to source ({exc}); using /tmp fallback.")
        fallback = Path(tempfile.gettempdir()) / runtime_name
        if list(ET.fromstring(text).iter("mesh")):
            text = _absolutize_mesh_paths(text, urdf_dir)
        _write(fallback)
        path = fallback

    if not keep_runtime_urdf:

        def _unlink_quiet(p: Path) -> None:
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

        atexit.register(_unlink_quiet, path)

    print(f"[info] Using runtime URDF: {path}")
    return path


def _compiler_line_for_scene_wrap(*, runtime_urdf: Path, source_urdf: Path) -> str:
    """Outer compiler meshdir: relative to URDF dir when possible, else absolute mesh folder."""
    tmp_root = Path(tempfile.gettempdir()).resolve()
    if runtime_urdf.parent.resolve() == tmp_root:
        meshes = (source_urdf.parent / ".." / "meshes").resolve()
        if meshes.is_dir():
            meshdir = meshes.as_posix()
            return f'  <compiler angle="radian" meshdir="{meshdir}" autolimits="true" strippath="false"/>'
        print("[warn] /tmp runtime URDF but mesh directory not found; wrapper meshdir may be wrong.")
    return '  <compiler angle="radian" meshdir="../meshes" autolimits="true" strippath="false"/>'


def _try_booster_scene_wrap(
    runtime_urdf: Path,
    *,
    source_urdf: Path,
    keep_aux_files: bool,
) -> tuple[Path, bool]:
    """
    Return (model_xml_path, used_wrap). On failure, fall back to runtime_urdf only.

    MuJoCo cannot <include> a URDF, so we compile the runtime URDF to MJCF via
    mj_saveLastXML, then merge that MJCF with a Booster-style scene MJCF.
    """
    scene = BOOSTER_LIKE_SCENE_XML.resolve()
    if not scene.is_file():
        print(f"[warn] Booster-like scene file missing ({BOOSTER_LIKE_SCENE_XML}); using URDF only.")
        return runtime_urdf, False

    try:
        robot_model = mj.MjModel.from_xml_path(str(runtime_urdf))
    except Exception as exc:
        print(f"[warn] Could not load runtime URDF for scene merge ({exc!r}); using URDF only.")
        return runtime_urdf, False

    compiled = runtime_urdf.parent / f".g1_compiled_{uuid.uuid4().hex}.xml"
    try:
        mj.mj_saveLastXML(str(compiled), robot_model)
    except Exception as exc:
        print(f"[warn] mj_saveLastXML failed ({exc!r}); using URDF only.")
        return runtime_urdf, False

    wrap = runtime_urdf.parent / f".g1_booster_scene_wrap_{uuid.uuid4().hex}.xml"
    compiler = _compiler_line_for_scene_wrap(runtime_urdf=runtime_urdf, source_urdf=source_urdf.resolve())
    scene_p = scene.as_posix()
    comp_p = compiled.resolve().as_posix()
    wrap.write_text(
        "<?xml version='1.0'?>\n"
        '<mujoco model="g1_goalkeeper_view">\n'
        f"{compiler}\n"
        f'  <include file="{comp_p}"/>\n'
        f'  <include file="{scene_p}"/>\n'
        "</mujoco>\n",
        encoding="utf-8",
    )

    try:
        mj.MjModel.from_xml_path(str(wrap))
    except Exception as exc:
        print(f"[warn] Scene + compiled MJCF failed to load ({exc!r}); using URDF only.")
        wrap.unlink(missing_ok=True)
        compiled.unlink(missing_ok=True)
        return runtime_urdf, False

    if not keep_aux_files:

        def _unlink_wrap_and_compiled() -> None:
            for p in (wrap, compiled):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass

        atexit.register(_unlink_wrap_and_compiled)

    print(f"[info] Booster-like scene merged (compiled robot MJCF + scene): {wrap}")
    return wrap, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-pt", type=Path, required=True, help="Path to .pt motion file.")
    parser.add_argument(
        "--joint-id",
        type=Path,
        default=Path("/home/isaak/BEP/ConvertData/export/motion_dataset/joint_id.txt"),
        help="Path to joint_id.txt",
    )
    parser.add_argument(
        "--g1-urdf",
        type=Path,
        default=_default_g1_urdf(),
        help="Path to G1 URDF file.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Playback FPS (default: from .pt or 30).")
    parser.add_argument(
        "--quat-order",
        choices=["xyzw", "wxyz"],
        default="xyzw",
        help="Quaternion order in base_pose in the source .pt file.",
    )
    parser.add_argument(
        "--allow-missing-meshes",
        action="store_true",
        help="Strip mesh <visual>/<collision> if STL files are missing (kinematic preview only).",
    )
    parser.add_argument(
        "--no-enable-floating-base",
        action="store_true",
        help="Do not uncomment floating_base_joint in the URDF (root pose from .pt will be ignored).",
    )
    parser.add_argument(
        "--keep-runtime-urdf",
        action="store_true",
        help="Keep the generated runtime URDF on disk (default: delete at exit).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load motion + model mapping, then exit without opening the MuJoCo viewer.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index.")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride.")
    parser.add_argument("--no-loop", action="store_true", help="Stop at last frame.")
    parser.add_argument(
        "--no-booster-scene",
        action="store_true",
        help="Do not merge Booster-style floor/sky/light MJCF (robot-only, like older G1 preview).",
    )
    args = parser.parse_args()

    data_dict = torch.load(args.motion_pt, map_location="cpu")
    base_position = to_numpy(data_dict["base_position"])
    base_pose = to_numpy(data_dict["base_pose"])
    joint_position = to_numpy(data_dict["joint_position"])
    joint_names = load_joint_names(args.joint_id)
    fps = (
        float(args.fps)
        if args.fps is not None
        else float(data_dict.get("frame_rate", data_dict.get("fps", 30.0)))
    )

    enable_floating = not args.no_enable_floating_base
    runtime_urdf = _prepare_runtime_urdf(
        args.g1_urdf,
        allow_missing_meshes=args.allow_missing_meshes,
        enable_floating_base=enable_floating,
        keep_runtime_urdf=args.keep_runtime_urdf,
    )

    if args.no_booster_scene:
        model_xml, used_scene_wrap = runtime_urdf, False
    else:
        model_xml, used_scene_wrap = _try_booster_scene_wrap(
            runtime_urdf,
            source_urdf=args.g1_urdf.resolve(),
            keep_aux_files=args.keep_runtime_urdf,
        )

    source_cols = min(joint_position.shape[1], len(joint_names))
    if source_cols < joint_position.shape[1]:
        print(
            f"[warn] joint_position has {joint_position.shape[1]} cols, "
            f"but joint_id has {len(joint_names)} names. Using first {source_cols}."
        )
    joint_names = joint_names[:source_cols]

    frame_count = min(len(base_position), len(base_pose), len(joint_position))
    if frame_count == 0:
        raise ValueError("No frames found in input .pt file.")

    model = mj.MjModel.from_xml_path(str(model_xml))
    free_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if free_joint_id == -1:
        print(
            "[warn] No joint named `floating_base_joint` in the loaded model. "
            "Using identity root; only joint angles from the .pt are applied."
        )

    if args.validate_only:
        print(
            "[ok] Validation successful (model loaded, mapping will use RobotMotionViewer layout). "
            f"scene_wrap={used_scene_wrap}"
        )
        return

    if not os.environ.get("DISPLAY"):
        print(
            "[warn] DISPLAY is not set; cannot open MuJoCo viewer. On a desktop, run from a GUI session."
        )
        print(f"[ok] Loaded {frame_count} frames at {fps} Hz — data OK.")
        return

    viewer = RobotMotionViewer(
        robot_type="unitree_g1",
        model_xml_path=str(model_xml),
        motion_fps=fps,
        visual_style_reference_xml=(
            None if used_scene_wrap else str(BOOSTER_T1_VISUAL_REFERENCE_XML)
        ),
        **ROBOT_MOTION_VIEWER_KWARGS,
    )
    model = viewer.model

    print(f"[info] Playing Unitree G1: {args.motion_pt} ({frame_count} frames @ {fps} Hz)")
    print(f"[info] Robot XML: {model_xml}" + ("" if used_scene_wrap else f" (runtime URDF: {runtime_urdf})"))
    print("[info] Close the viewer window to stop.")

    identity_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    try:
        frame = max(0, min(args.start_frame, frame_count - 1))
        while True:
            jp = joint_position[frame]
            dof = pack_hinge_dof_vector(model, joint_names, jp)

            if free_joint_id != -1:
                root_pos = base_position[frame]
                root_rot = reorder_quat(base_pose[frame], args.quat_order)
            else:
                root_pos = np.zeros(3, dtype=np.float64)
                root_rot = identity_wxyz.copy()

            viewer.step(
                root_pos,
                root_rot,
                dof,
                rate_limit=True,
                follow_camera=True,
            )

            frame += args.stride
            if frame >= frame_count:
                if args.no_loop:
                    break
                frame = 0
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
