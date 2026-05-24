import argparse
import pathlib
import pickle
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Ensure `general_motion_retargeting` is importable even when this script is run
# directly (without the wrapper that sets PYTHONPATH).
_RETARGETING_ROOT = pathlib.Path("/home/isaak/BEP/src/retargeting")
if _RETARGETING_ROOT.exists() and str(_RETARGETING_ROOT) not in sys.path:
    sys.path.insert(0, str(_RETARGETING_ROOT))


def _quat_identity_wxyz() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

def _swap_lr_name(name: str) -> str:
    if name.startswith("L"):
        return "R" + name[1:]
    if name.startswith("R"):
        return "L" + name[1:]
    return name

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)

def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = np.zeros_like(v)
    m = n[..., 0] > eps
    out[m] = v[m] / n[m]
    return out


def _rotmat_to_quat_wxyz(Rm: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion (w, x, y, z) using SciPy.
    Rm: (T, 3, 3) or (..., 3, 3)
    """
    from scipy.spatial.transform import Rotation as Rot

    Rm = np.asarray(Rm, dtype=np.float64)
    r = Rot.from_matrix(Rm.reshape(-1, 3, 3))
    q_xyzw = r.as_quat()  # (N, 4) in xyzw
    q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
    return q_wxyz.reshape(Rm.shape[:-2] + (4,)).astype(np.float32)

def _frame_from_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Build an orthonormal frame from (possibly non-orthogonal) axes.
    Returns rotation matrix with columns [x, y, z] in world coordinates.
    """
    x = _safe_normalize(x)
    y = y - (np.sum(x * y, axis=-1, keepdims=True) * x)
    y = _safe_normalize(y)
    z = np.cross(x, y)
    z = _safe_normalize(z)
    # Re-orthogonalize y to guarantee right-handedness
    y = np.cross(z, x)
    y = _safe_normalize(y)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def _compute_pelvis_quat_from_waist_markers_wxyz(
    LFWT: np.ndarray,
    RFWT: np.ndarray,
    MBWT: np.ndarray,
    MFWT: np.ndarray,
    *,
    y_up_to_z_up: bool,
) -> np.ndarray:
    """
    Estimate pelvis orientation from waist markers.

    Constructs a right-handed frame:
      - x: left direction (RFWT -> LFWT)
      - f: forward direction (MBWT -> MFWT)
      - u: up = x × f
      - f re-orthogonalized = u × x

    Returns quaternion (w, x, y, z). If y_up_to_z_up=True, converts to Z-up
    using the same rotation as the normal SMPL-X retarget pipeline.
    """
    x = _normalize(LFWT - RFWT)
    f = _normalize(MFWT - MBWT)
    u = _normalize(np.cross(x, f))
    f = _normalize(np.cross(u, x))

    # Rotation matrix whose columns are the body axes in world coordinates.
    Rm = np.stack([x, f, u], axis=-1)  # (T, 3, 3)

    if y_up_to_z_up:
        rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        Rm = (rot @ Rm @ rot.T).astype(np.float32)

    return _rotmat_to_quat_wxyz(Rm)


def _estimate_fps(time_s: np.ndarray) -> float:
    time_s = np.asarray(time_s, dtype=np.float64)
    if time_s.size < 2:
        return 30.0
    dt = np.diff(time_s)
    dt = dt[np.isfinite(dt)]
    dt = dt[dt > 1e-6]
    if dt.size == 0:
        return 30.0
    return float(1.0 / np.median(dt))


def _get_marker_xyz(df: pd.DataFrame, name: str) -> np.ndarray:
    return df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(dtype=np.float32)


def _mean_of_available(markers: Dict[str, np.ndarray]) -> np.ndarray:
    xs = [v for v in markers.values() if v is not None]
    if not xs:
        raise KeyError("No markers provided for mean()")
    return np.mean(np.stack(xs, axis=0), axis=0)


def _y_up_to_z_up(pos: np.ndarray) -> np.ndarray:
    # Match `general_motion_retargeting.utils.smpl.get_gvhmr_data_offline_fast`:
    # rotation_matrix = [[1,0,0],[0,0,-1],[0,1,0]]
    # (x, y, z)_y-up -> (x, -z, y)_z-up
    rot = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    return (pos @ rot.T).astype(np.float32)


def _ry_minus_90(pos: np.ndarray) -> np.ndarray:
    # Ry(-90°): x'=-z, y'=y, z'=+x
    rot = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    return (pos @ rot.T).astype(np.float32)

def _rz_pi(pos: np.ndarray) -> np.ndarray:
    # Rz(180°): x'=-x, y'=-y, z'=z
    rot = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return (pos @ rot.T).astype(np.float32)


def build_smplx_like_frames_from_markers(
    df: pd.DataFrame,
    *,
    units: str,
    apply_mosh_rotate: bool,
    apply_y_up_to_z_up: bool,
    forward_from: str,
    yaw180: bool,
    lr_swap: bool,
    orient_mode: str,
    toe_flip: bool,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], float]:
    """
    Produce per-frame "human_data" dict compatible with GMR's smplx IK configs.

    This is intentionally a *test* path: it uses sparse markers to approximate a
    subset of SMPL-X body frames (positions only; identity orientations).
    """
    time_s = df["Time"].to_numpy(dtype=np.float64) if "Time" in df.columns else None
    fps = _estimate_fps(time_s) if time_s is not None else 30.0

    if units == "auto":
        # Heuristic: waist width should be ~0.2–0.6 m. If it's > 2.0, assume mm.
        LFWT_raw = _get_marker_xyz(df, "LFWT")
        RFWT_raw = _get_marker_xyz(df, "RFWT")
        waist_width = np.nanmedian(np.linalg.norm(LFWT_raw - RFWT_raw, axis=1))
        units = "mm" if waist_width > 2.0 else "m"

    scale = 1.0 if units == "m" else 1e-3

    # Required by `smplx_to_t1.json` (and similar): pelvis, spine3, hips, knees,
    # feet, shoulders, elbows.
    # We approximate these from marker positions.
    marker_names_raw = {
        # pelvis from waist cluster
        "LFWT": _get_marker_xyz(df, "LFWT"),
        "RFWT": _get_marker_xyz(df, "RFWT"),
        "MBWT": _get_marker_xyz(df, "MBWT"),
        "MFWT": _get_marker_xyz(df, "MFWT"),
        # torso
        "T10": _get_marker_xyz(df, "T10") if "T10_x" in df.columns else None,
        "STRN": _get_marker_xyz(df, "STRN") if "STRN_x" in df.columns else None,
        "CLAV": _get_marker_xyz(df, "CLAV") if "CLAV_x" in df.columns else None,
        # legs
        "LTHI": _get_marker_xyz(df, "LTHI"),
        "RTHI": _get_marker_xyz(df, "RTHI"),
        "LKNE": _get_marker_xyz(df, "LKNE"),
        "RKNE": _get_marker_xyz(df, "RKNE"),
        "LANK": _get_marker_xyz(df, "LANK"),
        "RANK": _get_marker_xyz(df, "RANK"),
        "LHEE": _get_marker_xyz(df, "LHEE") if "LHEE_x" in df.columns else None,
        "RHEE": _get_marker_xyz(df, "RHEE") if "RHEE_x" in df.columns else None,
        # arms
        "LFSH": _get_marker_xyz(df, "LFSH"),
        "RFSH": _get_marker_xyz(df, "RFSH"),
        "LELB": _get_marker_xyz(df, "LELB"),
        "RELB": _get_marker_xyz(df, "RELB"),
        "LIWR": _get_marker_xyz(df, "LIWR") if "LIWR_x" in df.columns else None,
        "RIWR": _get_marker_xyz(df, "RIWR") if "RIWR_x" in df.columns else None,
        "LUPA": _get_marker_xyz(df, "LUPA") if "LUPA_x" in df.columns else None,
        "RUPA": _get_marker_xyz(df, "RUPA") if "RUPA_x" in df.columns else None,
        "LSHN": _get_marker_xyz(df, "LSHN") if "LSHN_x" in df.columns else None,
        "RSHN": _get_marker_xyz(df, "RSHN") if "RSHN_x" in df.columns else None,
        "LTOE": _get_marker_xyz(df, "LTOE") if "LTOE_x" in df.columns else None,
        "RTOE": _get_marker_xyz(df, "RTOE") if "RTOE_x" in df.columns else None,
        "LMT1": _get_marker_xyz(df, "LMT1") if "LMT1_x" in df.columns else None,
        "RMT1": _get_marker_xyz(df, "RMT1") if "RMT1_x" in df.columns else None,
        "LMT5": _get_marker_xyz(df, "LMT5") if "LMT5_x" in df.columns else None,
        "RMT5": _get_marker_xyz(df, "RMT5") if "RMT5_x" in df.columns else None,
        "C7": _get_marker_xyz(df, "C7") if "C7_x" in df.columns else None,
    }

    if lr_swap:
        # Swap L/R labels to match the normal pipeline (MoSh++ anatomical convention).
        marker_names = {_swap_lr_name(k): v for k, v in marker_names_raw.items()}
    else:
        marker_names = marker_names_raw

    # Choose best available for spine3.
    torso_candidates = {k: v for k, v in marker_names.items() if k in {"T10", "CLAV", "STRN"} and v is not None}
    if torso_candidates:
        spine3 = _mean_of_available(torso_candidates)
    else:
        spine3 = _mean_of_available({k: v for k, v in marker_names.items() if k in {"LFWT", "RFWT", "MBWT", "MFWT"}})

    pelvis = _mean_of_available({k: v for k, v in marker_names.items() if k in {"LFWT", "RFWT", "MBWT", "MFWT"}})
    left_hip = marker_names["LTHI"]
    right_hip = marker_names["RTHI"]
    left_knee = marker_names["LKNE"]
    right_knee = marker_names["RKNE"]

    # We'll compute foot centers after we apply axis conversions, so we can use
    # all available foot markers consistently (ankle/heel/toe/metatarsals).
    left_foot = marker_names["LANK"]
    right_foot = marker_names["RANK"]

    left_shoulder = marker_names["LFSH"]
    right_shoulder = marker_names["RFSH"]
    left_elbow = marker_names["LELB"]
    right_elbow = marker_names["RELB"]
    left_wrist = marker_names.get("LIWR")
    right_wrist = marker_names.get("RIWR")
    left_shank = marker_names.get("LSHN")
    right_shank = marker_names.get("RSHN")
    left_heel = marker_names.get("LHEE")
    right_heel = marker_names.get("RHEE")
    left_toe = marker_names.get("LTOE")
    right_toe = marker_names.get("RTOE")
    left_mt1 = marker_names.get("LMT1")
    right_mt1 = marker_names.get("RMT1")
    left_mt5 = marker_names.get("LMT5")
    right_mt5 = marker_names.get("RMT5")
    c7 = marker_names.get("C7")

    # Apply unit scaling and the same axis choices as the normal pipeline:
    # - Ry(-90°) in Y-up space to map Motive facing +X -> SMPL-X facing +Z
    # - then convert Y-up -> Z-up with the GMR rotation matrix.
    def prep(p: np.ndarray) -> np.ndarray:
        p = (p * scale).astype(np.float32)
        if apply_mosh_rotate:
            p = _ry_minus_90(p)
        if apply_y_up_to_z_up:
            p = _y_up_to_z_up(p)
        if yaw180:
            p = _rz_pi(p)
        return p

    if forward_from == "mb_to_mf":
        MB = marker_names["MBWT"] * scale
        MF = marker_names["MFWT"] * scale
    elif forward_from == "mf_to_mb":
        MB = marker_names["MFWT"] * scale
        MF = marker_names["MBWT"] * scale
    else:
        raise ValueError(f"Unsupported forward_from: {forward_from}")

    if apply_mosh_rotate:
        MB = _ry_minus_90(MB)
        MF = _ry_minus_90(MF)
    if apply_y_up_to_z_up:
        MB = _y_up_to_z_up(MB)
        MF = _y_up_to_z_up(MF)
    if yaw180:
        MB = _rz_pi(MB)
        MF = _rz_pi(MF)

    pelvis_quat = _compute_pelvis_quat_from_waist_markers_wxyz(
        marker_names["LFWT"] * scale,
        marker_names["RFWT"] * scale,
        MB,
        MF,
        y_up_to_z_up=apply_y_up_to_z_up,
    )
    if yaw180:
        from scipy.spatial.transform import Rotation as Rot
        q = pelvis_quat.reshape(-1, 4)
        r = Rot.from_quat(q[:, [1, 2, 3, 0]])  # wxyz -> xyzw
        rz = Rot.from_euler("z", np.pi)
        r2 = rz * r
        q2 = r2.as_quat()[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        pelvis_quat = q2.reshape(pelvis_quat.shape).astype(np.float32)

    pelvis = prep(pelvis)
    spine3 = prep(spine3)
    left_hip = prep(left_hip)
    right_hip = prep(right_hip)
    left_knee = prep(left_knee)
    right_knee = prep(right_knee)
    left_ankle = prep(left_foot)
    right_ankle = prep(right_foot)
    left_shoulder = prep(left_shoulder)
    right_shoulder = prep(right_shoulder)
    left_elbow = prep(left_elbow)
    right_elbow = prep(right_elbow)
    if left_wrist is not None:
        left_wrist = prep(left_wrist)
    if right_wrist is not None:
        right_wrist = prep(right_wrist)
    if left_shank is not None:
        left_shank = prep(left_shank)
    if right_shank is not None:
        right_shank = prep(right_shank)
    if left_toe is not None:
        left_toe = prep(left_toe)
    if right_toe is not None:
        right_toe = prep(right_toe)
    if left_heel is not None:
        left_heel = prep(left_heel)
    if right_heel is not None:
        right_heel = prep(right_heel)
    if left_mt1 is not None:
        left_mt1 = prep(left_mt1)
    if right_mt1 is not None:
        right_mt1 = prep(right_mt1)
    if left_mt5 is not None:
        left_mt5 = prep(left_mt5)
    if right_mt5 is not None:
        right_mt5 = prep(right_mt5)
    if c7 is not None:
        c7 = prep(c7)

    # Foot center targets: use as many foot markers as available.
    # This tends to stabilize foot placement vs using ankle/heel only.
    left_foot = _mean_of_available(
        {
            "ankle": left_ankle,
            "heel": left_heel,
            "toe": left_toe,
            "mt1": left_mt1,
            "mt5": left_mt5,
        }
    )
    right_foot = _mean_of_available(
        {
            "ankle": right_ankle,
            "heel": right_heel,
            "toe": right_toe,
            "mt1": right_mt1,
            "mt5": right_mt5,
        }
    )

    # Per-frame dict expected by GMR: body_name -> (pos(3,), quat_wxyz(4,))
    q = _quat_identity_wxyz()

    orient_mode = str(orient_mode)
    if orient_mode not in {"none", "pelvis", "feet", "segments"}:
        raise ValueError(f"Unsupported orient_mode: {orient_mode}")

    torso_quat = None
    left_thigh_quat = right_thigh_quat = None
    left_shank_quat = right_shank_quat = None
    left_foot_quat = right_foot_quat = None
    left_upperarm_quat = right_upperarm_quat = None
    left_forearm_quat = right_forearm_quat = None

    # Foot orientations can be estimated robustly from heel/toe even when we
    # don't trust the full-body segment orientation inference.
    waist_x = _safe_normalize(left_hip - right_hip)

    def foot_quat(ankle: np.ndarray, heel: np.ndarray, toe: np.ndarray, side_x: np.ndarray) -> np.ndarray:
        fwd = _safe_normalize(toe - heel)
        if toe_flip:
            fwd = -fwd
        up = _safe_normalize(ankle - heel)
        y = _safe_normalize(fwd)
        z = _safe_normalize(up)
        x = _safe_normalize(side_x)
        frame = _frame_from_axes(x, y, z)
        return _rotmat_to_quat_wxyz(frame)

    if left_toe is not None and left_heel is not None:
        left_foot_quat = foot_quat(left_ankle, left_heel, left_toe, waist_x)
    else:
        left_foot_quat = None
    if right_toe is not None and right_heel is not None:
        right_foot_quat = foot_quat(right_ankle, right_heel, right_toe, -waist_x)
    else:
        right_foot_quat = None

    if orient_mode == "segments":
        # Use additional markers to estimate orientations for major segments.
        torso_up = _safe_normalize(spine3 - pelvis)
        torso_y = _safe_normalize(np.cross(torso_up, waist_x))  # forward-ish
        torso_frame = _frame_from_axes(waist_x, torso_y, torso_up)
        torso_quat = _rotmat_to_quat_wxyz(torso_frame)

        def limb_quat(prox: np.ndarray, dist: np.ndarray, side_x: np.ndarray) -> np.ndarray:
            up = _safe_normalize(prox - dist)
            y = _safe_normalize(np.cross(up, side_x))
            frame = _frame_from_axes(side_x, y, up)
            return _rotmat_to_quat_wxyz(frame)

        left_thigh_quat = limb_quat(left_hip, left_knee, waist_x)
        right_thigh_quat = limb_quat(right_hip, right_knee, -waist_x)

        left_shank_quat = limb_quat(left_knee, left_foot, waist_x)
        right_shank_quat = limb_quat(right_knee, right_foot, -waist_x)

        # Prefer heel/toe-derived foot orientation if available; otherwise fall back.
        if left_foot_quat is None:
            left_foot_quat = left_shank_quat
        if right_foot_quat is None:
            right_foot_quat = right_shank_quat

        left_upperarm_quat = limb_quat(left_shoulder, left_elbow, waist_x)
        right_upperarm_quat = limb_quat(right_shoulder, right_elbow, -waist_x)
        left_forearm_quat = limb_quat(left_elbow, left_wrist, waist_x) if left_wrist is not None else left_upperarm_quat
        right_forearm_quat = limb_quat(right_elbow, right_wrist, -waist_x) if right_wrist is not None else right_upperarm_quat
    frames = []
    for i in range(len(df)):
        pelvis_q_i = pelvis_quat[i]
        if orient_mode == "none":
            pelvis_q_i = q
        torso_q_i = pelvis_q_i if orient_mode != "segments" else torso_quat[i]
        lf_q = pelvis_q_i
        rf_q = pelvis_q_i
        if orient_mode in {"feet", "segments"} and left_foot_quat is not None and right_foot_quat is not None:
            lf_q = left_foot_quat[i]
            rf_q = right_foot_quat[i]
        frames.append(
            {
                "pelvis": (pelvis[i], pelvis_q_i),
                "spine3": (spine3[i], torso_q_i),
                "left_hip": (left_hip[i], pelvis_q_i if orient_mode != "segments" else left_thigh_quat[i]),
                "right_hip": (right_hip[i], pelvis_q_i if orient_mode != "segments" else right_thigh_quat[i]),
                "left_knee": (left_knee[i], pelvis_q_i if orient_mode != "segments" else left_shank_quat[i]),
                "right_knee": (right_knee[i], pelvis_q_i if orient_mode != "segments" else right_shank_quat[i]),
                "left_foot": (left_foot[i], lf_q),
                "right_foot": (right_foot[i], rf_q),
                "left_shoulder": (left_shoulder[i], pelvis_q_i if orient_mode != "segments" else left_upperarm_quat[i]),
                "right_shoulder": (right_shoulder[i], pelvis_q_i if orient_mode != "segments" else right_upperarm_quat[i]),
                "left_elbow": (left_elbow[i], pelvis_q_i if orient_mode != "segments" else left_forearm_quat[i]),
                "right_elbow": (right_elbow[i], pelvis_q_i if orient_mode != "segments" else right_forearm_quat[i]),
            }
        )

    return frames, fps


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TEST: retarget directly from 39-marker CSV to robot using GMR, "
            "skipping MoSh++/SMPL-X fitting. This uses sparse markers to "
            "approximate SMPL-X joint frames."
        )
    )
    parser.add_argument("--csv", required=True, help="Path to *_filled39.csv (recommended).")
    parser.add_argument("--robot", default="booster_t1", help="Target robot type (e.g. booster_t1).")
    parser.add_argument("--out", default=None, help="Output robot motion pkl path.")
    parser.add_argument("--units", choices=["m", "mm", "auto"], default="auto", help="Units of marker coordinates in CSV.")
    parser.add_argument(
        "--forward-from",
        choices=["mb_to_mf", "mf_to_mb"],
        default="mb_to_mf",
        help="How to infer forward direction from waist markers.",
    )
    parser.add_argument(
        "--yaw180",
        action="store_true",
        help="Apply an additional 180° yaw rotation (fixes front/back flip).",
    )
    parser.add_argument(
        "--no-lr-swap",
        action="store_true",
        help="Disable the L/R label swap used in the normal MoSh++ pipeline.",
    )
    parser.add_argument(
        "--orient-mode",
        choices=["none", "pelvis", "feet", "segments"],
        default="segments",
        help="How to infer joint orientations from markers (test).",
    )
    parser.add_argument(
        "--toe-flip",
        action="store_true",
        help="Flip toe forward direction when computing foot orientation.",
    )
    parser.add_argument(
        "--show-human-names",
        action="store_true",
        help="Show human target frame labels in the viewer.",
    )
    parser.add_argument(
        "--ik-profile",
        choices=["default", "csv_pos", "csv_pos_feetrot"],
        default="csv_pos",
        help="IK config profile (csv_pos is position-heavy for marker-only input).",
    )
    parser.add_argument(
        "--no-mosh-rotate",
        action="store_true",
        help="Disable the normal MoSh++ rotate [0, -90, 0] step (Ry(-90°)).",
    )
    parser.add_argument(
        "--no-yup-to-zup",
        action="store_true",
        help="Disable Y-up -> Z-up conversion used by GMR (not recommended).",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Do not open the MuJoCo viewer; just write the motion file.",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        help="Limit viewer rate to match motion FPS.",
    )
    parser.add_argument(
        "--camera-azimuth",
        type=float,
        default=180.0,
        help="MuJoCo camera azimuth in degrees (default: 180 = front view).",
    )
    args = parser.parse_args()

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    frames, fps = build_smplx_like_frames_from_markers(
        df,
        units=args.units,
        apply_mosh_rotate=not args.no_mosh_rotate,
        apply_y_up_to_z_up=not args.no_yup_to_zup,
        forward_from=args.forward_from,
        yaw180=args.yaw180,
        lr_swap=not args.no_lr_swap,
        orient_mode=args.orient_mode,
        toe_flip=args.toe_flip,
    )

    # Import here so this script can be inspected without the full env.
    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer
    import general_motion_retargeting.params as gmr_params

    if args.robot == "booster_t1" and args.ik_profile in {"csv_pos", "csv_pos_feetrot"}:
        cfg_name = "smplx_to_t1_csv_pos.json" if args.ik_profile == "csv_pos" else "smplx_to_t1_csv_pos_feetrot.json"
        cfg_path = pathlib.Path(__file__).resolve().parent / "csv_to_gmr_ik" / cfg_name
        gmr_params.IK_CONFIG_DICT["smplx"]["booster_t1"] = cfg_path

    retarget = GMR(
        src_human="smplx",
        tgt_robot=args.robot,
        actual_human_height=None,
        verbose=False,
    )

    if args.out is None:
        out_path = csv_path.with_suffix("").with_name(f"{csv_path.stem}__csv_to_gmr__{args.robot}.pkl")
    else:
        out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    viewer = None
    if not args.no_viewer:
        viewer = RobotMotionViewer(robot_type=args.robot, motion_fps=fps, camera_follow=False)
        try:
            viewer.viewer.cam.azimuth = float(args.camera_azimuth)
        except Exception:
            pass

    qpos_list = []
    for frame in frames:
        qpos = retarget.retarget(frame, offset_to_ground=True)
        qpos_list.append(qpos)
        if viewer is not None:
            viewer.step(
                root_pos=qpos[:3],
                root_rot=qpos[3:7],
                dof_pos=qpos[7:],
                human_motion_data=retarget.scaled_human_data,
                show_human_body_name=args.show_human_names,
                rate_limit=args.rate_limit,
                follow_camera=True,
            )

    if viewer is not None:
        viewer.close()

    qpos_arr = np.asarray(qpos_list, dtype=np.float32)
    root_pos = qpos_arr[:, :3]
    root_rot_wxyz = qpos_arr[:, 3:7]
    root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
    dof_pos = qpos_arr[:, 7:]

    motion_data = {
        "fps": float(fps),
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with out_path.open("wb") as f:
        pickle.dump(motion_data, f)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

