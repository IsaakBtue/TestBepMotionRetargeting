"""Shared settings so G1 and Booster T1 goalkeeper viewers look/behave the same in MuJoCo."""

from __future__ import annotations

from pathlib import Path

BEP_ROOT = Path(__file__).resolve().parents[1]

# Same MJCF Booster uses for playback; we copy its MjVisual onto G1 URDF loads (headlight / haze / globals).
BOOSTER_T1_VISUAL_REFERENCE_XML = BEP_ROOT / "src/retargeting/assets/booster_t1/T1_serial.xml"

# kwargs passed to general_motion_retargeting.RobotMotionViewer for both robots
ROBOT_MOTION_VIEWER_KWARGS = {
    "camera_follow": True,
    "transparent_robot": 0,
    "record_video": False,
    "video_path": None,
}
