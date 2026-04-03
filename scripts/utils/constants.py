"""Shared constants for the dataset replay scripts.

No Isaac Sim imports — safe to import before SimulationApp is created.
"""

from pathlib import Path

import numpy as np

# ── Path anchors ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]          # dataset_replay/
DESCRIPTION_ROOT = PROJECT_ROOT.parent / "pandaorca_description"

# USD scenes
USD_PATH_SINGLE = DESCRIPTION_ROOT / "usd" / "pandaorca_single.usd"
USD_PATH_DUAL   = DESCRIPTION_ROOT / "usd" / "pandaorca_dual.usd"

# H5 data (default files per mode)
H5_PATH_SINGLE = PROJECT_ROOT / "data" / "20250804_104715.h5"
H5_PATH_DUAL   = PROJECT_ROOT / "data" / "20250829_180500.h5"

# Lula IK assets
LULA_DESCRIPTOR_PATH = DESCRIPTION_ROOT / "lula" / "fer_robot_descriptor.yaml"
URDF_PATH_LEFT  = DESCRIPTION_ROOT / "urdf" / "fer_orcahand_left_extended.urdf"
URDF_PATH_RIGHT = DESCRIPTION_ROOT / "urdf" / "fer_orcahand_right_extended.urdf"

# Object meshes
OBJECTS_DIR = PROJECT_ROOT / "objects"

# Video output
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ── Prim paths in the USD stage ───────────────────────────────────────────────
FRANKA_LEFT_PATH  = "/World/fer_orcahand_left_extended"
FRANKA_RIGHT_PATH = "/World/fer_orcahand_right_extended"

# ── DOF counts ────────────────────────────────────────────────────────────────
N_ARM_DOFS      = 7
N_ARM_POSE_DIMS = 7   # 3 position (xyz) + 4 quaternion (wxyz)
N_HAND_DOFS     = 17

# ── Joint names ───────────────────────────────────────────────────────────────
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

HAND_LEFT_JOINT_NAMES = [
    "left_wrist",
    "left_thumb_mcp",  "left_thumb_abd",  "left_thumb_pip",  "left_thumb_dip",
    "left_index_abd",  "left_index_mcp",  "left_index_pip",
    "left_middle_abd", "left_middle_mcp", "left_middle_pip",
    "left_ring_abd",   "left_ring_mcp",   "left_ring_pip",
    "left_pinky_abd",  "left_pinky_mcp",  "left_pinky_pip",
]

HAND_RIGHT_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp",  "right_thumb_abd",  "right_thumb_pip",  "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]

# ── Initial / home joint values ──────────────────────────────────────────────
ARM_JOINT_VALUES_INITIAL = np.array(
    [0.184280, -0.225290, -0.193213, -2.701355, -0.069782, 2.478938, 0.050881]
)
HAND_JOINT_VALUES_INITIAL = np.array([0.0] * N_HAND_DOFS)
HAND_HOME_JOINT_VALUES    = np.array([0.0] * N_HAND_DOFS)

# ── IK configuration ─────────────────────────────────────────────────────────
EE_FRAME_NAME_LEFT  = "fer_link8"
EE_FRAME_NAME_RIGHT = "fer_link8"

# The EE wrist (as recorded in H5 data) is offset from fer_link8 in fer_link8's
# local frame.  To position the EE wrist at a target, the IK must aim fer_link8
# at: target_pos - R_fer_link8 @ EE_WRIST_OFFSET_IN_LINK8.
EE_WRIST_OFFSET_IN_LINK8 = np.array([0.13, 0.0, 0.07])

# Home wrist target pose for IK.
# position  = [0.40, 0.0, 0.3]
# rotation  = x=[1,0,0], z=[0,0,-1]  →  R = [[1,0,0],[0,-1,0],[0,0,-1]]
WRIST_HOME_POSITION = np.array([0.40, 0.0, 0.3])
WRIST_HOME_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=float)

HOME_HOLD_SECONDS = 0.0

# H5 data uses a tool-frame convention where identity = hand pointing down.
# The URDF fer_link7/8 frame has Rx(180°) when the hand points down.
# Pre-multiply by Rx(180°) to convert from tool convention to URDF convention.
Q_TOOL_TO_URDF = np.array([0.0, 1.0, 0.0, 0.0])  # Rx(180°) in wxyz

# ── Camera calibration ───────────────────────────────────────────────────────
# Extrinsics: T_base_from_cam (4x4 homogeneous, transforms cam-frame → base-frame).
# T[:3, 3] = camera origin in base coordinates.
# To get camera world pose: T_world_cam = T_world_base @ T_base_from_cam

ARIA_EXTRINSICS = {
    "left": np.array([[-0.02199727, -0.80581615,  0.59175708,  0.20403467],
                      [-0.99905014,  0.03998766,  0.01731508, -0.25486327],
                      [-0.03761575, -0.59081411, -0.80593036,  0.43379187],
                      [ 0.        ,  0.        ,  0.        ,  1.        ]]),
    "right": np.array([[ 0.02933941, -0.83227828,  0.55358113,  0.17515134],
                       [-0.99642232,  0.01956109,  0.0822187 ,  0.34649483],
                       [-0.07925749, -0.55401284, -0.82872675,  0.46895363],
                       [ 0.        ,  0.        ,  0.        ,  1.        ]]),
}

# Aria Gen 1 intrinsics — full resolution (640×480).
# Derived from calibrated focal length 133.25430222 px at half-res,
# scaled ×2 for full-res.  cx/cy are principal point (image centre).
ARIA_INTRINSICS = {
    "width": 640,
    "height": 480,
    "fx": 133.25430222 * 2,   # 266.50860444
    "fy": 133.25430222 * 2,   # 266.50860444
    "cx": 320.0,
    "cy": 240.0,
}

# Aria Gen 1 intrinsics — half resolution (320×240).
ARIA_INTRINSICS_HALF = {
    "width": 320,
    "height": 240,
    "fx": 133.25430222,
    "fy": 133.25430222,
    "cx": 160.0,
    "cy": 120.0,
}

# Registry of camera configs.  Extend when adding new cameras (e.g., OAK-D).
CAMERA_CONFIGS = {
    "aria": {
        "extrinsics": ARIA_EXTRINSICS,
        "intrinsics": ARIA_INTRINSICS,
    },
    "aria_half": {
        "extrinsics": ARIA_EXTRINSICS,
        "intrinsics": ARIA_INTRINSICS_HALF,
    },
    # "oakd": { "extrinsics": OAKD_EXTRINSICS, "intrinsics": OAKD_INTRINSICS },
}

# ── H5 image datasets ──────────────────────────────────────────────────────
# Maps user-facing camera names to H5 dataset paths for image extraction.
H5_IMAGE_PATHS = {
    "aria": "observations/images/aria_rgb_cam/color",
    "oakd": "observations/images/oakd_front_view/color",
}

H5_DEFAULT_CAMERA = "aria"
