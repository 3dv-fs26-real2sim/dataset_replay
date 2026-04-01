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

# Home wrist target pose for IK.
# position  = [0.40, 0.0, 0.3]
# rotation  = x=[1,0,0], z=[0,0,-1]  →  R = [[1,0,0],[0,-1,0],[0,0,-1]]
WRIST_HOME_POSITION = np.array([0.40, 0.0, 0.3])
WRIST_HOME_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=float)

HOME_HOLD_SECONDS = 3.0

# H5 data uses a tool-frame convention where identity = hand pointing down.
# The URDF fer_link7/8 frame has Rx(180°) when the hand points down.
# Pre-multiply by Rx(180°) to convert from tool convention to URDF convention.
Q_TOOL_TO_URDF = np.array([0.0, 1.0, 0.0, 0.0])  # Rx(180°) in wxyz
