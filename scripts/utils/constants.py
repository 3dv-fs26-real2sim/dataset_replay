"""Robot kinematic constants (slim, single-arm).

These values are intrinsic to the robot model — joint names, home pose, the
EE-wrist offset, the H5 tool-frame quaternion convention. Scene parameters
(table, walls, AprilTag, camera, mount pose) live in :mod:`config`.

No Isaac Sim imports — safe to import before SimulationApp is created.
"""

from pathlib import Path

import numpy as np

# ── Output / asset anchors (kept here for compat with utils/capture.py) ──────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # dataset_replay/
OUTPUT_DIR   = PROJECT_ROOT / "outputs"


# ── Capture rate ──────────────────────────────────────────────────────────────
# Frame rate at which the current generation of H5 datasets is recorded.
# All replay / capture scripts default to this so MP4 outputs play back at
# the same speed as the source recording. Older 50 Hz datasets exist (e.g.
# the `*_50hz` files referenced in README §1); override via the shared
# ``--fps`` CLI flag (added by ``utils.app.add_common_args``) when you
# replay one of those.
H5_DEFAULT_FPS = 10.0   # Hz — H5 capture / MP4 output frame rate.


# ── Prim paths ────────────────────────────────────────────────────────────────
# Wrapper Xform that references the orcav1b USD's /Root. PhysicsArticulationRootAPI
# is composed in via the reference, so SingleArticulation should be created
# against this exact path.
ROBOT_PRIM_PATH = "/World/Robot"

# Visible robot base — the link the OAK-D extrinsic was calibrated against
# (and what we anchor object/camera math at if needed).
ROBOT_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/panda_link0"


# ── DOF counts ────────────────────────────────────────────────────────────────
N_ARM_DOFS      = 7
N_ARM_POSE_DIMS = 7   # 3 position (xyz) + 4 quaternion (wxyz)
N_HAND_DOFS     = 17


# ── Joint names ───────────────────────────────────────────────────────────────
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

# Right-hand 17-DOF OrcaHand chain — names match the orcav1b USD exactly.
HAND_JOINT_NAMES = [
    "right_wrist",
    "right_thumb_mcp",  "right_thumb_abd",  "right_thumb_pip",  "right_thumb_dip",
    "right_index_abd",  "right_index_mcp",  "right_index_pip",
    "right_middle_abd", "right_middle_mcp", "right_middle_pip",
    "right_ring_abd",   "right_ring_mcp",   "right_ring_pip",
    "right_pinky_abd",  "right_pinky_mcp",  "right_pinky_pip",
]


# ── Initial / home joint values ───────────────────────────────────────────────
HAND_HOME_JOINT_VALUES = np.zeros(N_HAND_DOFS)


# ── IK configuration ──────────────────────────────────────────────────────────
EE_FRAME_NAME = "panda_link8"

# The EE wrist (as recorded in H5 data) is offset from panda_link8 in
# panda_link8's local frame. To position the EE wrist at a target, the IK
# must aim panda_link8 at: target_pos - R_link8 @ EE_WRIST_OFFSET_IN_LINK8.
# Values provided by supervisors from the OrcaHand Wiki.
EE_WRIST_OFFSET_IN_LINK8 = np.array([0.13, 0.0, 0.07])

# Home wrist target pose for IK.
WRIST_HOME_POSITION = np.array([0.40, 0.0, 0.3])
WRIST_HOME_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=float)

# H5 data uses a tool-frame convention where identity = hand pointing down.
# The URDF panda_link7/8 frame has Rx(180°) when the hand points down.
# Pre-multiply by Rx(180°) (in wxyz: [0, 1, 0, 0]) to convert tool→URDF.
Q_TOOL_TO_URDF = np.array([0.0, 1.0, 0.0, 0.0])
