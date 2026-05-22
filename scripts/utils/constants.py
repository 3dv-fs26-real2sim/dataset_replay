"""Robot kinematic constants (slim, single-arm, Aria camera).

These values are intrinsic to the robot model — joint names, home pose, the
EE-wrist offset, the H5 tool-frame quaternion convention. Scene parameters
(table, camera mount pose) live in :mod:`config`.

The Aria-camera constants (intrinsics + a robot-base-relative extrinsic)
also live here because they describe the *physical* recording rig rather
than anything procedural the scene builder picks: the Aria sat in a fixed
place on the wearer's head during the egoverse recordings, so its pose
relative to the right-arm base is the same across sessions until a new
calibration changes it.

No Isaac Sim imports — safe to import before SimulationApp is created.
"""

from pathlib import Path

import numpy as np

# ── Output / asset anchors (kept here for compat with utils/capture.py) ──────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # dataset_replay/
OUTPUT_DIR   = PROJECT_ROOT / "outputs"


# ── Capture rate ──────────────────────────────────────────────────────────────
# Frame rate at which the egoverse H5 datasets are recorded. All replay /
# capture scripts default to this so MP4 outputs play back at the same
# speed as the source recording. Override via the shared ``--fps`` CLI
# flag (added by ``utils.app.add_common_args``) if you point a script at
# an H5 captured at a different rate.
#
# When ``--sample-every N`` is also passed, the writer's effective fps is
# ``H5_DEFAULT_FPS / N``: e.g. ``--sample-every 5`` on 50 Hz data gives
# a 10 Hz output video (recording every fifth simulated frame).
H5_DEFAULT_FPS = 50.0   # Hz — H5 capture / MP4 output frame rate.


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

# Offset from panda_link8 to the recorded "EE wrist" reference point, in
# panda_link8's local frame. IK aims panda_link8 at
# ``target_pos - R_link8 @ EE_WRIST_OFFSET_IN_LINK8``.
#
# Value [0.13, 0, 0.07] is the *teleop EE-wrist* convention shared with
# franka_teleop and the IsaacLab env cfg (teleop_manip_env_cfg.py:61).
# It sits ~3 cm forward of the URDF ``right_palm`` link, which composes
# from the link8→...→right_palm chain to [0.098, -0.003, 0.067]. The two
# differ because they reference different points on the hand; the teleop
# value is what the H5 was recorded against, so we use it here.
EE_WRIST_OFFSET_IN_LINK8 = np.array([0.13, 0.0, 0.07])

# Home wrist target pose for IK.
WRIST_HOME_POSITION = np.array([0.40, 0.0, 0.3])
WRIST_HOME_ROTATION = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1],
], dtype=float)

# ── H5 → URDF quaternion frame change ────────────────────────────────────────
# The H5 records wrist quaternions in Rokoko's left-handed sensor frame
# (their X-axis aligns with our URDF Z-axis). Three independent conventions
# differ between recording and replay; we fix all three in tool_quat_to_urdf:
#
#   1. Handedness   (LH → RH)        — negate x, z components
#   2. Axis labels  (their X ↔ our Z) — swap x, z components
#   3. Tool identity (down → URDF)   — pre-multiply Rx(180°)
#
# The three operations are orthogonal: handedness only fixes chirality, the
# axis swap only relabels, and Rx(180°) only flips "identity = hand-down"
# (recording) into "identity = Franka flange default" (URDF). Removing any
# one of them breaks the others; in particular, replacing Rx(180°) with
# Rz(180°) to "mirror" the teleop's rokoko_ingress.py:191 rotation is wrong
# — that Rz(180°) is upstream of the H5 and is already baked in.
Q_TOOL_TO_URDF = np.array([0.0, 1.0, 0.0, 0.0])  # Rx(180°) — tool-identity flip

# Which components of the H5 quaternion to negate BEFORE the axis swap +
# Rx(180°) premultiplication. Egoverse uses the same Rokoko-based recording
# path as maple, so the operational value is ``"negxz"`` — the handedness
# flip for Rokoko-LH → URDF-RH. Other values are diagnostic knobs for
# sign-ambiguity testing (Hamilton vs JPL, active vs passive).
#
#   "baseline"  no negation                   [w,  x,  y,  z]
#   "negx"      negate x                      [w, -x,  y,  z]
#   "negy"      negate y                      [w,  x, -y,  z]
#   "negz"      negate z                      [w,  x,  y, -z]
#   "negxy"     negate x, y    (= conj Rz)    [w, -x, -y,  z]
#   "negxz"     negate x, z    (= conj Ry)    [w, -x,  y, -z]   ← operational (egoverse / maple)
#   "negyz"     negate y, z    (= conj Rx)    [w,  x, -y, -z]
#   "conjugate" negate x, y, z (= inverse)    [w, -x, -y, -z]
TOOL_QUAT_NEGATE_PATTERN: str = "negxz"


# ── Aria camera intrinsics + extrinsic ────────────────────────────────────────
# These values were measured for the original main-branch capture rig and
# are reused verbatim for egoverse, which shares the same Aria glasses and
# recording pose. The 4×4 ``ARIA_EXTRINSICS_RIGHT`` is the camera-in-base
# transform — i.e., ``T_world_cam = T_world_base @ ARIA_EXTRINSICS_RIGHT``
# where ``T_world_base`` is the world pose of ``panda_link0``. Because the
# physical robot sits in the same place relative to the wearer in egoverse
# as in main, the base-relative camera transform carries over without
# re-measurement; only the absolute table-top Z differs (egoverse uses
# 0.75 m vs main's 1.0 m), which the world composition absorbs.
ARIA_INTRINSICS = {
    "width":  640,
    "height": 480,
    "fx": 266.50860444,   # = 133.25430222 × 2 (factory half-res × 2)
    "fy": 266.50860444,
    "cx": 320.0,
    "cy": 240.0,
}

ARIA_EXTRINSICS_RIGHT = np.array([
    [ 0.02933941, -0.83227828,  0.55358113,  0.17515134],
    [-0.99642232,  0.01956109,  0.0822187 ,  0.34649483],
    [-0.07925749, -0.55401284, -0.82872675,  0.46895363],
    [ 0.        ,  0.        ,  0.        ,  1.        ],
])


# ── Table-edge geometry for desk-based extrinsic refinement ───────────────────
# The refiner (``utils.calibrate_table.refine_aria_extrinsic``) aligns 3D
# table edges projected through the current camera to 2D SAM-mask lines
# extracted from the recorded video. The three line segments below must match the
# parametric ``TableConfig`` defaults (1.0 × 1.4 m centred at origin, top
# at z = 0.75). If you change ``TableConfig`` and want desk refinement to
# stay valid, rebuild these from the config — but for the default scene
# the hard-coded constants are correct and avoid pulling SceneConfig into
# the refiner's pure-CV path.
TABLE_TOP_Z  = 0.75
TABLE_X_HALF = 0.5
TABLE_Y_HALF = 0.7

# Far (back) edge of the combined table (at +X), running along Y.
TABLE_TOP_EDGE_WORLD  = np.array([[+TABLE_X_HALF, -TABLE_Y_HALF, TABLE_TOP_Z],
                                  [+TABLE_X_HALF, +TABLE_Y_HALF, TABLE_TOP_Z]])
# Left edge (at +Y), running along X.
TABLE_LEFT_EDGE_WORLD = np.array([[-TABLE_X_HALF, +TABLE_Y_HALF, TABLE_TOP_Z],
                                  [+TABLE_X_HALF, +TABLE_Y_HALF, TABLE_TOP_Z]])
# Physical seam between the two table cells, running along X at y = 0.
TABLE_SEAM_WORLD      = np.array([[-TABLE_X_HALF, 0.0,           TABLE_TOP_Z],
                                  [+TABLE_X_HALF, 0.0,           TABLE_TOP_Z]])
