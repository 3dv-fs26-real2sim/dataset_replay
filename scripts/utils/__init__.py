"""Utility package for dataset replay scripts.

Only re-exports that are safe before SimulationApp is created (no Isaac Sim
deps). Isaac-Sim-using modules — robot, ik, capture, object, scene, camera,
textures, apriltag — must be imported explicitly by entry scripts AFTER
SimulationApp has been instantiated.
"""

from .config import (
    SceneConfig,
    TableConfig,
    WallsConfig,
    AprilTagConfig,
    RobotMountConfig,
    CameraConfig,
)
from .constants import (
    ROBOT_PRIM_PATH,
    ROBOT_BASE_PRIM_PATH,
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    N_ARM_DOFS,
    N_ARM_POSE_DIMS,
    N_HAND_DOFS,
)
from .h5_loader import H5Reader, peek_schema
