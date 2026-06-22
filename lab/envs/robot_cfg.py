"""ArticulationCfg for the Franka Emika Panda + 17-DOF OrcaHand.

Points at ``dataset_replay/assets/pandaorca_right.usd`` (the same flat USD the
kinematic-replay scripts reference) and names its DOFs with the canonical
``panda_joint1..7`` / ``right_*`` order from :mod:`utils.constants`.

Imports Isaac Sim — must be imported only after ``SimulationApp`` is created.

Mount math
----------
``panda_link0`` sits at the USD ``/Root`` origin (no internal offset), so
setting ``init_state.pos`` on the ``{ENV_REGEX_NS}/Robot`` wrapper Xform places
``panda_link0`` exactly there. The env cfg sets this to ``mount_xyz`` in
``__post_init__``; the value baked here is only a transient warm-start.

Actuators
---------
Arm gravity is disabled: the deterministic PD tracks the recorded ``arm_qpos``
without gravity sag (matching the kinematic-replay intent); the manipulated
object keeps its own gravity. Arm PD 400/40, hand PD 300/20 — the gains the
upstream port verified for a stable open-loop grasp.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from utils.config import ASSETS

ROBOT_USD = ASSETS / "pandaorca_right.usd"

# Transient warm-start arm pose (the reset event snaps to demo frame 0 at once).
_ARM_HOME_JOINT_POS = {
    "panda_joint1":  1.033395,
    "panda_joint2": -0.528522,
    "panda_joint3": -1.255232,
    "panda_joint4": -2.385702,
    "panda_joint5": -0.596258,
    "panda_joint6":  2.118776,
    "panda_joint7": -1.406489,
}
# OrcaHand opens at qpos=0 across all 17 DOFs (right_wrist + 16 finger joints).
_HAND_HOME_JOINT_POS = {"right_.*": 0.0}


ORCA_FRANKA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD.resolve()),
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,            # PD tracks recorded arm_qpos w/o gravity sag
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,   # firmer/steadier grasp
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),                 # overridden to mount_xyz in env __post_init__
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={**_ARM_HOME_JOINT_POS, **_HAND_HOME_JOINT_POS},
    ),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=400.0, stiffness=400.0, damping=40.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=400.0, stiffness=400.0, damping=40.0,
        ),
        "orca_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_.*"],
            stiffness=300.0, damping=20.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Franka Emika Panda + 17-DOF OrcaHand articulation (right hand)."""
