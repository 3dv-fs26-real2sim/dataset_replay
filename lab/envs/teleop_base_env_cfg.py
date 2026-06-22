"""Shared residual-RL teleop env config (dataset-agnostic base).

Assembles the ManagerBasedRLEnv that learns a residual on top of a recorded
per-frame joint baseline (see :mod:`lab`). The EgoVerse and MAPLE variants
subclass :class:`TeleopBaseEnvCfg` and only change the manipulated object, the
demo sample rate, and any dataset-specific props.

Scene geometry (table size, top height) comes from :class:`utils.config`. The
manipulated object reference, robot baseline, and reset all come from the demo
attached at ``env.demo``.

Imports Isaac Sim — import only after ``SimulationApp`` is created.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
from isaaclab.utils import configclass

from utils.config import TableConfig

from . import mdp
from .objects import duck_spawn_cfg, object_cfg
from .robot_cfg import ORCA_FRANKA_CFG

# Scene geometry shared with the kinematic-replay rig (1.0 × 1.4 m, top z=0.75).
_TABLE = TableConfig()
# Verified residual-RL mount for the robot base (panda_link0) world pose. The
# object/wrist demo and the arm baseline are all anchored to this single frame,
# so the whole rig translates together; the grasp geometry is mount-invariant.
_MOUNT_XYZ = (-0.28, -0.35, 0.75)
# Solid block whose TOP sits at table.top_z (matches kinematic_replay's table).
_TABLE_BLOCK_HEIGHT = 1.0


def _table_cfg() -> AssetBaseCfg:
    Lx, Ly = _TABLE.combined_size_xy
    Lz = _TABLE_BLOCK_HEIGHT
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(_TABLE.centre_xy[0], _TABLE.centre_xy[1], _TABLE.top_z - Lz / 2.0),
        ),
        spawn=CuboidCfg(
            size=(Lx, Ly, Lz),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.18, 0.18, 0.18), roughness=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )


# ── Scene ─────────────────────────────────────────────────────────────────────
@configclass
class TeleopSceneCfg(InteractiveSceneCfg):
    """Franka + OrcaHand + table + one dynamic manipulated object."""

    ground = AssetBaseCfg(prim_path="/World/GroundPlane",
                          init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
                          spawn=GroundPlaneCfg())
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight",
                              spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))

    robot = ORCA_FRANKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    table = _table_cfg()

    # Dynamic manipulated object (duck by default; MAPLE swaps in the pan). Its
    # initial pose is rewritten by the reset event from demo frame 0.
    object: RigidObjectCfg = object_cfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=duck_spawn_cfg(),
        init_pos=(0.0, 0.0, _TABLE.top_z + 0.10),
    )

    # Optional dynamic bowl container (filled by attach_bowl when a pose is given).
    bowl: RigidObjectCfg | None = None

    # Optional MAPLE static props (filled by MapleTeleopEnvCfg.__post_init__).
    prop_box: RigidObjectCfg | None = None
    prop_carpet: RigidObjectCfg | None = None
    prop_heater: RigidObjectCfg | None = None


# ── Actions ───────────────────────────────────────────────────────────────────
@configclass
class ActionsCfg:
    joint_targets = mdp.RecordedQposResidualActionCfg(
        asset_name="robot",
        arm_joint_names_expr="panda_joint.*",
        hand_joint_names_expr="right_.*",
        residual_scale=0.1,  # overridden from TeleopBaseEnvCfg.residual_scale
    )


# ── Observations ──────────────────────────────────────────────────────────────
_OBJ = {"object_cfg": SceneEntityCfg("object")}


@configclass
class ObservationsCfg:
    """Asymmetric actor (``policy``) + critic groups, each with reference look-ahead."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame,
                                  params={"robot_cfg": SceneEntityCfg("robot"), **_OBJ})
        actions = ObsTerm(func=mdp.last_action)
        # reference look-ahead (demo frame to reach next step)
        ref_object_pos_delta = ObsTerm(func=mdp.ref_object_pos_delta, params=_OBJ)
        ref_object_quat = ObsTerm(func=mdp.ref_object_quat, params=_OBJ)
        ref_object_quat_delta = ObsTerm(func=mdp.ref_object_quat_delta, params=_OBJ)
        ref_base_action = ObsTerm(func=mdp.ref_base_action)
        ref_dof_delta = ObsTerm(func=mdp.ref_dof_delta)

        enable_corruption = False
        concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame,
                                  params={"robot_cfg": SceneEntityCfg("robot"), **_OBJ})
        object_lin_vel = ObsTerm(func=mdp.object_lin_vel_w, params=_OBJ)
        object_ang_vel = ObsTerm(func=mdp.object_ang_vel_w, params=_OBJ)
        object_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("object")})
        actions = ObsTerm(func=mdp.last_action)
        ref_object_pos_delta = ObsTerm(func=mdp.ref_object_pos_delta, params=_OBJ)
        ref_object_quat = ObsTerm(func=mdp.ref_object_quat, params=_OBJ)
        ref_object_quat_delta = ObsTerm(func=mdp.ref_object_quat_delta, params=_OBJ)
        ref_base_action = ObsTerm(func=mdp.ref_base_action)
        ref_dof_delta = ObsTerm(func=mdp.ref_dof_delta)

        enable_corruption = False
        concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ── Rewards ───────────────────────────────────────────────────────────────────
_HAND = {"asset_cfg": SceneEntityCfg("robot", joint_names=["right_.*"])}
_LIFT = {"object_cfg": SceneEntityCfg("object"), "lift_on": 0.02, "k_lift": 30.0}


@configclass
class RewardsCfg:
    track_obj_pos = RewTerm(func=mdp.track_object_pos, weight=5.0, params={**_OBJ, "k": 80.0})
    track_obj_rot = RewTerm(func=mdp.track_object_rot, weight=1.0, params={**_OBJ, "k": 3.0})
    track_hand_joints = RewTerm(func=mdp.track_joint_pos, weight=2.0, params={**_HAND, "k": 5.0})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)

    inhand_stability = RewTerm(
        func=mdp.inhand_object_stability, weight=1.5,
        params={**_OBJ, "robot_cfg": SceneEntityCfg("robot"), "k_lin": 10.0, "k_ang": 1.0,
                "lift_on": 0.02, "k_lift": 30.0},
    )
    # Lift-gated effort/jerk penalties (only act once the object is held).
    true_action_rate = RewTerm(func=mdp.true_action_rate_l2, weight=-5e-3, params=_LIFT)
    hand_joint_vel = RewTerm(func=mdp.hand_joint_vel_l2, weight=-2e-4, params={**_HAND, **_LIFT})
    applied_torque = RewTerm(func=mdp.applied_torque_l2, weight=0.0, params={**_HAND, **_LIFT})


# ── Terminations ──────────────────────────────────────────────────────────────
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropped = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": _TABLE.top_z - 0.10, "asset_cfg": SceneEntityCfg("object")},
    )


# ── Events ────────────────────────────────────────────────────────────────────
@configclass
class EventCfg:
    attach_demo = EventTerm(
        func=mdp.attach_demo, mode="startup",
        params={"npz_path": "", "demo_dt": 0.02, "max_seq_len": 1200,
                "world_z_offset": 0.0, "wrist_world_offset": (0.0, 0.0, 0.0),
                "obj_world_offset": None, "use_sam3_init": False},
    )
    reset_robot = EventTerm(
        func=mdp.reset_robot_to_demo_frame_0, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "arm_joint_names_expr": "panda_joint.*"},
    )
    reset_object = EventTerm(
        func=mdp.reset_object_to_demo_frame_0, mode="reset",
        params={**_OBJ, "position_noise": (0.02, 0.02, 0.0), "yaw_noise_rad": 0.175, "extra_z_clearance": 0.0},
    )
    # Force friction 2.0 on the hand + object at startup (USD-bound materials vary).
    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (2.0, 2.0), "dynamic_friction_range": (2.0, 2.0),
                "restitution_range": (0.0, 0.0), "num_buckets": 1},
    )
    object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("object"),
                "static_friction_range": (2.0, 2.0), "dynamic_friction_range": (2.0, 2.0),
                "restitution_range": (0.0, 0.0), "num_buckets": 1},
    )


# ── Env cfg ───────────────────────────────────────────────────────────────────
@configclass
class TeleopBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Residual-RL teleop env base (subclass per dataset)."""

    # Set by the runner script; the startup attach_demo event reads it.
    demo_npz_path: str = ""
    # Robot base (panda_link0) world pose — single anchor for the whole rig.
    mount_xyz: tuple[float, float, float] = _MOUNT_XYZ
    # Demo sample period (1/fps): 0.02 = 50 Hz (EgoVerse). MAPLE overrides to 0.10.
    demo_dt: float = 0.02
    # Residual scale + object-spawn randomization (training defaults).
    residual_scale: float = 0.1
    randomize_object_init: bool = True
    object_position_noise: tuple[float, float, float] = (0.02, 0.02, 0.0)
    object_yaw_noise_rad: float = 0.175
    max_demo_frames: int = 1200

    scene: TeleopSceneCfg = TeleopSceneCfg(num_envs=64, env_spacing=3.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    # 50 Hz control (decimation 2 × dt 0.01) over 100 Hz physics for contacts.
    decimation: int = 2
    episode_length_s: float = 26.0
    is_finite_horizon: bool = True

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 100.0,
        render_interval=2,
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            # Sized for an 8 GB laptop GPU (≤ a few hundred envs). The upstream
            # 2**23 patch cap preallocates a large GPU buffer that OOMs an 8 GB
            # card once the desktop also uses VRAM; 2**21 is ample here.
            gpu_max_rigid_contact_count=2 ** 20,
            gpu_max_rigid_patch_count=2 ** 21,
        ),
    )
    viewer = ViewerCfg(eye=(1.5, 1.5, 1.5), lookat=(0.0, 0.0, 0.75))

    def __post_init__(self):
        super().__post_init__()
        self.sim.render_interval = self.decimation

        # Wire the demo + frame offsets. The demo is in the panda_link0 frame, so
        # object/wrist get the full mount offset; world_z_offset covers any legacy
        # demo-world npz (table-top at z=0).
        self.events.attach_demo.params["npz_path"] = self.demo_npz_path
        self.events.attach_demo.params["demo_dt"] = self.demo_dt
        self.events.attach_demo.params["max_seq_len"] = self.max_demo_frames
        self.events.attach_demo.params["world_z_offset"] = _TABLE.top_z
        self.events.attach_demo.params["obj_world_offset"] = self.mount_xyz
        self.events.attach_demo.params["wrist_world_offset"] = self.mount_xyz

        # Robot base = mount_xyz (panda_link0 sits at the USD origin → no offset).
        self.scene.robot.init_state.pos = tuple(self.mount_xyz)

        self.actions.joint_targets.residual_scale = self.residual_scale

        if self.randomize_object_init:
            self.events.reset_object.params["position_noise"] = self.object_position_noise
            self.events.reset_object.params["yaw_noise_rad"] = self.object_yaw_noise_rad
        else:
            self.events.reset_object.params["position_noise"] = (0.0, 0.0, 0.0)
            self.events.reset_object.params["yaw_noise_rad"] = 0.0
