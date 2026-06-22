"""Observation terms for the residual-RL teleop task.

Stock proprioception (joint pos/vel, last action) comes from IsaacLab's ``mdp``;
this module adds the task-specific terms:

* :func:`object_position_in_robot_root_frame` — object pose relative to the base.
* The ``ref_*`` **reference look-ahead** terms (object-pose deltas, baseline
  action, delta-dof) that make the residual policy time-aware by showing it the
  demo frame it should reach next, instead of a phase scalar.
* :func:`object_lin_vel_w` / :func:`object_ang_vel_w` — privileged critic state.

Every term returns a fixed-width tensor even before ``env.demo`` is attached
(the ObservationManager probes term dims at construction, before the startup
``attach_demo`` event), falling back to zeros of the right shape.

Imports Isaac Sim — import only after ``SimulationApp`` is created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul, subtract_frame_transforms

from .actions import demo_index

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Single-frame look-ahead: show the policy the demo frame to reach next step.
_LOOKAHEAD = 1
ARM_JOINTS_EXPR = "panda_joint.*"
HAND_JOINTS_EXPR = "right_.*"


# ── helpers ──────────────────────────────────────────────────────────────────
def _demo(env: "ManagerBasedRLEnv"):
    return getattr(env, "demo", None)


def _ref_t(env: "ManagerBasedRLEnv", demo, offset: int = _LOOKAHEAD) -> torch.Tensor:
    """Rate-aware look-ahead frame: the demo frame ``offset`` steps ahead of now."""
    return demo.clamp_t(demo_index(env, demo) + offset)


def _joints(env: "ManagerBasedRLEnv"):
    """Resolve & cache (arm_ids, hand_ids, demo→articulation hand permutation).

    Mirrors the action term: arm via ``preserve_order=True`` so columns align
    with ``demo.arm_qpos``; hand via the demo→articulation permutation (resolved
    once a demo is attached).
    """
    cache = getattr(env, "_obs_joints", None)
    robot: Articulation = env.scene["robot"]
    if cache is None:
        arm_ids, _ = robot.find_joints(ARM_JOINTS_EXPR, preserve_order=True)
        hand_ids, hand_names = robot.find_joints(HAND_JOINTS_EXPR, preserve_order=False)
        cache = {"arm_ids": arm_ids, "hand_ids": hand_ids, "hand_names": hand_names, "hand_inv": None}
        env._obs_joints = cache  # type: ignore[attr-defined]
    if cache["hand_inv"] is None:
        demo = _demo(env)
        if demo is not None:
            art_idx = {n: i for i, n in enumerate(cache["hand_names"])}
            perm = torch.tensor([art_idx[n] for n in demo.hand_joint_names], dtype=torch.long, device=env.device)
            inv = torch.empty_like(perm)
            inv[perm] = torch.arange(len(perm), device=env.device)
            cache["hand_inv"] = inv
    return cache


def _base_action(env: "ManagerBasedRLEnv", offset: int = _LOOKAHEAD) -> torch.Tensor:
    """Baseline target dof = recorded ``[arm_qpos | hand_qpos]`` at the look-ahead
    frame, in articulation joint order. Zeros if no demo."""
    j = _joints(env)
    n_ctrl = len(j["arm_ids"]) + len(j["hand_ids"])
    demo = _demo(env)
    if demo is None or j["hand_inv"] is None:
        return torch.zeros(env.num_envs, n_ctrl, device=env.device)
    t = _ref_t(env, demo, offset)
    ei = torch.arange(env.num_envs, device=env.device)
    arm = demo.arm_qpos[ei, t]
    hand = demo.hand_qpos[ei, t][:, j["hand_inv"]]
    return torch.cat([arm, hand], dim=-1)


def _current_ctrl_q(env: "ManagerBasedRLEnv") -> torch.Tensor:
    j = _joints(env)
    robot: Articulation = env.scene["robot"]
    return robot.data.joint_pos[:, j["arm_ids"] + j["hand_ids"]]


def _obj_pos_envlocal(env: "ManagerBasedRLEnv", obj: RigidObject) -> torch.Tensor:
    """Object position in the ENV-LOCAL frame (world minus per-env origin), so it
    is comparable to ``demo.obj_pos`` (broadcast to all envs without origin)."""
    return obj.data.root_pos_w - env.scene.env_origins


# ── proprio / state ───────────────────────────────────────────────────────────
def object_position_in_robot_root_frame(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object world position expressed in the robot root frame ``(E, 3)``."""
    robot = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], obj.data.root_pos_w[:, :3],
    )
    return pos_b


def object_lin_vel_w(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_lin_vel_w


def object_ang_vel_w(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object")) -> torch.Tensor:
    return env.scene[object_cfg.name].data.root_ang_vel_w


# ── reference look-ahead (actor + critic) ─────────────────────────────────────
def ref_object_pos_delta(
    env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reference object position (next frame) minus current, ENV-LOCAL ``(E, 3)``."""
    demo = _demo(env)
    obj: RigidObject = env.scene[object_cfg.name]
    if demo is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    ei = torch.arange(env.num_envs, device=env.device)
    return demo.obj_pos[ei, _ref_t(env, demo)] - _obj_pos_envlocal(env, obj)


def ref_object_quat(
    env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Absolute reference object orientation (next frame), wxyz ``(E, 4)``."""
    demo = _demo(env)
    if demo is None:
        q = torch.zeros(env.num_envs, 4, device=env.device)
        q[:, 0] = 1.0
        return q
    ei = torch.arange(env.num_envs, device=env.device)
    return demo.obj_quat[ei, _ref_t(env, demo)]


def ref_object_quat_delta(
    env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Orientation error current→reference as a quaternion ``(E, 4)``."""
    demo = _demo(env)
    obj: RigidObject = env.scene[object_cfg.name]
    if demo is None:
        q = torch.zeros(env.num_envs, 4, device=env.device)
        q[:, 0] = 1.0
        return q
    ei = torch.arange(env.num_envs, device=env.device)
    tgt = demo.obj_quat[ei, _ref_t(env, demo)]
    return quat_mul(obj.data.root_quat_w, quat_conjugate(tgt))


def ref_base_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Baseline target dof the residual is correcting ``(E, n_arm + n_hand)``."""
    return _base_action(env)


def ref_dof_delta(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Target dof minus current joint pos over controlled joints ``(E, n_ctrl)``."""
    base = _base_action(env)
    if base.abs().sum() == 0.0:  # no demo yet → keep fixed width, zero signal
        return base
    return base - _current_ctrl_q(env)
