"""Reward terms for the residual-RL teleop task (read ``env.demo``).

* Tracking — exp-kernel rewards pulling the object pose and hand joints toward
  the demo reference (``track_object_pos`` / ``track_object_rot`` /
  ``track_joint_pos``). Object tracking is computed in the ENV-LOCAL frame so it
  stays valid at multi-env (the stock world-frame compare zeroes the reward for
  every env whose origin ≠ 0).
* Stabilization — ``inhand_object_stability`` (object held rigidly in the hand)
  plus lift-gated effort/jerk penalties (``true_action_rate_l2`` /
  ``hand_joint_vel_l2`` / ``applied_torque_l2``) that damp in-hand chatter only
  once the object is grasped, never taxing grasp discovery.
* Terminal — one-shot placement bonuses at the demo's final frame.

Imports Isaac Sim — import only after ``SimulationApp`` is created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_conjugate, quat_mul

from .actions import demo_index

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _demo(env: "ManagerBasedRLEnv"):
    return getattr(env, "demo", None)


def _obj_pos_envlocal(env: "ManagerBasedRLEnv", obj) -> torch.Tensor:
    return obj.data.root_pos_w - env.scene.env_origins


# ── hand-joint permutation cache ──────────────────────────────────────────────
def _hand_perm(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor | None:
    """demo→articulation permutation over the joints selected by ``asset_cfg``."""
    cache = getattr(env, "_reward_hand_remap", None)
    if cache is not None:
        return cache
    demo = _demo(env)
    asset = env.scene[asset_cfg.name]
    if demo is None or not asset_cfg.joint_ids:
        return None
    sel_names = [asset.data.joint_names[i] for i in asset_cfg.joint_ids]
    art_idx = {name: i for i, name in enumerate(sel_names)}
    remap = [art_idx[n] for n in demo.hand_joint_names if n in art_idx]
    if not remap:
        return None
    perm = torch.tensor(remap, dtype=torch.long, device=env.device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(len(perm), device=env.device)
    env._reward_hand_remap = inv  # type: ignore[attr-defined]
    return inv


# ── tracking ──────────────────────────────────────────────────────────────────
def track_joint_pos(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, k: float = 5.0) -> torch.Tensor:
    """exp(-k · mean|target_q − q|) on the joints selected by ``asset_cfg``."""
    demo = _demo(env)
    if demo is None:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids or list(range(asset.data.joint_pos.shape[-1]))
    q = asset.data.joint_pos[:, joint_ids]
    t = demo_index(env, demo)
    ei = torch.arange(env.num_envs, device=env.device)
    target_demo = demo.hand_qpos[ei, t]
    inv = _hand_perm(env, asset_cfg)
    target = target_demo[..., : len(joint_ids)] if (inv is None or inv.shape[0] != len(joint_ids)) else target_demo[:, inv]
    return torch.exp(-k * (target - q).abs().mean(dim=-1))


def track_object_pos(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, k: float = 80.0) -> torch.Tensor:
    """exp(-k · ‖demo.obj_pos[t] − obj_pos‖), ENV-LOCAL frame."""
    demo = _demo(env)
    if demo is None:
        return torch.zeros(env.num_envs, device=env.device)
    obj = env.scene[object_cfg.name]
    t = demo_index(env, demo)
    ei = torch.arange(env.num_envs, device=env.device)
    return torch.exp(-k * torch.norm(demo.obj_pos[ei, t] - _obj_pos_envlocal(env, obj), dim=-1))


def track_object_rot(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, k: float = 3.0) -> torch.Tensor:
    """exp(-k · geodesic_angle(target_quat, obj_quat)) — frame-independent."""
    demo = _demo(env)
    if demo is None:
        return torch.zeros(env.num_envs, device=env.device)
    obj = env.scene[object_cfg.name]
    t = demo_index(env, demo)
    ei = torch.arange(env.num_envs, device=env.device)
    diff = quat_mul(demo.obj_quat[ei, t], quat_conjugate(obj.data.root_quat_w))
    angle = 2.0 * torch.acos(torch.clamp(diff[:, 0].abs(), -1.0 + 1e-7, 1.0 - 1e-7))
    return torch.exp(-k * angle)


def action_rate_l2(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Penalize raw residual magnitude (smoothness proxy)."""
    return torch.sum(env.action_manager.action ** 2, dim=-1)


# ── stabilization + effort (lift-gated) ───────────────────────────────────────
def _lift_gate(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, lift_on: float = 0.02, k_lift: float = 30.0) -> torch.Tensor:
    """Smooth [0, 1] gate: 0 while the object rests, ~1 once lifted (≈ grasped)."""
    demo = _demo(env)
    if demo is None:
        return torch.zeros(env.num_envs, device=env.device)
    obj = env.scene[object_cfg.name]
    ei = torch.arange(env.num_envs, device=env.device)
    lift = obj.data.root_pos_w[:, 2] - demo.obj_pos[ei, 0, 2]
    return 1.0 - torch.exp(-k_lift * torch.clamp(lift - lift_on, min=0.0))


def true_action_rate_l2(
    env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg | None = None,
    lift_on: float = 0.02, k_lift: float = 30.0,
) -> torch.Tensor:
    """Sum of squared (action − prev_action): true jerk penalty, lift-gated."""
    am = env.action_manager
    raw = torch.sum((am.action - am.prev_action) ** 2, dim=-1)
    return raw if object_cfg is None else _lift_gate(env, object_cfg, lift_on, k_lift) * raw


def hand_joint_vel_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg | None = None,
    lift_on: float = 0.02, k_lift: float = 30.0,
) -> torch.Tensor:
    """Sum of squared hand joint velocities (effort/jerk proxy), lift-gated."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    qd = asset.data.joint_vel if joint_ids == slice(None) else asset.data.joint_vel[:, joint_ids]
    raw = torch.sum(qd ** 2, dim=-1)
    return raw if object_cfg is None else _lift_gate(env, object_cfg, lift_on, k_lift) * raw


def applied_torque_l2(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg | None = None,
    lift_on: float = 0.02, k_lift: float = 30.0,
) -> torch.Tensor:
    """Sum of squared applied joint torques (approximate PD effort), lift-gated.

    OPTIONAL fallback (default weight 0); prefer ``hand_joint_vel_l2``."""
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    tau = asset.data.applied_torque
    raw = torch.sum(tau ** 2, dim=-1) if joint_ids == slice(None) else torch.sum(tau[:, joint_ids] ** 2, dim=-1)
    return raw if object_cfg is None else _lift_gate(env, object_cfg, lift_on, k_lift) * raw


def _palm_body_idx(env, robot_cfg: SceneEntityCfg, body_name: str) -> int:
    cached = getattr(env, "_palm_body_idx_cache", None)
    if cached is not None:
        return cached
    body_ids, _ = env.scene[robot_cfg.name].find_bodies(body_name)
    env._palm_body_idx_cache = int(body_ids[0])  # type: ignore[attr-defined]
    return env._palm_body_idx_cache


def inhand_object_stability(
    env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg,
    k_lin: float = 10.0, k_ang: float = 1.0, lift_on: float = 0.02, k_lift: float = 30.0,
    palm_body_name: str = "connector_mount",
) -> torch.Tensor:
    """Bounded [0, 1] reward: object held rigidly in the hand (no in-hand chatter).

        gate(lift) · exp(-k_lin·‖v_slip‖ − k_ang·‖w_obj − w_palm‖)
        v_slip = v_obj − v_palm − w_palm × (p_obj − p_palm)   (rigid-grasp residual)

    ``v_slip`` is zero for a perfect rigid grasp regardless of wrist motion, and
    nonzero only when the object actually slips inside the hand — the true
    chatter signal. Lift-gated so it only shapes the hold/transport phase.
    """
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    pidx = _palm_body_idx(env, robot_cfg, palm_body_name)
    p_obj = obj.data.root_pos_w
    p_palm = robot.data.body_pos_w[:, pidx]
    w_palm = robot.data.body_ang_vel_w[:, pidx]
    v_slip = obj.data.root_lin_vel_w - robot.data.body_lin_vel_w[:, pidx] - torch.cross(w_palm, p_obj - p_palm, dim=-1)
    stab = torch.exp(-k_lin * torch.norm(v_slip, dim=-1) - k_ang * torch.norm(obj.data.root_ang_vel_w - w_palm, dim=-1))
    return _lift_gate(env, object_cfg, lift_on, k_lift) * stab


# ── terminal placement bonuses (fire once, at the demo's final frame) ─────────
def _time_out_mask(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """1.0 on the env's terminal time-out step (demo horizon reached), else 0.0."""
    tm = getattr(env, "termination_manager", None)
    if tm is None:
        return torch.zeros(env.num_envs, device=env.device)
    return tm.time_outs.float()


def terminal_track_object_pos(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, k: float = 80.0) -> torch.Tensor:
    """``track_object_pos`` delivered once, at the terminal frame."""
    return track_object_pos(env, object_cfg, k) * _time_out_mask(env)


def terminal_track_object_rot(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg, k: float = 3.0) -> torch.Tensor:
    """``track_object_rot`` delivered once, at the terminal frame."""
    return track_object_rot(env, object_cfg, k) * _time_out_mask(env)
