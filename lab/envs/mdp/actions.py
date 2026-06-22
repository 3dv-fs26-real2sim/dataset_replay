"""Residual action term: ``recorded_qpos[t] + residual_scale * policy``.

The deterministic baseline plays the demo's per-frame joint targets straight
onto the articulation — arm from ``demo.arm_qpos`` (the Lula-IK retarget) and
fingers from ``demo.hand_qpos`` — and the policy adds a small residual on top of
all 24 DOFs (7 arm + 17 hand). Because the baseline is the IK-of-wrist solution
the kinematic-replay scripts produce, no in-loop IK is needed here; the policy
only has to correct contact dynamics the open-loop baseline can't model.

Imports Isaac Sim — import only after ``SimulationApp`` is created.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Default demo sample period (s/frame) if the env doesn't set ``demo_dt``.
DEMO_DT = 0.02


def demo_index(env: "ManagerBasedRLEnv", demo, ebuf: torch.Tensor | None = None) -> torch.Tensor:
    """Rate-aware, clamped demo frame index.

    The env advances ``episode_length_buf`` by 1 each control step (``step_dt``
    s), but a demo frame lasts ``demo_dt`` s (50 Hz EgoVerse → 0.02; 10 Hz MAPLE
    → 0.10). The frame consumed is ``floor(ebuf · step_dt / demo_dt)`` so the
    reference plays back at the recording's real-time speed regardless of the
    control rate. Used by the action, observation, and reward terms alike so the
    baseline target, look-ahead, and tracking reward all reference one frame.
    """
    if ebuf is None:
        ebuf = getattr(env, "episode_length_buf", None)
    if ebuf is None:
        return demo.clamp_t(torch.zeros(env.num_envs, dtype=torch.long, device=env.device))
    step_dt = float(getattr(env, "step_dt", DEMO_DT))
    demo_dt = float(getattr(env, "demo_dt", DEMO_DT))
    if abs(step_dt - demo_dt) > 1e-9:
        ebuf = (ebuf.to(torch.float64) * (step_dt / demo_dt)).floor().to(torch.long)
    return demo.clamp_t(ebuf)


class RecordedQposResidualAction(ActionTerm):
    """Write ``recorded_qpos[t] + scale * residual`` to arm + hand joints."""

    cfg: "RecordedQposResidualActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "RecordedQposResidualActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # Arm joints in canonical order (preserve_order → columns align with the
        # demo's arm_qpos); hand joints by articulation order (remapped below).
        arm_ids, self._arm_names = self._asset.find_joints(cfg.arm_joint_names_expr, preserve_order=True)
        hand_ids, self._hand_names = self._asset.find_joints(cfg.hand_joint_names_expr, preserve_order=False)
        self._arm_ids = arm_ids
        self._hand_ids = hand_ids
        self._joint_ids = arm_ids + hand_ids
        self._n_dofs = len(self._joint_ids)

        # Demo hand column → articulation hand column permutation, resolved lazily
        # on first apply (needs demo.hand_joint_names).
        self._hand_demo_to_art_inv: torch.Tensor | None = None

        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        self._dof_lower = limits[..., 0]
        self._dof_upper = limits[..., 1]

        self._raw_actions = torch.zeros(self.num_envs, self._n_dofs, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self._n_dofs, device=self.device)

    # ── Term API plumbing ────────────────────────────────────────────────────
    @property
    def num_envs(self) -> int:
        return self._env.num_envs

    @property
    def device(self) -> torch.device:
        return self._env.device

    @property
    def action_dim(self) -> int:
        return self._n_dofs

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def _resolve_hand_remap(self, demo_hand_joint_names: Sequence[str]) -> None:
        art_idx = {name: i for i, name in enumerate(self._hand_names)}
        try:
            remap = [art_idx[name] for name in demo_hand_joint_names]
        except KeyError as exc:
            raise RuntimeError(
                f"demo hand joint '{exc.args[0]}' not in articulation joints {self._hand_names}"
            ) from exc
        perm = torch.tensor(remap, dtype=torch.long, device=self.device)
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(len(perm), device=self.device)
        self._hand_demo_to_art_inv = inv

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions

    def apply_actions(self) -> None:
        demo = getattr(self._env, "demo", None)
        if demo is None:
            base_target = torch.zeros_like(self._raw_actions)
        else:
            if self._hand_demo_to_art_inv is None:
                self._resolve_hand_remap(demo.hand_joint_names)
            t = demo_index(self._env, demo)
            env_idx = torch.arange(self.num_envs, device=self.device)
            arm_qpos = demo.arm_qpos[env_idx, t]                              # (E, 7)
            hand_qpos = demo.hand_qpos[env_idx, t][:, self._hand_demo_to_art_inv]  # (E, 17) art order
            base_target = torch.cat([arm_qpos, hand_qpos], dim=-1)

        target = base_target + self.cfg.residual_scale * self._raw_actions
        target = torch.clamp(target, self._dof_lower, self._dof_upper)
        self._processed_actions[:] = target
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions[:] = 0.0
            self._processed_actions[:] = 0.0
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class RecordedQposResidualActionCfg(ActionTermCfg):
    """Configuration for :class:`RecordedQposResidualAction`."""

    class_type: type[ActionTerm] = RecordedQposResidualAction

    arm_joint_names_expr: str = "panda_joint.*"
    hand_joint_names_expr: str = "right_.*"
    residual_scale: float = 0.1
    """Multiplier on the policy's residual delta added to the recorded qpos."""
