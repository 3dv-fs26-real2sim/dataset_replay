"""Gym registration for the residual-RL teleop tasks.

Importing this module registers both dataset variants:

* ``Teleop-Egoverse-OrcaFranka-v0`` — EgoVerse (Aria) duck-grasp.
* ``Teleop-Maple-OrcaFranka-v0``    — MAPLE (OAK-D) pan-manipulation.

Imports Isaac Sim (via the env-cfg modules) — import only after
``SimulationApp`` is created.
"""

from __future__ import annotations

import gymnasium as gym

from .agents.rsl_rl_ppo_cfg import TeleopPPORunnerCfg
from .egoverse_env_cfg import EgoverseTeleopEnvCfg
from .maple_env_cfg import MapleTeleopEnvCfg

gym.register(
    id="Teleop-Egoverse-OrcaFranka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": EgoverseTeleopEnvCfg,
        "rsl_rl_cfg_entry_point": TeleopPPORunnerCfg,
    },
)

gym.register(
    id="Teleop-Maple-OrcaFranka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": MapleTeleopEnvCfg,
        "rsl_rl_cfg_entry_point": TeleopPPORunnerCfg,
    },
)

__all__ = ["EgoverseTeleopEnvCfg", "MapleTeleopEnvCfg", "TeleopPPORunnerCfg"]
