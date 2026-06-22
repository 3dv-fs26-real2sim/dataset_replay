"""``lab`` — residual-RL training environments for the dataset_replay rigs.

Transfers the dexterous-manipulation residual-RL task (Franka Panda + 17-DOF
OrcaHand) onto the two dataset_replay datasets:

* :mod:`lab.envs.egoverse_env_cfg` — EgoVerse (Aria) duck-grasp demos.
* :mod:`lab.envs.maple_env_cfg`    — MAPLE (OAK-D) pan-manipulation demos.

The policy learns a residual on top of a deterministic per-frame baseline::

    joint_target[t] = recorded_qpos[t]  +  residual_scale * policy(obs)

where ``recorded_qpos`` (7 arm + 17 hand DOFs) comes from a demo built with
the *same* Lula IK retargeting the kinematic-replay scripts use
(:mod:`utils.ik`), so the training rig and the replay rig share one kinematic
convention.

Reuse boundary
--------------
This package builds on ``dataset_replay/scripts/utils`` rather than
duplicating it: joint names / EE offset / quaternion convention come from
:mod:`utils.constants`, scene geometry + asset paths from
:mod:`utils.config`, and demo IK from :mod:`utils.ik`. To make the ``utils``
package importable from anywhere, importing :mod:`lab` puts
``dataset_replay/scripts`` on ``sys.path`` (the same path the replay scripts
add via ``sys.path.insert``). This module performs NO Isaac Sim imports, so it
is safe to import before ``SimulationApp`` is created; the env-cfg submodules
under :mod:`lab.envs` must only be imported afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root = dataset_replay/ (this file is dataset_replay/lab/__init__.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"

# Make the shared ``utils`` package (dataset_replay/scripts/utils) importable as
# a top-level ``utils.*`` module, mirroring the replay scripts' path setup.
_SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if _SCRIPTS_DIR.is_dir() and str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

__all__ = ["PROJECT_ROOT", "ASSETS_DIR", "DATA_DIR"]
