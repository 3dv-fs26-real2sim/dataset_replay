# lab/ — residual-RL training

Based on [ManipTrans](https://github.com/ManipTrans/ManipTrans) implementation of a residual policy for dexterous manipulation. 

Trains a dexterous-manipulation **residual policy** for the Franka Panda +
17-DOF OrcaHand on the two dataset_replay rigs (EgoVerse, MAPLE). A
deterministic per-frame baseline plays a recorded demo onto the robot; a PPO
policy (rsl_rl) learns a small correction that makes the open-loop grasp robust
to contact dynamics and reset randomization.

```
joint_target[t] = recorded_qpos[t]  +  residual_scale * policy(obs)
                     (deterministic)        (PPO, trained)
```

There is no learned base imitator — the baseline is the demo's recorded joint
trajectory (7 arm + 17 hand DOFs), so training starts from a working grasp and
only learns the residual.

## How it connects to dataset_replay

The training env is built on the shared `dataset_replay/scripts/utils` toolkit
rather than duplicating it:

| Reused from `utils/` | Used for |
|---|---|
| `constants.py` (`ARM_JOINT_NAMES`, `HAND_JOINT_NAMES`, EE offset, quat conv) | joint ordering, demo↔articulation remap |
| `config.py` / `config_*.py` (`TableConfig`, `select_config`, asset paths) | scene geometry, robot/object USD paths |
| `ik.py` (`create_ik_solver`, `solve_ik_for_pose`) + `rotation.py` | arm IK retarget in `make_demo.py` |
| `h5_loader.py` (`H5Reader`) | reading raw recordings in `make_demo.py` |

`lab/__init__.py` puts `dataset_replay/scripts` on `sys.path`, so the env code
imports `utils.*` directly — one source of truth for the kinematic convention
shared with `kinematic_replay_*.py`.

## Layout

```
lab/
├── __init__.py              # path bootstrap (adds scripts/ for `utils`); no Isaac imports
├── train.py                 # rsl_rl PPO training entrypoint
├── play.py                  # eval / inference entrypoint (optional video)
├── envs/
│   ├── __init__.py          # gym.register both tasks
│   ├── teleop_base_env_cfg.py   # shared ManagerBasedRLEnvCfg (scene/actions/obs/rewards/term/events)
│   ├── egoverse_env_cfg.py  # EgoVerse variant (50 Hz, duck)
│   ├── maple_env_cfg.py     # MAPLE variant (10 Hz, pan + optional static props)
│   ├── robot_cfg.py         # ORCA_FRANKA_CFG ArticulationCfg (→ assets/pandaorca_right.usd)
│   ├── objects.py           # duck/ball/pan/bowl + maple-prop spawn cfgs
│   ├── demo_loader.py       # Demo dataclass + load_demo (per-frame targets, pure torch)
│   └── mdp/
│       ├── actions.py       # RecordedQposResidualAction (recorded qpos + residual)
│       ├── observations.py  # proprio + object-in-root + reference look-ahead
│       ├── rewards.py       # tracking + in-hand stability + lift-gated effort
│       └── events.py        # attach_demo (startup) + reset to demo frame 0
│   └── agents/
│       └── rsl_rl_ppo_cfg.py    # PPO runner cfg (rsl-rl 3.x schema)
└── scripts/
    └── make_demo.py         # build a demo npz from a raw H5 (Lula IK retarget)
└── rich/                    # simulation-rich rollout capture (MVR + contact force + montage)
    └── run_rich.py          # single entry — see lab/rich/README.md
```

## Rich rollout capture (`lab/rich/`)

After training, [`lab/rich/`](rich/README.md) re-runs a policy once and captures
a multi-viewpoint render (calibrated Aria POV + 4 free views, with
depth/normals/segmentation) plus per-skin-pad contact forces, then tiles them
into montages. All deps are in `dataset_replay`; see [`rich/README.md`](rich/README.md).

```bash
python lab/rich/run_rich.py --task egoverse \
    --demo data/egoverse/demos/egoverse_duck_104715.npz \
    --checkpoint logs/rsl_rl/teleop_residual/<run>/model_<n>.pt --headless
```

## Registered tasks

| Task id | `--task` | Object | Rate | Notes |
|---|---|---|---|---|
| `Teleop-Egoverse-OrcaFranka-v0` | `egoverse` | duck | 50 Hz | Aria duck-grasp demos |
| `Teleop-Maple-OrcaFranka-v0` | `maple` | pan | 10 Hz | OAK-D pan; optional static props |

Both expose asymmetric observations: a `policy` (actor) group and a privileged
`critic` group, each with a 1-frame reference look-ahead (object-pose deltas +
baseline action + delta-dof) so the residual policy is time-aware.

## Usage

```bash
conda activate dataset_replay

# Train (EgoVerse). num_envs 64 fits an 8 GB GPU; raise it on bigger cards.
python lab/train.py --task egoverse \
    --demo data/egoverse/demos/egoverse_duck_104715.npz \
    --num_envs 64 --headless --max_iterations 2000

# Train (MAPLE pan + static props)
python lab/train.py --task maple \
    --demo data/maple/demos/maple_pan_143954.npz \
    --maple-props data/maple/demos/maple_props_143954.npz \
    --num_envs 64 --headless

# Evaluate / record a rollout video
python lab/play.py --task egoverse \
    --demo data/egoverse/demos/egoverse_duck_104715.npz \
    --checkpoint logs/rsl_rl/teleop_residual/<run>/model_<n>.pt \
    --num_envs 1 --video

# Build a demo from a raw H5 (reuses utils/ik.py Lula retarget)
python lab/scripts/make_demo.py --dataset egoverse \
    --h5 data/egoverse/h5/20250804_104715.h5 --object duck \
    --out data/egoverse/demos/egoverse_duck_104715.npz
```

Useful `train.py` flags: `--residual-scale` (default 0.1), `--object {duck,ball,pan}`,
`--no-object-noise`, `--seed`, `--experiment_name`, `--logger {tensorboard,wandb}`,
`--video`. Logs/checkpoints land in `logs/rsl_rl/<experiment_name>/<timestamp>/`.

## Demos

A demo npz holds the per-frame training targets:

| key | shape | meaning |
|---|---|---|
| `arm_qpos` | (T, 7) | Panda joint baseline (Lula IK retarget of the wrist) |
| `hand_qpos` | (T, 17) | OrcaHand joint baseline |
| `obj_trajectory` | (T, 4, 4) | object SE(3) reference (panda_link0 frame) |
| `hand_joint_names` | (17,) | demo column order (validated vs `utils.constants`) |
| `wrist_pos` / `wrist_rot_aa` | (T, 3) | EE-wrist pose (informational) |
| `frame` | str | `panda_link0` → object/wrist offset by `mount_xyz` at load |

`make_demo.py` reads the wrist+hand trajectory from the H5, retargets the arm
with the warm-started Lula IK (`ee_target` frame), composes the object pose into
the panda_link0 frame (EgoVerse: `ARIA_EXTRINSICS_RIGHT @ T_cam_obj`), and
writes the npz that `load_demo` consumes.

> **Note on object fidelity.** `make_demo.py` uses the *nominal* Aria extrinsic
> (`constants.ARIA_EXTRINSICS_RIGHT`); the shipped demos additionally applied a
> per-clip SAM table-mask refinement (see `utils/calibrate_table.py`), which
> shifts the object trajectory by a few cm. For visually-tight overlays,
> refine the extrinsic before composing. MAPLE object trajectories come from an
> external 6D pose estimator and must be passed via `--object-traj`.

## Notes / tuning

- **Mount.** `mount_xyz = (-0.28, -0.35, 0.75)` is the verified residual-RL
  anchor (the grasp geometry is mount-invariant; this only places the rig over
  the table). Override via the cfg field if a clip needs it.
- **Robot.** Arm gravity is disabled and PD-tracks the recorded `arm_qpos`
  (400/40); hand PD is 300/20. Friction is forced to 2.0 on hand + object.
- **Rewards.** Object tracking is computed in the env-local frame so it stays
  valid at multi-env; effort/jerk penalties are lift-gated (active only once the
  object is grasped) so they never suppress grasp discovery.
