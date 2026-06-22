"""Residual-RL PPO training entrypoint (EgoVerse / MAPLE teleop tasks).

Usage::

    # EgoVerse duck-grasp (train on the first 500 frames; bowl auto-spawned)
    python lab/train.py --task egoverse \
        --demo data/egoverse/demos/egoverse_duck_20250804_104715.npz \
        --bowl-pose data/egoverse/demos/egoverse_bowl_20250804_104715.npz --bowl-pose-frame panda_link0 \
        --max-frames 500 --num_envs 256 --headless

    # MAPLE pan, whole clip, with static props
    python lab/train.py --task maple \
        --demo data/maple/demos/maple_pan_20250922_143954.npz \
        --maple-props data/maple/demos/maple_props_20250922_143954.npz \
        --num_envs 256 --headless

Parses CLI and boots ``SimulationApp`` BEFORE importing anything that touches
Isaac Sim (per dataset_replay's convention).
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Make ``lab`` importable (and, via lab/__init__, the shared ``utils`` package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab  # noqa: E402,F401  (runs the path bootstrap)

_TASK_IDS = {
    "egoverse": "Teleop-Egoverse-OrcaFranka-v0",
    "maple": "Teleop-Maple-OrcaFranka-v0",
}

parser = argparse.ArgumentParser(description="Train residual PPO on an EgoVerse/MAPLE teleop demo.")
parser.add_argument("--task", type=str, default="egoverse", choices=sorted(_TASK_IDS),
                    help="Dataset / registered task to train.")
parser.add_argument("--demo", type=Path, required=True, help="Cleaned demo npz path.")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--residual-scale", type=float, default=None,
                    help="Override residual action scale (env default 0.1).")
parser.add_argument("--object", type=str, default=None, choices={"duck", "ball", "pan"},
                    help="Override the manipulated object (default: duck for egoverse, pan for maple).")
parser.add_argument("--ball-radius", type=float, default=0.048, help="Sphere radius for --object ball.")
parser.add_argument("--maple-props", type=Path, default=None,
                    help="MAPLE only: props npz to spawn static obstacles (box/carpet/heater).")
parser.add_argument("--bowl-pose", type=Path, default=None,
                    help="EgoVerse only: bowl pose npz ((4,4) or (N,4,4)); spawns the bowl container.")
parser.add_argument("--bowl-pose-frame", type=str, default=None, choices={"aria_camera", "panda_link0"},
                    help="Frame of --bowl-pose (default: aria_camera for egoverse).")
parser.add_argument("--no-object-noise", action="store_true", help="Disable object-spawn randomization.")
parser.add_argument("--episode-buffer", type=float, default=2.0,
                    help="Seconds added to the clip length for the episode horizon (default 2.0).")
parser.add_argument("--max-frames", type=int, default=0,
                    help="Train on only the first N demo frames (0 = whole clip).")
parser.add_argument("--experiment_name", type=str, default=None)
parser.add_argument("--run_name", type=str, default=None)
parser.add_argument("--logger", type=str, default=None, choices={"wandb", "tensorboard", "neptune"})
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# Accept the EULA non-interactively + warn early on low VRAM / leftover GPU procs.
from lab._preflight import preflight  # noqa: E402
preflight()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-boot imports ─────────────────────────────────────────────────────────
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import lab.envs  # noqa: E402,F401  (registers the gym tasks)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main() -> None:
    task_id = _TASK_IDS[args_cli.task]
    spec = gym.spec(task_id)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()

    # Wire the demo path. The cfg's __post_init__ already ran at construction, so
    # set the RESOLVED attach_demo param (not just the field) — same for the
    # other overrides below that __post_init__ would otherwise have consumed.
    demo_path = str(args_cli.demo.resolve())
    env_cfg.demo_npz_path = demo_path
    env_cfg.events.attach_demo.params["npz_path"] = demo_path

    # Size the episode to the clip: T·demo_dt + buffer (so time-out fires just
    # after the demo ends, no long "hold final pose" tail). The cfg can't do
    # this itself — the demo only loads at the startup event.
    from lab.envs.demo_loader import peek_demo_length
    n_frames = peek_demo_length(args_cli.demo)
    if args_cli.max_frames > 0:
        n_frames = min(n_frames, args_cli.max_frames)
    env_cfg.max_demo_frames = n_frames
    env_cfg.episode_length_s = n_frames * env_cfg.demo_dt + args_cli.episode_buffer
    print(f"[train] training on {n_frames} demo frames @ {1/env_cfg.demo_dt:.0f} Hz → "
          f"episode_length_s={env_cfg.episode_length_s:.1f}s "
          f"(clip {n_frames * env_cfg.demo_dt:.1f}s + {args_cli.episode_buffer:.1f}s buffer)", flush=True)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.residual_scale is not None:
        env_cfg.residual_scale = args_cli.residual_scale
        env_cfg.actions.joint_targets.residual_scale = args_cli.residual_scale
    if args_cli.no_object_noise:
        env_cfg.events.reset_object.params["position_noise"] = (0.0, 0.0, 0.0)
        env_cfg.events.reset_object.params["yaw_noise_rad"] = 0.0
    if args_cli.task == "maple" and args_cli.maple_props is not None:
        if not args_cli.maple_props.exists():
            raise SystemExit(f"--maple-props npz not found: {args_cli.maple_props}")
        from lab.envs.maple_env_cfg import attach_maple_props
        attach_maple_props(env_cfg, str(args_cli.maple_props.resolve()))
    if args_cli.task == "egoverse" and args_cli.bowl_pose is not None:
        if not args_cli.bowl_pose.exists():
            raise SystemExit(f"--bowl-pose npz not found: {args_cli.bowl_pose}")
        from lab.envs.egoverse_env_cfg import attach_bowl
        attach_bowl(env_cfg, str(args_cli.bowl_pose.resolve()), args_cli.bowl_pose_frame)

    # Optional manipulated-object override (applied after __post_init__'s default).
    if args_cli.object is not None:
        from lab.envs.objects import ball_spawn_cfg, duck_spawn_cfg, pan_spawn_cfg
        spawn = {"duck": duck_spawn_cfg, "pan": pan_spawn_cfg}.get(args_cli.object)
        env_cfg.scene.object.spawn = (ball_spawn_cfg(args_cli.ball_radius) if args_cli.object == "ball" else spawn())
        print(f"[train] object → {args_cli.object}", flush=True)

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.experiment_name is not None:
        agent_cfg.experiment_name = args_cli.experiment_name
    if args_cli.run_name is not None:
        agent_cfg.run_name = args_cli.run_name
    if args_cli.logger is not None:
        agent_cfg.logger = args_cli.logger
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root, log_dir)
    print(f"[train] task={task_id}  demo={args_cli.demo.name}  logging→ {log_dir}", flush=True)

    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        env = gym.wrappers.RecordVideo(env, video_folder=os.path.join(log_dir, "videos", "train"),
                                       step_trigger=lambda s: s % args_cli.video_interval == 0,
                                       video_length=args_cli.video_length, disable_logger=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
