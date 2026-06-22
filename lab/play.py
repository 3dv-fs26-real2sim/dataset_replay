"""Play / evaluate a trained residual policy (EgoVerse / MAPLE teleop tasks).

Usage::

    python lab/play.py --task egoverse \
        --demo data/egoverse/demos/egoverse_duck_104715.npz \
        --checkpoint logs/rsl_rl/teleop_residual/<run>/model_*.pt \
        --num_envs 1 --video

Parses CLI and boots ``SimulationApp`` before any Isaac Sim import.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lab  # noqa: E402,F401

_TASK_IDS = {
    "egoverse": "Teleop-Egoverse-OrcaFranka-v0",
    "maple": "Teleop-Maple-OrcaFranka-v0",
}

parser = argparse.ArgumentParser(description="Play a trained residual policy on a teleop demo.")
parser.add_argument("--task", type=str, default="egoverse", choices=sorted(_TASK_IDS))
parser.add_argument("--demo", type=Path, required=True, help="Cleaned demo npz path.")
parser.add_argument("--checkpoint", type=Path, default=None,
                    help="rsl_rl model_*.pt to load. Omit to run the zero-residual baseline.")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--object", type=str, default=None, choices={"duck", "ball", "pan"})
parser.add_argument("--ball-radius", type=float, default=0.048)
parser.add_argument("--maple-props", type=Path, default=None)
parser.add_argument("--bowl-pose", type=Path, default=None, help="EgoVerse: bowl pose npz to spawn the bowl.")
parser.add_argument("--bowl-pose-frame", type=str, default=None, choices={"aria_camera", "panda_link0"})
parser.add_argument("--steps", type=int, default=0, help="Control steps to roll out (0 = full demo).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=0, help="Video frames (0 = full demo).")

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

from lab._preflight import preflight  # noqa: E402
preflight()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import lab.envs  # noqa: E402,F401


def main() -> None:
    task_id = _TASK_IDS[args_cli.task]
    spec = gym.spec(task_id)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()

    demo_path = str(args_cli.demo.resolve())
    env_cfg.demo_npz_path = demo_path
    env_cfg.events.attach_demo.params["npz_path"] = demo_path

    from lab.envs.demo_loader import peek_demo_length
    n_frames = peek_demo_length(args_cli.demo)
    env_cfg.max_demo_frames = n_frames
    env_cfg.episode_length_s = n_frames * env_cfg.demo_dt + 2.0

    env_cfg.scene.num_envs = args_cli.num_envs
    # Deterministic eval: zero the object-spawn randomization (__post_init__ ran).
    env_cfg.events.reset_object.params["position_noise"] = (0.0, 0.0, 0.0)
    env_cfg.events.reset_object.params["yaw_noise_rad"] = 0.0
    if args_cli.task == "maple" and args_cli.maple_props is not None:
        from lab.envs.maple_env_cfg import attach_maple_props
        attach_maple_props(env_cfg, str(args_cli.maple_props.resolve()))
    if args_cli.task == "egoverse" and args_cli.bowl_pose is not None:
        from lab.envs.egoverse_env_cfg import attach_bowl
        attach_bowl(env_cfg, str(args_cli.bowl_pose.resolve()), args_cli.bowl_pose_frame)
    if args_cli.object is not None:
        from lab.envs.objects import ball_spawn_cfg, duck_spawn_cfg, pan_spawn_cfg
        spawn = {"duck": duck_spawn_cfg, "pan": pan_spawn_cfg}.get(args_cli.object)
        env_cfg.scene.object.spawn = (ball_spawn_cfg(args_cli.ball_radius) if args_cli.object == "ball" else spawn())
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # One env.step = one CONTROL step (step_dt = decimation * sim.dt). The demo
    # advances by demo_dt per real-time second, so playing the whole N-frame clip
    # takes N * demo_dt / step_dt control steps (e.g. 10 Hz MAPLE on 50 Hz control
    # → 5 steps per demo frame). The video records one frame per control step, so
    # its real-time fps is the CONTROL rate (1/step_dt), not the demo rate.
    step_dt = env_cfg.decimation * env_cfg.sim.dt
    n_steps_full = int(round(n_frames * env_cfg.demo_dt / step_dt))

    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name, "play"))
    env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        vid_fps = int(round(1.0 / step_dt))
        vid_len = args_cli.video_length if args_cli.video_length > 0 else n_steps_full
        vid_dir = os.path.join(log_dir, "videos")
        env = gym.wrappers.RecordVideo(env, video_folder=vid_dir,
                                       step_trigger=lambda s: s == 0,
                                       video_length=vid_len, disable_logger=True,
                                       name_prefix="rollout", fps=vid_fps)
        print(f"[play] recording {vid_len}-frame video @ {vid_fps} fps → {vid_dir}", flush=True)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    if args_cli.checkpoint is not None:
        runner.load(str(args_cli.checkpoint.resolve()))
        print(f"[play] loaded checkpoint {args_cli.checkpoint}", flush=True)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        print("[play] no checkpoint → zero-residual baseline", flush=True)
        policy = None

    n_steps = args_cli.steps if args_cli.steps > 0 else n_steps_full
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(n_steps):
            actions = policy(obs) if policy is not None else torch.zeros(
                env.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
            obs, _, _, _ = env.step(actions)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    sys.exit(0)
