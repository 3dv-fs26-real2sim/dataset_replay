"""Simulation-rich rollout — the single entry point.

Re-runs a trained teleop_manip policy once and captures, in one rollout:
  * multi-viewpoint render: the calibrated Aria POV + 4 free views (front/side/top/hero),
    with depth / world-normals / robot-grouped segmentation on the Aria view;
  * per-skin-pad contact force vs the object, and object/hand state.

Then renders the animated contact-force hand and tiles the montages. Outputs:
  outputs/rich/mvr/      per-viewpoint + Aria-modality videos, cam_params.npz
  outputs/rich/cfm/      contacts.npz, state.npz, contact_hand.mp4
  outputs/rich/montage/  mvr_views.mp4, mvr_modalities.mp4, master_montage.mp4

Usage:
  python -u scripts/rich/run_rich.py \
      --demo data/teleop/cleaned_20250804_104715_260602v3.npz \
      --checkpoint logs/<model_path> --headless --device cuda:0
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))          # sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))      # repo root → `lab`
import lab  # noqa: E402,F401  (path bootstrap for `utils`)

_TASK_IDS = {"egoverse": "Teleop-Egoverse-OrcaFranka-v0", "maple": "Teleop-Maple-OrcaFranka-v0"}


def _find_latest_checkpoint(experiment: str) -> str:
    root = os.path.join("logs", "rsl_rl", experiment)
    for run in sorted(glob.glob(os.path.join(root, "*")), key=os.path.getmtime, reverse=True):
        ck = glob.glob(os.path.join(run, "model_*.pt"))
        if ck:
            return max(ck, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))
    raise FileNotFoundError(f"No model_*.pt under {root}/*/")


parser = argparse.ArgumentParser(description="Simulation-rich rollout capture.")
parser.add_argument("--task", type=str, default="egoverse", choices=sorted(_TASK_IDS),
                    help="Dataset / registered task to roll out.")
parser.add_argument("--demo", "--demo-npz", dest="demo", type=Path, required=True)
parser.add_argument("--checkpoint", type=Path, default=None, help="logs/<model_path>; else newest under --experiment.")
parser.add_argument("--experiment", type=str, default="teleop_residual")
parser.add_argument("--out", type=Path, default=Path("outputs/rich"))
parser.add_argument("--steps", type=int, default=0, help="0 -> demo length.")
parser.add_argument("--res", type=int, default=480, help="Free-cam tile height (px); width = 4:3.")
parser.add_argument("--fps", type=int, default=50, help="Sim rate (decimation 2 x dt 0.01 = 50 Hz).")
parser.add_argument("--residual-scale", type=float, default=None, help="0.0 -> deterministic baseline.")
parser.add_argument("--maple-props", type=Path, default=None, help="MAPLE: props npz (box/carpet/heater static).")
parser.add_argument("--bowl-pose", type=Path, default=None, help="EgoVerse: bowl pose npz (dynamic bowl).")

from isaaclab.app import AppLauncher  # noqa: E402
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

_use_policy = args_cli.residual_scale is None or args_cli.residual_scale != 0.0
_ckpt = (str(args_cli.checkpoint) if args_cli.checkpoint else
         _find_latest_checkpoint(args_cli.experiment)) if _use_policy else None
_steps = args_cli.steps or int(np.load(args_cli.demo, allow_pickle=True)["arm_qpos"].shape[0])
print(f"[rich] checkpoint={_ckpt} steps={_steps} out={args_cli.out}", flush=True)

from lab._preflight import preflight  # noqa: E402
preflight()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import lab.envs                                  # noqa: F401, E402  (registers gym tasks)

import rich_capture                              # noqa: E402
import mvr_capture                               # noqa: E402

_TASK_ID = _TASK_IDS[args_cli.task]


def main():
    """Run one rollout, save per-stream videos + contact/state; return output dirs."""
    cfm_dir = args_cli.out / "cfm"
    mvr_dir = args_cli.out / "mvr"
    for d in (cfm_dir, mvr_dir):
        d.mkdir(parents=True, exist_ok=True)

    spec = gym.spec(_TASK_ID)
    env_cfg = spec.kwargs["env_cfg_entry_point"]()
    agent_cfg = spec.kwargs["rsl_rl_cfg_entry_point"]()
    demo_path = str(args_cli.demo.resolve())
    env_cfg.demo_npz_path = demo_path
    env_cfg.events.attach_demo.params["npz_path"] = demo_path
    from lab.envs.demo_loader import peek_demo_length
    env_cfg.max_demo_frames = peek_demo_length(args_cli.demo)
    if args_cli.task == "maple" and args_cli.maple_props is not None:
        from lab.envs.maple_env_cfg import attach_maple_props
        attach_maple_props(env_cfg, str(args_cli.maple_props.resolve()))
    if args_cli.task == "egoverse" and args_cli.bowl_pose is not None:
        from lab.envs.egoverse_env_cfg import attach_bowl
        attach_bowl(env_cfg, str(args_cli.bowl_pose.resolve()), "panda_link0")
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device
    env_cfg.randomize_object_init = False
    env_cfg.events.reset_object.params["position_noise"] = (0.0, 0.0, 0.0)
    env_cfg.events.reset_object.params["yaw_noise_rad"] = 0.0
    env_cfg.events.reset_object.params["extra_z_clearance"] = 0.0
    if args_cli.residual_scale is not None:
        env_cfg.actions.joint_targets.residual_scale = args_cli.residual_scale
        env_cfg.residual_scale = args_cli.residual_scale

    rich_capture.inject_contact_sensors(env_cfg)

    env = gym.make(_TASK_ID, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if _use_policy:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(_ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
    else:
        _z = torch.zeros(env.num_envs, env.num_actions, device=env.unwrapped.device)
        policy = lambda _o: _z  # noqa: E731

    uenv = env.unwrapped
    sensors = rich_capture.get_sensors(uenv)
    pad_ids = rich_capture.pad_body_ids(uenv.scene["robot"])

    cams, cam_params = mvr_capture.build_cameras(res=args_cli.res, dataset=args_cli.task)
    import omni.usd
    mvr_capture.label_scene(omni.usd.get_context().get_stage())
    reader = mvr_capture.MVRReader(cams, cam_params)
    for name, prm in cam_params.items():
        print(f"[rich] camera {name} eye={np.round(prm['T_world_cam'][:3,3],3)} "
              f"{prm['w']}x{prm['h']}", flush=True)

    writers = {s: imageio.get_writer(str(mvr_dir / f"{s}.mp4"), fps=args_cli.fps, codec="libx264", quality=8)
               for s in mvr_capture.STREAMS}
    forces, nets, opos, oquat, olin, oang, ppos = ([] for _ in range(7))

    obs = env.get_observations()
    with torch.inference_mode():
        for k in range(_steps):
            if not simulation_app.is_running():
                break
            obs, _r, _d, _i = env.step(policy(obs))
            uenv.render()
            frames = reader.read()
            for s, w in writers.items():
                w.append_data(frames[s])
            fc, nc = rich_capture.read_contacts(sensors, uenv.device)
            forces.append(fc.cpu().clone()); nets.append(nc.cpu().clone())
            st = rich_capture.read_state(uenv, pad_ids)
            opos.append(st["obj_pos"].cpu().clone()); oquat.append(st["obj_quat"].cpu().clone())
            olin.append(st["obj_lin_vel"].cpu().clone()); oang.append(st["obj_ang_vel"].cpu().clone())
            ppos.append(st["pad_pos"].cpu().clone())
    for w in writers.values():
        w.close()

    forces_w = torch.stack(forces).numpy(); net_w = torch.stack(nets).numpy()
    opos = torch.stack(opos).numpy()
    contacts = rich_capture.save_contacts(cfm_dir, forces_w, net_w, opos[:, 2])
    rich_capture.save_state(cfm_dir, opos, torch.stack(oquat).numpy(), torch.stack(olin).numpy(),
                            torch.stack(oang).numpy(), torch.stack(ppos).numpy())
    np.savez(mvr_dir / "cam_params.npz",
             **{f"{n}_T_world_cam": cam_params[n]["T_world_cam"] for n in cam_params},
             **{f"{n}_fx": cam_params[n]["fx"] for n in cam_params})
    print(f"[rich] duck lift {opos[:,2].max()-opos[0,2]:+.3f} m; "
          f"peak pad force {np.linalg.norm(forces_w, axis=-1).sum(1).max():.1f} N", flush=True)
    env.close()
    return {"cfm": cfm_dir, "mvr": mvr_dir, "montage": args_cli.out / "montage", "contacts": contacts}


def _run_offline(paths):
    """Render the contact hand + build montages in a fresh process, isolated from
    the live omni/EGL context (simulation_app.close() ends this process, so the
    offline step can't run after it here)."""
    import subprocess
    rich_dir = str(Path(__file__).resolve().parent)
    hand = paths["cfm"] / "contact_hand.mp4"
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from viz_contact_hand import render_contact_hand; "
        "from montage_build import build_montages; "
        "render_contact_hand(%r, %r, fps=%d); "
        "build_montages(%r, %r, %r, fps=%d)"
        % (rich_dir, str(paths["contacts"]), str(hand), args_cli.fps,
           str(paths["mvr"]), str(hand), str(paths["montage"]), args_cli.fps)
    )
    subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    paths = main()
    _run_offline(paths)
    print("\n[rich] outputs:")
    print(f"  mvr videos -> {paths['mvr']}/  ({', '.join(s + '.mp4' for s in mvr_capture.STREAMS)})")
    print(f"  cfm        -> {paths['cfm']}/  (contacts.npz, state.npz, contact_hand.mp4)")
    print(f"  montages   -> {paths['montage']}/  (mvr_views.mp4, mvr_modalities.mp4, master_montage.mp4)")
    sys.stdout.flush()
    simulation_app.close()
    sys.exit(0)
