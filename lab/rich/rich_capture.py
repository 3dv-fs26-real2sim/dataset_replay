"""Contact-force + object/hand state capture helpers.

Per-pad ContactSensors (force vs the manipulated object) are injected onto the
scene cfg at runtime, so no shared task file is touched. The per-link force is the
sum over that link's colliders (skin pad + structural), since PhysX reports
contact per rigid body. Also reads object 6-DoF + velocity and pad-link poses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# 5 distal tips + 5 mid pads + palm; order reused by every downstream consumer.
PAD_LINKS = [
    "right_index_ip", "right_middle_ip", "right_ring_ip", "right_pinky_ip", "right_thumb_dp",
    "right_index_pp", "right_middle_pp", "right_ring_pp", "right_pinky_pp", "right_thumb_ip",
    "right_palm",
]


def inject_contact_sensors(env_cfg, pad_links=PAD_LINKS):
    """Add one ContactSensor per pad link (force vs the object) onto env_cfg.scene.

    force_matrix_w (filtered) reporting is one-sensor-body to many-filter-shapes,
    so it must be one sensor per pad. Picked up by InteractiveScene because it
    iterates the scene cfg's __dict__.
    """
    from isaaclab.sensors import ContactSensorCfg
    env_cfg.scene.robot.spawn.activate_contact_sensors = True
    if getattr(env_cfg.scene, "object", None) is not None:
        env_cfg.scene.object.spawn.activate_contact_sensors = True
    for ln in pad_links:
        setattr(env_cfg.scene, f"contact_{ln}", ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + ln,
            update_period=0.0, history_length=0,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]))


def get_sensors(uenv, pad_links=PAD_LINKS):
    return {ln: uenv.scene.sensors[f"contact_{ln}"] for ln in pad_links}


def pad_body_ids(robot, pad_links=PAD_LINKS):
    return [robot.body_names.index(ln) for ln in pad_links]


def read_contacts(sensors, device, pad_links=PAD_LINKS):
    """(force_vs_object (11,3), net_force (11,3)) for env 0 this step."""
    import torch
    fk, nk = [], []
    for ln in pad_links:
        s = sensors[ln]
        fm = s.data.force_matrix_w               # (N,1,M,3) vs object; sum over object hulls
        fk.append(fm[0, 0].sum(dim=0) if fm is not None else torch.zeros(3, device=device))
        nf = s.data.net_forces_w                 # (N,1,3) total contact on the body
        nk.append(nf[0, 0] if nf is not None else torch.zeros(3, device=device))
    return torch.stack(fk), torch.stack(nk)


def read_state(uenv, body_ids):
    """Object pose/velocity and pad-link world positions for env 0 this step."""
    obj = uenv.scene["object"]
    robot = uenv.scene["robot"]
    return {
        "obj_pos": obj.data.root_pos_w[0],
        "obj_quat": obj.data.root_quat_w[0],
        "obj_lin_vel": obj.data.root_lin_vel_w[0],
        "obj_ang_vel": obj.data.root_ang_vel_w[0],
        "pad_pos": robot.data.body_pos_w[0, body_ids],
    }


def save_contacts(out_dir, forces_w, net_w, duck_z, pad_links=PAD_LINKS):
    """Write contacts.npz; returns its path."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    fmag = np.linalg.norm(forces_w, axis=-1)
    path = out / "contacts.npz"
    np.savez(path, forces_w=forces_w, force_mag=fmag, total_mag=fmag.sum(1),
             net_forces_w=net_w, body_names=np.array(pad_links),
             frames=np.arange(len(forces_w)), duck_z=duck_z)
    return path


def save_state(out_dir, obj_pos, obj_quat, obj_lin_vel, obj_ang_vel, pad_pos, pad_links=PAD_LINKS):
    """Write state.npz; returns its path."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    path = out / "state.npz"
    np.savez(path, obj_pos=obj_pos, obj_quat=obj_quat, obj_lin_vel=obj_lin_vel,
             obj_ang_vel=obj_ang_vel, pad_pos=pad_pos, pad_links=np.array(pad_links),
             frames=np.arange(len(obj_pos)))
    return path
