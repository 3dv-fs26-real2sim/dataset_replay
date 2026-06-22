#!/usr/bin/env python3
"""Build an EgoVerse bowl pose npz (panda_link0 frame) for the training env.

The bowl's WORLD pose lives in ``all_object_world_poses.npz`` under the key
``<session>__bowl`` (4x4). Those poses are in the original capture world whose
table top is at ``z ≈ 1.0`` m, whereas the lab env's table top is at 0.75 m
(``utils.config`` ``TableConfig.top_z``); this script shifts z by that delta so
the bowl rests on the env's table, then converts world → panda_link0 by
subtracting the robot-base mount (exactly like ``make_maple_props.py``). The
egoverse env re-adds ``mount_xyz`` on spawn (``attach_bowl --bowl-pose ...
--bowl-pose-frame panda_link0``), keeping the bowl rigid with the rig.

Usage::

    python lab/scripts/make_bowl_pose.py \
        --world-poses .claude/context/all_object_world_poses.npz \
        --session 20250804_104715 \
        --out data/egoverse/demos/egoverse_bowl_20250804_104715.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import lab  # noqa: E402,F401  (bootstrap for utils)
from utils.config import select_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world-poses", type=Path, required=True,
                    help="all_object_world_poses.npz (keys '<session>__bowl')")
    ap.add_argument("--session", type=str, required=True, help="e.g. 20250804_104715")
    ap.add_argument("--out", type=Path, required=True, help="output bowl pose npz")
    ap.add_argument("--capture-table-top", type=float, default=1.0,
                    help="table-top z of the capture world the bowl pose is in (default 1.0).")
    args = ap.parse_args()

    d = np.load(args.world_poses, allow_pickle=True)
    key = f"{args.session}__bowl"
    if key not in d.files:
        raise SystemExit(f"[bowl] {key!r} not in {args.world_poses.name}; "
                         f"available bowl keys: {[k for k in d.files if k.endswith('__bowl')]}")

    cfg = select_config("egoverse")
    mount = np.asarray(cfg.robot.mount_xyz, dtype=np.float64)
    env_top = cfg.table.top_z

    T_world = np.asarray(d[key], dtype=np.float64).reshape(4, 4)
    # Re-reference the capture world (table top ≈ capture_table_top) to the env
    # world (table top env_top) by shifting z, then world → panda_link0 (− mount).
    T_env = T_world.copy()
    T_env[2, 3] += env_top - args.capture_table_top
    T_link0 = T_env.copy()
    T_link0[:3, 3] -= mount

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, pose=T_link0.astype(np.float64))
    print(f"[bowl] {key}: world xyz={np.round(T_world[:3,3],3)} → env xyz="
          f"{np.round(T_env[:3,3],3)} → link0 xyz={np.round(T_link0[:3,3],3)}")
    print(f"[bowl] wrote {args.out}  (frame=panda_link0, mount={tuple(mount)})")


if __name__ == "__main__":
    main()
