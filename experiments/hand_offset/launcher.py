"""Run scripts/kinematic_replay.py with monkey-patched constants for the
hand-offset investigation.

This wrapper applies a small set of overrides to ``scripts.utils.constants``
*before* the main replay imports any Isaac Sim modules, then forwards every
remaining CLI argument to ``kinematic_replay.py``. The produced overlay MP4
is moved to ``experiments/hand_offset/outputs/<exp_label>.mp4`` so that
successive runs do not clobber each other.

The move is registered via ``atexit`` because ``SimulationApp.close()``
shuts down the Kit process before control returns from ``runpy``.

Usage:
    python experiments/hand_offset/launcher.py \\
        --exp-label v2_ee_y_pos15mm \\
        --exp-ee-offset 0.13,0.015,0.07 \\
        --refined-extrinsic data/sam_masks_aria_extrinsic.npz \\
        --record-overlay 0.50

Wrapper-only flags (consumed here, not forwarded):
    --exp-label STR              required: tag used for the output filename.
    --exp-ee-offset X,Y,Z        override EE_WRIST_OFFSET_IN_LINK8.
    --exp-cx-shift PX            add to ARIA_INTRINSICS["cx"].
    --exp-cy-shift PX            add to ARIA_INTRINSICS["cy"].
    --exp-fx-scale F             multiply ARIA_INTRINSICS["fx"].
    --exp-fy-scale F             multiply ARIA_INTRINSICS["fy"].
    --exp-base-y-shift M         add to ROBOT_BASE_WORLD_POSITIONS["right"][1].
    --exp-no-refine              strip --refined-extrinsic from the forwarded argv.
"""
import argparse
import atexit
import shutil
import sys
from pathlib import Path


def _split_argv():
    """Pull our --exp-* flags out of sys.argv and leave the rest for kinematic_replay."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--exp-label", required=True)
    parser.add_argument("--exp-ee-offset", default=None,
                        help="X,Y,Z floats (meters), comma-separated")
    parser.add_argument("--exp-cx-shift", type=float, default=0.0)
    parser.add_argument("--exp-cy-shift", type=float, default=0.0)
    parser.add_argument("--exp-fx-scale", type=float, default=1.0)
    parser.add_argument("--exp-fy-scale", type=float, default=1.0)
    parser.add_argument("--exp-base-y-shift", type=float, default=0.0)
    parser.add_argument("--exp-no-refine", action="store_true")
    return parser.parse_known_args()


def _drop_refined_flag(argv):
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--refined-extrinsic":
            skip = True
            continue
        if a.startswith("--refined-extrinsic="):
            continue
        out.append(a)
    return out


def main():
    exp_args, remaining = _split_argv()

    project_root = Path(__file__).resolve().parents[2]
    scripts_dir = project_root / "scripts"
    exp_out = Path(__file__).resolve().parent / "outputs"
    exp_out.mkdir(parents=True, exist_ok=True)

    if exp_args.exp_no_refine:
        remaining = _drop_refined_flag(remaining)

    sys.path.insert(0, str(scripts_dir))
    sys.argv = ["kinematic_replay.py", *remaining]

    # Patch constants BEFORE the main script imports them.
    from utils import constants
    import numpy as np

    if exp_args.exp_ee_offset is not None:
        parts = [float(x.strip()) for x in exp_args.exp_ee_offset.split(",")]
        if len(parts) != 3:
            raise SystemExit("--exp-ee-offset expects exactly 3 comma-separated floats")
        constants.EE_WRIST_OFFSET_IN_LINK8 = np.array(parts, dtype=float)
        print(f"[exp] EE_WRIST_OFFSET_IN_LINK8 = {parts}")

    intr = constants.CAMERA_CONFIGS["aria"]["intrinsics"]
    intr_changed = (
        exp_args.exp_cx_shift != 0.0
        or exp_args.exp_cy_shift != 0.0
        or exp_args.exp_fx_scale != 1.0
        or exp_args.exp_fy_scale != 1.0
    )
    if intr_changed:
        new_intr = dict(intr)
        new_intr["cx"] = intr["cx"] + exp_args.exp_cx_shift
        new_intr["cy"] = intr["cy"] + exp_args.exp_cy_shift
        new_intr["fx"] = intr["fx"] * exp_args.exp_fx_scale
        new_intr["fy"] = intr["fy"] * exp_args.exp_fy_scale
        constants.CAMERA_CONFIGS["aria"]["intrinsics"] = new_intr
        print(f"[exp] aria intrinsics override -> "
              f"fx={new_intr['fx']:.3f} fy={new_intr['fy']:.3f} "
              f"cx={new_intr['cx']:.2f} cy={new_intr['cy']:.2f}")

    if exp_args.exp_base_y_shift != 0.0:
        rbp = dict(constants.ROBOT_BASE_WORLD_POSITIONS)
        rx, ry, rz = rbp["right"]
        rbp["right"] = (rx, ry + exp_args.exp_base_y_shift, rz)
        constants.ROBOT_BASE_WORLD_POSITIONS = rbp
        print(f"[exp] ROBOT_BASE_WORLD_POSITIONS[right] = {rbp['right']}")

    print(f"[exp] Forwarding to kinematic_replay: {sys.argv[1:]}")

    # Register the rename for atexit -- SimulationApp.close() tears down the
    # process before runpy returns, so post-replay code in this function
    # never runs. atexit fires inside the shutdown path.
    out_dir = constants.OUTPUT_DIR
    label = exp_args.exp_label

    def _move_overlay():
        try:
            matches = sorted(
                out_dir.glob("*_overlay_a*.mp4"),
                key=lambda p: p.stat().st_mtime,
            )
            if not matches:
                print(f"[exp] WARNING: no overlay output found in {out_dir}")
                return
            src = matches[-1]
            dst = exp_out / f"{label}.mp4"
            shutil.move(str(src), str(dst))
            print(f"[exp] {src.name} -> {dst}")
        except Exception as exc:
            print(f"[exp] WARNING: rename hook failed: {exc!r}")

    atexit.register(_move_overlay)

    # Run kinematic_replay.py as __main__.
    import runpy
    runpy.run_path(str(scripts_dir / "kinematic_replay.py"), run_name="__main__")


if __name__ == "__main__":
    main()
