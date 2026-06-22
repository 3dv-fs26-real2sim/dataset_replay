# dataset_replay

Single-arm **kinematic replay** of teleoperated manipulation recordings in
Isaac Sim. Given an H5 trajectory, it rebuilds the capture scene, drives a
Panda + OrcaHand robot through the recorded motion frame-by-frame, optionally
replays a tracked object, and renders the result — including sim/real overlays
for visually checking calibration.

Two recording rigs share one codebase:

- **Maple** — OAK-D camera, U-shape walls, an AprilTag on the table.
- **Egoverse** — Aria glasses (head-mounted), open desk (no walls, no tag).

The codebase is split so the two rigs share the entire scene / robot / IK /
capture stack via dataclass-typed configs; the only dataset-specific code is
the two replay scripts plus the per-rig calibration library. Calibration runs
**in-process at startup** against the H5 image data — there are no on-disk
extrinsic artefacts to manage.

---

## The two rigs: Maple vs Egoverse

Both rigs replay the **same robot** (7-DOF Franka Panda arm + 17-DOF right
OrcaHand) on the **same scene** (two 1.0 × 0.7 m tables side-by-side, combined
surface 1.0 × 1.4 m centred at the world origin, top at `Z = 0.75 m`), using
the **same IK solver** and the **same capture pipeline**. They differ only in
the camera, the surrounding scene furniture, and the calibration method:

| | **Maple** | **Egoverse** |
|---|---|---|
| Camera | OAK-D Pro AF, **world-fixed** above the open −X edge | Aria glasses, **head-mounted**, posed relative to the robot base |
| Intrinsics | doc K, `fx=fy=367.16` @ 480×270 | `fx=fy=266.51` @ 640×480 |
| Render viewport | 1280×720 (16:9) | 1280×960 (2× 640×480, 4:3) |
| Scene furniture | U-shape **walls** + **AprilTag** flat on the table | open desk — **no walls, no tag** |
| Calibration | AprilTag detect + joint PnP/RANSAC, scanning the H5 video | SAM table-mask edge fit + LM refinement |
| Calibration input | the H5 image stream itself (in-process) | `data/egoverse/desk/<stem>_desk.npz` |
| Nominal-pose fallback | configured OakD lookat pose | `T_world_base @ ARIA_EXTRINSICS_RIGHT` |
| H5 arm key | `observations/qpos_arm_right` | `observations/qpos_arm` |
| H5 hand key | `observations/qpos_hand_right` | `observations/qpos_hand` |
| H5 image key | `observations/images/oakd_front_view/color` | `observations/images/aria_rgb_cam/color` |
| Top-level actions | none | `actions_arm` / `actions_hand` (`--use-actions`) |
| Per-frame extrinsics in H5 | yes (`T_robot_cam`) | no |
| Per-frame intrinsics in H5 | yes | no (static, in `constants.py`) |
| Object-trajectory frame (default) | `camera` (`T_cam_obj`) | `camera` (`T_cam_obj`) |
| Capture rate | 10 Hz | 50 Hz |
| Robot mount xyz | `(-0.255, -0.35, 0.75)` | `(-0.246, -0.350, 0.75)` |

The split is enforced structurally: shared modules consume a
`BaseSceneConfig` and branch on its decision hooks `cfg.has_walls()`,
`cfg.has_apriltag()`, `cfg.viewport_size()` — never on `isinstance`. Maple
flips both walls/tag hooks on; Egoverse leaves them off. See
`utils/config_maple.py` and `utils/config_egoverse.py` for the concrete
values, and the geometry sketches in each.

---

## Setup

Everything runs in one conda env (`3dv`): IsaacSim 5.1.0 + IsaacLab 2.3.x +
rsl-rl 3.1.2 on Python 3.11 / Torch 2.7 (CUDA 12.8). The pinned spec lives in
[`environment.yml`](environment.yml) (it replaces the old `requirements.txt`).

```bash
# 1. Create the env (IsaacSim + Torch + rsl-rl + replay/capture extras).
conda env create -f environment.yml      # creates env "dataset_replay"
conda activate dataset_replay            # (or reuse an existing "3dv" env)

# 2. Install IsaacLab from source (sub-packages are not on PyPI).
git clone --depth 1 --branch v2.3.0 https://github.com/isaac-sim/IsaacLab.git ../IsaacLab
pip install -e ../IsaacLab/source/isaaclab
pip install -e ../IsaacLab/source/isaaclab_assets
pip install -e "../IsaacLab/source/isaaclab_rl[rsl-rl]"
pip install -e ../IsaacLab/source/isaaclab_tasks
```

IsaacLab is only needed for the **residual-RL training env** under
[`lab/`](lab/) — kinematic replay (the `scripts/` entry points) needs only
IsaacSim. The first IsaacSim launch needs `OMNI_KIT_ACCEPT_EULA=YES` (cached
afterwards).

Data goes under `data/<dataset>/`:

- H5 recordings → `data/<dataset>/h5/`
- object pose trajectories → `data/<dataset>/pose/`
- SAM table masks (egoverse only for calibration) → `data/egoverse/desk/<h5_stem>_desk.npz`
- residual-RL demos → `data/<dataset>/demos/` (built by `lab/scripts/make_demo.py`)

Object meshes live under `assets/objects/<name>/<name>.obj` (kinematic replay)
and `assets/objects/<name>/<name>_vhacd.usd` (RL grasp colliders).

---

## Residual-RL training (`lab/`)

Beyond kinematic replay, [`lab/`](lab/) trains a **residual policy** that makes
the recorded grasp robust under physics: a deterministic per-frame baseline
plays the demo's joint targets, and a PPO policy learns a small correction on
top.

```
joint_target[t] = recorded_qpos[t]  +  residual_scale * policy(obs)
```

The baseline arm joints (`arm_qpos`) come from the **same Lula IK** the replay
scripts use (`utils/ik.py`), so replay and training share one kinematic
convention. Two gym tasks are registered, one per rig — `egoverse` (duck) and
`maple` (pan + optional static props). See [`lab/README.md`](lab/README.md) for
the full design; quick start:

```bash
conda activate dataset_replay

# EgoVerse duck-grasp
python lab/train.py --task egoverse \
    --demo data/egoverse/demos/egoverse_duck_104715.npz --num_envs 64 --headless

# MAPLE pan (with static props)
python lab/train.py --task maple \
    --demo data/maple/demos/maple_pan_143954.npz \
    --maple-props data/maple/demos/maple_props_143954.npz --num_envs 64 --headless

# Build a new demo from a raw H5 (dataset_replay-style Lula IK retarget)
python lab/scripts/make_demo.py --dataset egoverse \
    --h5 data/egoverse/h5/20250804_104715.h5 --object duck \
    --out data/egoverse/demos/egoverse_duck_104715.npz

# Simulation-rich rollout capture (multi-view render + per-pad contact force + montage)
python lab/rich/run_rich.py --task egoverse \
    --demo data/egoverse/demos/egoverse_duck_104715.npz \
    --checkpoint logs/rsl_rl/teleop_residual/<run>/model_<n>.pt --headless
```

---

## How kinematic replay works

"Kinematic" means **nothing is dynamically simulated**: the robot's joints are
*set* directly each frame and the object's pose is *prescribed* each frame.
PhysX never integrates contact forces or gravity for the driven bodies — they
follow the recorded trajectory exactly, with no drift, no settling, no
collision response. This is what makes a replay a faithful playback of the
recording rather than a physics rollout.

A run proceeds in four stages.

### 1. Build the scene (`utils/scene.build_scene`)

A fresh USD stage is created from scratch — Z-up, metric (1 m units) — and
populated procedurally:

- a physics scene with gravity along −Z (present but inert for the driven bodies),
- a ground collision plane at `z = 0`,
- a distant light,
- `cfg.table.n_tables` cuboid table cells tiled along Y,
- **walls** (Maple only, via `cfg.has_walls()`) — a U opening toward −X,
- an **AprilTag** plane (Maple only, via `cfg.has_apriltag()`),
- the **robot**, referenced from `assets/orcav1b_franka_vmnt_v10_flattened.usd`
  with a wrapper transform that places `panda_link0` exactly at
  `cfg.robot.mount_xyz`.

`build_scene` is called with `robot_collision=False` for replay. That disables
the articulation's self-collisions (`physxArticulation:enabledSelfCollisions =
False`) and adds a `FilteredPairsAPI` against the tables, walls, and ground.
The consequence: when a replayed pose makes the hand intersect the table or
itself, PhysX does **not** push back — the joints stay exactly where they were
teleported, so the robot never jitters.

### 2. Set up the robot (`utils/robot.setup_robot`)

The articulation (7 arm DOFs + 17 hand DOFs) is registered with the world, DOF
indices are resolved by joint name, and a **home pose** is solved once with the
Lula IK solver. `setup_robot` returns a `set_positions(wrist_pose, hand_q)`
callable that the replay loop drives every frame.

### 3. Resolve the camera pose (per-rig calibration)

The camera's world pose `T_world_cam` is calibrated **once at startup** from the
H5 data — no on-disk extrinsics:

- **Maple** scans the H5 video for the AprilTag and runs joint PnP+RANSAC
  against the tag's measured world pose (`utils/calibrate_april.py`).
- **Egoverse** loads the SAM table mask and fits the table edges with LM
  refinement off the nominal Aria extrinsic (`utils/calibrate_table.py`).

Either falls back to a configured nominal pose if its input is unavailable (tag
not detected / mask missing), and the resolved pose is written onto a USD
camera prim that frames the rendered viewport.

### 4. The replay loop

The H5 yields, per frame: a wrist target `[x, y, z, qw, qx, qy, qz]` and 17
hand joint angles. For each frame `i`:

1. **Arm via IK.** The wrist quaternion is converted from the recording's
   tool convention into the URDF convention (handedness flip + axis swap +
   `Rx(180°)`, see `utils/rotation.tool_quat_to_urdf`). The IK target is shifted
   from the recorded EE-wrist point back to `panda_link8`'s origin using
   `EE_WRIST_OFFSET_IN_LINK8 = [0.13, 0, 0.07]`. Lula IK solves the 7 arm
   joints, **warm-started** from the previous frame for temporal continuity. On
   an IK failure the arm holds its previous joints and a counter is bumped
   (reported at the end of the run).
2. **Hand directly.** The 17 hand joints are set as `hand_home + q_hand` — no
   IK, a straight joint copy.
3. **Teleport.** `articulation.set_joint_positions(...)` writes the full joint
   vector. This is a position *set*, not a torque command — hence "kinematic".
4. **Object (optional).** If an object trajectory is loaded, the object prim is
   teleported to `T_world_obj[i]` (`utils/object.set_object_world_pose`). The
   object is a kinematic rigid body with gravity disabled and collision off, so
   it too just follows its prescribed path.
5. **Step + render.** `world.step(render=True)` advances one sim/render tick and
   (if recording) the viewport frame is captured.

The loop runs `min(n_h5_frames, n_object_frames)` frames.

### Objects and trajectory frames

Objects are spawned from `assets/objects/<name>/<name>.obj` in **kinematic,
no-collision** mode (`utils/object.spawn_object`). An object can be spawned
without an `--h5` (handy for eyeballing placement); when a trajectory is
supplied the per-frame poses override the static placement.

Object trajectories are `(N, 4, 4)` stacks of homogeneous transforms in an
`.npz` (single array). The 6D pose estimator outputs object poses in the
**camera frame** (`T_cam_obj` per frame), so **both rigs default to `camera`**:
each pose is composed at replay time as `T_world_obj = T_world_cam @ T_cam_obj`,
landing the sim object where the real object was relative to the camera. This
requires the camera to be set up (don't pass `--no-camera`).

Pass `--object-traj-frame world` only for the rare NPZ that is already
world-frame (used verbatim, no composition).

### Recording outputs

With recording flags, each replayed frame can be written to:

- `--record-sim` — the raw Isaac Sim viewport,
- `--record-sidebyside` — sim and the H5 frame placed side-by-side,
- `--record-overlay A [B ...]` — sim alpha-blended over the H5 frame (one MP4
  per alpha) — the primary tool for checking sim/real calibration.

Output FPS is `--fps` (defaults to the per-rig capture rate), divided by
`--sample-every N` if subsampling. The replay teleports joints once per H5
frame regardless; only the *output video* is retimed.

---

## Usage

```bash
# ── Egoverse (Aria) ─────────────────────────────────────────────────────────
# No --h5: build the scene, set home pose, hold (test_setup-style).
python scripts/kinematic_replay_egoverse.py

# Full replay; SAM table-mask refines the Aria pose at startup if the mask
# is present, else nominal pose with a warning.
python scripts/kinematic_replay_egoverse.py --h5 data/egoverse/h5/20250804_104715.h5

# Replay with the duck, driven by its tracked trajectory. --object-traj-frame
# defaults to `camera`, matching the (1199, 4, 4) T_cam_obj duck NPZ; each pose
# is composed with T_world_cam at replay time.
python scripts/kinematic_replay_egoverse.py \
    --h5 data/egoverse/h5/20250804_104715.h5 \
    --object duck \
    --object-traj data/egoverse/pose/20250804_104715_duck.npz

# Recorded overlay (headless).
python scripts/kinematic_replay_egoverse.py --headless \
    --h5 data/egoverse/h5/20250804_104715.h5 --record-overlay 0.5

# Drive from top-level actions instead of recorded qpos.
python scripts/kinematic_replay_egoverse.py \
    --h5 data/egoverse/h5/20250804_104715.h5 --use-actions

# ── Maple (OakD) ────────────────────────────────────────────────────────────
# No --h5: scene + home pose, holds. OakD uses nominal lookat (no image, no calib).
python scripts/kinematic_replay_maple.py

# Full replay; AprilTag-PnP refines T_world_cam at startup by scanning the
# H5 video (~5 s; full scan).
python scripts/kinematic_replay_maple.py --h5 data/maple/h5/20250922_143954.h5

# ── Utilities ───────────────────────────────────────────────────────────────
# Extract H5 camera video to MP4 (no GPU needed). FPS picked from camera
# name prefix: oakd_* → 10, aria_* → 50. Override with --fps.
python scripts/record_h5.py --h5 data/maple/h5/20250922_143954.h5 --camera oakd_front_view
python scripts/record_h5.py --h5 data/egoverse/h5/20250804_104715.h5 --camera aria_rgb_cam

# Spot-check the runtime calibration (writes NPZ + overlay PNG, optionally MP4).
python scripts/calibrate/calibrate_april.py --h5 data/maple/h5/20250922_143954.h5 --viz-mp4
python scripts/calibrate/calibrate_table.py --h5 data/egoverse/h5/20250804_104715.h5 --viz-mp4
```

All replay flags are documented inline (`python scripts/kinematic_replay_<dataset>.py --help`).

---

## Calibration

Both rigs auto-calibrate at startup from the H5 image stream. **No on-disk
extrinsic files** — calibration is in-process every run.

- **Maple**: `utils.calibrate_april.calibrate_from_h5` scans every H5 frame,
  detects the AprilTag, runs joint PnP+RANSAC. ~5 s for a 300-frame recording.
  Falls back to the nominal `OakDCameraConfig` lookat pose if the tag is
  undetectable. Pass `--no-calibrate` to skip the scan entirely.
- **Egoverse**: `utils.calibrate_table.refine_aria_extrinsic` loads
  `data/egoverse/desk/<stem>_desk.npz`, extracts the top/left/seam edge fits,
  runs LM refinement off `T_world_base @ ARIA_EXTRINSICS_RIGHT`. Falls back to
  the nominal Aria pose if the mask file is missing. Pass `--no-refine` to skip.

> **Intrinsics note (Maple).** PnP and the sim render share **one** K — the
> OAK-D spec-sheet ("doc") K, `fx=fy=367.16` at 480×270 — not the H5-stored K
> (`fx≈299`). Both stages must use the same K, or the sim feed is zoomed
> relative to the H5 in overlay/side-by-side videos. See
> `OakDCameraConfig` for the rationale.

The diagnostic scripts under `scripts/calibrate/` invoke the same library
functions and emit NPZ + overlay PNG/MP4 to `outputs/calibration/` for visual
spot-checking.

---

## Project Structure

```text
scripts/
├── kinematic_replay_maple.py     # Replay + auto-AprilTag-calib at startup
├── kinematic_replay_egoverse.py  # Replay + auto-SAM-table-calib at startup
├── record_h5.py                  # Schema-agnostic H5 → MP4 extractor
├── calibrate/                    # Sanity-check tools (NPZ + overlay PNG/MP4)
│   ├── calibrate_april.py
│   └── calibrate_table.py
└── utils/
    ├── __init__.py
    │
    ├── app.py                    # SimulationApp + shared argparse (--headless, --fps)
    ├── capture.py                # Viewport video capture / side-by-side / overlay
    ├── constants.py              # Shared constants (paths, joints, Aria K/extrinsic, FPS dict)
    ├── rotation.py               # Quaternion math (pure numpy)
    ├── ik.py                     # IK solver creation + per-frame position setter
    ├── object.py                 # Object spawning + per-frame pose updates
    ├── poses.py                  # 6D pose trajectory loader
    ├── robot.py                  # Articulation setup + home pose
    ├── viewport.py               # omni.kit.viewport.utility wrapper
    ├── textures.py               # USD material/texture helpers
    ├── generate_lula_description.py
    │
    ├── config.py                 # BASE: TableConfig, RobotMountConfig,
    │                             #       BaseCameraConfig, BaseSceneConfig, select_config()
    ├── config_maple.py           # WallsConfig, AprilTagConfig, OakDCameraConfig, MapleSceneConfig
    ├── config_egoverse.py        # AriaCameraConfig, EgoverseSceneConfig
    │
    ├── scene.py                  # Shared builder; reads cfg.has_walls() / cfg.has_apriltag()
    ├── camera.py                 # Shared USD-camera prim writer (pose comes from caller)
    │
    ├── apriltag.py               # MAPLE-ONLY USD tag plane builder
    ├── calibrate_april.py        # MAPLE-ONLY: AprilTag detect + PnP → T_world_cam
    ├── calibrate_table.py        # EGOVERSE-ONLY: SAM table-edge refinement → T_world_cam
    │
    └── h5_loader.py              # Dataset-keyed schema: H5Schema, H5Reader(dataset=...)
```

**File naming rule:** `*_maple.py` / `*_egoverse.py` suffixes mark
dataset-specific modules. Files without a suffix are shared. `apriltag.py`,
`calibrate_april.py`, `calibrate_table.py` don't carry suffixes because the
name itself is already rig-specific.

**Config dispatch rule:** shared modules consume `BaseSceneConfig` and branch
on `cfg.has_walls()` / `cfg.has_apriltag()` / `cfg.viewport_size()`. No
`isinstance(cfg, MapleSceneConfig)` in shared code.

---

## Data layout

H5 stems are recording timestamps (`YYYYMMDD_HHMMSS`). Object-pose and SAM-mask
files are keyed to the H5 stem they belong to. Current contents:

```text
data/
├── maple/                                   # OakD recordings, 10 Hz
│   ├── h5/
│   ├── pose/
│   └── video/
└── egoverse/                                # Aria recordings, 50 Hz
    ├── h5/
    ├── pose/
    ├── desk/
    └── video/
```
