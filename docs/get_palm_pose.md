# get_palm_pose.py

Extracts a "palm point" pose trajectory from the H5 wrist data by applying a
configurable offset+rotation in the wrist's local frame, and optionally
visualises the result as a sphere in Isaac Sim.

## Motivation

`kinematic_replay.py` drives the arm from H5 `arm_<side>` rows (a wrist pose
per frame). The H5 wrist pose is in **tool convention** (identity = hand
pointing down) and is the EE-wrist frame used by IK:

```
wrist_position = fer_link8_position + R_urdf @ EE_WRIST_OFFSET_IN_LINK8
              = fer_link8_position + R_urdf @ [0.13, 0, 0.07]      (URDF frame)
```

So the H5 wrist already lands in the palm region — no FK through the revolute
`left_wrist` / `right_wrist` joint is needed to get a useful palm-like point.
From there, `--offset-cm` in the wrist's local frame lets you push a few cm
deeper into the palm without running IK or any simulation.

### Frames

The H5 wrist pose is expressed in the **robot base frame** (the frame the
Lula IK solver consumes in `utils/ik.py` — the solver assumes the robot
base is at the origin). The robot base itself is placed at a static world
pose in the USD scene (right arm at world `(-0.262, -0.386, 1.0)` in
`pandaorca_single.usd`). To produce a world-frame palm trajectory, the
script composes one of:

```
--from-link wrist  (default):
  T_world_palm = T_world_base @ T_base_wrist @ T_local_offset

--from-link palm:
  T_world_palm = T_world_base @ T_base_fer_link8 @ T_fer_link8_palm(q) @ T_local_offset
```

`T_world_base` is looked up from the USD (via plain `pxr` on the no-sim
path, or from the live Isaac Sim stage via `utils.camera.get_prim_world_transform`
on the `--visualize` path — both return the same static value).

`T_local_offset` is the 4×4 transform built from `--offset-cm` (translation
in cm → metres) and `--rotation-deg` (extrinsic XYZ Euler in deg), applied
in whichever source link you chose.

### `--from-link wrist` vs `--from-link palm`

- **`wrist` (default, current behaviour):** `T_base_wrist` comes straight
  from the H5 row — position in the robot base frame, quaternion in
  "tool convention" (identity = hand pointing down). This is the same
  frame that drives IK. It's fast and has no hand DOF dependency.
  Caveat: the `<side>_wrist` revolute joint lives *downstream* of this
  frame. When the hand flexes at the wrist, the true palm rotates but
  `T_base_wrist` does not — so an offset applied here drifts relative to
  the real palm as the wrist moves.

- **`palm`:** compute `T_base_palm` by forward-kinematic-ing along the
  URDF chain `fer_link8 → connector_mount → (orcahand) world → <side>_tower
  → <side>_wrist_jointbody → <side>_palm`. Only one joint in this chain
  is moveable (`<side>_wrist`, revolute about ±x), and its value is read
  per frame from `hand_<side>[:, 0]`. The offset is therefore applied in
  the actual palm link frame and tracks the palm correctly regardless of
  wrist flexion.

### How the palm FK chain is built

`scripts/get_palm_pose.py` hard-codes the six joint transforms from
`pandaorca_description/urdf/fer_orcahand_<side>_extended.urdf`:

| Joint | Kind | Source (URDF line, left / right) |
|-------|------|----------------------------------|
| `fer_link8_to_connector_mount` | fixed | left:247, right:247 |
| `connector_mount_to_orcahand_world` | fixed | left:254, right:254 |
| `world2<side>_tower_fixed` | fixed | left:377, right:366 |
| `world2<side>_tower_fixed_offset` | fixed (identity) | left:382, right:371 |
| `<side>_wrist` | revolute (axis ±x) | left:429, right:418 |
| `<side>_wrist_offset` | fixed | left:436, right:425 |

URDF `rpy` is fixed-axis roll-pitch-yaw, which equals scipy's extrinsic
`"xyz"` Euler convention, so each `<origin xyz rpy>` becomes a 4×4 via
`scipy.spatial.transform.Rotation.from_euler("xyz", rpy)` + a translation.

**Which H5 column drives the revolute wrist?** Column 0 of
`hand_<side>`. `utils/constants.py` defines `HAND_LEFT_JOINT_NAMES[0] ==
"left_wrist"` and `HAND_RIGHT_JOINT_NAMES[0] == "right_wrist"`, and
`utils/robot.py:setup_arms_ik` uses that list to map to articulation DOF
indices. `kinematic_replay.py` then passes `data["hand_<side>"][frame]`
straight through to the position setter, which does `buf[hand_idx] =
hand_home + q_hand` with `hand_home = 0`. So `hand_<side>[frame, 0]` is
the wrist angle in radians.

**Verification.** Running Isaac Sim on frame 0 of `20250804_104715.h5`
(wrist at 0.309 rad) and comparing our FK output to the live
`right_palm` prim world transform gave position error ~1 mm and rotation
error ~0.2°. The small residual is Lula IK solver tolerance, not a chain
error — the FK is correct.

## What it does

1. Loads `arm_<side>` from the H5 file (same loader as `kinematic_replay.py`,
   including the wxyz / xyzw auto-detection).
2. Builds `T_wrist_palm` from `--offset-cm` (cm → m) and `--rotation-deg`
   (extrinsic XYZ Euler, in deg) in the wrist's local frame.
3. Per frame: `T_world_palm = T_world_wrist @ T_wrist_palm`.
4. Saves the `(N, 4, 4)` trajectory to NPZ, along with the parameters used.
5. With `--visualize`, replays the scene in Isaac Sim (same articulations /
   IK as `kinematic_replay.py`), sets the viewport to the calibrated camera
   (default `aria`, same as `kinematic_replay.py --camera aria`), renders a
   sphere that tracks the palm point each frame, **and** draws an
   always-on-top debug-draw point so the marker remains visible even when
   the hand geometry is between the camera and the palm.

The computation only depends on `numpy`, `scipy`, and `h5py` — Isaac Sim is
**only** loaded when `--visualize` is set.

## Output

| File | Description |
|------|-------------|
| `data/palm_poses_<h5-stem>_<side>_from-<wrist\|palm>_frame-<camera\|world>_o<X>_<Y>_<Z>[_r<RX>_<RY>_<RZ>].npz` | Default output. Side, source link, output frame, offset (cm), and rotation (deg) are encoded in the filename. Frame tag is the camera name (e.g. `aria`) when `--out-frame=camera`, or `world` when `--out-frame=world`. Negative numbers use `m` (e.g. `-5` → `m5`) and decimals use `p` (e.g. `0.5` → `0p5`). The `_r...` rotation suffix is omitted when `--rotation-deg` is `0 0 0`. Examples: `palm_poses_20250804_104715_right_from-palm_frame-aria_o0_3_0.npz`, `palm_poses_20250804_104715_right_from-palm_frame-world_o0_5_0.npz`. |

### NPZ schema

| Key | Shape | Meaning |
|-----|-------|---------|
| `poses` | `(N, 4, 4)` float64 | Palm pose per H5 frame, in the frame given by `out_frame`. By default this is the Aria camera frame (`T_cam_palm`), matching the convention used by the object-pose NPZs in `data/poses_*.npz`. Pass `--out-frame world` to save `T_world_palm` instead. |
| `offset_cm` | `(3,)` | Echo of `--offset-cm` used. |
| `rotation_deg` | `(3,)` | Echo of `--rotation-deg` used. |
| `side` | `()` str | Which arm (`"left"` / `"right"`). |
| `from_link` | `()` str | Source frame used (`"wrist"` or `"palm"`). |
| `out_frame` | `()` str | `"camera"` or `"world"`. |
| `camera` | `()` str | Camera name used to build `T_world_cam` (e.g. `"aria"`), empty when `out_frame="world"`. |
| `h5_path` | `()` str | Source H5 file. |

## Usage

```bash
# Extract the EE-wrist point directly (default; already near the palm).
python scripts/get_palm_pose.py

# Visualise, with stdout unbuffered so status lines appear promptly.
python -u scripts/get_palm_pose.py --visualize

# Move the marker 10 cm below the palm so it sticks out of the hand and is
# obviously visible (useful for sanity-checking the trajectory before tuning
# the real palm offset).
python -u scripts/get_palm_pose.py --visualize --offset-cm 0 0 -10

# Track the true <side>_palm link (accounts for the revolute wrist joint).
# Default offset puts the marker at the palm link origin.
python -u scripts/get_palm_pose.py --visualize --from-link palm

# Palm link + 3 cm along +y in the palm's local frame.
python -u scripts/get_palm_pose.py --visualize --from-link palm --offset-cm 0 3 0

# Push 5 cm along +y in the wrist frame (deeper into the palm).
python scripts/get_palm_pose.py --offset-cm 0 5 0

# Left arm, dual-mode scene, no NPZ write (just print + optional sim).
python scripts/get_palm_pose.py --mode dual --side left --no-save --visualize

# Visualise the EE wrist itself with a red sphere.
python scripts/get_palm_pose.py --visualize

# Visualise a palm candidate 5 cm deep with a 3 cm green sphere.
python scripts/get_palm_pose.py --offset-cm 0 5 0 \
    --visualize --sphere-radius 0.03 --sphere-color 0 1 0

# Visualise from the aria_half viewport with a larger overlay dot.
python scripts/get_palm_pose.py --offset-cm 0 5 0 \
    --visualize --camera aria_half --overlay-size 40

# 3D sphere only, no always-on-top overlay.
python scripts/get_palm_pose.py --offset-cm 0 5 0 --visualize --no-overlay

# Headless visualisation (still runs Isaac Sim, no window).
python scripts/get_palm_pose.py --offset-cm 0 5 0 --visualize --headless

# Actions instead of observations.
python scripts/get_palm_pose.py --use-actions

# Custom output path.
python scripts/get_palm_pose.py --offset-cm 0 5 0 \
    --output data/my_palm_trajectory.npz
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `single` | Scene / H5 selection (`single`, `dual`). |
| `--side` | `right` | Which arm's wrist to offset from (`left`, `right`). Only `right` is available in single mode. |
| `--use-actions` | off | Use `actions_arm_*` instead of `observations/qpos_arm_*`. |
| `--offset-cm X Y Z` | `0 0 0` | Translation in the wrist's local frame, in **cm**. `0 0 0` = the EE-wrist itself. |
| `--rotation-deg RX RY RZ` | `0 0 0` | Extra rotation applied in the source link's frame (extrinsic XYZ Euler, in **deg**). |
| `--from-link` | `wrist` | Which link to treat as the source frame for `--offset-cm` / `--rotation-deg`. `wrist` = H5 tool-convention wrist frame (fast, no hand DOF dependency). `palm` = walk the URDF chain to the true `<side>_palm` link (accounts for the revolute wrist joint). See above. |
| `--out-frame` | `camera` | Reference frame of the saved `poses` array. `camera` uses `--camera`'s calibrated extrinsics (`T_cam_palm`) — consistent with the object-pose NPZs. `world` saves `T_world_palm` in the USD world frame. |
| `--output` | auto | NPZ output path. Auto: `data/palm_poses_<h5-stem>_<side>_from-<wrist\|palm>_frame-<camera\|world>_o<X>_<Y>_<Z>[_r<RX>_<RY>_<RZ>].npz`. |
| `--no-save` | off | Skip writing the NPZ. |
| `--visualize` | off | Open Isaac Sim and render a tracking sphere. |
| `--camera` | `aria` | Calibrated camera (`aria`, `aria_half`). Used as the viewport camera when `--visualize`, and as the reference frame for the NPZ when `--out-frame=camera`. |
| `--headless` | off | Run Isaac Sim without GUI (only with `--visualize`). |
| `--fps` | `50.0` | Simulation frame rate hint (only with `--visualize`). |
| `--sphere-radius` | `0.02` | Sphere radius in **metres** (only with `--visualize`). |
| `--sphere-color R G B` | `1 0.2 0.2` | Sphere RGB colour in `[0, 1]` (used for both the sphere and the debug overlay). |
| `--no-overlay` | off | Disable the always-on-top debug-draw overlay point. |
| `--overlay-size` | `25` | Debug-draw overlay point size in pixels. |

## Frame conventions

`--offset-cm` and `--rotation-deg` are applied in the **H5 wrist frame**
(tool convention, identity = hand pointing down) — the same frame the
quaternion in `arm_<side>[frame, 3:7]` describes. This is neither the
`fer_link8` URDF frame nor the true `left_palm` / `right_palm` link frame;
it is the frame the H5 data is authored in, which makes direction reasoning
("+y", "+z") match what you see in the H5 pose.

If the recorded quaternion is near identity, the hand points down and:

- `+x` points forward (away from the robot base)
- `+y` points to the hand's left
- `+z` points up

The default `--offset-cm 0 0 0` therefore yields the EE wrist — approximately
the centre of the palm-skin area — and `--offset-cm 0 5 0` pushes 5 cm into
the hand-left side of the palm.

## How the sim visualisation works

When `--visualize` is passed, the script boots Isaac Sim exactly like
`kinematic_replay.py`:

1. `SimulationApp(...)` with the configured headless flag.
2. Open the `single` or `dual` USD scene.
3. `add_articulations` + `setup_arms_ik` — same IK pipeline.
4. `utils.camera.setup_camera(...)` — sets the viewport to the calibrated
   `--camera` (default `aria`).
5. Spawn a `UsdGeom.Sphere` under `/World/palm_marker` with the requested
   radius and display colour.
6. Acquire the Isaac Sim debug-draw interface
   (`isaacsim.util.debug_draw._debug_draw`, with a fallback to the legacy
   `omni.isaac.debug_draw`). Debug-draw points are rendered with depth-test
   disabled, so they remain visible through the hand mesh.
7. Per frame:
   - Drive both arms via `set_positions(arm_row, hand_row)`.
   - Update the sphere pose with `utils.object.set_object_world_pose`.
   - Push an always-on-top point at the palm position via
     `draw.draw_points([...], [...], [overlay_size])`.
   - `world.step(render=True)`.

The sphere is a pure visual prim — no collision, no rigid-body, no physics
interaction. It will not affect the replay. The debug-draw overlay is a
runtime overlay, not a USD prim, so it doesn't persist in the stage.

### Visibility recipes

- **Can't see the sphere at all?** This almost always means the marker is
  *inside* the palm mesh and the hand is occluding it from the Aria camera
  view. The sphere is a USD prim with normal depth testing, so geometry in
  front of it wins. Two fixes:
  - Push it outside the hand with `--offset-cm`. Try `--offset-cm 0 0 -10`
    (10 cm below the palm, hanging under the hand) to confirm it's being
    drawn, then dial in the direction / magnitude you actually want.
  - Look at the **debug-draw overlay dot** — it ignores depth test and is
    visible through the hand. Bump `--overlay-size 50` to make it
    unmissable.
- **No progress output in the terminal?** Run with `python -u` (the script
  also reconfigures stdout to line-buffered, but the `-u` flag is the most
  reliable). Isaac Sim's kit logger occasionally buffers aggressively.
- **Want only the overlay dot, no 3D sphere?** Use `--sphere-radius 0.001`
  (sphere is effectively invisible) or pick a colour that blends in.
- **Want only the 3D sphere, no overlay?** Pass `--no-overlay`.
- **Camera not framing the robot?** Try `--camera aria_half` or omit
  `--visualize` — the NPZ is computed regardless of viewport.

### Diagnostic lines to look for

After Isaac Sim starts up, the script prints the following. If any of these
are missing, that's the failure mode to chase:

```
[palm] Extension enabled: isaacsim.util.debug_draw
[camera] Viewport set to /World/AriaCamera
[palm] T_world_base (from stage) translation = [-0.262 -0.386  1.   ]
[palm] Computed 1199 poses (world frame). First xyz = [ 0.19366904 -0.34439914  1.28763861]
[palm] Spawned VisualSphere at /World/palm_marker (radius=0.03 m).
[palm] Sphere at world xyz = (0.1937, -0.3444, 1.2876), radius = 0.03 m, color = [1.0, 0.2, 0.2]
[palm] Debug-draw overlay acquired via isaacsim.util.debug_draw.
[palm] Visualizing 1199 frames...  (Ctrl-C to stop)
```

The first-frame world xyz should be above the table (`z ≈ 1.29 m`, since the
robot base is at `z = 1.0 m`). If you see `z ≈ 0.29 m` instead, the
world-frame transform got skipped and the sphere will appear below the
table.

If `[palm] VisualSphere unavailable` appears instead, the script has fallen
back to a raw `UsdGeom.Sphere` which doesn't bind a material — the sphere
will render in Isaac Sim's default shading, which is less obvious. If
`[palm] Debug-draw unavailable` appears, the overlay was silently disabled;
the 3D sphere still works.

## Loading the output

```python
import numpy as np

d = np.load(
    "data/palm_poses_20250804_104715_right_from-palm_frame-aria_o0_3_0.npz",
    allow_pickle=False,
)
poses = d["poses"]                  # (N, 4, 4) float64, camera frame by default
positions = poses[:, :3, 3]         # (N, 3) — in Aria camera coords
rotations = poses[:, :3, :3]        # (N, 3, 3)

print("offset (cm):", d["offset_cm"])
print("rotation (deg):", d["rotation_deg"])
print("side:", d["side"])
print("from_link:", d["from_link"])
print("out_frame:", d["out_frame"], "camera:", d["camera"])
print("source:", d["h5_path"])
```

### Converting camera-frame → world-frame after the fact

If you later need the same trajectory in world frame, multiply by the camera's
world pose (same formula `utils/object.py` uses for object-pose NPZs):

```python
from utils.constants import CAMERA_CONFIGS
# T_world_base for the relevant arm, e.g. right in single mode:
T_world_base = np.eye(4); T_world_base[:3, 3] = [-0.262, -0.386, 1.0]
T_world_cam  = T_world_base @ CAMERA_CONFIGS["aria"]["extrinsics"]["right"]
poses_world  = T_world_cam @ poses
```

## Design notes

- **No edits to other files.** All logic lives in `scripts/get_palm_pose.py`.
  Helpers that would normally come from `utils/app.py` (`resolve_h5_path`,
  `resolve_usd_path`) are inlined because importing `utils/app.py` pulls in
  `isaacsim` at module load time, which is unwanted on the no-sim code path.
- **Why approximate, not exact FK to `left_palm`.** The true `left_palm`
  frame is downstream of the revolute `left_wrist` joint (part of the hand
  DOFs, not the arm DOFs). Building a dedicated FK path just for this
  experiment is more machinery than the use case — extracting a point "near
  the palm" for labelling / downstream pose data — actually needs. The
  `--offset-cm` / `--rotation-deg` pair lets you tune the approximation by
  visual inspection with `--visualize`.
- **Reproducibility.** The parameters used are saved into the NPZ so the
  trajectory can be regenerated exactly from a given H5 file.

## Related scripts

- `kinematic_replay.py` — Full IK replay the wrist data comes from. Uses the
  same `arm_<side>` rows this script reads.
- `utils/rotation.py` — `detect_quaternion_order`, `wxyz_to_rotation_matrix`.
- `utils/h5_loader.py` — `load_h5`.
- `utils/object.py` — `set_object_world_pose` (used for the sphere in
  `--visualize`).
