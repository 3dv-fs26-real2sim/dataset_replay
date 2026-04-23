# refine_camera_extrinsic.py

Refines the calibrated Aria camera's world-frame pose by aligning the sim-projected table edges with the real-world table edges extracted from SAM masks. The refined pose is saved as an NPZ that `kinematic_replay.py` consumes via `--refined-extrinsic`.

## Why this is needed

The real Aria camera is hand-calibrated per capture session — slightly tilted, slightly shifted. The nominal extrinsic stored in `CAMERA_CONFIGS["aria"]["extrinsics"]` is correct *on average* but off by a few centimetres and a few degrees for any specific session. When we render the sim from that nominal pose and compare the masked table against the SAM mask of the same video, they disagree: IoU ≈ 0.80, with a visibly misaligned top edge and left edge.

The refiner produces a per-dataset correction of the camera's world-frame pose so the sim view matches the real view. Typical correction magnitude on the 20250804 dataset: **~96 mm translation, ~6° rotation.**

## What it does

1. Loads the SAM table-mask NPZ (`(N, H, W)` uint8 in `{0, 1}`).
2. Aggregates the masks into a per-pixel table-probability map `sam_freq = masks.mean(axis=0)` — this suppresses transient occlusions (hand, object).
3. Extracts three dominant 2D lines from `sam_freq`:
   - **Top edge** — far edge of the table (near-horizontal in image).
   - **Left edge** — slanted left side of the table.
   - **Seam** — the physical seam running along the table centre.
4. Opens the USD scene with stand-alone pxr to read the nominal `T_world_base` of the right arm (and left arm in `--mode dual`), then reconstructs the nominal `T_world_cam = T_world_base @ extrinsics["right"]`.
5. Runs Levenberg–Marquardt over a 6-DOF twist correction `ξ = (tx, ty, tz, rx, ry, rz)` applied as `T_refined = se3(ξ) · T_nominal`. The residual per sampled point on each 3D line is its signed perpendicular distance to the corresponding 2D SAM line.
6. Saves `T_refined` plus diagnostics as NPZ.

Isaac Sim is **not** required to run the refiner — only `pxr`, numpy, scipy, and (optionally for the diagnostic PNG) matplotlib.

## Output

| File | Description |
|------|-------------|
| `<sam_stem>_<camera>_extrinsic.npz` | Refined extrinsic + metadata (see schema below) |
| `<sam_stem>_<camera>_extrinsic.png` | *(optional, `--diagnostic-png`)* Before/after overlay on SAM frequency map |

### NPZ schema

| Key | Dtype/Shape | Description |
|-----|-------------|-------------|
| `T_world_cam` | `float64 (4, 4)` | Refined camera-in-world pose (column-vector convention) |
| `T_world_cam_nominal` | `float64 (4, 4)` | Nominal pose used as LM starting point |
| `xi` | `float64 (6,)` | LM-found twist correction `(tx, ty, tz, rx, ry, rz)` |
| `residual_rms_px` | `float64` | Final RMS of signed-distance residuals |
| `camera` | `str` | Camera name (e.g. `"aria"`) — consumed by the replay for dispatch |
| `mode` | `str` | Scene mode at refine time (`"single"` or `"dual"`) |
| `sam_masks_path` | `str` | Absolute path of the source SAM NPZ |

## Usage

```bash
# Single-arm scene (default). Writes data/sam_masks_aria_extrinsic.npz
python scripts/refine_camera_extrinsic.py --sam-masks data/sam_masks.npz

# With before/after diagnostic PNG
python scripts/refine_camera_extrinsic.py \
    --sam-masks data/sam_masks.npz \
    --diagnostic-png

# Dual-arm scene, explicit output path
python scripts/refine_camera_extrinsic.py \
    --sam-masks data/sam_masks_20250829.npz \
    --mode dual \
    --output data/20250829_extrinsic.npz

# Manual seam search band (when the nominal pose is so far off the auto-prediction misses)
python scripts/refine_camera_extrinsic.py \
    --sam-masks data/sam_masks.npz \
    --seam-u-range 150 400
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--sam-masks` | **required** | Path to SAM NPZ with key `mask` of shape `(N, H, W)` |
| `--camera` | `aria` | Camera calibration being refined (key into `CAMERA_CONFIGS`) |
| `--mode` | `single` | Scene mode used to compute the nominal pose (`single` or `dual`) |
| `--output` | auto | Output NPZ path; defaults to `<sam_dir>/<sam_stem>_<camera>_extrinsic.npz` |
| `--mask-threshold` | `0.5` | SAM frequency threshold for the binary mask used in line fitting |
| `--seam-u-range` | auto | Column window searched for the seam; auto = nominal-predicted ±100 px |
| `--diagnostic-png` | off | Also save a nominal-vs-refined overlay PNG next to the NPZ |

### Auto seam band

By default the seam search band is *predicted* from the nominal pose: the 3D seam line is projected, and a ±100 px window is taken around its midpoint. This adapts when the nominal pose drifts between datasets. Override with `--seam-u-range U0 U1` if the prediction still misses (extremely large pose error, lighting causing SAM to segment a different strip, etc.).

## Using the refined extrinsic in kinematic replay

`kinematic_replay.py` gained a `--refined-extrinsic PATH` flag. When set, it loads the NPZ and applies the refinement to **two** places in the pipeline:

1. **Viewport camera** — the USD camera prim is placed at `T_world_cam` instead of computing it from `CAMERA_CONFIGS`.
2. **Object trajectory** — the `(N, 4, 4)` object-in-camera trajectory loaded from `data/poses_<object>_*.npz` is mapped to world with `traj_world[i] = T_world_cam_refined @ traj_cam[i]` instead of with the nominal `T_world_cam`.

Both are gated on the camera name in the NPZ matching the relevant CLI arg (`--camera` and `--object-pose-camera` respectively), so mismatched setups fall back safely to the nominal.

```bash
# Replay with refined extrinsic (applied to camera + object)
python scripts/kinematic_replay.py \
    --refined-extrinsic data/sam_masks_aria_extrinsic.npz \
    --record-sim --headless
```

## Why the object trajectory must also use the refined pose

**Short answer: yes, the refinement changes the object's world-frame pose. Empirically measured for the 20250804 duck trajectory:**

| | |
|--|--|
| Trajectory length | 1199 frames |
| World-pos Δ, min / mean / max | **65.5 mm / 77.1 mm / 92.5 mm** |

The object's pose estimator outputs poses *in camera frame* (one `(4, 4)` per frame). Converting to world requires `T_world_cam`, so if we refine the extrinsic we must refine the object trajectory too. Otherwise the scene is internally inconsistent:

| | Camera pose | Object world pose | Visible in the render? |
|---|---|---|---|
| Neither refined (nominal replay) | nominal | via nominal `T_world_cam` | Table edges misaligned ~30 px against SAM; object looks right *relative to the camera*. |
| Camera only refined | refined | **still via nominal** | Table edges aligned; object drifts by the refinement delta (~96 mm world-frame) relative to the camera → small but real visible mis-render. |
| Both refined ✅ | refined | via refined `T_world_cam` | Table edges aligned; object visually sits in the same place on the table as in the source video. |
| Object only refined | nominal | via refined | Object drifts relative to the mis-aligned camera; worse than nominal. |

### Why this can look nearly identical visually

The *camera-relative* pose of the object is `T_cam_world @ T_world_obj`. If both `T_world_cam` and `T_world_obj` are rotated/translated by the same amount, the camera-relative pose is unchanged and the rendered pixels are nearly identical. That's what happens in both the "neither refined" and "both refined" rows above. The visible *difference* between them is mainly the **table edges** (which live in world frame, so they move when you refine the camera but don't when you refine the object).

So when you run

```bash
python scripts/kinematic_replay.py --refined-extrinsic data/sam_masks_aria_extrinsic.npz
```

and it looks very similar to the nominal replay, that's **expected and correct**: the object is being projected into the image at roughly the same pixel location (because its world pose co-moved with the camera), but its underlying world-frame position actually shifted by ~77 mm on average. What does visibly change:

- The **table mask** rendered by the sim now lines up with the real table (the entire point of the exercise).
- The **robot base** is fixed in world, so the camera looks at it from a slightly different angle — subtle, ~6° — which you can see at frame borders if you flip between recordings.

### Downstream consequences of the refined world object pose

Even when the visual difference is small, the world-frame position does matter for:

- **Collision physics** — in `dynamic_replay.py`, the object collides with the (world-fixed) table and the arm. The refined world pose lands it at the right height above the table.
- **IK targets** — any grasp plan built against the object's world pose uses a position that's now ~77 mm different from the nominal.
- **Comparisons to ground truth** — if object poses are later compared to recorded tracker data (also in world frame), the refined version is the one that matches.

## How the alignment math works (one paragraph)

The three SAM-detected lines are all on the table-top plane `z = 1.0`. Each 3D line has two known world-frame endpoints; projecting them through the current `T_world_cam` gives a 2D line. Each sampled point `p_s` on the 3D line projects to some `(u_s, v_s)`; its residual is `n_x u_s + n_y v_s + c` — the signed perpendicular distance to the fitted SAM line. We sample 40 points per line × 3 lines = 120 residuals, and LM finds the 6-DOF `ξ` that drives the residual vector to zero. Because three coplanar lines provide exactly `3 × 2 = 6` independent constraints on the 8-DOF planar homography induced by the 6-DOF pose, the problem is just-determined — the solver converges in sub-second to residual RMS ≈ 0 px. See `scripts/utils/pose_refine.py` for the implementation and `tmp/table_align/PLAN.md` for the exploration notebook that led to this design.

## Failure modes and what to try

| Symptom | Likely cause | Remedy |
|---------|--------------|--------|
| `Could not extract 'seam' line` | Seam not in the default `u` band | Pass `--seam-u-range U0 U1` covering the seam |
| `Could not extract 'left' line` | Left edge off-image (camera pointing too far right) | Try `--mask-threshold 0.3` to keep more of the mask |
| Residual RMS > ~1 px after refine | Line fits are noisy (too-few SAM frames) | Feed more SAM frames, or raise RANSAC `thresh` in `sam_table_features.py` |
| |T_world_cam_refined − T_world_cam_nominal| > 30 cm | Something is off (wrong USD, wrong SAM masks) | Inspect `--diagnostic-png`; likely the seam was misidentified as a table border |

## Related files

- `scripts/utils/sam_table_features.py` — line extraction from the SAM frequency map.
- `scripts/utils/pose_refine.py` — `se3_from_twist`, `project_world_points`, `refine_world_pose`.
- `scripts/utils/constants.py` — `TABLE_TOP_EDGE_WORLD`, `TABLE_LEFT_EDGE_WORLD`, `TABLE_SEAM_WORLD`.
- `scripts/utils/camera.py` — `setup_camera(..., world_pose_override=...)`.
- `scripts/utils/object.py` — `load_object_world_trajectory(..., T_world_cam_override=...)`.
- `scripts/kinematic_replay.py` — `--refined-extrinsic` plumbs the override into both camera and object.
- `tmp/table_align/PLAN.md` — original exploration and algorithm design notes.
