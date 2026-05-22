# calibrate_extrinsic_table.py

Refines the egoverse Aria camera's world-frame pose by aligning the
sim-projected table edges with the real-world table edges extracted from
SAM masks. The refined pose is saved as an NPZ that `kinematic_replay.py`
consumes via `--refined-extrinsic`.

## Why this is needed

The Aria glasses are mounted on the wearer's head — the pose relative to
the right-arm base is **only nominally** the same every session. The
default `T_world_cam` we compute as
`T_world_panda_link0 @ ARIA_EXTRINSICS_RIGHT` is correct on average but
typically off by a few centimetres and a few degrees for any specific
capture. When we render the sim from that nominal pose and compare the
masked table against the SAM mask of the same video, they disagree
visibly along the top and left edges.

The refiner produces a per-dataset correction so the sim view matches the
real view.

## What it does

1. Loads the SAM table-mask NPZ (`(N, H, W)` uint8 in `{0, 1}`).
2. Aggregates the masks into a per-pixel table-probability map
   `sam_freq = masks.mean(axis=0)` — this suppresses transient occlusions
   (hand, object).
3. Extracts three dominant 2D lines from `sam_freq`:
   - **Top edge** — far edge of the table (near-horizontal in image).
   - **Left edge** — slanted left side of the table.
   - **Seam** — the physical seam between the two table cells along world Y=0.
4. Reconstructs the nominal `T_world_cam` from `SceneConfig` (mount xyz +
   `ARIA_EXTRINSICS_RIGHT`) without opening Isaac Sim.
5. Runs Levenberg–Marquardt over a 6-DOF twist correction
   `ξ = (tx, ty, tz, rx, ry, rz)` applied as
   `T_refined = se3(ξ) · T_nominal`. The residual per sampled point on
   each 3D line is its signed perpendicular distance to the corresponding
   2D SAM line.
6. Saves `T_refined` plus diagnostics as NPZ.

Isaac Sim is **not** required to run the refiner — only `numpy`, `scipy`,
and (optionally for the diagnostic PNG) `matplotlib`.

## NPZ schema

| Key | Dtype/Shape | Description |
|-----|-------------|-------------|
| `T_world_cam` | `float64 (4, 4)` | Refined camera-in-world pose (column-vector convention) |
| `T_world_cam_nominal` | `float64 (4, 4)` | Nominal pose used as LM starting point |
| `xi` | `float64 (6,)` | LM-found twist correction `(tx, ty, tz, rx, ry, rz)` |
| `residual_rms_px` | `float64` | Final RMS of signed-distance residuals |
| `camera` | `str` | Camera name (`"aria_rgb_cam"`) |
| `sam_masks_path` | `str` | Absolute path of the source SAM NPZ |

## Usage

```bash
conda activate 3dv

# Writes data/sam_masks_aria_rgb_cam_extrinsic.npz
python scripts/calibrate_extrinsic_table.py --sam-masks data/sam_masks.npz

# With before/after diagnostic PNG
python scripts/calibrate_extrinsic_table.py \
    --sam-masks data/sam_masks.npz \
    --diagnostic-png

# Manual seam search band (when the nominal pose is so far off the auto
# prediction misses)
python scripts/calibrate_extrinsic_table.py \
    --sam-masks data/sam_masks.npz \
    --seam-u-range 250 450
```

## Using the refined extrinsic in kinematic replay

```bash
python scripts/kinematic_replay.py \
    --h5 data/h5/session.h5 \
    --refined-extrinsic data/sam_masks_aria_rgb_cam_extrinsic.npz \
    --record-sim --headless
```

When `--refined-extrinsic` is set, `kinematic_replay.py` loads the NPZ and
applies the refinement to:

1. **Viewport camera** — `setup_camera(..., world_pose_override=T_refined)`
   instead of the base-relative computation.
2. **Object trajectory** *(only if you spawn one)* — the `(N, 4, 4)`
   object-in-camera trajectory loaded from `data/poses_<object>_*.npz` is
   mapped to world with `traj_world[i] = T_refined @ traj_cam[i]` instead
   of with the nominal `T_world_cam`. Both are gated on the `camera`
   field inside the NPZ matching `cfg.camera.name` so mismatched setups
   fall back safely.

## How the alignment math works (one paragraph)

The three SAM-detected lines are all on the table-top plane `z = 0.75`.
Each 3D line has two known world-frame endpoints; projecting them through
the current `T_world_cam` gives a 2D line. Each sampled point `p_s` on
the 3D line projects to some `(u_s, v_s)`; its residual is
`n_x u_s + n_y v_s + c` — the signed perpendicular distance to the
fitted SAM line. We sample 40 points per line × 3 lines = 120 residuals,
and LM finds the 6-DOF `ξ` that drives the residual vector to zero.
Because three coplanar lines provide exactly `3 × 2 = 6` independent
constraints on the 8-DOF planar homography induced by the 6-DOF pose,
the problem is just-determined — the solver converges in sub-second to
residual RMS ≈ 0 px.

## Failure modes

| Symptom | Likely cause | Remedy |
|---------|--------------|--------|
| `Could not extract 'seam' line` | Seam not in the default `u` band | Pass `--seam-u-range U0 U1` covering the seam |
| `Could not extract 'left' line` | Left edge off-image | Try `--mask-threshold 0.3` |
| Residual RMS > ~1 px after refine | Line fits noisy (too-few SAM frames) | Feed more SAM frames |
| |Δpos| > 30 cm | Likely wrong USD or wrong SAM masks | Inspect `--diagnostic-png` |

## Related files

- `scripts/utils/calibrate_table.py` — `extract_feature_lines` (SAM line extraction), `refine_world_pose`, `project_world_points`, `se3_from_twist` (LM refinement).
- `scripts/utils/constants.py` — `TABLE_TOP_EDGE_WORLD`, `TABLE_LEFT_EDGE_WORLD`, `TABLE_SEAM_WORLD`, `ARIA_EXTRINSICS_RIGHT`.
- `scripts/utils/camera.py` — `setup_camera(..., world_pose_override=...)`.
- `scripts/kinematic_replay.py` — `--refined-extrinsic` plumbs the override into the camera setup.
