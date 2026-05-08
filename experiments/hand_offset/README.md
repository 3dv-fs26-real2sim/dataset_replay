# Hand-offset investigation under `--refined-extrinsic`

## Symptom

When running

```bash
python scripts/kinematic_replay.py --refined-extrinsic data/sam_masks_aria_extrinsic.npz
```

the rendered hand sits a little to the right of where the H5 video shows it
— a small lateral shift that does not look perfectly axis-aligned. The
table, robot base and viewport framing look correct: the discrepancy is
isolated to the hand.

This is consistent with the geometry of the refinement: `T_world_cam` is
solved against table landmarks (residual ~3.5e-13 px against the table
corners), so anything that lies on the table plane — and only the table
plane — is fit perfectly. A hand at ~0.5–0.7 m camera-frame depth still
sees the projection of any small *rotational* refinement error around an
axis through the table, so a near-zero table residual leaves room for a
visible hand-depth residual.

## Hypotheses

The hand pose in image space is the projection of a chain of transforms:

```
image_px = K_aria @ T_cam_world @ T_world_base @ FK(qpos)_link8 @ T_link8_wrist
```

Each link in the chain is a candidate source for the offset:

| # | Where | Hypothesis |
|---|-------|------------|
| H1 | `EE_WRIST_OFFSET_IN_LINK8` lateral | The orcahand is mounted with a small Y offset relative to panda_link8, which is currently set to `0.0`. A non-zero Y shifts the wrist sideways in world frame. |
| H2 | `EE_WRIST_OFFSET_IN_LINK8` reach | The 0.13 m forward offset along link8 +X (`0.13, 0.0, 0.07`) was provided by supervisors but not measured here; a longer/shorter offset moves the wrist along the gripper's pointing axis. |
| H3 | `ARIA_INTRINSICS["cx"]` | The intrinsic principal point is set to the image centre (320, 240), which is an assumption. If the actual `cx` is offset by a few pixels, every projected point shifts by the same amount in image space — exactly the symptom. |
| H4 | `ARIA_INTRINSICS["fx"]` / `fy` | The focal length is `133.25 × 2 = 266.5`. If the reported half-res focal length was rounded, projections at non-zero image offsets are scaled slightly wrong, which manifests as a position-dependent shift. |
| H5 | Refinement quality at hand depth | The refined extrinsic perfectly fits table corners but is unconstrained at hand depth; a sanity check is to render with the *nominal* extrinsic so we can see whether refinement improved or worsened the hand region overall. |

## Variants

Each variant runs the standard kinematic replay with
`--refined-extrinsic data/sam_masks_aria_extrinsic.npz --record-overlay 0.50`,
swapping in one override at a time. Outputs land in `outputs/<label>.mp4`.

| Label | Override | Tests |
|-------|----------|-------|
| `v0_baseline` | (none) | The user's current view. Reference for diffing. |
| `v1_no_refine` | drop `--refined-extrinsic` | H5 — does refinement help at all in the hand region? |
| `v2_ee_y_pos15mm` | `EE_WRIST_OFFSET_IN_LINK8 = (0.13, +0.015, 0.07)` | H1, sign A |
| `v3_ee_y_neg15mm` | `EE_WRIST_OFFSET_IN_LINK8 = (0.13, -0.015, 0.07)` | H1, sign B |
| `v4_ee_x_pos20mm` | `EE_WRIST_OFFSET_IN_LINK8 = (0.15,  0.000, 0.07)` | H2 |
| `v5_cx_minus15` | `ARIA_INTRINSICS["cx"] -= 15 px` | H3, sign A |
| `v6_cx_plus15` | `ARIA_INTRINSICS["cx"] += 15 px` | H3, sign B |
| `v7_focal_scale_1p03` | `fx, fy *= 1.03` | H4 |

### Reading the overlays

At alpha = 0.50, the simulated robot/hand/duck and the H5 RGB are equally
weighted. The hand offset shows up as a "double image" of the hand; the
variant whose double-image is *most collapsed* is the one closest to the
true configuration.

The duck (rendered from the same refined camera but driven by an
independent 6D-pose trajectory) is a free differential signal:

* If the **duck shifts in the same direction as the hand**, the offset
  is camera-side (`cx`/`cy`/focal length / extrinsic) — robot-side
  variants will not fix it.
* If the **duck stays put while only the hand shifts**, the offset is
  robot-side (EE wrist offset, base placement, IK). Camera-side variants
  will not fix it.

This narrows the search before you study individual variants.

* If `v2` or `v3` collapses the double image but the others don't, the
  fix is **the EE_WRIST_OFFSET_IN_LINK8 Y component**: bake the chosen sign
  and magnitude into `scripts/utils/constants.py`.
* If `v4` collapses it, the fix is the **X component** (length along
  link8) — same place to edit.
* If `v5` or `v6` collapses it but the table double-image *also* shifts,
  the fix is in **`ARIA_INTRINSICS["cx"]`**. (Watch the duck and table
  edges too: an intrinsic fix moves *every* projected world point.)
* If `v7` collapses it (and table edges scale as expected), the focal
  length is the culprit.
* If `v1_no_refine` looks closer than `v0_baseline` in the hand region,
  the refinement is being over-constrained by the table and pulling the
  camera in a direction that worsens hand-depth projection — investigate
  the refinement objective rather than tuning here.
* If *none* of these variants improves on `v0_baseline`, the offset most
  likely comes from somewhere outside this hypothesis set; promising
  follow-ups include lens distortion (Aria is a wide-angle fisheye but
  is rendered here with a pinhole model) or a robot-base world-pose
  miscalibration.

## How to run

```bash
conda activate 3dv
bash experiments/hand_offset/run_all.sh
```

Run a subset (e.g. only the EE-offset variants) by setting `ONLY`:

```bash
ONLY=v0_baseline,v2_ee_y_pos15mm,v3_ee_y_neg15mm \
    bash experiments/hand_offset/run_all.sh
```

Append flags (e.g. `--object none` to remove the duck for cleaner overlays):

```bash
EXTRA_ARGS="--object none" bash experiments/hand_offset/run_all.sh
```

Run a single variant directly:

```bash
python experiments/hand_offset/launcher.py \
    --exp-label v2_ee_y_pos15mm \
    --exp-ee-offset 0.13,0.015,0.07 \
    --refined-extrinsic data/sam_masks_aria_extrinsic.npz \
    --record-overlay 0.50
```

## How the launcher works

`launcher.py` peels off the wrapper-only `--exp-*` flags, monkey-patches
`scripts/utils/constants.py` *before* any Isaac Sim import, then runs
`scripts/kinematic_replay.py` as `__main__`. The kinematic replay reads
the patched constants exactly the same way as a fresh run; nothing about
the main pipeline changes. After the replay closes, the produced overlay
MP4 (a single `*_overlay_a0.50_*.mp4` in the project's `outputs/`) is
moved to `experiments/hand_offset/outputs/<exp_label>.mp4` so the next
variant does not overwrite it.

Wrapper flags consumed locally:

| Flag | Effect |
|------|--------|
| `--exp-label` | Output filename stem (required). |
| `--exp-ee-offset X,Y,Z` | Replaces `EE_WRIST_OFFSET_IN_LINK8`. |
| `--exp-cx-shift PX` | Adds to `ARIA_INTRINSICS["cx"]`. |
| `--exp-cy-shift PX` | Adds to `ARIA_INTRINSICS["cy"]`. |
| `--exp-fx-scale F` / `--exp-fy-scale F` | Multiplies the focal lengths. |
| `--exp-base-y-shift M` | Adds to `ROBOT_BASE_WORLD_POSITIONS["right"][1]`. |
| `--exp-no-refine` | Strips `--refined-extrinsic` from the forwarded argv. |

Every other CLI argument is forwarded verbatim to `kinematic_replay.py`.

## Once you have picked a winner

The launcher does *not* edit the source tree — it only patches the
in-memory constants for the duration of one run. To make a chosen fix
permanent, edit the corresponding constant in
`scripts/utils/constants.py`:

* H1 / H2 → `EE_WRIST_OFFSET_IN_LINK8`
* H3 / H4 → `ARIA_INTRINSICS`
