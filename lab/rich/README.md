# lab/rich/ — simulation-rich rollout capture

Note: Please add meshes of ORCA hand to assets/meshes/orcahand/ for contact-force measurement visualisastion to work. You can find them from [`orcahand_description`](https://github.com/orcahand/orcahand_description).

Re-runs a trained (or baseline) residual policy **once** and captures, in a
single rollout:

* **Multi-viewpoint render (MVR)** — the calibrated Aria POV + 4 free views
  (front / side / top / hero); the Aria view also carries depth, world-normals,
  and robot-grouped semantic segmentation.
* **Contact-force measurement (CFM)** — per-skin-pad contact force vs the
  manipulated object (11 pads), plus object 6-DoF + velocity and pad poses.

Then it renders an animated contact-force hand and tiles everything into
montages.

```bash
conda activate dataset_replay
python lab/rich/run_rich.py --task egoverse \
    --demo data/egoverse/demos/egoverse_duck_104715.npz \
    --checkpoint logs/rsl_rl/teleop_residual/<run>/model_<n>.pt \
    --headless --device cuda:0

# Deterministic baseline (no policy / checkpoint needed):
python lab/rich/run_rich.py --task egoverse \
    --demo data/egoverse/demos/egoverse_duck_104715.npz \
    --residual-scale 0.0 --headless
```

Outputs (under `--out`, default `outputs/rich/`):

```
outputs/rich/mvr/      per-viewpoint + Aria-modality videos, cam_params.npz
outputs/rich/cfm/      contacts.npz, state.npz, contact_hand.mp4
outputs/rich/montage/  mvr_views.mp4, mvr_modalities.mp4, master_montage.mp4
```

## How it connects

* The MVR cameras are built **independently of the env scene** by
  `mvr_capture.build_cameras` — the Aria POV is reconstructed from
  dataset_replay's `ARIA_EXTRINSICS_RIGHT` + `ARIA_INTRINSICS`
  (`utils/constants.py`) composed with the rig mount, so the training env does
  not need to carry a camera.
* Per-pad `ContactSensor`s are injected onto the env cfg at runtime
  (`rich_capture.inject_contact_sensors`) — no task file is edited.
* The contact-hand visualization loads the OrcaHand from
  `assets/urdf/orcahand_right_extended.urdf` (+ `assets/meshes/orcahand/`) via
  `yourdfpy`/`trimesh`, rendered offline with `pyrender` (EGL, headless-safe).

## Modules

| File | Role |
|---|---|
| `run_rich.py` | single entry: rollout → MVR+CFM capture → contact-hand + montage |
| `mvr_capture.py` | free + Aria cameras, Replicator annotators, per-stream frames |
| `rich_capture.py` | per-pad contact sensors, contact/state readout + save |
| `semantics.py` | semantic-segmentation labelling + colourization |
| `viz_common.py` | depth / normals colourization |
| `hand_geometry.py` | OrcaHand URDF FK → skin-pad / structure meshes |
| `viz_contact_hand.py` | pyrender animation of per-pad contact force on the hand |
| `montage_build.py` | tile the MVR views / modalities + master montage |

All Python deps (`trimesh`, `yourdfpy`, `pyrender`, `matplotlib`, `imageio`,
`opencv`) are in the `dataset_replay` env — no extra installs. Run `--headless` (the
capture forces `--enable_cameras`).
