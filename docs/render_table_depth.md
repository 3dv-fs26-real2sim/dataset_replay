# render_table_depth.py

Renders a depth image of the table surface from the calibrated Aria camera viewpoint using Isaac Sim's depth annotator.

## What it does

1. Opens the USD scene (same as `test_setup.py`)
2. Sets up the Aria camera using calibrated extrinsics/intrinsics
3. Hides the robot arm(s) so only the table and ground plane are visible
4. Captures a depth image using `omni.replicator.core`'s `distance_to_image_plane` annotator
5. Saves the result as both a raw `.npz` and a colorized `.png`

The depth values are **Z-depth to the image plane** in metres (same convention as `calculate_table_depth.py`'s `depth_z`).

## Output

| File | Description |
|------|-------------|
| `outputs/table_depth.npz` | Raw float32 depth array, key `"depth"`, shape `(480, 640)` |
| `outputs/table_depth.png` | Colorized depth visualization (turbo colormap) |
| `outputs/table_depth_masked.npz` | Same as above but with `--mask` — non-table pixels set to 0 |
| `outputs/table_depth_masked.png` | Masked visualization — only table surface shown |

### Depth ranges (aria camera, single mode)

| Region | Depth range |
|--------|------------|
| Table surface | ~0.33 -- 0.77 m |
| Gap (nothing) | 0.77 -- 1.81 m |
| Ground plane | ~1.81 -- 6.23 m |

The default mask threshold of **1.0 m** sits in the natural gap between table and ground.

## Usage

```bash
# Basic — full depth map
python dataset_replay/scripts/render_table_depth.py --headless

# Masked — table-only pixels, rest set to 0
python dataset_replay/scripts/render_table_depth.py --headless --mask

# Custom threshold
python dataset_replay/scripts/render_table_depth.py --headless --mask --mask-threshold 0.9

# Dual-arm mode
python dataset_replay/scripts/render_table_depth.py --headless --mode dual
```

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--headless` | off | Run without GUI |
| `--camera` | `aria` | Camera calibration (`aria`, `aria_half`) |
| `--mode` | `single` | Arm mode (`single`, `dual`) |
| `--mask` | off | Keep only table pixels; set rest to 0 |
| `--mask-threshold` | `1.0` | Depth cutoff in metres for `--mask` |
| `--output-dir` | `outputs/` | Output directory |

## Loading the output

```python
import numpy as np

data = np.load("dataset_replay/outputs/table_depth.npz")
depth = data["depth"]  # (480, 640) float32, metres

# For masked version:
data = np.load("dataset_replay/outputs/table_depth_masked.npz")
depth = data["depth"]  # 0 = non-table pixel
table_mask = depth > 0
```

## How it works

The script reuses the same camera setup pipeline as the replay scripts (`utils/camera.py`):

1. The camera world pose is computed from robot base transforms and calibrated extrinsics: `T_world_cam = T_world_base @ T_base_from_cam`
2. A `UsdGeom.Camera` prim is created with intrinsics mapped to USD focal length / aperture
3. Robot prims are hidden via `UsdGeom.Imageable.MakeInvisible()` (transforms remain valid for camera positioning)
4. A Replicator render product is created at the camera's native resolution (640x480) and the `distance_to_image_plane` annotator is attached to capture Z-depth

## Related scripts

- `calculate_table_depth.py` — Analytical depth at table corners (no Isaac Sim required)
- `test_setup.py` — Interactive scene preview with robot and objects visible
