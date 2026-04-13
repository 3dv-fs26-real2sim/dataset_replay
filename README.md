# dataset_replay

## Setup

Create a conda environment and install the dependencies. What I prefer to do is install Isaac Sim and all dependencies into a single conda environment. If you have a different installation method for Isaac Sim, you would need to link it somehow. For the dependencies in requirements.txt, I may have missed some out -- please add any additional dependencies needed, and install new dependencies as you go when you run into errors.

```bash
# Create and activate conda environment
conda create -n 3dv python=3.11
conda activate 3dv

# Install Isaac Sim (https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
pip install isaacsim[all,extscache]==5.1.0 --extra-index-url https://pypi.nvidia.com

# Install other dependencies
pip install -r requirements.txt
```

Download h5 files from Euler cluster. I chose object_in_bowl_processed_50hz/20250804_104715.h5 bag_groceries/20250829_180500.h5 as they were the smallest files. You can try other files too.

```bash
# Copy with scp
scp USERNAME@euler.ethz.ch:/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/object_in_bowl_processed_50hz/20250804_104715.h5 data/

scp USERNAME@euler.ethz.ch:/cluster/work/cvg/data/Egoverse/raw_timesynced_h5/bag_groceries/20250829_180500.h5 data/
```

Download object 6d pose trajectory npz files, they should be in the format (N, 4, 4) where N is the number of frames and (4, 4) is the transform from camera to object. It should also match the number of frames in the h5 file. We generate them using depth estimation + FoundationPose. Move them to `data/`. File paths have been hard-coded as experimental code but you can change them. 

Now run the replay file. **Make sure that the h5 file paths and USD file paths are correct.**

```bash
# ── Setup & verification ──────────────────────────────────────────────────
# Test home pose with duck object (defaults: --mode single --object duck)
python scripts/test_setup.py

# Test home pose without object
python scripts/test_setup.py --object none

# Dual arm setup with camera preview
python scripts/test_setup.py --mode dual --camera aria

# ── Kinematic replay (object follows trajectory exactly, no physics) ──────
# Default: single arm, duck object, aria camera
python scripts/kinematic_replay.py

# Record Isaac Sim viewport → outputs/20250804_104715_replay_qpos_duck.mp4
python scripts/kinematic_replay.py --record-sim

# Side-by-side comparison (Isaac Sim + H5 original)
python scripts/kinematic_replay.py --record-comparison

# Dual arm replay (no object)
python scripts/kinematic_replay.py --mode dual

# Without object spawning
python scripts/kinematic_replay.py --object none

# ── Dynamic replay (object follows physics with spring-damper tracking) ───
# Default: single arm, duck object, aria camera, physics-enabled
python scripts/dynamic_replay.py

# Record with default stiffness/damping
python scripts/dynamic_replay.py --record-sim

# Higher stiffness for tighter trajectory tracking
python scripts/dynamic_replay.py --stiffness 5000 --damping 500

# Override object mass
python scripts/dynamic_replay.py --object-mass 0.5

# ── Utilities ─────────────────────────────────────────────────────────────
# Extract original H5 camera video (no GPU needed)
python scripts/record_h5.py
```

## Inspection of Dataset

You can inspect the h5 files with `notebooks/inspect_h5.ipynb`. Feel free to modify or add any scripts. You should choose the kernel to be the conda environment you created.

## Project Structure

```text
scripts/
├── test_setup.py              # Scene setup: IK home pose + optional object spawn + camera preview
├── kinematic_replay.py        # Kinematic replay: object follows trajectory exactly (no physics)
├── dynamic_replay.py          # Dynamic replay: object follows physics with D6 spring-damper tracking
├── record_h5.py               # Extract H5 camera images to MP4 (no GPU needed)
├── calculate_table_depth.py   # Compute camera→table-corner depths analytically (no GPU needed)
└── utils/
    ├── __init__.py            # Convenience re-exports (safe before SimulationApp)
    ├── constants.py           # All shared constants (paths, joint names, home poses, etc.)
    ├── app.py                 # SimulationApp creation + shared argparse flags
    ├── robot.py               # Articulation setup, DOF index resolution
    ├── rotation.py            # Quaternion math (pure numpy/scipy, no Isaac Sim deps)
    ├── ik.py                  # IK solver creation, solving, position-setter factory
    ├── h5_loader.py           # HDF5 trajectory loading
    ├── poses.py               # 6D pose trajectory loader, frame transforms, pose averaging
    ├── camera.py              # Camera setup from calibration extrinsics/intrinsics
    ├── capture.py             # Viewport video capture + side-by-side comparison pipeline
    ├── viewport.py            # Shared omni.kit.viewport.utility import helper
    ├── object.py              # Object spawning, pose updates, D6 tracking joints, collision filtering
    └── generate_lula_description.py  # One-time utility to generate Lula YAML from URDF
```

### Scripts

| Script | Description |
| --- | --- |
| `test_setup.py` | Loads a USD scene, computes IK home arm joints, optionally spawns an object, sets the home pose, and holds it. Defaults to `--mode single --object duck`. Supports `--camera` to preview calibrated viewpoints and `--object`/`--position`/`--scale` for object spawning. Runs until the window is closed. Use this to verify the scene and home pose look correct before running replay. |
| `kinematic_replay.py` | Kinematic replay. Defaults to `--mode single --camera aria --object duck`. Loads H5 wrist pose + hand joint data, solves IK per frame, and replays the trajectory in Isaac Sim. The object is spawned as a kinematic rigid body that follows the 6D pose trajectory exactly (no physics). Supports `--record-sim` for video, `--record-comparison` for side-by-side. See `docs/kinematic_replay.html` for detailed documentation. |
| `dynamic_replay.py` | Dynamic replay. Same defaults as kinematic replay (`--mode single --camera aria --object duck`). The object is a dynamic rigid body that follows physics (gravity, collisions). A kinematic anchor follows the trajectory, and a D6 joint with spring-damper drives pulls the object toward it. Tune with `--stiffness` (default 500), `--damping` (default 100), and `--object-mass` (default 0.1 kg). See `docs/dynamic_replay_strategy.html` for design rationale. |
| `record_h5.py` | Standalone script to extract original camera images from H5 files to MP4. No Isaac Sim or GPU required. Supports `--h5-camera` (`aria`, `oakd`), `--mode`, `--fps`, `--h5-path`. |
| `calculate_table_depth.py` | Computes camera-to-table-corner depths analytically using calibration data. No Isaac Sim required. |

### Utility Modules (`scripts/utils/`)

| Module | Isaac Sim deps? | Description |
| --- | --- | --- |
| `constants.py` | No | Central source of truth for all constants: prim paths, joint names, DOF counts, file paths (USD, URDF, H5, objects), home poses, IK configuration, camera calibration (Aria extrinsics/intrinsics), object pose trajectory paths, and per-arm configuration (`ARM_CONFIGS`). All paths are anchored to `PROJECT_ROOT`. |
| `app.py` | Minimal | `add_common_args(parser)` adds `--headless`, `--fps`, `--mode` to any argparse parser. `create_app(args)` creates the SimulationApp. `resolve_usd_path(mode)` / `resolve_h5_path(mode)` map mode to file paths. |
| `robot.py` | Yes (deferred) | `setup_articulation()` creates a robot from a USD prim. `resolve_dof_indices()` maps joint names to DOF indices with alias (`panda_joint` ↔ `fer_joint`) and suffix fallback. `print_dof_info()` prints DOF names for debugging. `add_articulations(world, mode)` registers one or two robot articulations based on mode. `setup_arms_ik(arms)` creates IK solvers, resolves DOFs, sets home poses, and builds per-frame position setter closures for each arm. |
| `rotation.py` | No | Pure quaternion math: `rotation_matrix_to_wxyz()`, `wxyz_to_rotation_matrix()`, `quat_multiply()`, `tool_quat_to_urdf()` (H5 tool-frame → URDF convention via Rx(180°)), `detect_quaternion_order()` (auto-detect wxyz vs xyzw). |
| `ik.py` | Yes (deferred) | `create_ik_solver()` builds a Lula IK solver from URDF + descriptor. `solve_ik_for_pose()` solves IK for a target pose. `make_ik_position_setter()` returns a per-frame closure with warm-start tracking and EE wrist offset support. |
| `h5_loader.py` | No | `load_h5()` loads arm wrist poses and hand joint angles from HDF5 files. Supports both `observations/qpos_*` and `actions_*` key schemas, single and dual arm modes. Also provides `get_available_cameras()`, `get_h5_image_dims()`, and `open_h5_images()` for H5 image data access. |
| `poses.py` | No | `load_pose_trajectory()` loads (N, 4, 4) transforms from `.npz`. `transform_trajectory()` re-expresses trajectories in a different frame. `translation_matrix()` and `average_poses()` are shared pure-numpy utilities used by camera.py and calculate_table_depth.py. |
| `camera.py` | Yes (deferred) | Camera setup from real-world calibration. `compute_camera_world_pose()` computes camera world pose from extrinsics + robot base transforms. `create_camera_prim()` creates a USD camera with intrinsics. `set_viewport_camera()` sets the active viewport. Supports Aria Gen 1 (extensible to OAK-D). |
| `capture.py` | Yes (deferred) | Video capture pipeline: `setup_recording(args, h5_path, n_frames, video_suffix, ...)` is the high-level entry point that wires up recording from CLI flags. Underlying primitives: `setup_capture()` / `capture_frame_to_writer()` / `close_recorder()` for sim viewport recording (with deferred encoding for speed; supports memory-only mode when `output_path=None`), and `setup_sidebyside()` / `capture_sidebyside_frame()` / `close_sidebyside()` for side-by-side comparison videos. |
| `viewport.py` | Yes (deferred) | Shared `get_viewport_utility()` helper that imports and returns `omni.kit.viewport.utility`, auto-enabling the extension if needed. Used by camera.py and capture.py. |
| `object.py` | Yes (deferred) | `load_object_world_trajectory()` is the high-level entry point: resolves the .npz path, loads the camera-frame trajectory, transforms it to world frame, and warns on frame-count mismatch. `resolve_object_pose_path()` maps an object name to its .npz pose file. `spawn_object()` loads an OBJ mesh into the scene with rigid body physics (supports kinematic mode for trajectory replay). `set_object_world_pose()` updates pose from a 4x4 transform via `XformCommonAPI`. `create_d6_tracking_joint()` creates a kinematic anchor + D6 joint with spring-damper drives for dynamic replay. `filter_collision_pair()` disables collision between two prims. |

**Import ordering note:** Modules marked "Yes (deferred)" import Isaac Sim types and must be imported by scripts *after* `create_app()` returns. Modules marked "No" are safe to import at any time.

### Docs

Docs have been generated with the help of Claude to explain how things are done and brainstorm ideas. Look at them as reference to better understand how everything works. They may be incomplete or slightly out of date. Feel free to add your own documentations. 
