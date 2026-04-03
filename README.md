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

Now run the replay file. **Make sure that the h5 file paths and USD file paths are correct.**

```bash
# Run the replay script
python scripts/test_replay.py
```

## Inspection of Dataset

You can inspect the h5 files with `notebooks/inspect_h5.ipynb`. Feel free to modify or add any scripts. You should choose the kernel to be the conda environment you created.

## Project Structure

```
scripts/
├── test_setup.py              # Minimal robot init — load scene, set home pose, run sim loop
├── test_replay.py             # Replay H5 trajectories with IK solving + optional video recording
├── test_object_spawn.py       # Spawn objects on the table with physics
└── utils/
    ├── __init__.py            # Convenience re-exports (safe before SimulationApp)
    ├── constants.py           # All shared constants (paths, joint names, home poses, etc.)
    ├── app.py                 # SimulationApp creation + shared argparse flags
    ├── robot.py               # Articulation setup, DOF index resolution, collision control
    ├── rotation.py            # Quaternion math (pure numpy/scipy, no Isaac Sim deps)
    ├── ik.py                  # IK solver creation, solving, position-setter factory
    ├── h5_loader.py           # HDF5 trajectory loading
    ├── capture.py             # Viewport video capture pipeline
    ├── object.py              # Object spawning + physics (collision, gravity)
    └── generate_lula_description.py  # One-time utility to generate Lula YAML from URDF
```

### Scripts

| Script | Description |
|---|---|
| `test_setup.py` | Loads a USD scene, creates robot articulations, sets home joint values, and runs the simulation loop. Use this to verify the scene loads correctly. |
| `test_replay.py` | The main script. Loads H5 wrist pose + hand joint data, solves IK per frame, and replays the trajectory in Isaac Sim. Supports `--record` for video, `--no-collision` to disable table collision, `--camera aria` to view from the calibrated Aria camera. See `docs/test_replay.html` for detailed documentation. |
| `test_object_spawn.py` | Spawns a selected object (ball, duck, fish, grape, shovel) onto the table with rigid body physics. Use `--object`, `--position`, `--scale` to configure. |

### Utility Modules (`scripts/utils/`)

| Module | Isaac Sim deps? | Description |
|---|---|---|
| `constants.py` | No | Central source of truth for all constants: prim paths, joint names, DOF counts, file paths (USD, URDF, H5, objects), home poses, and IK configuration. All paths are anchored to `PROJECT_ROOT`. |
| `app.py` | Minimal | `add_common_args(parser)` adds `--headless`, `--fps`, `--mode` to any argparse parser. `create_app(args)` creates the SimulationApp. `resolve_usd_path(mode)` / `resolve_h5_path(mode)` map mode to file paths. |
| `robot.py` | Yes (deferred) | `setup_articulation()` creates a robot from a USD prim. `resolve_dof_indices()` maps joint names to DOF indices with alias and suffix fallback. `set_collision_enabled()` toggles collision on any prim. |
| `rotation.py` | No | Pure quaternion math: `rotation_matrix_to_wxyz()`, `quat_multiply()`, `tool_quat_to_urdf()` (H5 tool-frame → URDF convention via Rx(180°)), `detect_quaternion_order()` (auto-detect wxyz vs xyzw). |
| `ik.py` | Yes (deferred) | `create_ik_solver()` builds a Lula IK solver from URDF + descriptor. `solve_ik_for_pose()` solves IK for a target pose. `make_ik_position_setter()` returns a per-frame closure with warm-start tracking. |
| `h5_loader.py` | No | `load_h5()` loads arm wrist poses and hand joint angles from HDF5 files. Supports both `observations/qpos_*` and `actions_*` key schemas, single and dual arm modes. |
| `camera.py` | Yes (deferred) | Camera setup from real-world calibration. Computes camera world pose from extrinsics + robot base transforms, creates a USD camera prim with intrinsics, and sets it as the active viewport. Supports Aria Gen 1 (extensible to OAK-D). |
| `capture.py` | Yes (deferred) | `setup_capture()` / `capture_frame_to_writer()` / `close_recorder()` handle viewport-to-MP4 recording via imageio. Handles multiple Kit buffer formats (numpy, memoryview, raw pointer, PyCapsule). |
| `object.py` | Yes (deferred) | `spawn_object()` loads an OBJ mesh into the scene and enables rigid body physics with convex hull collision. |

**Import ordering note:** Modules marked "Yes (deferred)" import Isaac Sim types and must be imported by scripts *after* `create_app()` returns. Modules marked "No" are safe to import at any time.
