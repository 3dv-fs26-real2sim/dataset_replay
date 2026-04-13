"""Compute camera→table-corner depths analytically (no Isaac Sim required).

Method A: pure transforms.

Uses the same calibration data and base→camera composition convention as
utils/camera.py, but in pure NumPy so it can run without SimulationApp.

Usage:
    python dataset_replay/scripts/calculate_table_depth.py [--camera aria] [--mode single]
"""

import argparse

import numpy as np

from utils.constants import CAMERA_CONFIGS
from utils.poses import translation_matrix, average_poses


# ── Scene constants (mirrored from pandaorca_single.usda / pandaorca_dual.usda) ──
# /World/Cube: translate=(0,0,0.5), scale=(1, 1.4, 1), local extent ±0.5.
# Top face is at world z = 0.5 + 0.5*1 = 1.0; x in ±0.5, y in ±0.7 (0.5*1.4).
CUBE_TRANSLATE = np.array([0.0, 0.0, 0.5])
CUBE_SCALE     = np.array([1.0, 1.4, 1.0])
CUBE_HALF      = np.array([0.5, 0.5, 0.5])

# Robot base translates in /World (from the USDA files; both arms have identity rotation).
BASE_POS = {
    "single": {
        "right": np.array([-0.262, -0.386, 1.0]),
    },
    "dual": {
        "right": np.array([-0.262, -0.386, 1.0]),
        "left":  np.array([-0.262,  0.386, 1.0]),
    },
}


def cube_top_corners() -> np.ndarray:
    """Return the four world-frame corners of the cube's top face (4, 3)."""
    hx, hy, _hz = CUBE_HALF * CUBE_SCALE
    z = CUBE_TRANSLATE[2] + CUBE_HALF[2] * CUBE_SCALE[2]
    cx, cy, _ = CUBE_TRANSLATE
    return np.array([
        [cx - hx, cy - hy, z],
        [cx + hx, cy - hy, z],
        [cx + hx, cy + hy, z],
        [cx - hx, cy + hy, z],
    ])



def compute_camera_world_pose(camera_name: str, mode: str) -> np.ndarray:
    """Reproduce utils.camera.compute_camera_world_pose without USD.

    Returns the 4x4 camera-in-world matrix in CV convention (+Z forward).
    """
    extrinsics = CAMERA_CONFIGS[camera_name]["extrinsics"]

    poses = []
    T_world_base_r = translation_matrix(BASE_POS[mode]["right"])
    poses.append(T_world_base_r @ extrinsics["right"])

    if mode == "dual":
        T_world_base_l = translation_matrix(BASE_POS["dual"]["left"])
        poses.append(T_world_base_l @ extrinsics["left"])

    return poses[0] if len(poses) == 1 else average_poses(poses)


def project_pinhole(p_cam: np.ndarray, K: dict):
    """Project a camera-frame point (CV convention) to pixel (u, v)."""
    X, Y, Z = p_cam
    if Z <= 0:
        return None
    u = K["fx"] * X / Z + K["cx"]
    v = K["fy"] * Y / Z + K["cy"]
    return u, v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", default="aria", choices=list(CAMERA_CONFIGS.keys()))
    parser.add_argument("--mode", default="single", choices=["single", "dual"])
    args = parser.parse_args()

    intrinsics = CAMERA_CONFIGS[args.camera]["intrinsics"]
    T_world_cam = compute_camera_world_pose(args.camera, args.mode)
    T_cam_world = np.linalg.inv(T_world_cam)

    cam_pos = T_world_cam[:3, 3]
    print(f"[cam] camera world position: [{cam_pos[0]:+.4f}, {cam_pos[1]:+.4f}, {cam_pos[2]:+.4f}]")
    print(f"[cam] intrinsics: {intrinsics['width']}x{intrinsics['height']}, "
          f"fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
          f"cx={intrinsics['cx']}, cy={intrinsics['cy']}")

    corners_w = cube_top_corners()
    corners_w_h = np.hstack([corners_w, np.ones((4, 1))])
    corners_c = (T_cam_world @ corners_w_h.T).T[:, :3]

    print()
    print(f"{'#':>2} {'world (x,y,z)':>26} {'cam (X,Y,Z)':>26} "
          f"{'depth_z':>9} {'range':>9} {'(u,v)':>20} {'in':>4}")
    print("-" * 100)
    W, H = intrinsics["width"], intrinsics["height"]
    for i, (pw, pc) in enumerate(zip(corners_w, corners_c)):
        depth_z = pc[2]
        rng = float(np.linalg.norm(pc))
        uv = project_pinhole(pc, intrinsics)
        if uv is None:
            uv_str = "(behind cam)"
            in_frame = "no"
        else:
            u, v = uv
            uv_str = f"({u:7.1f},{v:7.1f})"
            in_frame = "yes" if (0 <= u < W and 0 <= v < H) else "no"
        print(f"{i:>2} ({pw[0]:+.3f},{pw[1]:+.3f},{pw[2]:+.3f}) "
              f"({pc[0]:+.3f},{pc[1]:+.3f},{pc[2]:+.3f}) "
              f"{depth_z:>9.4f} {rng:>9.4f} {uv_str:>20} {in_frame:>4}")


if __name__ == "__main__":
    main()
