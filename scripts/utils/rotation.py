"""Quaternion and rotation utilities.

Pure math — only numpy and scipy. No Isaac Sim dependencies.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .constants import Q_TOOL_TO_URDF


def rotation_matrix_to_wxyz(rot_matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a wxyz quaternion."""
    q_xyzw = Rotation.from_matrix(rot_matrix).as_quat()  # scipy returns xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def wxyz_to_rotation_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy takes xyzw


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton quaternion product q1*q2, both in wxyz convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def tool_quat_to_urdf(q_tool_wxyz: np.ndarray) -> np.ndarray:
    """Convert tool-convention quaternion (identity=down) to URDF convention."""
    return quat_multiply(Q_TOOL_TO_URDF, q_tool_wxyz)


def detect_quaternion_order(arm_data: np.ndarray, label: str) -> np.ndarray:
    """Detect whether quaternion columns 3:7 are wxyz or xyzw, reorder to wxyz if needed.

    Heuristic: for small rotations near identity, the scalar part w is close to 1.0
    and dominates the other components. Whichever column (3 or 6) has the higher mean
    absolute value is likely w.
    """
    w_if_wxyz = np.mean(np.abs(arm_data[:, 3]))
    w_if_xyzw = np.mean(np.abs(arm_data[:, 6]))

    print(f"[quat] {label}: mean|col3|={w_if_wxyz:.4f}  mean|col6|={w_if_xyzw:.4f}")

    if w_if_xyzw > w_if_wxyz:
        print(f"[quat] {label}: detected xyzw ordering. Reordering to wxyz.")
        reordered = arm_data.copy()
        reordered[:, 3] = arm_data[:, 6]  # w
        reordered[:, 4] = arm_data[:, 3]  # x
        reordered[:, 5] = arm_data[:, 4]  # y
        reordered[:, 6] = arm_data[:, 5]  # z
        return reordered

    print(f"[quat] {label}: appears to be wxyz ordering. Using as-is.")
    return arm_data
